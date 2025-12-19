from typing import List
import json
import json5
from langchain_core.messages import (
    AIMessage, 
    SystemMessage,
    ToolCall, 
    AnyMessage, 
    HumanMessage
)

def extract_fn(text: str) -> tuple[str, str]:
    """Extract function name and arguments from tool call text."""
    fn_name, fn_args = '', ''
    fn_name_s = '"name": "'
    fn_name_e = '", "'
    fn_args_s = '"arguments": '
    
    i = text.find(fn_name_s)
    k = text.find(fn_args_s)
    
    if i > 0:
        _text = text[i + len(fn_name_s):]
        j = _text.find(fn_name_e)
        if j > -1:
            fn_name = _text[:j]
    
    if k > 0:
        fn_args = text[k + len(fn_args_s):]
    
    fn_args = fn_args.strip()
    if len(fn_args) > 2:
        fn_args = fn_args[:-1]
    else:
        fn_args = ''
    
    return fn_name, fn_args


def postprocess_ai_message(
    ai_message: AIMessage,
) -> AIMessage:
    """
    Convert AIMessage with <tool_call> tags to proper LangChain message with tool calls and leave it in a list to integrate with MessagesState.
    Assumes all content is text (no multimodal).
    """
    tool_id = 1
    
    content: str = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
    
    # Handle <think> tags - skip tool call parsing inside thinking
    if '<think>' in content:
        if '</think>' not in content:
            # Incomplete thinking, add as regular message
            return ai_message
        
        # Split thinking from rest of content
        parts = content.split('</think>')
        content = parts[-1]
        
    
    # Find tool calls in content
    if '<tool_call>' not in content:
        # No tool calls, add as regular message
        return AIMessage(content=content.strip())
    
    # Split content by tool calls
    tool_call_list = content.split('<tool_call>')
    pre_text = tool_call_list[0].strip()
    tool_calls: List[ToolCall] = []
    
    # Process each tool call
    for txt in tool_call_list[1:]:
        if not txt.strip():
            continue
        
        # Handle incomplete tool calls (no closing tag)
        if '</tool_call>' not in txt:
            fn_name, fn_args = extract_fn(txt)
            if fn_name:
                tool_calls.append(
                    ToolCall(
                        name=fn_name,
                        args=json.loads(fn_args) if fn_args else {},
                        id=str(tool_id),
                    )
                )
                tool_id += 1
                # new_messages.append(AIMessage(content='', tool_calls=tool_calls))
            continue
        
        # Handle complete tool calls
        one_tool_call_txt = txt.split('</tool_call>')[0].strip()
        
        try:
            # Try to parse as JSON
            fn = json5.loads(one_tool_call_txt)
            if 'name' in fn and 'arguments' in fn:
                tool_calls.append(
                    ToolCall(
                        name=fn['name'],
                        args=fn['arguments'],
                        id=str(tool_id),
                    )
                )
                tool_id += 1
                # new_messages.append(AIMessage(content='', tool_calls=tool_calls))
        except Exception:
            # Fallback to manual extraction
            fn_name, fn_args = extract_fn(one_tool_call_txt)
            if fn_name:
                tool_calls.append(
                    ToolCall(
                        name=fn_name,
                        args=json.loads(fn_args) if fn_args else {},
                        id=str(tool_id),
                    )
                )
                tool_id += 1
                # new_messages.append(AIMessage(content='', tool_calls=tool_calls))
        
    if tool_calls:
        return AIMessage(content=pre_text, tool_calls=tool_calls)
    elif pre_text:
        return AIMessage(content=pre_text)
    else:
        return AIMessage(content=content)


def preprocess_messages(
    conversation: List[AnyMessage],
) -> List[AnyMessage]:
    """
    Convert LangChain messages with tool calls to plaintext format with <tool_call> tags.
    Converts ToolMessages to <tool_response> tags.
    Assumes all content is text (no multimodal).
    """
    new_conversation: List[AnyMessage] = []
    start_index: int = 0

    system_content: str = ""
    while start_index < len(conversation) and conversation[start_index].type == "system":
        system_content += str(conversation[start_index].content)
        start_index += 1

    if system_content:
        new_conversation.append(SystemMessage(content=system_content))

    for msg in conversation[start_index:]:
        # Pass through human conversation as-is
        if msg.type == "human":
            new_conversation.append(msg)
            continue
        # Handle AI conversation with tool calls
        elif msg.type == "ai":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            
            # Convert tool calls to plaintext format
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    fc = {
                        'name': tool_call['name'],
                        'arguments': tool_call['args']
                    }
                    fc_str = json.dumps(fc, ensure_ascii=False)
                    tool_call_text = f'<tool_call>\n{fc_str}\n</tool_call>'
                    
                    # Append to content
                    if content:
                        content += '\n' + tool_call_text
                    else:
                        content = tool_call_text
            
            # Merge consecutive AI conversation
            if new_conversation and new_conversation[-1].type == "ai":
                prev_content = str(new_conversation[-1].content)
                if prev_content and not prev_content.endswith('\n'):
                    prev_content += '\n'
                new_conversation[-1] = AIMessage(content=prev_content + content)
            else:
                new_conversation.append(AIMessage(content=content))
            continue
        # Handle tool conversation - convert to <tool_response> wrapped in HumanMessage
        elif msg.type == "tool":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            response_text = f'<tool_response>\n{content}\n</tool_response>'
            if new_conversation and new_conversation[-1].type == "human":
                prev_content = str(new_conversation[-1].content)
                prev_content += '\n' + response_text
                new_conversation[-1] = HumanMessage(content=prev_content)
            else:
                new_conversation.append(HumanMessage(content=response_text))
            continue
    
    return new_conversation
