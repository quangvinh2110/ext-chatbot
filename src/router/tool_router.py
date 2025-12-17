from typing import List, TypedDict
from functools import partial
import json
import json5
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool
from langchain_core.messages import (
    AIMessage, 
    ToolCall, 
    AnyMessage, 
    SystemMessage, 
    HumanMessage,
)

from ..utils.date import get_today_date_vi
from ..prompts.router import TOOL_ROUTING_PROMPT


class ToolRouterState(TypedDict):
    messages: List[AnyMessage]


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
    state: ToolRouterState,
    system_prompt: str,
) -> List[AnyMessage]:
    """
    Convert LangChain messages with tool calls to plaintext format with <tool_call> tags.
    Converts ToolMessages to <tool_response> tags.
    Assumes all content is text (no multimodal).
    """
    if not hasattr(state, "messages"):
        raise ValueError("messages not found in state")
    messages: List[AnyMessage] = state["messages"]
    new_messages: List[AnyMessage] = []

    if messages[0].type == "system":
        new_messages.append(messages[0])
    else:
        date_info = "Hôm nay là {date}.\n".format(date=get_today_date_vi())
        new_messages.append(SystemMessage(
            content=date_info + system_prompt
        ))
        messages = [SystemMessage(content=date_info + system_prompt)] + messages

    for msg in messages[1:]:
        # Pass through human messages as-is
        if msg.type == "human":
            new_messages.append(msg)
            continue
        # Handle AI messages with tool calls
        elif msg.type == "ai":
            # ignore if the content contains other type of data (e.g. images, voice, etc.)
            content = str(msg.content)
            
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
            
            # Merge consecutive AI messages
            if new_messages and new_messages[-1].type == "ai":
                # ignore if the content contains other type of data (e.g. images, voice, etc.)
                prev_content = str(new_messages[-1].content)
                if prev_content and not prev_content.endswith('\n'):
                    prev_content += '\n'
                new_messages[-1] = AIMessage(content=prev_content + content)
            else:
                new_messages.append(AIMessage(content=content))
            continue
        # Handle tool messages - convert to <tool_response> wrapped in HumanMessage
        elif msg.type == "tool":
            # ignore if the content contains other type of data (e.g. images, voice, etc.)
            content = str(msg.content)
            response_text = f'<tool_response>\n{content}\n</tool_response>'
            if new_messages and new_messages[-1].type == "human":
                # ignore if the content contains other type of data (e.g. images, voice, etc.)
                prev_content = str(new_messages[-1].content)
                prev_content += '\n' + response_text
                new_messages[-1] = HumanMessage(content=prev_content)
            else:
                new_messages.append(HumanMessage(content=response_text))
            continue
    
    return new_messages


class SQLAssistantInput(BaseModel):
    command: str = Field(description="Mệnh lệnh bằng Tiếng Việt")


@tool("call_sql_assistant", args_schema=SQLAssistantInput)
def call_sql_assistant(command: str) -> str:
    """Gọi trợ lý ảo SQL Assistant để thực hiện các câu truy vấn"""
    return ""


async def route(
    state: ToolRouterState,
    chat_model: BaseChatModel,
) -> ToolRouterState:
    """
    Route the message to the appropriate tool.
    """
    orchestrator_chain = (
        RunnableLambda(partial(
            preprocess_messages, 
            system_prompt=TOOL_ROUTING_PROMPT
        )) | chat_model.bind_tools([call_sql_assistant])
        | postprocess_ai_message
    )
    ai_messages = await orchestrator_chain.ainvoke(state["messages"])
    state["messages"].append(ai_messages)
    return state