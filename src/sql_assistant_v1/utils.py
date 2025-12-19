from typing import List

from langchain_core.messages import AnyMessage


def format_conversation(conversation: List[AnyMessage]) -> str:
    formatted_conversation = ""
    end_index = len(conversation) - 1 
    for ind in range(len(conversation) - 1, -1, -1):
        if conversation[ind].type == "human":
            end_index = ind
            break
    for message in conversation[:end_index]:
        if message.type == "human":
            formatted_conversation += f"Customer: {message.content}\n"
        elif message.type == "ai":
            formatted_conversation += f"Support Team: {message.content}\n"
    
    formatted_conversation += f"\nLatest Customer Message: {conversation[end_index].content}"
    return formatted_conversation