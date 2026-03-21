"""
LangChain 메시지 직렬화/역직렬화 유틸리티
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

def serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    """BaseMessage 리스트를 JSON 직렬화 가능한 dict 리스트로 변환"""
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"type": "ai", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            result.append({"type": "system", "content": msg.content})
    return result


def deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    """dict 리스트를 BaseMessage 리스트로 복원"""
    result = []
    for item in data:
        msg_type = item.get("type")
        content = item.get("content", "")
        if msg_type == "human":
            result.append(HumanMessage(content=content))
        elif msg_type == "ai":
            result.append(AIMessage(content=content))
        elif msg_type == "system":
            result.append(SystemMessage(content=content))
    return result
