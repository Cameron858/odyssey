from typing import Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str
