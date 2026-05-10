from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    user_query: str
    chat_state: list
    memory_state: list


# class STTResponse(BaseModel):
#     type: Literal["partial", "final", "error", "closed"]
#     text: Optional[str] = None
#     timestamp: str = datetime.utcnow().isoformat()
#     code: Optional[int] = None
#     reason: Optional[str] = None


class STTResponse(BaseModel):
    type: Literal["partial", "final", "error", "closed"]
    text: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    code: Optional[int] = None
    reason: Optional[str] = None
