# common.py
"""
OpenAI compatible response schemas.
"""

from pydantic import BaseModel, Field
from typing import Literal
import time, uuid

# --- common response ---

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


# class Usage(BaseModel):
#     prompt_tokens: int = 0
#     completion_tokens: int = 0
#     total_tokens: int = 0

class ChatCompletionRequest(BaseModel):
    model: Literal["openai", "gemini"] = "gemini"
    messages: list[ChatMessage]
    temperature: float | None = 0.3

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: Literal["openai", "gemini"] = "gemini"
    choices: list[Choice] 
    #usage: Optional[Usage] = None


def make_chat_response(model: str, content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=ChatMessage(role="assistant", content=content))]
    )


# --- streaming resposne ---

class Delta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChoiceChunk(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: str | None = None

class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "gemini-2.5-flash"
    choices: list[ChoiceChunk] 


def make_chunk(model: str, text_piece: str = "", end: bool = False) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        model=model,
        choices=[ChoiceChunk(
            delta=Delta(content=None if end else text_piece),
            finish_reason="stop" if end else None
        )]
    )