from typing import Literal, Optional

from pydantic import BaseModel, Field


class SessionChatRequest(BaseModel):
    message: str = Field(max_length=4000)
    conversation_id: str | None = None
    model: str = "deepseek-v4-flash"
    provider: str = "openai_compatible"
    stream: bool = True


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    model: str | None
    provider: str | None
    created_at: str
    updated_at: str


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse]


RoleLiteral = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: RoleLiteral
    content: str
    emotion: str | None = None  # Optional


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = True
    provider: Optional[str] = Field(
        default=None, description="Optional provider override."
    )


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]


class ChatCompletionChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[Literal["stop"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
