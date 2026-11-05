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
