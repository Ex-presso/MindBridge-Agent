from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    provider: str = "gemini"


class ChatResponse(BaseModel):
    message: str
    
