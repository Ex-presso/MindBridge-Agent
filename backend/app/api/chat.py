from fastapi import APIRouter
from app.schemas.conversation import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/completion")
def chat(request: ChatRequest):
    return ChatResponse(message="Hello, World!")