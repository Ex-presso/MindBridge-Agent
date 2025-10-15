from fastapi import APIRouter, HTTPException

from app.schemas.conversation import ChatRequest, ChatResponse
from app.services.chat import run_chat

router = APIRouter()


@router.post("/completion", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        reply = run_chat(request.provider, request.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ChatResponse(message=reply)
