from fastapi import APIRouter
from app.schemas.model import Model, ModelResponse

router = APIRouter()

@router.get("/models", response_model=ModelResponse)
def list_models():
   
    return ModelResponse(
        data=[
            Model(id="gemini-2.5-flash", owned_by="google_genai"),
            Model(id="gpt-5", owned_by="openai"),
        ]
    )

@router.get("/models/{model_id}", response_model=Model)
def get_model(model_id: str):
    
    return Model(id=model_id)