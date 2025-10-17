from pydantic import BaseModel

class Model(BaseModel):
    id: str
    object: str = "model"
    owned_by: str | None = None

class ModelResponse(BaseModel):
    object: str = "list"
    data: list[Model]