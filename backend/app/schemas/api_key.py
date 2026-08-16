from pydantic import BaseModel, Field

VALID_PROVIDERS = {"openai", "anthropic", "google_genai", "openai_compatible", "anthropic_compatible"}


class ApiKeySaveRequest(BaseModel):
    provider: str = Field(description="One of: openai, anthropic, google_genai, openai_compatible, anthropic_compatible")
    api_key: str = Field(min_length=1, max_length=500)
    base_url: str | None = Field(default=None, max_length=500)
    model_id: str | None = Field(default=None, max_length=100, description="Model name for compatible endpoints")
    display_name: str | None = Field(default=None, max_length=100)


class ApiKeyResponse(BaseModel):
    provider: str
    api_key_masked: str  # only last 4 chars visible
    base_url: str | None
    model_id: str | None
    display_name: str | None
    configured: bool = True


class ProviderModelsResponse(BaseModel):
    provider: str
    configured: bool
    models: list[dict]  # [{"id": "gpt-4o", "name": "GPT-4o"}, ...]
