from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import settings

def get_llm(provider: str) -> BaseChatModel:
    provider = provider.lower()

    if provider == "openai":
        return ChatOpenAI(
            model=settings.OPENAI_MODEL, 
            temperature=settings.MODEL_TEMPERATURE, 
            api_key=settings.OPENAI_API_KEY
        )    
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=settings.MODEL_TEMPERATURE, 
            api_key=settings.GEMINI_API_KEY
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
