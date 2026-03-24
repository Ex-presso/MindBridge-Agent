"""
MindBridge Agent - Gradio Frontend Interface

A modern chat interface for mental health counseling powered by LangGraph and RAG.
"""

import json
import os
import time
from typing import Generator

import gradio as gr
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
API_ENDPOINT = os.getenv("API_ENDPOINT", "/api/v1/chat/completions")
CHAT_API_URL = f"{API_BASE_URL}{API_ENDPOINT}"

MODELS = {
    "google_genai": ["gemini-2.5-flash"],
    "openai": ["gpt-5"],
}

DEFAULT_PROVIDER = "google_genai"
DEFAULT_MODEL = "gemini-2.5-flash"


def chat_with_agent(
    message: str,
    history: list[dict],
    provider: str,
    model: str,
) -> Generator[str, None, None]:
    """
    Send a message to the backend API and stream the response.

    Args:
        message: User's input message
        history: Conversation history in Gradio messages format (OpenAI-style dicts)
        provider: LLM provider (google_genai or openai)
        model: Model name to use

    Yields:
        Accumulated response text as it streams
    """
    # History is already in OpenAI format when type="messages"
    # Just copy it and add the current message
    messages = history.copy() if history else []
    
    # Add current message
    messages.append({"role": "user", "content": message})

    # Prepare API request
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "provider": provider,
    }

    try:
        with requests.post(
            CHAT_API_URL,
            json=payload,
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()

            accumulated_text = ""

            # Process streaming response
            for line in response.iter_lines():
                if not line:
                    continue

                line_text = line.decode("utf-8")

                # Skip empty lines and [DONE] marker
                if not line_text.strip() or line_text.strip() == "data: [DONE]":
                    continue

                # Parse SSE format: "data: {...}"
                if line_text.startswith("data: "):
                    json_str = line_text[6:]  # Remove "data: " prefix

                    try:
                        chunk_data = json.loads(json_str)
                        choices = chunk_data.get("choices", [])

                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")

                            if content:
                                accumulated_text += content
                                yield accumulated_text
                                time.sleep(0.005)

                    except json.JSONDecodeError:
                        continue

            if accumulated_text:
                yield accumulated_text
            else:
                yield "I apologize, but I didn't receive a response. Please try again."

    except requests.exceptions.ConnectionError:
        yield "❌ **Connection Error**: Unable to reach the backend API. Please ensure the backend server is running at `http://127.0.0.1:8080`."
    except requests.exceptions.Timeout:
        yield "⏱️ **Timeout Error**: The request took too long. Please try again with a shorter message."
    except requests.exceptions.RequestException as e:
        yield f"❌ **Error**: {str(e)}"
    except Exception as e:
        yield f"❌ **Unexpected Error**: {str(e)}"


def update_model_choices(provider: str) -> gr.Radio:
    """Update available models when provider changes."""
    models = MODELS.get(provider, MODELS[DEFAULT_PROVIDER])
    return gr.Radio(
        choices=models,
        value=models[0],
        label="Model",
    )


# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""


with gr.Blocks(
    fill_height=True,
    theme=gr.themes.Soft(),
    css=custom_css,
    title="MindBridge Agent",
) as demo:
    gr.Markdown("""
        # 🧠 MindBridge Agent - Your Mental Health Companion

        <div style='background: linear-gradient(90deg, #f6f8fa 0%, #e2eafc 100%); border-radius: 10px; padding: 16px 18px; font-size: 16px; margin-bottom: 10px;'>
        <b>🤖 What does MindBridge do?</b>
        <br><br>
        MindBridge is your AI-powered mental health companion practicing person-centered therapy! <br>
        I can help you with <b>
            💭 Emotional support, 
            🧘 Coping strategies,
            ⚡ Stress management,
            🌱 Self-understanding 
        </b> and provide empathetic guidance through your mental health journey. <br><br>
        <b>How to use MindBridge?</b><br><br>
        💬 Simply <b>share your thoughts and feelings</b> in natural language, and I'll listen with empathy and understanding. 
        <br><br>
        <b>⚠️ Important:</b> This is not a substitute for professional medical advice. In case of emergency, contact local crisis services!
        </div>
        """
    )

    with gr.Row():
        provider_radio = gr.Radio(
            choices=list(MODELS.keys()),
            value=DEFAULT_PROVIDER,
            label="Provider",
            scale=1,
        )
        model_radio = gr.Radio(
            choices=MODELS[DEFAULT_PROVIDER],
            value=DEFAULT_MODEL,
            label="Model",
            scale=1,
        )
    

    chat = gr.ChatInterface(
        fn=chat_with_agent,
        additional_inputs=[
            provider_radio,
            model_radio,
        ],
        additional_inputs_accordion=gr.Accordion(label="Settings", open=False, visible=False),
        type="messages",
        chatbot=gr.Chatbot(
            height=600,
            show_copy_button=True,
            type="messages",
        ),
        textbox=gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
        ),
        submit_btn="Send",
        stop_btn="Stop",
        multimodal=False,
        save_history=True,
    )
    
    
    provider_radio.change(
        fn=update_model_choices,
        inputs=[provider_radio],
        outputs=[model_radio],
    )


if __name__ == "__main__":
    print("🚀 Starting MindBridge Agent Frontend...")
    print(f"📡 Backend API: {CHAT_API_URL}")
    print(f"🌐 Frontend will be available at: http://127.0.0.1:7860")
    print("\n⚠️  Make sure the backend server is running before using the interface!\n")

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        show_error=True,
    )
