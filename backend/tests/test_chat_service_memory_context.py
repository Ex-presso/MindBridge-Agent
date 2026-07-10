"""Chat orchestration forwards per-run privacy context to the Agent."""

import asyncio

from app.services import chat as chat_service


class _AgentSpy:
    def __init__(self) -> None:
        self.nonstream_call = None
        self.stream_call = None

    async def ainvoke(self, message, **kwargs):
        self.nonstream_call = (message, kwargs)
        return "reply"

    async def astream_tokens(self, message, **kwargs):
        self.stream_call = (message, kwargs)
        yield "token"


def test_session_chat_services_forward_user_consent_and_thread():
    async def scenario():
        agent = _AgentSpy()
        usage = {}

        reply = await chat_service.run_chat_session(
            agent,
            "hello",
            "thread-1",
            user_id="user-1",
            memory_enabled=True,
            usage_sink=usage,
        )
        streamed = [
            token
            async for token in chat_service.stream_chat_session(
                agent,
                "hello again",
                "thread-2",
                user_id="user-2",
                memory_enabled=False,
                usage_sink=usage,
            )
        ]

        assert reply == "reply"
        assert streamed == ["token"]
        assert agent.nonstream_call[1] == {
            "thread_id": "thread-1",
            "usage_sink": usage,
            "user_id": "user-1",
            "memory_enabled": True,
        }
        assert agent.stream_call[1] == {
            "thread_id": "thread-2",
            "usage_sink": usage,
            "user_id": "user-2",
            "memory_enabled": False,
        }

    asyncio.run(scenario())
