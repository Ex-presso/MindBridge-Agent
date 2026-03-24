from collections.abc import AsyncGenerator, Sequence
import logging
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from app.core.llm.provider import get_llm
from app.services.rag import get_retriever

logger = logging.getLogger(__name__)


class State(TypedDict):
    """State container for the LangGraph chat flow."""

    messages: Annotated[list[BaseMessage], add_messages]
    tool_iterations: int


class Agent:
    """LangGraph-driven mental health agent with conditional RAG tool usage."""

    def __init__(self, provider: str, checkpointer=None):
        base_llm = get_llm(provider)

        self.system_prompt = """
        You are a compassionate mental health assistant practicing Rogerian (person-centered) therapy principles.

        Core approach:
        - Show unconditional positive regard and genuine empathy
        - Reflect users' feelings and thoughts to help them feel heard and understood
        - Be non-directive: explore rather than advise, validate rather than solve
        - Respond naturally and conversationally, adapting your language to each unique situation

        Your responses must stay focused on mental health support, emotional well-being, coping strategies, and related counseling topics.
        When the user asks about subjects outside mental health, gently decline and redirect to their emotional well-being.

        Use retrieved counselor-style examples (if available) to guide your reply. Paraphrase insights rather than copying them verbatim.
        Never provide medical diagnoses, prescribe medication, or offer crisis intervention advice.
        """

        self._rag_top_k = 3
        self._tools = self._build_tools()
        self.llm = base_llm.bind_tools(list(self._tools.values()))
        self._system_message = SystemMessage(content=self.system_prompt.strip())
        self._max_tool_calls = 3
        self._redirect_instruction = (
            "If the user's request is not related to mental health, emotions, stress, coping, or counseling, "
            "respond briefly to explain that you can only help with mental or emotional well-being and invite them "
            "to share how they are feeling instead of answering the unrelated question."
        )

        self.app = self._build_graph().compile(checkpointer=checkpointer)

    def _build_tools(self) -> dict[str, StructuredTool]:
        """Create the tool set exposed to the LLM."""

        async def fetch_mental_health_examples(query: str) -> str:
            logger.info("RAG tool invoked with raw query=%r", query)
            normalized_query = self._normalize_content(query).strip()
            if not normalized_query:
                logger.info("RAG tool received empty query after normalization.")
                return "No relevant counselor examples found because the query was empty."

            retriever = await get_retriever(k=self._rag_top_k)
            logger.info(
                "RAG retriever invoked with top_k=%s; normalized_query=%r",
                self._rag_top_k,
                normalized_query,
            )
            results = await retriever.ainvoke(normalized_query)

            if not results:
                logger.info("RAG returned zero documents.")
                return "No similar counseling examples were retrieved."

            snippets: list[str] = []
            for idx, doc in enumerate(results, start=1):
                content = doc.page_content.strip()
                if content:
                    snippets.append(f"[Example {idx}]\n{content}")

            if not snippets:
                logger.info("RAG results contained no non-empty snippets.")
                return "No similar counseling examples were retrieved."

            logger.info("RAG tool assembled %s snippets.", len(snippets))
            return "\n\n".join(snippets)

        description = (
            "Retrieve counselor-style examples from the mental health knowledge base. "
            "Call this tool only when the user is asking about emotions, coping strategies, or other mental health topics."
        )

        rag_tool = StructuredTool.from_function(
            coroutine=fetch_mental_health_examples,
            name="fetch_mental_health_examples",
            description=description,
        )

        return {rag_tool.name: rag_tool}

    async def _chat_node(self, state: State) -> State:
        messages = list(state["messages"])
        has_user_message = any(isinstance(msg, HumanMessage) for msg in messages)

        if not has_user_message:
            fallback = AIMessage(
                content=(
                    "Hi, I'm your mental health companion. I'm here to listen and support you—"
                    "feel free to share what's on your mind whenever you're ready."
                )
            )
            return cast(
                State,
                {
                    "messages": [fallback],
                    "tool_iterations": state.get("tool_iterations", 0),
                },
            )

        last_user_index = next(
            (idx for idx in range(len(messages) - 1, -1, -1) if isinstance(messages[idx], HumanMessage)),
            None,
        )
        if last_user_index is None:
            logger.debug("No human message found despite guard; returning fallback.")
            return cast(
                State,
                {
                    "messages": [AIMessage(content="How are you feeling today?")],
                    "tool_iterations": state.get("tool_iterations", 0),
                },
            )

        # Reset tool iterations when this is a fresh user turn
        # (last message is HumanMessage = new user input just added by checkpointer)
        last_msg = messages[-1] if messages else None
        if isinstance(last_msg, HumanMessage):
            tool_iterations = 0
        else:
            tool_iterations = state.get("tool_iterations", 0)

        last_user_msg = cast(HumanMessage, messages[last_user_index])
        normalized_user = self._normalize_content(last_user_msg.content)
        is_related = self._is_mental_health_related(normalized_user)
        augmented_user = HumanMessage(
            content=self._augment_user_message(normalized_user, is_related=is_related)
        )

        working_messages = list(messages)
        working_messages[last_user_index] = augmented_user

        llm_input: list[BaseMessage] = [self._system_message, *working_messages]

        response = await self.llm.ainvoke(llm_input)
        return cast(
            State,
            {
                "messages": [response],
                "tool_iterations": tool_iterations,
            },
        )

    async def _tool_node(self, state: State) -> State:
        messages = list(state["messages"])
        last_ai_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage) and msg.tool_calls),
            None,
        )

        if last_ai_message is None:
            logger.info("Tool node skipped: no pending tool calls.")
            return cast(
                State,
                {
                    "messages": [],
                    "tool_iterations": state.get("tool_iterations", 0),
                },
            )

        current_iterations = state.get("tool_iterations", 0)
        if current_iterations >= self._max_tool_calls:
            logger.warning(
                "Tool invocation limit reached (%s). Returning error message.",
                self._max_tool_calls,
            )
            limit_message = ToolMessage(
                content=(
                    "Tool invocation limit reached; proceeding without additional retrieval."
                ),
                tool_call_id="fetch_mental_health_examples_limit",
                status="error",
            )
            return cast(
                State,
                {
                    "messages": [limit_message],
                    "tool_iterations": current_iterations,
                },
            )

        tool_messages: list[ToolMessage] = []

        for call in last_ai_message.tool_calls:
            name, args, call_id = self._unwrap_tool_call(call)
            tool = self._tools.get(name)

            if tool is None:
                logger.warning("Requested tool '%s' is not registered.", name)
                error_message = ToolMessage(
                    content=f"Requested tool '{name}' is not available.",
                    tool_call_id=call_id,
                    status="error",
                )
                tool_messages.append(error_message)
                continue

            if "query" not in args:
                args["query"] = ""

            try:
                logger.info(
                    "Invoking tool '%s' with args=%s (call_id=%s)",
                    name,
                    args,
                    call_id,
                )
                output = await tool.ainvoke(args)
                logger.info("Tool '%s' completed successfully (call_id=%s)", name, call_id)
            except Exception as exc:
                logger.exception("Tool '%s' failed during execution (call_id=%s)", name, call_id)
                error_message = ToolMessage(
                    content=f"Tool '{name}' failed with error: {exc}",
                    tool_call_id=call_id,
                    status="error",
                )
                tool_messages.append(error_message)
                continue

            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call_id))

        return cast(
            State,
            {
                "messages": [*tool_messages],
                "tool_iterations": current_iterations + 1,
            },
        )

    @staticmethod
    def _unwrap_tool_call(call: ToolCall | dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        """Extract the tool name, args, and id from a tool call payload."""
        if isinstance(call, dict):
            name = cast(str, call.get("name", ""))
            args_payload = call.get("args", {})
            call_id = call.get("id")
        else:
            name = cast(str, getattr(call, "name", ""))
            args_payload = getattr(call, "args", {})
            call_id = getattr(call, "id", None)

        if isinstance(args_payload, dict):
            args_dict = dict(args_payload)
        elif args_payload is None:
            args_dict = {}
        else:
            args_dict = {"query": str(args_payload)}

        call_id_str = str(call_id) if call_id not in (None, "") else name or "rag_tool_call"
        return name, args_dict, call_id_str

    def _route_from_chat(self, state: State) -> str:
        """Determine the next step after the LLM response."""
        latest = state["messages"][-1] if state["messages"] else None
        current_iterations = state.get("tool_iterations", 0)
        if isinstance(latest, AIMessage) and latest.tool_calls:
            if current_iterations >= self._max_tool_calls:
                logger.warning(
                    "Routing decision: tool call requested but limit %s reached; sending to tool node for error.",
                    self._max_tool_calls,
                )
            else:
                logger.info("Routing decision: tool call detected, moving to tool node.")
            return "use_tool"
        logger.info("Routing decision: no tool call, finishing response.")
        return "finish"

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    value = item.get("text")
                    if isinstance(value, str):
                        parts.append(value)
            return " ".join(parts)
        return str(content)

    def _augment_user_message(self, user_message: str, *, is_related: bool) -> str:
        """
        Augment user message with instructions that guide the LLM to respond
        in a Rogerian (person-centered) therapy style.
        """
        guidance = (
            "Response guidelines for this message:\n"
            "- Start by reflecting what you sense in the user's words (e.g., 'It sounds like...', "
            "'What I'm hearing is...'), but express this naturally—don't limit yourself to these exact phrases.\n"
            "- Reference retrieved counselor examples when available, weaving them naturally into your response.\n"
            f"- {self._redirect_instruction}\n"
            "- Keep your reply warm, conversational, and free of clinical jargon.\n"
        )

        if not is_related:
            guidance += (
                "- Note: Initial keyword scan suggests this may not be directly about mental health, "
                "but use your own judgment to assess if there are underlying emotional concerns.\n"
                "If the user's request is not related to mental health, "
                "gently acknowledge it, then invite them to share how they are feeling instead.\n"
            )

        return f"{guidance}\n\nUser: {user_message}"

    @staticmethod
    def _is_mental_health_related(message: str) -> bool:
        lowered = message.lower()
        keywords = [
            "anxiety", "anxious", "stress", "stressed", "depress", "depressed", "lonely", "loneliness",
            "panic", "fear", "sad", "overwhelmed", "therapy", "counsel", "mental", "emotion",
            "feel", "cope", "coping", "support", "burnout", "grief", "trauma",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(State)
        graph.add_node("chat", self._chat_node)
        graph.add_node("use_tool", self._tool_node)
        graph.add_edge(START, "chat")
        graph.add_conditional_edges(
            "chat",
            self._route_from_chat,
            {
                "use_tool": "use_tool",
                "finish": END,
            },
        )
        graph.add_edge("use_tool", "chat")
        return graph

    def invoke(self, messages: Sequence[BaseMessage]) -> str:
        """
        Synchronous invoke for backward compatibility.
        Invoke the agent with the system prompt and messages.
        """
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self.ainvoke_legacy(messages))

    async def ainvoke_legacy(self, messages: Sequence[BaseMessage]) -> str:
        """Stateless invocation (no checkpointer) — takes full message history."""
        state: State = {"messages": list(messages), "tool_iterations": 0}
        result = await self.app.ainvoke(state)
        ai_msg = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        if ai_msg is None:
            raise RuntimeError("Agent produced no assistant message.")
        return str(ai_msg.content)

    async def ainvoke(self, new_message: HumanMessage, thread_id: str) -> str:
        """Invoke with checkpointer — only pass the new message, history is in checkpoint."""
        config = {"configurable": {"thread_id": thread_id}}
        result = await self.app.ainvoke({"messages": [new_message]}, config=config)
        ai_msg = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        if ai_msg is None:
            raise RuntimeError("Agent produced no assistant message.")
        return str(ai_msg.content)

    async def astream_tokens(self, new_message: HumanMessage, thread_id: str) -> AsyncGenerator[str, None]:
        """Async generator that yields text tokens as they stream from the LLM."""
        config = {"configurable": {"thread_id": thread_id}}
        async for event in self.app.astream_events(
            {"messages": [new_message]},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield str(chunk.content)
