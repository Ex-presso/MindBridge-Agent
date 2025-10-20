from collections.abc import Sequence
import logging
from typing import Annotated, Any, TypedDict, cast
from typing import Sequence, Optional

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


# === NEW: Rogerian reflection helper ==========================================

def rogerian_reflection(user_message: str) -> str:
    """
    Reformulate the user's message in a reflective, non-directive, and empathetic tone.
    Inspired by Rogerian therapy and ELIZA.
    """
    reflections = [
        ("i feel", "It sounds like you feel"),
        ("i think", "It seems like you think"),
        ("i am", "You seem to be"),
        ("i’m", "You seem to be"),
        ("i was", "It seems you were"),
        ("i can’t", "You find it difficult to"),
        ("i want", "You wish to"),
        ("because", "What makes you feel that way about"),
        ("sometimes", "Sometimes it feels that way, doesn’t it"),
        ("maybe", "Perhaps you’re uncertain about"),
    ]

    lower_msg = user_message.lower()
    for trigger, reflection in reflections:
        if trigger in lower_msg:
            rest = user_message[lower_msg.index(trigger) + len(trigger):].strip()
            if rest:
                return f"{reflection} {rest}?"
            return f"{reflection}?"
    # fallback when no trigger found
    return f"Can you tell me more about that?"


# === Agent ====================================================================

class State(TypedDict):
    """State container for the LangGraph chat flow."""

    messages: Annotated[list[BaseMessage], add_messages]
    tool_iterations: int


class Agent:
    """LangGraph-driven mental health agent with conditional RAG tool usage."""

    def __init__(self, provider: str):
        base_llm = get_llm(provider)

        self.system_prompt = """
        You are a compassionate and knowledgeable mental health assistant modeled after a professional counselor.
        Your responses must stay focused on mental health support, emotional well-being, coping strategies, and related counseling topics.
        When the user asks about subjects outside mental health, gently decline to provide a detailed answer, explain that your purpose is to support their emotional well-being, and invite them to share how they are feeling instead.
        Use retrieved counselor-style examples (if available) to guide your reply. Paraphrase insights rather than copying them verbatim.
        Offer validation, warmth, and encouragement. Never provide medical diagnoses, prescribe medication, or offer crisis intervention advice.
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

        self.app = self._build_graph().compile()

    def _build_tools(self) -> dict[str, StructuredTool]:
        """Create the tool set exposed to the LLM."""

        def fetch_mental_health_examples(query: str) -> str:
            logger.info("RAG tool invoked with raw query=%r", query)
            normalized_query = self._normalize_content(query).strip()
            if not normalized_query:
                logger.info("RAG tool received empty query after normalization.")
                return "No relevant counselor examples found because the query was empty."

            retriever = get_retriever(k=self._rag_top_k)
            logger.info(
                "RAG retriever invoked with top_k=%s; normalized_query=%r",
                self._rag_top_k,
                normalized_query,
            )
            results = retriever.invoke(normalized_query)

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
            func=fetch_mental_health_examples,
            name="fetch_mental_health_examples",
            description=description,
        )

        return {rag_tool.name: rag_tool}

    def _chat_node(self, state: State) -> State:
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

        last_user_msg = cast(HumanMessage, messages[last_user_index])
        normalized_user = self._normalize_content(last_user_msg.content)
        is_related = self._is_mental_health_related(normalized_user)
        augmented_user = HumanMessage(
            content=self._augment_user_message(normalized_user, is_related=is_related)
        )

        working_messages = list(messages)
        working_messages[last_user_index] = augmented_user

        llm_input: list[BaseMessage] = [self._system_message, *working_messages]

        response = self.llm.invoke(llm_input)
        return cast(
            State,
            {
                "messages": [response],
                "tool_iterations": state.get("tool_iterations", 0),
            },
        )

    def _tool_node(self, state: State) -> State:
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
                output = tool.invoke(args)
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

    # === MODIFIED: includes Rogerian reflection ==================================
    def _augment_user_message(self, user_message: str, *, is_related: bool) -> str:
        reflection = rogerian_reflection(user_message)

        guidance = (
            "Instructions for the assistant:\n"
            "- Offer empathetic mental health support tailored to the user's emotions and concerns.\n"
            "- Reflect the user's feelings and thoughts in a Rogerian (non-directive) style.\n"
            "- Reference retrieved counselor examples when available, paraphrasing naturally.\n"
            f"- {self._redirect_instruction}\n"
            "- Keep the reply compassionate, brief, and free of clinical diagnoses or medication advice.\n"
        )

        if not is_related:
            guidance += (
                "- The latest user request appears unrelated to mental health. "
                "Gently decline the topic and invite them to share how they are feeling instead.\n"
            )

        return f"{guidance}\nUser message:\n{user_message}\n\nReflective cue for assistant:\n{reflection}"
    # ============================================================================

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

    # === MODIFIED: emotion-aware invoke ==========================================
    def invoke(self, messages: Sequence[BaseMessage], emotion: Optional[str] = None) -> str:
        """
        Invoke the agent with emotion-aware reflection.
        Adjusts system prompt and reflection cues based on the detected user emotion.
        """
        reflection_prompt = self._get_reflection_prompt(emotion)
        if emotion:
            system_msg = SystemMessage(
                content=(
                    f"You are a compassionate assistant. The user seems {emotion.lower()}. "
                    f"Adjust your response accordingly: {reflection_prompt}."
                )
            )
        else:
            system_msg = self._system_message

        state: State = {
            "messages": [system_msg, *messages],
            "tool_iterations": 0,
        }

        result = self.app.invoke(state)
        ai_message = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )
        if ai_message is None:
            raise RuntimeError("Agent did not produce an assistant message.")
        return cast(str, ai_message.content)

    def _get_reflection_prompt(self, emotion: Optional[str]) -> str:
        """Return response behavior based on emotion."""
        if not emotion:
            return "Respond naturally and kindly."
        mapping = {
            "sadness": "Be gentle and supportive, reflect empathy softly.",
            "joy": "Be enthusiastic but balanced, mirror positivity.",
            "anger": "Be calm, acknowledge frustration, and de-escalate softly.",
            "fear": "Be reassuring, emphasize safety and control.",
            "disgust": "Be understanding and neutral, avoid amplifying emotion.",
            "surprise": "Acknowledge curiosity and invite further sharing.",
            "neutral": "Respond naturally and helpfully.",
            "anxiety": "Be calming and comforting, reduce intensity in tone."
        }
        return mapping.get(emotion.lower(), "Respond naturally and kindly.")
