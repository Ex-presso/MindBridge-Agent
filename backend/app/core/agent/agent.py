from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from app.core.llm.provider import get_llm
from app.services.rag import get_retriever


def merge_context(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Aggregator that safely merges retrieved context snippets across nodes."""
    existing_list = list(existing) if existing else []
    if not new:
        return existing_list
    return [*existing_list, *new]


class State(TypedDict):
    """State container for the LangGraph chat flow."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: Annotated[list[str], merge_context]


class Agent:
    """Single-node LangGraph agent that calls an LLM and returns its reply."""

    def __init__(self, provider: str):
        self.llm = get_llm(provider)
        
        self.system_prompt = """
        You are a compassionate and knowledgeable mental health assistant modeled after a professional counselor.
        Your responses should be warm, empathetic, and supportive while remaining factual and safe.
        Use the retrieved context—which consists of counselor responses from real or synthetic mental health conversations—to inspire and guide your answer.
        Paraphrase or summarize relevant insights from the context rather than copying text verbatim.
        When the retrieved context is sparse or missing, offer a gentle introduction, validate the user's feelings, and invite them to share more.
        Do not provide medical diagnoses, prescribe medication, or offer crisis intervention advice.
        """

        self.app = self._build_graph().compile()


    def _chat_node(self, state: State) -> State:
        last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)

        if last_user_msg is None:
            fallback = AIMessage(
                content=(
                    "Hi, I'm your mental health companion. I'm here to listen and support you—"
                    "feel free to share what's on your mind whenever you're ready."
                )
            )
            return State(messages=[fallback], context=state.get("context", []))

        # Extract message content safely (handle both str and list types)
        content = last_user_msg.content
        if isinstance(content, list):
            # If content is a list, extract text content
            content = " ".join(str(item) if isinstance(item, str) else str(item.get("text", "")) for item in content)
        
        # Use the current context from this retrieval cycle
        current_context = state.get("context", [])
        enhanced_messages = self._build_enhanced_messages(content, current_context)
        
        sys = SystemMessage(content=self.system_prompt)

        response = self.llm.invoke([sys, enhanced_messages])
        return State(messages=[response], context=state.get("context", []))
       
    

    def _retrieve_context(self, state: State) -> State:
        last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        
        if not last_user_msg:
            return State(messages=[], context=[])
        
        # Extract content safely (handle both str and list types)
        content = last_user_msg.content
        if isinstance(content, list):
            content = " ".join(str(item) if isinstance(item, str) else str(item.get("text", "")) for item in content)
        
        query: str = content if content else ""
        
        retriever = get_retriever(k=3)
        results = retriever.invoke(query) if query else []

        return State(messages=[], context=[r.page_content for r in results])
    

    def _build_enhanced_messages(self, user_message: str, contexts: list[str]) -> HumanMessage:
        ctx_block = ""
        if contexts:
            ctx_block = (
                "Below are example counselor responses retrieved from similar mental health conversations:\n"
                + "\n---\n".join(contexts)
                + "\n\n"
            )

        prompt = (
            f"{ctx_block}"
            f"User message:\n{user_message}\n\n"
            "Craft a supportive, empathetic reply that validates the user's experience. "
            "Reference relevant ideas from the retrieved examples when they exist, phrasing them in your own words. "
            "When examples are unavailable or do not apply, introduce yourself warmly, encourage the user to share more, and offer gentle, non-clinical guidance."
        )

        return HumanMessage(content=prompt)
    

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(State)
        graph.add_node("chat", self._chat_node)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_edge(START, "retrieve_context")
        graph.add_edge("retrieve_context", "chat")
        graph.add_edge("chat", END)
        return graph

    def invoke(self, message: str) -> str:
        result = self.app.invoke({"messages": [HumanMessage(content=message)], "context": []})
        return result["messages"][-1].content
