from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

from app.core.llm.provider import get_llm
from app.core.rag.vectore_store import VectorStore
import operator


class State(TypedDict):
    """State container for the LangGraph chat flow."""

    messages: Annotated[list[BaseMessage], operator.add]
    context: Annotated[list[str], operator.add]


class Agent:
    """Single-node LangGraph agent that calls an LLM and returns its reply."""

    def __init__(self, provider: str):
        self.llm = get_llm(provider)
        self.vectorstore = VectorStore()
        
        self.system_prompt = """
        You are a compassionate and knowledgeable mental health assistant modeled after a professional counselor.
        Your responses should be warm, empathetic, and supportive while remaining factual and safe.
        Use the retrieved context—which consists of counselor responses from real or synthetic mental health conversations—to inspire and guide your answer.
        Paraphrase or summarize relevant insights from the context rather than copying text verbatim.
        If the context does not contain enough relevant information, respond with "I don't know" and, if appropriate, gently encourage the user to seek professional support.
        Do not provide medical diagnoses, prescribe medication, or offer crisis intervention advice.
        """

        self.app = self._build_graph().compile()


    def _chat_node(self, state: State) -> State:
        last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)

        if last_user_msg is None:
            return {"messages": [HumanMessage(content="User message is empty")]}


        enhanced_messages = self._build_enhanced_messages(last_user_msg.content, state.get("context", []))
        
        sys = SystemMessage(content=self.system_prompt)

        response = self.llm.invoke([sys, enhanced_messages])
        return {"messages": [response]}
       
    
 
    def _retrieve_context(self, state: State) -> State:
        last_user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        query = last_user_msg.content if last_user_msg else ""

        self.vectorstore.load()

        results = self.vectorstore.get_retriever(k=3).invoke(query)

        return {"context": [r.page_content for r in results]}
    

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
            "The context above contains sample counselor responses and psychological insights related to similar topics. "
            "Write a supportive and empathetic reply to the user's message using ideas or patterns from the context where appropriate. "
            "Do not copy sentences directly. "
            "If the context is not relevant or insufficient, respond with 'I don't know' and, if suitable, gently suggest that the user seek professional guidance."
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
        result = self.app.invoke({"messages": [HumanMessage(content=message)]})
        return result["messages"][-1].content
