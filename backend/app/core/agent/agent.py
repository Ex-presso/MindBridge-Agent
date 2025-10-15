from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from app.core.llm.provider import get_llm


class State(TypedDict):
    """State container for the LangGraph chat flow."""

    messages: Annotated[List[BaseMessage], add_messages]


class Agent:
    """Single-node LangGraph agent that calls an LLM and returns its reply."""

    def __init__(self, provider: str):
        self.llm = get_llm(provider)
        self.app = self._build_graph().compile()

    def _chat_node(self, state: State) -> State:
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(State)
        graph.add_node("chat", self._chat_node)
        graph.add_edge(START, "chat")
        graph.add_edge("chat", END)
        return graph

    def invoke(self, message: str) -> str:
        result = self.app.invoke({"messages": [HumanMessage(content=message)]})
        return result["messages"][-1].content
