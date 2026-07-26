"""Wires the call-classifier pipeline together as a LangGraph graph.

transcribe -> classify is a fixed ETL pipeline (skip translation for
English calls). From kickoff_agent onward it's a ReAct-style tool-calling
agent (StateGraph + ToolNode + tools_condition) that decides for itself
which ticketing tool to call.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from . import nodes
from .config import get_llm
from .state import CallState
from .tools import TOOLS


def agent_node(state: CallState) -> dict:
    llm_with_tools = get_llm().bind_tools(TOOLS)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(CallState)

    graph.add_node("transcribe", nodes.transcribe_node)
    graph.add_node("translate", nodes.translate_node)
    graph.add_node("extract_entities", nodes.extract_entities_node)
    graph.add_node("classify", nodes.classify_node)
    graph.add_node("kickoff_agent", nodes.kickoff_agent_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "transcribe")
    graph.add_conditional_edges(
        "transcribe",
        nodes.route_after_transcribe,
        {"translate": "translate", "extract_entities": "extract_entities"},
    )
    graph.add_edge("translate", "extract_entities")
    graph.add_edge("extract_entities", "classify")
    graph.add_edge("classify", "kickoff_agent")
    graph.add_edge("kickoff_agent", "agent")

    # tools_condition inspects the last message: "tools" if the agent
    # produced tool calls, otherwise END.
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
