from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import (
    scanner_agent,
    pick_agent,
    preprocessor_agent,
    neural_network_agent,
    rag_agent,
    ensemble_agent,
    score_agent,
    planning_agent,
    messaging_agent
)


# Shared information available to agents
class DealState(TypedDict, total=False):
    deals: list[dict]
    current: dict
    description: str
    dnn_price: float
    rag_price: float
    estimate: float
    similar: list[str]
    results: list[dict]
    best: dict | None
    message: str


# Decide what to do after scanning
def after_scan(state):
    if state["deals"]:
        return "pick"

    return "plan"


# Decide whether to process another deal
def after_score(state):
    if len(state["results"]) < len(state["deals"]):
        return "pick"

    return "plan"


# Create graph
builder = StateGraph(DealState)


# Register agents as graph nodes
nodes = [
    ("scan", scanner_agent),
    ("pick", pick_agent),
    ("preprocess", preprocessor_agent),
    ("dnn", neural_network_agent),
    ("rag", rag_agent),
    ("ensemble", ensemble_agent),
    ("score", score_agent),
    ("plan", planning_agent),
    ("message", messaging_agent)
]

for name, function in nodes:
    builder.add_node(name, function)


# Connect nodes
builder.add_edge(START, "scan")

builder.add_conditional_edges(
    "scan",
    after_scan
)

builder.add_edge("pick", "preprocess")
builder.add_edge("preprocess", "dnn")
builder.add_edge("dnn", "rag")
builder.add_edge("rag", "ensemble")
builder.add_edge("ensemble", "score")

builder.add_conditional_edges(
    "score",
    after_score
)

builder.add_edge("plan", "message")
builder.add_edge("message", END)


# Compile workflow
app = builder.compile()