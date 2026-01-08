from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import Nodes

def build_graph(retriever):
    workflow = StateGraph(AgentState)
    nodes = Nodes(retriever)

    # Define the nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)

    # Build the edges (The Flow)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional logic: If docs are bad -> transform query. If good -> generate.
    workflow.add_conditional_edges(
        "grade_documents",
        lambda x: x["search_again"],
        {
            "yes": "transform_query",
            "no": "generate"
        }
    )
    
    workflow.add_edge("transform_query", "retrieve") # Loop back to search again!
    workflow.add_edge("generate", END)

    return workflow.compile()