from src.api.v1.states.rag_state import RAGState
from src.core.db import get_vector_store


def vector_search_node(state: RAGState):
    print("====== INSIDE vector_search_node: searching the vector db")
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(state["query"], k=20)
    print(
        "======= INSIDE vector_search_node: Searched the Vector DB - Retrieved Docs Count:",
        len(docs),
    )
    return {**state, "retrieved_docs": docs}


## add the other tools for fts and hybrid search
