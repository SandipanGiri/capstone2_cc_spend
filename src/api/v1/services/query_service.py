
from fastapi import Request
from src.core.guardrails import guard_input, guard_output

# -----------------------------
# Non streaming
# -----------------------------


async def query_documents(request: Request, query: str, thread_id: str):

    print(query)

    try:

        # Input guardrail
        guard_input(query)

        # get graph from FastAPI app state
        rag_graph = request.app.state.rag_graph

        initial_state = {
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "response": {},
            "images": [],
            "evaluation": {},
            "is_good": False,
            "attempts": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}

        result = await rag_graph.ainvoke(initial_state, config=config)

        response = result.get("response", {})

        # Output guardrail
        if response.get("answer"):

            response["answer"] = guard_output(response["answer"])

        return response

    except Exception as e:

        print(f"Error in query_documents: {e}")

        raise


# -----------------------------
# Streaming
# -----------------------------


async def query_documents_stream(request: Request, query: str, thread_id: str):

    try:

        print(query)

        # Input guardrail
        guard_input(query)

        rag_graph = request.app.state.rag_graph

        initial_state = {
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "response": {},
            "images": [],
            "evaluation": {},
            "is_good": False,
            "attempts": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}

        async for event in rag_graph.astream_events(
            initial_state, config=config, version="v2"
        ):

            kind = event.get("event")

            # LLM token streaming

            if kind == "on_chat_model_stream":

                chunk = event["data"]["chunk"]

                content = chunk.content

                if content:

                    # output guardrail
                    content = guard_output(content)

                    yield {"content": content}

        # after stream completed get state

        final_state = rag_graph.get_state(config)

        values = final_state.values

        yield {
            "done": True,
            "sources": values.get("sources", []),
            "images": values.get("images", []),
            "policy_citations": values.get("policy_citations", ""),
            "page_no": values.get("page_no", ""),
            "document_name": values.get("document_name", ""),
        }

    except Exception as e:

        print(f"Streaming error: {e}")

        yield {"error": str(e)}
