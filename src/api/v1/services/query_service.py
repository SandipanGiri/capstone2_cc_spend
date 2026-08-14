from src.api.v1.agents.agents import run_search_agent_stream, run_search_agent
from src.core.guardrails import guard_input, guard_output


# method for non streaming response
def query_documents(query: str, user_id: str):
    print(query)
    # input guardrail: toxicity
    guard_input(query)
    result = run_search_agent(query, user_id)
    if isinstance(result, dict) and result.get("response"):
        result["response"] = guard_output(result["response"])
    return result


# method for streaming response
async def query_documents_stream(query: str):
    # just return async generator
    return run_search_agent_stream(query)
