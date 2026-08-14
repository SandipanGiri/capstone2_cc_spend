from src.api.v1.agents.agents import run_search_agent_stream, run_search_agent


# for non streaming response
def query_documents(query:str,thread_id:str):
    #query=request["query"]
    #print(query)
    #return run_search_agent(query)
    return run_search_agent(query,thread_id)
   


# method for streaming response
async def query_documents_stream(query: str):
    # just return async generator
    return run_search_agent_stream(query)
