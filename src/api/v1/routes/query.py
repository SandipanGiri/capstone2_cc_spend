from fastapi import APIRouter
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents,query_documents_stream
from fastapi.responses import StreamingResponse


router = APIRouter(prefix="/api/v1/query")


# for non streaming repsonse
@router.post("/")
def query_endpoint(request: QueryRequest) -> QueryResponse:
   response = query_documents(request.query,request.thread_id)
   #return docs
   #print("REQUEST THREAD ID:", request.thread_id)
   #print("GRAPH RESPONSE:", response)
   

   return QueryResponse(
    query=request.query,
    thread_id=request.thread_id,
    answer=response.get("answer", ""),
    policy_citations=response.get("policy_citations", ""),
    page_no=response.get("page", ""),
    document_name=response.get("document_name", ""),
    sql_query_executed=response.get("sql_query_executed"),
    images=response.get("images", [])
    )


#for streaming response
@router.post("/stream")
async def stream_query_endpoint(request: QueryRequest) -> QueryResponse:
   """
   endpoint that return an SSE steam of the agent's response
   """
   generator = await query_documents_stream(request.query)
   return StreamingResponse(generator, media_type="text/event-stream")



