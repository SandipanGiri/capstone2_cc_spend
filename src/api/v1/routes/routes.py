from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.v1.schemas.query_schemas import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents, query_documents_stream
from src.core.guardrails import GuardrailViolation

router = APIRouter(prefix="/api/v1/query")


# for non streaming response
@router.post("/")
def query_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        docs = query_documents(request.query, request.user_id)
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )
    return docs


# for streaming response
@router.post("/stream")
async def stream_query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    endpoint that return an SSE steam of the agent's response
    """
    generator = await query_documents_stream(request.query)
    return StreamingResponse(generator, media_type="text/event-stream")
