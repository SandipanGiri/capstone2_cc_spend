import base64
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.v1.schemas.query_schema import (
    QueryRequest,
    QueryResponse,
)
from src.api.v1.services.query_service import (
    query_documents,
    query_documents_stream,
)
from src.core.guardrails import GuardrailViolation

router = APIRouter(prefix="/api/v1/query")


# ============================================================
# IMAGE ENCODING
# ============================================================


def encode_image(image_path: str) -> str:

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded


def encode_images(images) -> list[str]:

    encoded_images = []

    for image_item in images or []:

        try:

            if isinstance(image_item, dict):

                image_path = (
                    image_item.get("image_path")
                    or image_item.get("path")
                    or image_item.get("file_path")
                )

            else:
                image_path = image_item

            if not image_path:
                continue

            encoded_images.append(encode_image(image_path))

        except Exception as e:

            print(f"Image encoding failed: {e}")

    return encoded_images


# ============================================================
# NON-STREAMING QUERY
# ============================================================


@router.post(
    "/",
    response_model=QueryResponse,
)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:

    try:

        # ----------------------------------------------------
        # The service gets the already initialized graph from
        # request.app.state.rag_graph.
        # ----------------------------------------------------

        response = await query_documents(
            request,
            body.query,
            body.thread_id,
        )

        print(
            "SERVICE RESPONSE:",
            response,
        )

    except GuardrailViolation as violation:

        raise HTTPException(
            status_code=400,
            detail={
                "guardrail": violation.guard,
                "message": violation.message,
            },
        )

    except Exception as e:

        print(
            "Query endpoint error:",
            e,
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    # --------------------------------------------------------
    # Validate response
    # --------------------------------------------------------

    if response is None:

        raise HTTPException(
            status_code=500,
            detail="Agent returned empty response",
        )

    # --------------------------------------------------------
    # Encode images
    # --------------------------------------------------------

    images = encode_images(response.get("images", []))

    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return QueryResponse(
        query=body.query,
        thread_id=body.thread_id,
        answer=response.get(
            "answer",
            "",
        ),
        policy_citations=response.get(
            "policy_citations",
            "",
        ),
        page_no=response.get(
            "page_no",
            "",
        ),
        document_name=response.get(
            "document_name",
            "",
        ),
        sql_query_executed=response.get("sql_query_executed"),
        images=images,
    )


# ============================================================
# STREAMING QUERY
# ============================================================


@router.post(
    "/stream",
)
async def stream_query_endpoint(
    request: Request,
    body: QueryRequest,
):

    async def event_generator():

        try:

            # ------------------------------------------------
            # query_documents_stream should use the same graph
            # stored in request.app.state.rag_graph.
            # ------------------------------------------------

            async for chunk in query_documents_stream(
                request,
                body.query,
                body.thread_id,
            ):

                # ============================================
                # TOKEN
                # ============================================

                if isinstance(chunk, str):

                    yield (
                        "event: token\n" f"data: {json.dumps({'content': chunk})}\n\n"
                    )

                    continue

                # ============================================
                # DICT RESPONSE
                # ============================================

                if isinstance(chunk, dict):

                    # ----------------------------------------
                    # Token
                    # ----------------------------------------

                    content = chunk.get("content")

                    if content:

                        yield (
                            "event: token\n"
                            f"data: {json.dumps({'content': content})}\n\n"
                        )

                    # ----------------------------------------
                    # Final metadata
                    # ----------------------------------------

                    if chunk.get("done"):

                        # Encode images before sending them
                        images = encode_images(
                            chunk.get(
                                "images",
                                [],
                            )
                        )

                        metadata = {
                            "sources": chunk.get(
                                "sources",
                                [],
                            ),
                            "images": images,
                            "policy_citations": chunk.get(
                                "policy_citations",
                                "",
                            ),
                            "page_no": chunk.get(
                                "page_no",
                                "",
                            ),
                            "document_name": chunk.get(
                                "document_name",
                                "",
                            ),
                        }

                        yield ("event: metadata\n" f"data: {json.dumps(metadata)}\n\n")

            # ------------------------------------------------
            # Streaming completed
            # ------------------------------------------------

            yield ("event: done\n" f"data: {json.dumps({'status': 'completed'})}\n\n")

        # ====================================================
        # GUARDRAIL ERROR
        # ====================================================

        except GuardrailViolation as violation:

            yield ("event: guardrail_error\n" f"data: {json.dumps({
                    'guardrail': violation.guard,
                    'message': violation.message,
                })}\n\n")

        # ====================================================
        # GENERAL ERROR
        # ====================================================

        except Exception as e:

            print(
                "Streaming endpoint error:",
                e,
            )

            yield ("event: error\n" f"data: {json.dumps({
                    'message': str(e),
                })}\n\n")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
