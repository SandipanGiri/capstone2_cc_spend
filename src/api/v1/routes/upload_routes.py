from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from src.api.v1.services.upload_services import upload_document

router = APIRouter(prefix="/api/v1/documents")


@router.post("/")
async def upload(
    file: UploadFile = File(...),
):

    print("========== AT UPLOAD ROUTE ==========")

    try:

        response = await upload_document(file)

        return response

    except Exception as e:

        print(
            "Document upload error:",
            e,
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
