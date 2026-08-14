from fastapi import APIRouter, UploadFile, File
from services.upload_services import upload_document

router = APIRouter(prefix="/api/v1/documents")


@router.post("/")
async def upload(file: UploadFile = File(...)):

    response = await upload_document(file)

    return response
