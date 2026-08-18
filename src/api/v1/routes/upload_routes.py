from fastapi import APIRouter, UploadFile, File
from src.api.v1.services.upload_services import upload_document

router = APIRouter(prefix="/api/v1/documents")


@router.post("/")
async def upload(file: UploadFile = File(...)):

    response = await upload_document(file)

    return response
