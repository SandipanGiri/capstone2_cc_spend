from pathlib import Path
from fastapi import UploadFile

from src.ingestion.ingestion import ingest_pdf


async def upload_document(file: UploadFile):
    print("am in uploadservice")
    print("filename ", file)
    Data = Path("data")
    Data.mkdir(exist_ok=True)
    file_path = Data / file.filename

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = ingest_pdf(file_path)
    print("*****file ingetsed", result)
    return {
        # "message": "File uploaded successfully",
        # "filename": file.filename,
        "message": result
    }
