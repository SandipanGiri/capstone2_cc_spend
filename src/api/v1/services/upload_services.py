from pathlib import Path
from fastapi import UploadFile
import shutil
from src.ingestion.ingestion import run_ingestion

from src.ingestion.ingestion import run_ingestion


async def upload_document(file: UploadFile):
    print("am in uploadservice")
    print("filename ", file)
    Data = Path("data")
    Data.mkdir(exist_ok=True)
    file_path = Data / file.filename
    print("file_name", file_path)

    #     with open(file_path, "wb") as buffer:
    #         buffer.write(await file.read())

    result = run_ingestion(file_path)
    print("*****file ingetsed", result)
    return {
        "message": result["message"],
        # "filename": file.filename,
        "success": result["status"],
    }
