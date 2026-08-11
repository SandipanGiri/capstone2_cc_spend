from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/chat")
def chat(req: ChatRequest):
    return {
        "answer": "Example response from your RAG pipeline.",
        "sources": [
            {
                "title": "Example Document",
                "content": "Retrieved passage...",
                "score": 0.94,
                "metadata": {"page": 3},
            }
        ],
    }


@app.post("/upload")
async def upload(file: UploadFile):

    contents = await file.read()
    rag.ingest(contents, filename=file.filename)
    return {"status": "indexed"}


from fastapi.responses import StreamingResponse

# @app.post("/chat")
# async def chat():

#     async def generator():

#         for token in llm.stream(prompt):
#             yield token

#     return StreamingResponse(
#         generator(),
#         media_type="text/plain"
#     )
