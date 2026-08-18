from fastapi import FastAPI
from src.api.v1.routes import query
from src.api.v1.routes.upload_routes import router as upload_router
from src.api.v1.agents.agents import build_rag_graph

from contextlib import asynccontextmanager

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(query.router)


# Upload routes
app.include_router(upload_router)


from src.core.checkpoint import init_checkpoint, close_checkpoint


@app.on_event("startup")
async def startup():

    checkpoint = await init_checkpoint()

    app.state.rag_graph = build_rag_graph(checkpoint)


@app.on_event("shutdown")
async def shutdown():

    await close_checkpoint()
