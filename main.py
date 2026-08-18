from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.routes.query import router
from src.api.v1.agents.agents import create_rag_graph


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("========== STARTING APPLICATION ==========")

    async with create_rag_graph() as rag_graph:

        # Store the graph on FastAPI application state
        app.state.rag_graph = rag_graph

        print("========== RAG GRAPH READY ==========")

        yield

    print("========== APPLICATION SHUTDOWN ==========")


app = FastAPI(
    title="Agentic RAG API",
    lifespan=lifespan,
)

app.include_router(router)
