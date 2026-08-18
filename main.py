from fastapi import FastAPI
from src.api.v1.routes import query
from src.api.v1.routes.upload_routes import router as upload_router

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