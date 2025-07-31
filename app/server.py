from fastapi import FastAPI
from app.api.endpoint import router as api_router

app = FastAPI(
    title="U-Hyu Recommendation API",
    version="1.0.0",
    debug=True
)

app.include_router(api_router)