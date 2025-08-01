import os
from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.endpoint import router as api_router

load_dotenv()  # .env 파일 로드
debug_mode = os.getenv("DEBUG", "false").lower() == "true"

app = FastAPI(
    title="U-Hyu Recommendation API",
    version="1.0.0",
    debug=debug_mode
)

app.include_router(api_router)