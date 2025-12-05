from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import centralized config
try:
    from backend.config import settings
    SERVER_PORT = settings.SERVER_PORT
    FRONTEND_PORT = settings.FRONTEND_PORT
except ImportError:
    # Fallback if running standalone
    SERVER_PORT = 8002
    FRONTEND_PORT = 5181

app = FastAPI(
    title="Magnus API",
    description="Backend API for Magnus Financial & Sports Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS - uses centralized port config
origins = [
    f"http://localhost:{FRONTEND_PORT}",  # React frontend (Vite)
    f"http://localhost:{SERVER_PORT}",  # API itself
    f"http://127.0.0.1:{FRONTEND_PORT}",
    f"http://127.0.0.1:{SERVER_PORT}",
    "http://localhost:3000",  # Fallback React port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Magnus API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=SERVER_PORT, reload=True)
