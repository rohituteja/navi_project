from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from parent directory
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)
print(f"Loaded .env from: {env_path}")
print(f"OPENAI_API_KEY present: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

from app.api import endpoints

app = FastAPI(title="Navi Debrief Engine")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP, allow all. In prod, restrict to frontend URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router)

@app.get("/")
async def root():
    return {"message": "Navi Debrief Engine API is running"}
