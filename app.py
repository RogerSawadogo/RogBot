from fastapi import FastAPI, Request
from pydantic import BaseModel
from apps.rag import create_index, load_index_and_docs
from apps.generator import generate_response_deepseek
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load .env variables
load_dotenv()
API_KEY = os.getenv("API_KEY_LLAMA")

# Load or create index
if not (os.path.exists('index.faiss') and os.path.exists('docs.pkl')):
    create_index('data/data.txt')

embedding_model, index, docs = load_index_and_docs()

# Create FastAPI app
app = FastAPI()

# Allow CORS if you want to call it from frontends (e.g., React, Gradio frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    message: str
    history: list = []

# Endpoint
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    answer, _ = generate_response_deepseek(request.message, embedding_model, index, docs, API_KEY)
    return {"response": answer}


