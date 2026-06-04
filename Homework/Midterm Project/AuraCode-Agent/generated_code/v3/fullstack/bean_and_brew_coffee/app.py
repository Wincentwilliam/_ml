import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Client for Ollama (OpenAI compatible API)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # Required but ignored by Ollama
)

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Try available models in order of preference
    models_to_try = ["qwen3.5:9b", "llama3.2:latest", "gemma4:31b-cloud"]
    selected_model = models_to_try[0] 
    
    system_prompt = (
        "You are a helpful AI assistant for Bean & Brew, a premium artisanal coffee shop. "
        "Be friendly, concise, and helpful. Answer questions about the menu (Cappuccino, "
        "Caramel Cloud, Midnight Cold Brew, Zen Matcha Latte, Pure Gold Espresso, Croissants), "
        "hours (7am-8pm daily), location (123 Roast Avenue, NY), and ordering. "
        "Use coffee emojis occasionally ☕️🥐."
    )

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                *request.history,
                {"role": "user", "content": request.message}
            ]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        # Fallback response if Ollama is not running
        return {"response": "I'm currently steaming some milk and can't think straight! Please try again in a moment. ☕️"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)