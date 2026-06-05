from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve CSS, JS, images langsung dari folder yang sama
app.mount("/static", StaticFiles(directory="."), name="static")

class ChatRequest(BaseModel):
    message: str
    history: list = []

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    timeout=60
)

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.get("/style.css")
async def get_css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/script.js")
async def get_js():
    return FileResponse("script.js", media_type="application/javascript")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        system_prompt = {
            "role": "system",
            "content": "You are BrewBot, a friendly AI barista assistant for Bean & Brew Coffee Shop. Help customers with menu questions, recommendations, and coffee knowledge. Be warm, friendly, and concise."
        }
        messages = [system_prompt]
        history_slice = request.history[-8:] if request.history else []
        messages.extend(history_slice)
        messages.append({"role": "user", "content": request.message})
        response = client.chat.completions.create(
            model="qwen3.5:9b",
            messages=messages,
            temperature=0.7,
            max_tokens=250
        )
        reply = response.choices[0].message.content
        return JSONResponse({"response": reply})
    except Exception as e:
        return JSONResponse(
            {"response": "Hi! I'm BrewBot your AI barista. I'm having a small technical issue right now, but I'll be back shortly! ?"},
            status_code=200
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
