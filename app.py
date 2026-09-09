import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from google import genai


# =========================
# Gemini setup
# =========================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=api_key)

# =========================
# App setup
# =========================

app = FastAPI(title="My AI Chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")


# Stores one Gemini chat per browser session
sessions = {}


# =========================
# Homepage
# =========================

@app.get("/")
async def home():
    return FileResponse("static/index.html")


# =========================
# Create a new session
# =========================

@app.post("/session")
async def create_session():
    session_id = str(uuid.uuid4())

    sessions[session_id] = client.chats.create(
        model="gemini-3.6-flash"
    )

    return {
        "session_id": session_id
    }


# =========================
# Send message
# =========================

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()

    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id:
        return {"error": "Missing session_id"}

    if not message:
        return {"error": "Missing message"}

    if session_id not in sessions:
        return {"error": "Session not found"}

    gemini_chat = sessions[session_id]

    def generate():
        response = gemini_chat.send_message_stream(message)

        for chunk in response:
            if chunk.text:
                yield chunk.text

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )


# =========================
# Run locally
# =========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860
    )