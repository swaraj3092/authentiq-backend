from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import base64
import json
import os

app = FastAPI()

# ─── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Groq Client ───────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ─── Health Check ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AuthentiQ API is running"}


# ─── Text Analysis ─────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

@app.post("/analyze-text")
async def analyze_text(input: TextInput):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""You are an expert AI content detector. Analyze the following text and determine if it is AI-generated or written by a human.

Return ONLY a valid JSON object with exactly these fields (no extra text, no markdown, no backticks):
{{
  "authenticity_score": <number 0-100, higher means more authentic/human-written>,
  "ai_probability": <number 0-100, likelihood it was AI generated>,
  "verdict": "<AUTHENTIC or AI-GENERATED>",
  "language_consistency": <number 0-100>,
  "source_credibility": <number 0-100>,
  "summary": "<one sentence explanation of your verdict>"
}}

Text to analyze:
{input.text}"""
        }],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "authenticity_score": 50,
            "ai_probability": 50,
            "verdict": "INCONCLUSIVE",
            "language_consistency": 50,
            "source_credibility": 50,
            "summary": "Unable to determine authenticity. Please try again."
        }

    return result


# ─── Image Analysis ────────────────────────────────────────────────────────
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    ext = file.filename.split(".")[-1].lower() if file.filename else "jpeg"
    media_type = f"image/{ext}" if ext in ["png", "gif", "webp"] else "image/jpeg"

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """You are an expert AI image detector. Analyze this image for signs of AI generation or manipulation.

Return ONLY a valid JSON object with exactly these fields (no extra text, no markdown, no backticks):
{
  "authenticity_score": <number 0-100, higher means more likely real/authentic>,
  "ai_probability": <number 0-100, likelihood it was AI generated>,
  "verdict": "<AUTHENTIC or AI-GENERATED>",
  "artifacts_detected": <true or false>,
  "source_credibility": <number 0-100>,
  "summary": "<one sentence explanation of your verdict>"
}"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_image}"
                    }
                }
            ]
        }],
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "authenticity_score": 50,
            "ai_probability": 50,
            "verdict": "INCONCLUSIVE",
            "artifacts_detected": False,
            "source_credibility": 50,
            "summary": "Unable to analyze image. Please try a different file."
        }

    return result
