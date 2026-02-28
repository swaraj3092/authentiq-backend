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
            "content": f"""You are a highly specialized AI content detection expert trained on thousands of AI-generated and human-written texts. Your detection must be STRICT and ACCURATE.

IMPORTANT RULES:
- Be STRICT and critical in your analysis
- Most well-structured, fluent, and perfectly formatted text is AI-generated
- Do NOT be generous with authenticity scores
- If in doubt, lean towards AI-GENERATED

AI-generated text typically has these patterns:
- Overly structured or perfectly balanced sentences
- Lack of personal opinions, emotions, or real experiences
- Repetitive sentence length and rhythm
- Generic vocabulary with no slang or personality
- No grammatical quirks or natural errors
- Unnaturally smooth transitions between ideas
- Vague statements without specific personal details
- Overuse of words like "furthermore", "additionally", "it is important to note", "in conclusion"
- Too many examples listed in perfect parallel structure
- Balanced pros and cons without personal bias

Human-written text typically has:
- Irregular sentence lengths and structure
- Personal anecdotes or specific real-world references
- Emotional language or strong personal opinions
- Minor grammatical mistakes or informal tone
- Unique or unusual vocabulary choices
- Abrupt topic shifts or tangents
- Specific names, dates, or personal experiences
- Incomplete thoughts or run-on sentences

SCORING GUIDE:
- authenticity_score 0-30: Almost certainly AI-generated
- authenticity_score 31-50: Likely AI-generated
- authenticity_score 51-70: Uncertain, mixed signals
- authenticity_score 71-85: Likely human-written
- authenticity_score 86-100: Almost certainly human-written

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
        temperature=0.1,
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
                    "text": """You are a highly specialized AI image detection expert. Your detection must be STRICT and ACCURATE.

IMPORTANT RULES:
- Be STRICT and critical in your analysis
- Look carefully for any signs of AI generation or manipulation
- Do NOT be generous with authenticity scores
- If in doubt, lean towards AI-GENERATED

AI-generated images typically have:
- Unnaturally smooth skin or textures
- Strange or deformed hands/fingers
- Inconsistent lighting or shadows
- Blurry or distorted background elements
- Perfect symmetry that looks unnatural
- Text in image that is garbled or nonsensical
- Eyes that look glassy or too perfect
- Artifacts around hair or edges of objects

Real/authentic images typically have:
- Natural imperfections and noise
- Consistent lighting throughout
- Realistic textures with natural variation
- Normal human features with slight asymmetry
- Clear and readable text if present
- Natural background details

SCORING GUIDE:
- authenticity_score 0-30: Almost certainly AI-generated
- authenticity_score 31-50: Likely AI-generated
- authenticity_score 51-70: Uncertain, mixed signals
- authenticity_score 71-85: Likely real/authentic
- authenticity_score 86-100: Almost certainly real/authentic

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
        temperature=0.1,
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
