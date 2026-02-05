import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.schemas import QuestionRequest, AnswerResponse
from app.prompt import (
    SYSTEM_PROMPT,
    PROFILE_CONTEXT,
    SKILLS_CONTEXT,
    PROJECTS_CONTEXT,
    VPC_ARCHITECTURE_CONTEXT
)

# ======================================================
# LOAD ENV
# ======================================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print("OPENROUTER_API_KEY loaded:", bool(OPENROUTER_API_KEY))

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found")

# ======================================================
# OPENROUTER CONFIG
# ======================================================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://harshpurohit.dev",  # can be any URL
    "X-Title": "Harsh Portfolio AI Assistant",
}

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(title="Harsh Purohit AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE = {}

# ======================================================
# INTENT DETECTION
# ======================================================
def detect_intent(user_q: str) -> str:
    q = user_q.lower()
    if any(x in q for x in ["hi", "hello", "hey", "hii"]):
        return "greeting"
    if any(x in q for x in ["architecture", "design", "flow"]):
        return "architecture"
    if any(x in q for x in ["project", "vpc", "smartdocx", "nova", "vizismart"]):
        return "projects"
    if any(x in q for x in ["python", "fastapi", "langchain", "langgraph", "aws"]):
        return "skills"
    return "profile"


def get_context(intent: str) -> str:
    if intent == "architecture":
        return VPC_ARCHITECTURE_CONTEXT
    if intent == "projects":
        return PROJECTS_CONTEXT
    if intent == "skills":
        return SKILLS_CONTEXT
    return PROFILE_CONTEXT


# ======================================================
# API ENDPOINT
# ======================================================
@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    user_q = req.question.strip()

    if user_q.lower() in ["hi", "hello", "hey", "hii"]:
        return {
            "answer": (
                "Hi 👋 I’m Harsh’s AI assistant.\n\n"
                "You can ask me about:\n"
                "- AI projects (VPC, SmartDocX, NOVA, VIZISmart)\n"
                "- GenAI & RAG systems\n"
                "- Backend & system design"
            )
        }

    intent = detect_intent(user_q)

    if user_q in CACHE and intent != "architecture":
        return {"answer": CACHE[user_q]}

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": get_context(intent)},
            {"role": "user", "content": user_q},
        ],
        "temperature": 0.2,
        "max_tokens": 150,
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(response.text)

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()

        CACHE[user_q] = answer
        return {"answer": answer}

    except Exception as e:
        print("❌ AI ERROR:", repr(e))
        return {
            "answer": "AI assistant temporarily unavailable. Please try again."
        }
