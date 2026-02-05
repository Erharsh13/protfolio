import threading
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json


from app.schemas import QuestionRequest, AnswerResponse
from app.prompt import (
    SYSTEM_PROMPT,
    PROFILE_CONTEXT,
    SKILLS_CONTEXT,
    PROJECTS_CONTEXT,
    VPC_ARCHITECTURE_CONTEXT
)


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

# HARD limits for speed + brevity
OLLAMA_OPTIONS = {
    "temperature": 0.2,
    "num_predict": 80,   # ⬅ prevents essays
    "num_ctx": 4096
}

app = FastAPI(title="Harsh Purohit AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE = {}


def detect_intent(user_q: str) -> str:
    if any(x in user_q for x in ["hi", "hello", "hey"]):
        return "greeting"
    if any(x in user_q for x in ["architecture", "design", "flow"]):
        return "architecture"
    if any(x in user_q for x in ["project", "vpc", "smartdocx", "nova"]):
        return "projects"
    if any(x in user_q for x in ["python", "fastapi", "langchain", "langgraph", "aws"]):
        return "skills"
    return "profile"

def build_prompt(question: str, intent: str) -> str:
    if intent == "architecture":
        context = VPC_ARCHITECTURE_CONTEXT
    elif intent == "projects":
        context = PROJECTS_CONTEXT
    elif intent == "skills":
        context = SKILLS_CONTEXT
    else:
        context = PROFILE_CONTEXT

    return f"""
{SYSTEM_PROMPT}

Relevant Context:
{context}

User Question:
{question}

Answer concisely in bullet points:
""".strip()


# Warm-up (reduces first-query delay)
def warmup():
    try:
        requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "Say hello in one sentence.",
                "stream": True,
                "options": {"num_predict": 10}
            },
            timeout=30
        )
        print("🔥 Model warmed up")
    except:
        pass

threading.Thread(target=warmup, daemon=True).start()

@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    user_q = req.question.strip().lower()

    # 👋 Handle greetings explicitly
    if user_q in ["hi", "hello", "hey", "hii"]:
        return {
            "answer": "Hi 👋 How can I help? You can ask about my projects, GenAI work, or system design."
        }
    intent = detect_intent(user_q)

    # Cache only for non-architecture queries
    if user_q in CACHE and intent != "architecture":
        return {"answer": CACHE[user_q]}

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": build_prompt(req.question, intent),
                "stream": True,
                "options": OLLAMA_OPTIONS
            },
            stream=True,
            timeout=180
        )

        if response.status_code != 200:
            raise Exception(f"Ollama error {response.status_code}: {response.text}")

        answer_chunks = []

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            if data.get("done"):
                break

            chunk = data.get("response")
            if chunk:
                answer_chunks.append(chunk)

        answer = "".join(answer_chunks).strip()

        CACHE[user_q] = answer



        # ✅ SUCCESS RETURN (THIS WAS MISSING)
        return {"answer": answer}

    except Exception as e:
        print("❌ AI ERROR:", repr(e))
        return {
            "answer": f"AI ERROR: {str(e)}"
        }
