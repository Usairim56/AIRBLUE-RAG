# app.py
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from retriever import HybridRetriever
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables (GROQ_API_KEY will come from HF secrets)
load_dotenv()

# Init FastAPI
app = FastAPI(title="Airblue RAG Chatbot")

# Allow frontend calls (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load retriever + LLM once
retriever = HybridRetriever(faiss_path="data", k_search=15, top_n=10, verbose=False)
llm = ChatGroq(model="gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

# Request schema
class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(req: QueryRequest):
    query = req.question

    # Step 1: Retrieve docs
    docs = retriever.retrieve(query)
    context_texts = "\n\n".join([f"- {d['text']}" for d in docs])

    # Step 2: Strong system prompt
    system_prompt = """You are Airblue’s customer assistant, built as a **Retrieval-Augmented Generation (RAG) demo chatbot**. 
Important background:
- This chatbot is a **portfolio project**, not an official Airblue product. 
- It is powered by documents manually collected from **publicly available sources** (Airblue website, policies, public info).
- It uses chunked data (flights, routes, baggage, check-in, offices, BlueMiles, fleet, history, passenger rights, etc.) stored in a vector database.

Your role:
- Be polite, professional, and conversational. Handle greetings, introductions, and meta-questions about the project.
- Always synthesize multiple relevant chunks into one detailed answer (4–8 sentences).
- Use lists when appropriate.
- Only answer from context; if missing, say politely you don’t have it.
- If query is vague, infer intent politely (e.g., “history” → “Airblue’s history”) and expand into a useful answer.
- Never invent facts. Stay within the data.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context_texts}\n\nQuestion: {query}")
    ]

    try:
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        return {"error": str(e)}
