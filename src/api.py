from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.embeddings import load_vector_store
from src.llm import get_llm
from src.prompt import get_prompt
from src.retrievers.hybrid_retriever import retrieve_documents
from src.memory import get_memory
from src.chat import chat

# Initialize FastAPI app
app = FastAPI(
    title="HR Policy Bot API",
    description="A RAG-based API for employee policy queries",
    version="1.0.0"
)

# --- GLOBAL COMPONENTS (Load once on startup) ---
# Pre-loading these ensures fast response times for the API
print("🚀 Initializing HR Bot Components...")
vectordb = load_vector_store()
llm = get_llm()
prompt = get_prompt()

# --- REQUEST/RESPONSE MODELS ---
class ChatRequest(BaseModel):
    question: str
    user_id: str = "default_user"  # Matches the session_id logic in main.py

class ChatResponse(BaseModel):
    answer: str
    user_id: str
    requires_department: bool # Notifies the UI if the user is still in the 'Login' phase

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "HR Policy Bot API is running."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles policy queries and proactive department locking.
    """
    try:
        #  Get memory for this specific user session (Syncs with main.py Step 2)
        memory = get_memory(session_id=request.user_id)
        active_filters = memory.get_filters()

        #  PROACTIVE GREETING LOGIC (Syncs with main.py GREETING block)
        # If no department is set and the user sends a generic message or empty string
        if not active_filters.get("department"):
            if request.question.strip() == "" or "hello" in request.question.lower():
                welcome_msg = "🤖 Bot: Welcome! Before we begin, please let me know which **department** you belong to?"
                
                # Add message to memory so the LLM knows the last thing said was the greeting
                memory.add_ai_message(welcome_msg)
                
                return ChatResponse(
                    answer=welcome_msg,
                    user_id=request.user_id,
                    requires_department=True
                )

        # 3. CHAT LOGIC (Syncs with main.py Step 3)
        # The chat function handles: Dept extraction, Access Denied, and RAG retrieval.
        answer = chat(
            memory=memory,
            llm=llm,
            prompt=prompt,
            retrieve_documents=retrieve_documents,
            vectordb=vectordb,
            question=request.question
        )

        # 4. Determine if a department is now set after the chat processing
        has_dept = bool(memory.get_filters().get("department"))

        return ChatResponse(
            answer=answer,
            user_id=request.user_id,
            requires_department=not has_dept
        )

    except Exception as e:
        print(f"⚠️ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")