from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Literal

from backend.agent.ai_agent import agent

router = APIRouter()


class HistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    history: Optional[List[HistoryItem]] = None


@router.post("/chat")
def chat(request: ChatRequest):

    try:

        history = None
        if request.history:
            history = [{"role": h.role, "content": h.content} for h in request.history]

        result = agent.process_message(
            request.message,
            thread_id=request.thread_id,
            history=history
        )

        return {
            "response": result["response"]
        }

    except Exception as e:

        return {
            "response": f"Server error: {str(e)}"
        }