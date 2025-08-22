from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from chatbot import ChatBot

app = FastAPI(
    title="CodeBot API",
    description="API for CodeBot - Programming Assistant with conversation history",
    version="1.0.0"
)

chatbot_instance = None


def get_chatbot():
    global chatbot_instance
    if chatbot_instance is None:
        try:
            chatbot_instance = ChatBot()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize ChatBot: {e}")
    return chatbot_instance


class MessageRequest(BaseModel):
    message: str


class MessageResponse(BaseModel):
    response: str
    success: bool


class HistoryResponse(BaseModel):
    history: List[Dict[str, str]]
    count: int


class StatusResponse(BaseModel):
    status: str
    model_info: str


@app.get("/", response_model=StatusResponse)
async def root():
    """Get API status and model information"""
    bot = get_chatbot()
    return StatusResponse(
        status="CodeBot API is running",
        model_info=bot.get_model_info()
    )


@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Send a message to ChatBot and get response"""
    try:
        bot = get_chatbot()
        response = bot.get_response(request.message)

        if response:
            return MessageResponse(response=response, success=True)
        else:
            raise HTTPException(status_code=500, detail="Failed to get response from ChatBot")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {e}")


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    """Get conversation history"""
    try:
        bot = get_chatbot()
        history = bot.get_history()
        return HistoryResponse(history=history, count=len(history))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {e}")


@app.delete("/history")
async def clear_history():
    """Clear conversation history"""
    try:
        bot = get_chatbot()
        bot.clear_history()
        return {"message": "History cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
