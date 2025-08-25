from fastapi import FastAPI, HTTPException
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from chatbot import ChatBot
from pydantic_models import (
    MessageRequest,
    MessageResponse,
    HistoryResponse,
    StatusResponse,
    MetricsResponse
)


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
            chatbot_instance = ChatBot(use_rag=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize ChatBot: {e}")
    return chatbot_instance


@app.get("/", response_model=StatusResponse)
async def root():
    bot = get_chatbot()
    return StatusResponse(
        status="CodeBot API is running",
        model_info=bot.get_model_info()
    )


@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
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
    try:
        bot = get_chatbot()
        history = bot.get_history()
        return HistoryResponse(history=history, count=len(history))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {e}")


@app.delete("/history")
async def clear_history():
    try:
        bot = get_chatbot()
        bot.clear_history()
        return {"message": "History cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {e}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    try:
        bot = get_chatbot()
        metrics = bot.get_metrics_summary()
        return MetricsResponse(metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {e}")


@app.post("/metrics/export")
async def export_metrics():
    try:
        bot = get_chatbot()
        filepath = "api_metrics_export.json"
        bot.export_metrics(filepath)
        return {"message": f"Metrics exported to {filepath}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting metrics: {e}")


@app.delete("/metrics")
async def clear_metrics():
    try:
        bot = get_chatbot()
        bot.clear_metrics()
        return {"message": "Metrics cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing metrics: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
