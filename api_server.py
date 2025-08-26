from fastapi import FastAPI, HTTPException
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from chatbot import ChatBot
from config_utils import get_api_config
from pydantic_models import (
    MessageRequest,
    MessageResponse,
    HistoryResponse,
    StatusResponse,
    MetricsResponse,
    AgentProcessRequest,
    AgentProcessResponse
)

api_config = get_api_config()

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
        model_info=bot.get_model_info(),
        agent_system_active=hasattr(bot, 'use_agents') and bot.use_agents
    )


@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    try:
        bot = get_chatbot()

        if hasattr(bot, 'use_agents') and bot.use_agents and bot.agent_graph:
            result = bot.agent_graph.process_query(request.message, bot.get_history())

            return MessageResponse(
                response=result.get("response", ""),
                success=bool(result.get("response")),
                rag_used="research" in result.get("agents_used", []),
                agents_used=result.get("agents_used", []),
                intent=result.get("intent", ""),
                quality_score=result.get("quality_score", 0.0),
                research_results=result.get("research_results", []),
                metadata=result.get("metadata", {})
            )
        else:
            response = bot.get_response(request.message)
            rag_used = hasattr(bot, 'last_rag_used') and bot.last_rag_used

            if response:
                return MessageResponse(
                    response=response,
                    success=True,
                    rag_used=rag_used,
                    agents_used=[],
                    intent="",
                    quality_score=0.0,
                    research_results=[],
                    metadata={}
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to get response from ChatBot")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {e}")


@app.post("/agents/process", response_model=AgentProcessResponse)
async def process_with_agents(request: AgentProcessRequest):
    try:
        bot = get_chatbot()

        if not (hasattr(bot, 'use_agents') and bot.use_agents and bot.agent_graph):
            raise HTTPException(status_code=400, detail="Agent system is not available")

        start_time = time.time()
        result = bot.agent_graph.process_query(request.query, request.conversation_history)
        processing_time = time.time() - start_time

        return AgentProcessResponse(
            response=result.get("response", ""),
            quality_score=result.get("quality_score", 0.0),
            agents_used=result.get("agents_used", []),
            intent=result.get("intent", ""),
            research_results=result.get("research_results", []),
            metadata=result.get("metadata", {}),
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with agents: {e}")


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


@app.get("/agents/info")
async def get_agents_info():
    try:
        bot = get_chatbot()

        if hasattr(bot, 'use_agents') and bot.use_agents:
            agent_types = [
                {"name": "Router Agent", "icon": "🎯", "purpose": "Query intent classification"},
                {"name": "Research Agent", "icon": "📚", "purpose": "Knowledge base search"},
                {"name": "Code Agent", "icon": "💻", "purpose": "Code generation and examples"},
                {"name": "Reviewer Agent", "icon": "✅", "purpose": "Quality assessment and feedback"}
            ]

            return {
                "agent_system_active": True,
                "available_agents": agent_types,
                "total_agents": len(agent_types)
            }
        else:
            return {
                "agent_system_active": False,
                "available_agents": [],
                "total_agents": 0
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agents info: {e}")


@app.get("/health")
async def health_check():
    bot = get_chatbot()
    return {
        "status": "healthy",
        "agent_system": hasattr(bot, 'use_agents') and bot.use_agents,
        "rag_enabled": hasattr(bot, 'use_rag') and bot.use_rag
    }


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
    uvicorn.run(app, host=api_config["host"], port=api_config["port"])
