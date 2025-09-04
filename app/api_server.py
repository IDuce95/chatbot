from fastapi import FastAPI, HTTPException

from .chatbot import ChatBot
from .config_utils import get_api_config
from .pydantic_models import (
    HistoryResponse,
    MessageRequest,
    MessageResponse,
    MetricsResponse,
    StatusResponse,
)

api_config = get_api_config()

app = FastAPI(
    title="CodeBot API",
    description="API for CodeBot - Programming Assistant with conversation history",
    version="1.0.0",
)

chatbot_instance = None


def get_chatbot():
    global chatbot_instance
    if chatbot_instance is None:
        try:
            chatbot_instance = ChatBot(use_rag=True)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize ChatBot: {e}"
            )
    return chatbot_instance


@app.get("/", response_model=StatusResponse)
async def root():
    bot = get_chatbot()
    return StatusResponse(
        status="CodeBot API is running",
        model_info=bot.get_model_info(),
        agent_system_active=hasattr(bot, "use_agents") and bot.use_agents,
    )


@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    try:
        bot = get_chatbot()

        response = bot.get_response(request.message)
        rag_used = hasattr(bot, "last_rag_used") and bot.last_rag_used

        if response:
            agents_used = []
            intent = ""
            quality_score = 0.0
            research_results = []
            metadata = {}

            if hasattr(bot, "use_agents") and bot.use_agents:
                conversation_history = bot.get_history()
                if len(conversation_history) >= 2:
                    agents_used = getattr(bot, "_last_agents_used", [])
                    intent = getattr(bot, "_last_intent", "")
                    quality_score = getattr(bot, "_last_quality_score", 0.0)
                    research_results = getattr(bot, "_last_research_results", [])
                    metadata = getattr(bot, "_last_metadata", {})

            return MessageResponse(
                response=response,
                success=True,
                rag_used=rag_used,
                agents_used=agents_used,
                intent=intent,
                quality_score=quality_score,
                research_results=research_results,
                metadata=metadata,
            )
        else:
            raise HTTPException(
                status_code=500, detail="Failed to get response from ChatBot"
            )

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


@app.get("/agents/info")
async def get_agents_info():
    try:
        bot = get_chatbot()

        if hasattr(bot, "use_agents") and bot.use_agents:
            agent_types = [
                {"name": "Router Agent", "purpose": "Query intent classification"},
                {"name": "Research Agent", "purpose": "Knowledge base search"},
                {"name": "Code Agent", "purpose": "Code generation and examples"},
                {
                    "name": "Reviewer Agent",
                    "purpose": "Quality assessment and feedback",
                },
            ]

            return {
                "agent_system_active": True,
                "available_agents": agent_types,
                "total_agents": len(agent_types),
            }
        else:
            return {
                "agent_system_active": False,
                "available_agents": [],
                "total_agents": 0,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agents info: {e}")


@app.get("/health")
async def health_check():
    bot = get_chatbot()
    return {
        "status": "healthy",
        "agent_system": hasattr(bot, "use_agents") and bot.use_agents,
        "rag_enabled": hasattr(bot, "use_rag") and bot.use_rag,
    }


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    try:
        bot = get_chatbot()
        metrics = bot.get_metrics_summary()
        return MetricsResponse(metrics=metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {e}")


@app.get("/metrics/detailed")
async def get_detailed_metrics():
    try:
        bot = get_chatbot()
        if bot.use_rag and bot.rag_manager:
            session_metrics = bot.rag_manager.metrics.session_metrics
            summary = bot.rag_manager.metrics.get_session_summary()

            response_times = [m.get("response_time", 0) for m in session_metrics]
            quality_scores = [m.get("quality_score", 0.0) for m in session_metrics]
            latest_interaction = {}

            if session_metrics:
                latest = session_metrics[-1]
                latest_interaction = {
                    "response_word_count": latest.get("generation_metrics", {}).get(
                        "response_word_count", 0
                    ),
                    "perplexity_approx": latest.get("generation_metrics", {}).get(
                        "perplexity_approx", 0
                    ),
                    "unique_word_ratio": latest.get("generation_metrics", {}).get(
                        "unique_word_ratio", 0
                    ),
                    "quality_score": latest.get("quality_score", 0.0),
                    "agents_used": (
                        latest.get("context", "")
                        .split("Agents: ")[-1]
                        .split(" |")[0]
                        .split(", ")
                        if "Agents:" in latest.get("context", "")
                        else []
                    ),
                    "response_time": latest.get("response_time", 0),
                }

            detailed_summary = summary.copy()
            detailed_summary.update(
                {
                    "response_time_history": response_times,
                    "quality_score_history": quality_scores,
                    "latest_interaction": latest_interaction,
                    "avg_quality_score": (
                        sum(quality_scores) / len(quality_scores)
                        if quality_scores
                        else 0
                    ),
                    "session_start_time": (
                        session_metrics[0]["timestamp"] if session_metrics else None
                    ),
                }
            )

            return {"detailed_metrics": detailed_summary}
        else:
            return {
                "detailed_metrics": {
                    "total_interactions": 0,
                    "status": "RAG not enabled",
                }
            }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting detailed metrics: {e}"
        )


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
