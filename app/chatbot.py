import os
from typing import Dict, List, Optional

import openai
import toml
from dotenv import load_dotenv

from .rag_manager import RAGManager

load_dotenv()


class ChatBot:
    def __init__(self, config_path: str = "config.toml", use_rag: bool = True):
        self.config = toml.load(config_path)

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key in environment variables")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.model_info = (
            f"CodeBot initialized with model: {self.config['model']['name']}"
        )
        self.last_rag_used = False

        self.use_rag = use_rag
        self.rag_manager = None
        if use_rag:
            try:
                self.rag_manager = RAGManager()
            except Exception as e:
                print(f"Warning: Could not initialize RAG manager: {e}")
                self.use_rag = False

        self.agent_graph = None
        self.use_agents = False
        try:
            from .agents.agent_graph import AgentGraph

            self.agent_graph = AgentGraph(self, self.config)
            self.use_agents = True
            print("Multi-Agent System initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize Agent Graph: {e}")
            self.use_agents = False

    def get_response(self, user_message: str) -> Optional[str]:
        if not self.use_agents or not self.agent_graph:
            raise RuntimeError(
                "Agent system not available. Please ensure proper initialization."
            )

        return self.get_response_with_agents(user_message)

    def get_response_with_agents(self, user_message: str) -> Optional[str]:
        try:
            result = self.agent_graph.process_query(
                user_message, self.conversation_history
            )

            response_text = result.get("response", "")
            agents_used = result.get("agents_used", [])
            intent = result.get("intent", "")
            quality_score = result.get("quality_score", 0.0)
            research_results = result.get("research_results", [])

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )

            self._last_agents_used = agents_used
            self._last_intent = intent
            self._last_quality_score = quality_score
            self._last_research_results = research_results
            self._last_metadata = result.get("metadata", {})

            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            self.last_rag_used = "research" in agents_used

            print(
                f"Agents used: {' --> '.join(agent for agent in agents_used)} | Intent: {intent}"
            )
            return response_text

        except Exception as e:
            print(f"Agent system error: {e}")
            raise RuntimeError(f"Agent system failed: {e}")

    def clear_history(self):
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

    def get_model_info(self) -> str:
        return self.model_info

    def get_metrics_summary(self) -> dict:
        if self.use_rag and self.rag_manager:
            return self.rag_manager.get_metrics_summary()
        return {"status": "RAG not enabled"}

    def export_metrics(self, filepath: str = "chatbot_metrics.json"):
        if self.use_rag and self.rag_manager:
            self.rag_manager.export_metrics(filepath)
        else:
            print("RAG not enabled, no metrics to export")

    def clear_metrics(self):
        if self.use_rag and self.rag_manager:
            self.rag_manager.clear_metrics()
        else:
            print("RAG not enabled, no metrics to clear")
