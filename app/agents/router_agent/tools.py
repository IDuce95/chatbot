from typing import Any, Dict, List

from ...pydantic_models import RouterDecision
from ..base_tool import BaseTool


class ClassifierTool(BaseTool):
    def __init__(self, chatbot, config: Dict[str, Any]):
        self.chatbot = chatbot
        self.config = config

    def execute(self, query: str) -> RouterDecision:
        classification_prompt = self.config["agents"]["router"]["prompt"]

        messages = [
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": query}
        ]

        try:
            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.config["model"]["name"],
                messages=messages,
                max_tokens=self.config["agent_parameters"]["classification_max_tokens"],
                temperature=self.config["agent_parameters"]["classification_temperature"]
            )

            content = response.choices[0].message.content.strip()

            import json
            try:
                result = json.loads(content)
                return RouterDecision(
                    intent=result.get("intent", "GENERAL"),
                    confidence=float(result.get("confidence", 0.75))
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                return self._classify_by_keywords(query)

        except Exception as e:
            print(f"Classification error: {e}")
            return self._classify_by_keywords(query)

    def _classify_by_keywords(self, query: str) -> RouterDecision:
        query_lower = query.lower()

        code_keywords = self.config["keyword_classification"]["code_keywords"]
        doc_keywords = self.config["keyword_classification"]["documentation_keywords"]
        conv_keywords = self.config["keyword_classification"]["conversation_keywords"]

        if any(word in query_lower for word in code_keywords):
            return RouterDecision(intent="CODE", confidence=0.8)

        if any(word in query_lower for word in doc_keywords):
            return RouterDecision(intent="DOCUMENTATION", confidence=0.8)

        if any(word in query_lower for word in conv_keywords):
            return RouterDecision(intent="GENERAL", confidence=0.8)

        return RouterDecision(intent="GENERAL", confidence=0.6)


class DelegationTool(BaseTool):

    def __init__(self, chatbot, config: Dict[str, Any]):
        self.chatbot = chatbot
        self.config = config
        self.agent_mapping = {
            "CODE": "code_agent",
            "DOCUMENTATION": "research_agent",
            "GENERAL": "conversation_agent"
        }

    def execute(self, decision: RouterDecision, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        target_agent = self.agent_mapping.get(decision.intent, "conversation_agent")

        delegation_info = {
            "target_agent": target_agent,
            "original_intent": decision.intent,
            "confidence": decision.confidence,
            "query": query,
            "conversation_history": conversation_history or [],
            "routing_metadata": {
                "router_decision": decision.intent,
                "confidence_level": self._get_confidence_level(decision.confidence),
                "fallback_options": self._get_fallback_options(decision.intent, decision.confidence)
            }
        }

        return delegation_info

    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"

    def _get_fallback_options(self, intent: str, confidence: float) -> List[str]:
        fallbacks = []

        if confidence < 0.6:
            if intent == "CODE":
                fallbacks.extend(["research_agent", "conversation_agent"])
            elif intent == "DOCUMENTATION":
                fallbacks.extend(["conversation_agent", "code_agent"])
            elif intent == "GENERAL":
                fallbacks.extend(["research_agent"])

        return fallbacks
