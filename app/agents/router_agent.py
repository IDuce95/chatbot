import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from agents.base_agent import BaseAgent
from agents.state import AgentState
from pydantic_models import RouterDecision


class RouterAgent(BaseAgent):
    def __init__(self, chatbot, config):
        super().__init__(
            chatbot=chatbot,
            name="Router",
            prompt=config["agents"]["router"]["prompt"]
        )

    def process(self, state: AgentState) -> AgentState:
        return self.route_query(state)

    def _classify_intent(self, state: AgentState) -> RouterDecision:
        classification_prompt = f"{self.prompt}\n\nPlease classify the user's query considering the conversation context."

        response = self._call_llm_with_context(state, classification_prompt, max_tokens=100, temperature=0.1)

        try:
            response_json = json.loads(response)
            return RouterDecision(**response_json)
        except Exception:
            return RouterDecision(intent="GENERAL", confidence=0.5)

    def route_query(self, state: AgentState) -> AgentState:
        user_query = state["user_query"]

        try:
            decision = self._classify_intent(state)

            state["intent_classification"] = decision.intent
            state["metadata"]["router_confidence"] = decision.confidence

            print(f"🎯 Router analyzing: {user_query[:50]}...")
            print(f"🎯 Intent classified as: {decision.intent} (confidence: {decision.confidence:.2f})")

        except Exception as e:
            print(f"Router error: {e}")
            state["intent_classification"] = "GENERAL"
            state["metadata"]["router_error"] = str(e)

        return self._update_state(state)

    def should_use_research(self, state: AgentState) -> bool:
        intent = state.get("intent_classification", "GENERAL")
        return intent in ["DOCUMENTATION", "CODE"]

    def should_use_direct(self, state: AgentState) -> bool:
        return not self.should_use_research(state)
