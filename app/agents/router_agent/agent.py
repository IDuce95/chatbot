from ..base_agent import BaseAgent
from ..state import AgentState
from ...pydantic_models import RouterDecision
from .tools import ClassifierTool, DelegationTool


class RouterAgent(BaseAgent):
    def __init__(self, chatbot, config):
        prompt = config["agents"]["router"]["prompt"]
        super().__init__(chatbot, "Router", prompt)
        self.config = config
        self.classifier = ClassifierTool(chatbot, config)
        self.delegator = DelegationTool(chatbot, config)

    def process(self, state: AgentState) -> AgentState:
        return self.route_query(state)

    def _classify_intent(self, state: AgentState) -> RouterDecision:
        user_query = state["user_query"]
        conversation_history = state.get("conversation_history", [])

        return self.classifier.execute(user_query, conversation_history)

    def route_query(self, state: AgentState) -> AgentState:
        user_query = state["user_query"]

        try:
            decision = self._classify_intent(state)

            delegation_info = self.delegator.execute(
                decision,
                user_query,
                state.get("conversation_history", [])
            )

            state["intent_classification"] = decision.intent
            state["target_agent"] = delegation_info["target_agent"]
            state["metadata"]["router_confidence"] = decision.confidence
            state["metadata"]["routing_info"] = delegation_info["routing_metadata"]

            print(f"🎯 Router analyzing: {user_query[:50]}...")
            print(f"🎯 Intent classified as: {decision.intent} (confidence: {decision.confidence:.2f})")
            print(f"🎯 Delegating to: {delegation_info['target_agent']}")

        except Exception as e:
            print(f"Router error: {e}")
            state["intent_classification"] = "GENERAL"
            state["target_agent"] = "conversation_agent"
            state["metadata"]["router_error"] = str(e)

        return self._update_state(state)

    def should_use_research(self, state: AgentState) -> bool:
        target_agent = state.get("target_agent", "conversation_agent")
        return target_agent == "research_agent"

    def get_target_agent(self, state: AgentState) -> str:
        return state.get("target_agent", "conversation_agent")

    def should_use_direct(self, state: AgentState) -> bool:
        return not self.should_use_research(state)
