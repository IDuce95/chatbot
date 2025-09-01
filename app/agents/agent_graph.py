from typing import List

from langgraph.graph import END, StateGraph

from .code_agent import CodeAgent
from .conversation_agent import ConversationAgent
from .presenter_agent import PresenterAgent
from .research_agent import ResearchAgent
from .router_agent import RouterAgent
from .state import AgentState


class AgentGraph:
    def __init__(self, chatbot, config: dict):
        self.chatbot = chatbot
        self.config = config

        self.router_agent = RouterAgent(chatbot, config)
        self.research_agent = ResearchAgent(chatbot, config)
        self.code_agent = CodeAgent(chatbot, config)
        self.conversation_agent = ConversationAgent(chatbot, config)
        self.presenter_agent = PresenterAgent(chatbot, config)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:

        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("code", self._code_node)
        workflow.add_node("conversation", self._conversation_node)
        workflow.add_node("presenter", self._presenter_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "research": "research",
                "conversation": "conversation"
            }
        )

        workflow.add_conditional_edges(
            "research",
            self._after_research_decision,
            {
                "code": "code",
                "presenter": "presenter"
            }
        )

        workflow.add_edge("code", "presenter")
        workflow.add_edge("conversation", "presenter")

        workflow.add_edge("presenter", END)

        return workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        state = self.router_agent.process(state)
        self._add_trace_entry(state, "router", {
            "step": "classification",
            "intent": state.get("intent_classification", ""),
            "confidence": state.get("metadata", {}).get("router_confidence", 0),
            "target_agent": state.get("target_agent", "")
        })
        return state

    def _research_node(self, state: AgentState) -> AgentState:
        print(f"📚 Research conducting for: {state['intent_classification']}")

        state = self.research_agent.process(state)
        self._add_trace_entry(state, "research", {
            "step": "retrieve",
            "docs_found": len(state.get("research_results", [])),
            "relevance_score": state.get("metadata", {}).get("relevance_score", 0),
            "sources": [r.get("source", "") for r in state.get("research_results", [])][:3]
        })
        return state

    def _code_node(self, state: AgentState) -> AgentState:
        print("💻 Code generation starting...")
        state = self.code_agent.process(state)
        self._add_trace_entry(state, "code", {
            "step": "code_generation",
            "code_generated": bool(state.get("generated_code")),
            "has_context": state.get("metadata", {}).get("context_used", False)
        })
        return state

    def _conversation_node(self, state: AgentState) -> AgentState:
        state = self.conversation_agent.process(state)
        self._add_trace_entry(state, "conversation", {
            "step": "conversation",
            "type": state.get("metadata", {}).get("conversation_type", "general"),
            "rag_bypassed": True
        })
        return state

    def _presenter_node(self, state: AgentState) -> AgentState:
        print("🎨 Presenter formatting final response...")

        # Get the response content from state
        response_content = state.get("final_response", "")
        metadata = state.get("metadata", {})

        # Add agent information to metadata
        metadata["agent_type"] = state.get("current_agent", "unknown")
        metadata["agents_used"] = state.get("agents_visited", [])
        metadata["intent"] = state.get("intent_classification", "")

        # Add specific data based on agent type
        if state.get("research_results"):
            metadata["research_results"] = state["research_results"]
        if state.get("generated_code"):
            metadata["generated_code"] = state["generated_code"]

        # Process through presenter
        presentation_result = self.presenter_agent.process(response_content, metadata)

        # Update state with formatted response
        state["final_response"] = presentation_result["response"]
        state["metadata"]["presentation"] = presentation_result["metadata"]
        state["metadata"]["presentation_quality"] = presentation_result["presentation_quality"]

        # Add trace entry
        self._add_trace_entry(state, "presenter", {
            "step": "presentation",
            "quality_level": presentation_result["presentation_quality"]["level"],
            "format_applied": True,
            "original_length": len(response_content),
            "final_length": len(presentation_result["response"])
        })

        state["current_agent"] = "presenter"
        if "presenter" not in state["agents_visited"]:
            state["agents_visited"].append("presenter")

        return state

    def _route_decision(self, state: AgentState) -> str:
        target_agent = state.get("target_agent", "conversation_agent")

        if target_agent == "research_agent":
            return "research"
        elif target_agent == "code_agent":
            return "research"
        else:
            return "conversation"

    def _after_research_decision(self, state: AgentState) -> str:
        intent = state.get("intent_classification", "")
        if intent == "CODE" or self.code_agent.should_generate_code(state):
            return "code"
        else:
            return "presenter"

    def _add_trace_entry(self, state: AgentState, agent_name: str, trace_data: dict):
        if "trace" not in state["metadata"]:
            state["metadata"]["trace"] = []

        trace_entry = {
            "agent": agent_name,
            "timestamp": __import__('time').time(),
            **trace_data
        }
        state["metadata"]["trace"].append(trace_entry)

    def initialize_state(self, user_query: str, conversation_history: List[dict] = None) -> AgentState:
        return {
            "user_query": user_query,
            "conversation_history": conversation_history or [],
            "intent_classification": "",
            "target_agent": "",
            "research_results": [],
            "generated_code": None,
            "final_response": "",
            "quality_score": 0.0,
            "metadata": {"trace": []},
            "current_agent": "",
            "agents_visited": []
        }

    def process_query(self, user_query: str, conversation_history: List[dict] = None) -> dict:
        import time
        print(f"\n🚀 Starting agent processing for: {user_query}")

        start_time = time.time()
        initial_state = self.initialize_state(user_query, conversation_history)

        try:
            final_state = self.graph.invoke(initial_state)
            end_time = time.time()
            response_time = end_time - start_time

            print(f"✅ Processing complete. Agents visited: {final_state['agents_visited']}")

            response = final_state["final_response"]

            if hasattr(self.chatbot, 'rag_manager') and self.chatbot.rag_manager:
                rag_used = "research" in final_state['agents_visited']

                retrieved_docs = []
                if rag_used and final_state.get("research_results"):
                    retrieved_docs = final_state["research_results"]

                context = ""
                if retrieved_docs:
                    context = "\n".join([doc.get('content', '') for doc in retrieved_docs if isinstance(doc, dict)])

                try:
                    self.chatbot.rag_manager.metrics.log_interaction(
                        query=user_query,
                        retrieved_docs=retrieved_docs,
                        response=response,
                        context=context,
                        response_time=response_time,
                        rag_used=rag_used,
                        quality_score=final_state.get("quality_score", 0.0)
                    )
                except Exception as e:
                    print(f"Warning: Failed to log metrics: {e}")

            return {
                "response": response,
                "quality_score": final_state["quality_score"],
                "agents_used": final_state["agents_visited"],
                "intent": final_state["intent_classification"],
                "research_results": final_state.get("research_results", []),
                "metadata": final_state["metadata"],
                "trace": final_state["metadata"].get("trace", []),
                "presentation_quality": final_state["metadata"].get("presentation_quality", {})
            }

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ Graph execution error: {e}")

            error_response = f"I'm sorry, I encountered an error processing your request: {e}"

            if hasattr(self.chatbot, 'rag_manager') and self.chatbot.rag_manager:
                try:
                    self.chatbot.rag_manager.metrics.log_interaction(
                        query=user_query,
                        retrieved_docs=[],
                        response=error_response,
                        context="",
                        response_time=response_time,
                        rag_used=False,
                        quality_score=0.0
                    )
                except Exception as log_error:
                    print(f"Warning: Failed to log error metrics: {log_error}")

            return {
                "response": error_response,
                "quality_score": 0.0,
                "agents_used": ["error"],
                "intent": "error",
                "metadata": {"error": str(e)}
            }
