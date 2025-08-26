import sys
import os
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import StateGraph, END
from chatbot import ChatBot
from agents.state import AgentState
from agents.router_agent import RouterAgent
from agents.research_agent import ResearchAgent
from agents.code_agent import CodeAgent
from agents.reviewer_agent import ReviewerAgent


class AgentGraph:
    def __init__(self, chatbot: ChatBot, config: dict):
        self.chatbot = chatbot
        self.config = config

        self.router_agent = RouterAgent(chatbot, config)
        self.research_agent = ResearchAgent(chatbot, config)
        self.code_agent = CodeAgent(chatbot, config)
        self.reviewer_agent = ReviewerAgent(chatbot, config)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:

        workflow = StateGraph(AgentState)

        workflow.add_node("router", self._router_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("code", self._code_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("direct_response", self._direct_response_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "research": "research",
                "direct": "direct_response"
            }
        )

        workflow.add_conditional_edges(
            "research",
            self._after_research_decision,
            {
                "code": "code",
                "reviewer": "reviewer"
            }
        )

        workflow.add_edge("code", "reviewer")

        workflow.add_conditional_edges(
            "reviewer",
            self._review_decision,
            {
                "improve": "research",
                "end": END
            }
        )

        workflow.add_edge("direct_response", END)

        return workflow.compile()

    def _router_node(self, state: AgentState) -> AgentState:
        return self.router_agent.process(state)

    def _research_node(self, state: AgentState) -> AgentState:
        print(f"📚 Research conducting for: {state['intent_classification']}")
        return self.research_agent.process(state)

    def _code_node(self, state: AgentState) -> AgentState:
        print("💻 Code generation starting...")
        return self.code_agent.process(state)

    def _reviewer_node(self, state: AgentState) -> AgentState:
        print("✅ Quality review in progress...")
        return self.reviewer_agent.process(state)

    def _direct_response_node(self, state: AgentState) -> AgentState:
        print("🔄 Direct response generation...")

        try:
            messages = []

            system_prompt = self.chatbot.config["system"]["preprompt"]
            messages.append({"role": "system", "content": system_prompt})

            conversation_history = state.get("conversation_history", [])
            messages.extend(conversation_history)

            messages.append({"role": "user", "content": state["user_query"]})

            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.config["model"]["name"],
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )

            state["final_response"] = response.choices[0].message.content.strip()
            state["quality_score"] = 3.5
            state["feedback_loop"] = False

        except Exception as e:
            state["final_response"] = f"I'm sorry, I encountered an error: {e}"
            state["quality_score"] = 1.0
            state["feedback_loop"] = False

        state["current_agent"] = "direct"
        state["agents_visited"].append("direct")

        return state

    def _route_decision(self, state: AgentState) -> str:
        if self.router_agent.should_use_research(state):
            return "research"
        else:
            return "direct"

    def _after_research_decision(self, state: AgentState) -> str:
        intent = state.get("intent_classification", "")
        if intent == "CODE" or self.code_agent.should_generate_code(state):
            return "code"
        else:
            return "reviewer"

    def _review_decision(self, state: AgentState) -> str:
        current_iteration = state.get("iteration_count", 0)

        if self.reviewer_agent.should_improve(state):
            state["iteration_count"] = current_iteration + 1
            print(f"🔄 Feedback loop - iteration {state['iteration_count']}")
            return "improve"
        else:
            if current_iteration >= 1:
                print(f"🛑 Max iterations reached ({current_iteration}), ending feedback loop")
            else:
                print("✅ Quality acceptable, ending process")
            return "end"

    def initialize_state(self, user_query: str, conversation_history: List[dict] = None) -> AgentState:
        return {
            "user_query": user_query,
            "conversation_history": conversation_history or [],
            "intent_classification": "",
            "research_results": [],
            "generated_code": None,
            "final_response": "",
            "quality_score": 0.0,
            "feedback_loop": False,
            "metadata": {},
            "current_agent": "",
            "agents_visited": [],
            "iteration_count": 0
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
                        rag_used=rag_used
                    )
                except Exception as e:
                    print(f"Warning: Failed to log metrics: {e}")

            return {
                "response": response,
                "quality_score": final_state["quality_score"],
                "agents_used": final_state["agents_visited"],
                "intent": final_state["intent_classification"],
                "research_results": final_state.get("research_results", []),
                "metadata": final_state["metadata"]
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
                        rag_used=False
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
