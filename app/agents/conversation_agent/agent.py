from ..base_agent import BaseAgent
from ..state import AgentState


class ConversationAgent(BaseAgent):
    def __init__(self, chatbot, config):
        super().__init__(
            chatbot=chatbot,
            name="Conversation",
            prompt=config["agents"]["conversation"]["prompt"],
        )
        self.config = config

    def process(self, state: AgentState) -> AgentState:
        state = self._update_state(state)
        try:
            print(
                "ConversationAgent: Using direct LLM call for natural conversation (bypassing RAG)"
            )

            messages = self._build_conversation_context(state)

            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.config["model"]["name"],
                messages=messages,
                max_tokens=self.config["agent_parameters"]["conversation_max_tokens"],
                temperature=self.config["agent_parameters"]["conversation_temperature"],
            )

            response_text = response.choices[0].message.content.strip()

            state["final_response"] = response_text
            state["quality_score"] = self._assess_conversation_quality(
                response_text, state["user_query"]
            )

            state["metadata"]["rag_bypassed"] = True
            state["metadata"]["response_style"] = "conversational"

            print(
                f"ConversationAgent: Response generated - Quality: {state['quality_score']:.1f}"
            )

        except Exception as e:
            print(f"❌ ConversationAgent error: {e}")
            state["final_response"] = (
                f"I apologize, but I encountered an error while processing your message: {e}"
            )
            state["quality_score"] = 2.0

        return state

    def _build_conversation_context(self, state: AgentState) -> list:
        messages = []

        conversation_prompt = self.prompt
        messages.append({"role": "system", "content": conversation_prompt})

        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            recent_history = conversation_history[-6:]
            messages.extend(recent_history)

        messages.append({"role": "user", "content": state["user_query"]})

        return messages

    def _assess_conversation_quality(self, response: str, query: str) -> float:
        quality_score = 4.0

        if len(response) < 20:
            quality_score -= 1.0
        elif len(response) > 1000:
            quality_score -= 0.5

        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))

        if overlap < 2:
            quality_score -= 0.5

        conversational_elements = [
            "I think",
            "In my experience",
            "You might",
            "Let me explain",
            "For example",
        ]
        if any(
            element.lower() in response.lower() for element in conversational_elements
        ):
            quality_score += 0.5

        return max(1.0, min(5.0, quality_score))

    def should_handle_conversation(self, state: AgentState) -> bool:
        query = state["user_query"].lower()

        conversation_indicators = [
            "hello",
            "hi",
            "how are you",
            "what's up",
            "thanks",
            "thank you",
            "what is",
            "what are",
            "why",
            "explain",
            "tell me about",
            "discuss",
            "opinion",
            "think",
            "believe",
        ]

        technical_indicators = [
            "code",
            "function",
            "class",
            "import",
            "def",
            "return",
            "python",
            "javascript",
            "html",
            "css",
            "sql",
            "algorithm",
            "debug",
            "error",
            "exception",
        ]

        has_conversation_indicators = any(
            indicator in query for indicator in conversation_indicators
        )
        has_technical_indicators = any(
            indicator in query for indicator in technical_indicators
        )

        return (has_conversation_indicators and not has_technical_indicators) or len(
            query.split()
        ) <= 3
