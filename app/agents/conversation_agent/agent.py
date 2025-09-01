from ..base_agent import BaseAgent
from ..state import AgentState


class ConversationAgent(BaseAgent):
    def __init__(self, chatbot, config):
        super().__init__(
            chatbot=chatbot,
            name="Conversation",
            prompt=config["agents"]["conversation"]["prompt"]
        )
        self.config = config

    def process(self, state: AgentState) -> AgentState:
        try:
            print(f"💬 Conversation agent processing: {state['user_query'][:50]}...")

            messages = self._build_conversation_context(state)

            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.config["model"]["name"],
                messages=messages,
                max_tokens=self.config["agent_parameters"]["conversation_max_tokens"],
                temperature=self.config["agent_parameters"]["conversation_temperature"]
            )

            response_text = response.choices[0].message.content.strip()

            state["final_response"] = response_text
            state["quality_score"] = self._assess_conversation_quality(response_text, state["user_query"])
            state["feedback_loop"] = False
            state["current_agent"] = "conversation"
            state["agents_visited"].append("conversation")

            state["metadata"]["conversation_type"] = self._classify_conversation_type(state["user_query"])
            state["metadata"]["rag_bypassed"] = True
            state["metadata"]["response_style"] = "conversational"

            print(f"💬 Conversation response generated (quality: {state['quality_score']:.1f})")

        except Exception as e:
            print(f"❌ Conversation agent error: {e}")
            state["final_response"] = f"I apologize, but I encountered an error while processing your message: {e}"
            state["quality_score"] = 2.0
            state["feedback_loop"] = False

        return state

    def _build_conversation_context(self, state: AgentState) -> list:
        messages = []

        conversation_prompt = self._get_conversation_system_prompt()
        messages.append({"role": "system", "content": conversation_prompt})

        conversation_history = state.get("conversation_history", [])
        if conversation_history:
            recent_history = conversation_history[-6:]
            messages.extend(recent_history)

        messages.append({"role": "user", "content": state["user_query"]})

        return messages

    def _get_conversation_system_prompt(self) -> str:
        return """You are a friendly and knowledgeable AI assistant. You excel at:

- Having natural, engaging conversations
- Answering general knowledge questions
- Discussing concepts and ideas
- Providing thoughtful explanations on various topics
- Being helpful with everyday questions and small-talk

Your conversation style:
- Be warm, friendly, and approachable
- Show genuine interest in the user's questions
- Provide clear, well-structured responses
- Use examples when helpful
- Admit when you don't know something
- Keep responses conversational but informative

You do NOT need to write code or search technical documentation for these types of questions.
Focus on being a great conversation partner and knowledge resource."""

    def _classify_conversation_type(self, query: str) -> str:
        query_lower = query.lower()

        small_talk_keywords = ["hello", "hi", "how are you", "what's up", "good morning", "good evening"]
        if any(keyword in query_lower for keyword in small_talk_keywords):
            return "small_talk"

        question_keywords = ["what is", "what are", "why", "how does", "explain", "tell me about"]
        if any(keyword in query_lower for keyword in question_keywords):
            return "conceptual_question"

        discussion_keywords = ["think", "opinion", "believe", "feel", "discuss"]
        if any(keyword in query_lower for keyword in discussion_keywords):
            return "discussion"

        return "general_conversation"

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

        conversational_elements = ["I think", "In my experience", "You might", "Let me explain", "For example"]
        if any(element.lower() in response.lower() for element in conversational_elements):
            quality_score += 0.5

        return max(1.0, min(5.0, quality_score))

    def should_handle_conversation(self, state: AgentState) -> bool:
        query = state["user_query"].lower()

        conversation_indicators = [
            "hello", "hi", "how are you", "what's up", "thanks", "thank you",
            "what is", "what are", "why", "explain", "tell me about",
            "discuss", "opinion", "think", "believe"
        ]

        technical_indicators = [
            "code", "function", "class", "import", "def", "return",
            "python", "javascript", "html", "css", "sql",
            "algorithm", "debug", "error", "exception"
        ]

        has_conversation_indicators = any(indicator in query for indicator in conversation_indicators)
        has_technical_indicators = any(indicator in query for indicator in technical_indicators)

        return (has_conversation_indicators and not has_technical_indicators) or len(query.split()) <= 3
