import logging
from abc import ABC, abstractmethod

from ..chatbot import ChatBot
from .state import AgentState


class BaseAgent(ABC):
    def __init__(self, chatbot: ChatBot, name: str, prompt: str):
        self.chatbot = chatbot
        self.name = name
        self.prompt = prompt

    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        pass

    def _call_llm(self, messages: list, max_tokens: int = 500, temperature: float = 0.1) -> str:
        try:
            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.config["model"]["name"],
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error("Exception in _call_llm: %s", e, exc_info=True)
            return "An error occurred while processing your request."

    def _format_conversation_context(self, conversation_history: list, max_messages: int = 6) -> str:
        if not conversation_history:
            return ""

        recent_history = conversation_history[-max_messages:]

        context_lines = []
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                context_lines.append(f"User: {content}")
            elif role == "assistant":
                truncated = content[:200] + "..." if len(content) > 200 else content
                context_lines.append(f"Assistant: {truncated}")

        return "\n".join(context_lines)

    def _call_llm_with_context(self, state: AgentState, system_prompt: str,
                               max_tokens: int = 500, temperature: float = 0.1) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        conversation_context = self._format_conversation_context(state.get("conversation_history", []))
        if conversation_context:
            context_message = f"Previous conversation context:\n{conversation_context}\n\nCurrent question: {state['user_query']}"
            messages.append({"role": "user", "content": context_message})
        else:
            messages.append({"role": "user", "content": state["user_query"]})

        return self._call_llm(messages, max_tokens, temperature)

    def _update_state(self, state: AgentState) -> AgentState:
        state["current_agent"] = self.name.lower()
        if self.name.lower() not in state["agents_visited"]:
            state["agents_visited"].append(self.name.lower())
        return state
