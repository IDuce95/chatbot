from typing import Any, Dict, List

from ..base_tool import BaseTool


class SmallTalkHandler(BaseTool):
    def __init__(self, chatbot, config: Dict[str, Any]):
        self.chatbot = chatbot
        self.config = config

    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        query_lower = query.lower().strip()

        small_talk_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What would you like to know?",
            "how are you": "I'm doing well, thank you for asking! How can I assist you?",
            "good morning": "Good morning! What can I help you with today?",
            "good evening": "Good evening! How may I assist you?",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "thanks": "You're welcome! Feel free to ask if you need anything else.",
            "bye": "Goodbye! Have a great day!",
            "goodbye": "Goodbye! Come back anytime if you have more questions."
        }

        for phrase, response in small_talk_responses.items():
            if phrase in query_lower:
                return {
                    'response': response,
                    'confidence': 0.9,
                    'response_type': 'small_talk',
                    'requires_llm': False
                }

        return {
            'response': None,
            'confidence': 0.5,
            'response_type': 'small_talk',
            'requires_llm': True
        }


class ConceptualExplainer(BaseTool):
    def __init__(self, chatbot, config: Dict[str, Any]):
        self.chatbot = chatbot
        self.config = config

    def execute(self, topic: str, complexity_level: str = "intermediate") -> Dict[str, Any]:
        explanation_prompt = self._build_explanation_prompt(complexity_level)

        messages = [
            {"role": "system", "content": explanation_prompt},
            {"role": "user", "content": f"Explain: {topic}"}
        ]

        try:
            response = self.chatbot.client.chat.completions.create(
                model=self.chatbot.config["model"]["name"],
                messages=messages,
                max_tokens=600,
                temperature=0.4
            )

            explanation = response.choices[0].message.content.strip()

            return {
                'explanation': explanation,
                'topic': topic,
                'complexity_level': complexity_level,
                'response_type': 'conceptual_explanation',
                'educational_value': self._assess_educational_value(explanation)
            }

        except Exception as e:
            return {
                'explanation': f"I apologize, but I couldn't generate an explanation for '{topic}' at the moment: {e}",
                'topic': topic,
                'complexity_level': complexity_level,
                'response_type': 'error',
                'educational_value': 0.0
            }

    def _build_explanation_prompt(self, complexity_level: str) -> str:
        level_instructions = {
            "beginner": "Explain in very simple terms, avoiding technical jargon. Use analogies and examples.",
            "intermediate": "Provide a balanced explanation with some technical detail but keep it accessible.",
            "advanced": "Give a comprehensive explanation with technical depth and nuances."
        }

        instruction = level_instructions.get(complexity_level, level_instructions["intermediate"])
        explanation_prompt = self.config["agents"]["conversation"]["explanation_prompt"]
        return explanation_prompt.format(instruction=instruction, topic="{topic}")

    def _assess_educational_value(self, explanation: str) -> float:
        score = 3.0

        educational_indicators = [
            "for example", "imagine", "think of it as", "in other words",
            "this means", "to understand", "the key point", "importantly"
        ]

        for indicator in educational_indicators:
            if indicator.lower() in explanation.lower():
                score += 0.2

        word_count = len(explanation.split())
        if 50 <= word_count <= 300:
            score += 0.5
        elif word_count < 20:
            score -= 1.0

        return max(1.0, min(5.0, score))


class ContextualConversation(BaseTool):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_memory = {}

    def execute(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        if not conversation_history:
            return {
                'context_type': 'new_conversation',
                'suggested_style': 'friendly_introduction',
                'continuation_topics': []
            }

        recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history

        topics = self._extract_topics(recent_messages)

        context_type = self._analyze_conversation_flow(recent_messages, query)

        suggested_style = self._suggest_response_style(context_type, topics)

        return {
            'context_type': context_type,
            'suggested_style': suggested_style,
            'recent_topics': topics,
            'continuation_topics': self._suggest_continuation_topics(topics),
            'conversation_depth': len(conversation_history)
        }

    def _extract_topics(self, messages: List[Dict]) -> List[str]:
        topics = []

        for message in messages:
            content = message.get('content', '').lower()
            words = content.split()

            potential_topics = [word for word in words if len(word) > 4 and word.isalpha()]
            topics.extend(potential_topics[:2])

        return list(set(topics))

    def _analyze_conversation_flow(self, recent_messages: List[Dict], current_query: str) -> str:
        if len(recent_messages) < 2:
            return "starting_conversation"

        question_indicators = ["what about", "how about", "can you also", "what if", "and"]
        if any(indicator in current_query.lower() for indicator in question_indicators):
            return "follow_up_question"

        last_user_message = None
        for msg in reversed(recent_messages):
            if msg.get('role') == 'user':
                last_user_message = msg.get('content', '')
                break

        if last_user_message:
            current_words = set(current_query.lower().split())
            last_words = set(last_user_message.lower().split())
            overlap = len(current_words.intersection(last_words))

            if overlap < 2:
                return "topic_change"

        return "continuing_conversation"

    def _suggest_response_style(self, context_type: str, topics: List[str]) -> str:
        style_map = {
            "starting_conversation": "warm_and_welcoming",
            "follow_up_question": "detailed_and_building",
            "topic_change": "acknowledging_and_transitioning",
            "continuing_conversation": "natural_and_flowing"
        }

        return style_map.get(context_type, "balanced_and_helpful")

    def _suggest_continuation_topics(self, current_topics: List[str]) -> List[str]:
        continuation_suggestions = []

        for topic in current_topics[:3]:

            if "programming" in topic:
                continuation_suggestions.extend(["algorithms", "best practices", "debugging"])
            elif "science" in topic:
                continuation_suggestions.extend(["research", "experiments", "theories"])
            elif "technology" in topic:
                continuation_suggestions.extend(["innovation", "future trends", "applications"])

        return continuation_suggestions[:5]
