import pytest
from unittest.mock import Mock

from app.agents.router_agent.tools import ClassifierTool, DelegationTool
from app.agents.presenter_agent.tools import TextFormatterTool
from app.pydantic_models import RouterDecision


class TestAgentTools:
    @pytest.fixture
    def mock_chatbot(self):
        class MockChatbot:
            def __init__(self):
                self.config = {
                    "model": {"name": "gpt-4o-mini"},
                    "agent_parameters": {
                        "classification_max_tokens": 200,
                        "classification_temperature": 0.1,
                    },
                }
                self.client = None

        return MockChatbot()

    @pytest.fixture
    def mock_config(self):
        return {
            "agents": {
                "router": {"prompt": "You are a router agent."},
                "research": {"min_relevance": 0.3},
            },
            "agent_parameters": {
                "classification_max_tokens": 200,
                "classification_temperature": 0.1,
            },
            "keyword_classification": {
                "code_keywords": ["function", "code", "implement", "write", "create"],
                "documentation_keywords": [
                    "documentation",
                    "docs",
                    "langchain",
                    "explain",
                    "what is",
                ],
                "conversation_keywords": ["hello", "hi", "thanks", "how are you"],
            },
        }

    def test_classifier_tool_keyword_classification_code(
        self, mock_chatbot, mock_config
    ):
        classifier = ClassifierTool(mock_chatbot, mock_config)
        result = classifier._classify_by_keywords("Write a function to sort data")

        assert result.intent == "CODE"
        assert result.confidence == 0.8

    def test_classifier_tool_keyword_classification_documentation(
        self, mock_chatbot, mock_config
    ):
        classifier = ClassifierTool(mock_chatbot, mock_config)
        result = classifier._classify_by_keywords("What is LangChain documentation?")

        assert result.intent == "DOCUMENTATION"
        assert result.confidence == 0.8

    def test_classifier_tool_keyword_classification_conversation(
        self, mock_chatbot, mock_config
    ):
        classifier = ClassifierTool(mock_chatbot, mock_config)
        result = classifier._classify_by_keywords("Hello, how are you?")

        assert result.intent == "GENERAL"
        assert result.confidence == 0.8

    def test_classifier_tool_fallback_classification(self, mock_chatbot, mock_config):
        classifier = ClassifierTool(mock_chatbot, mock_config)
        result = classifier._classify_by_keywords("Random unclassifiable text xyz")

        assert result.intent == "GENERAL"
        assert result.confidence == 0.6

    def test_delegation_tool_code_mapping(self, mock_chatbot, mock_config):
        delegator = DelegationTool(mock_chatbot, mock_config)
        decision = RouterDecision(intent="CODE", confidence=0.9)
        result = delegator.execute(decision, "Write a function", [])

        assert result["target_agent"] == "code_agent"
        assert result["original_intent"] == "CODE"
        assert result["confidence"] == 0.9

    def test_delegation_tool_documentation_mapping(self, mock_chatbot, mock_config):
        delegator = DelegationTool(mock_chatbot, mock_config)
        decision = RouterDecision(intent="DOCUMENTATION", confidence=0.85)
        result = delegator.execute(decision, "What is LangChain?", [])

        assert result["target_agent"] == "research_agent"
        assert result["original_intent"] == "DOCUMENTATION"
        assert result["confidence"] == 0.85

    def test_delegation_tool_general_mapping(self, mock_chatbot, mock_config):
        delegator = DelegationTool(mock_chatbot, mock_config)
        decision = RouterDecision(intent="GENERAL", confidence=0.75)
        result = delegator.execute(decision, "Hello there", [])

        assert result["target_agent"] == "conversation_agent"
        assert result["original_intent"] == "GENERAL"
        assert result["confidence"] == 0.75

    def test_delegation_tool_confidence_levels(self, mock_chatbot, mock_config):
        delegator = DelegationTool(mock_chatbot, mock_config)

        decision_high = RouterDecision(intent="CODE", confidence=0.95)
        result_high = delegator.execute(decision_high, "test", [])
        assert result_high["routing_metadata"]["confidence_level"] == "very_high"

        decision_medium = RouterDecision(intent="CODE", confidence=0.65)
        result_medium = delegator.execute(decision_medium, "test", [])
        assert result_medium["routing_metadata"]["confidence_level"] == "medium"

        decision_low = RouterDecision(intent="CODE", confidence=0.45)
        result_low = delegator.execute(decision_low, "test", [])
        assert result_low["routing_metadata"]["confidence_level"] == "low"

    def test_delegation_tool_fallback_options(self, mock_chatbot, mock_config):
        delegator = DelegationTool(mock_chatbot, mock_config)
        decision = RouterDecision(intent="CODE", confidence=0.4)
        result = delegator.execute(decision, "test", [])

        fallbacks = result["routing_metadata"]["fallback_options"]
        assert "research_agent" in fallbacks
        assert "conversation_agent" in fallbacks

    def test_text_formatter_tool_empty_content(self):
        mock_agent = Mock()
        formatter = TextFormatterTool(mock_agent)
        result = formatter.process("", {})

        assert result == "I don't have a specific answer for that question."

    def test_text_formatter_tool_code_formatting(self):
        mock_agent = Mock()
        formatter = TextFormatterTool(mock_agent)
        code_content = "def hello():\n    print('Hello World')"

        result = formatter.process(code_content, {"agent_type": "code"})

        assert "```python" in result
        assert "def hello():" in result
        assert "```" in result

    def test_text_formatter_tool_research_formatting(self):
        mock_agent = Mock()
        formatter = TextFormatterTool(mock_agent)
        research_content = "This is research content about programming."
        metadata = {
            "agent_type": "research",
            "sources": ["source1.pdf", "source2.pdf", "source3.pdf"],
        }

        result = formatter.process(research_content, metadata)

        assert "This is research content about programming." in result
        assert "**Sources:**" in result
        assert "source1.pdf" in result
        assert "source2.pdf" in result

    def test_text_formatter_tool_conversation_formatting(self):
        mock_agent = Mock()
        formatter = TextFormatterTool(mock_agent)

        short_response = "Hi!"
        result_short = formatter.process(short_response, {"agent_type": "conversation"})
        assert "✨ Hi!" in result_short

        long_response = "Hello there, how can I help you today?"
        result_long = formatter.process(long_response, {"agent_type": "conversation"})
        assert result_long == long_response

    def test_text_formatter_tool_truncation(self):
        mock_agent = Mock()
        formatter = TextFormatterTool(mock_agent)
        long_content = "x" * 6000

        result = formatter.process(long_content, {})

        assert len(result) < 5000
        assert "*[Response truncated for readability]*" in result

    def test_text_formatter_tool_error_handling(self):
        mock_agent = Mock()
        formatter = TextFormatterTool(mock_agent)

        content = "Test content"

        result = formatter.process(content, {})
        assert result == content
