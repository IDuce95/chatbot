import pytest
import sys
import os
import time
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.chatbot import ChatBot


class TestChatBot:
    @pytest.fixture
    def bot(self):
        return ChatBot(use_rag=False)

    @pytest.fixture
    def bot_with_rag(self):
        return ChatBot(use_rag=True)

    @pytest.fixture
    def mock_config(self):
        return {
            "model": {
                "name": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 4000
            },
            "system": {
                "preprompt": "You are a test assistant"
            }
        }

    def test_chatbot_initialization(self, bot):
        assert bot.config["model"]["name"] == "gpt-4o-mini"
        assert bot.config["model"]["temperature"] == 0.1
        assert bot.config["model"]["max_tokens"] == 4000
        assert bot.config["system"]["preprompt"] is not None
        assert len(bot.config["system"]["preprompt"]) > 0

    def test_basic_response(self, bot):
        response = bot.get_response("Hello")
        assert response is not None
        assert len(response) > 0

    def test_empty_message(self, bot):
        response = bot.get_response("")
        assert response is not None

    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            ChatBot(config_path="nonexistent_config.toml")

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Missing OpenAI API key"):
            ChatBot()

    @patch('chatbot.toml.load')
    @patch('chatbot.os.getenv')
    @patch('chatbot.openai.OpenAI')
    def test_chatbot_initialization_unit(self, mock_openai, mock_getenv, mock_toml_load, mock_config):
        mock_toml_load.return_value = mock_config
        mock_getenv.return_value = "test-api-key"
        mock_client = Mock()
        mock_openai.return_value = mock_client

        with patch('builtins.print'):
            bot = ChatBot(use_rag=False)

        assert bot.config["model"]["name"] == "gpt-4o-mini"
        assert bot.config["model"]["temperature"] == 0.1
        assert bot.config["model"]["max_tokens"] == 4000
        assert bot.config["system"]["preprompt"] == "You are a test assistant"

    @patch('chatbot.toml.load')
    @patch('chatbot.os.getenv')
    @patch('chatbot.openai.OpenAI')
    @patch('agents.agent_graph.AgentGraph')
    def test_get_response_success_unit(self, mock_agent_graph, mock_openai, mock_getenv, mock_toml_load, mock_config):
        mock_toml_load.return_value = mock_config
        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_agent_instance = Mock()
        mock_agent_instance.process_query.return_value = {
            "response": "Test response",
            "agents_used": ["router"],
            "intent": "test",
            "quality_score": 4.0,
            "research_results": [],
            "metadata": {}
        }
        mock_agent_graph.return_value = mock_agent_instance

        with patch('builtins.print'):
            bot = ChatBot()

        response = bot.get_response("Test question")
        assert response == "Test response"
        mock_agent_instance.process_query.assert_called_once()

    @patch('chatbot.toml.load')
    @patch('chatbot.os.getenv')
    @patch('chatbot.openai.OpenAI')
    @patch('agents.agent_graph.AgentGraph')
    def test_get_response_error_unit(self, mock_agent_graph, mock_openai, mock_getenv, mock_toml_load, mock_config):
        mock_toml_load.return_value = mock_config
        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_openai.return_value = mock_client

        mock_agent_instance = Mock()
        mock_agent_instance.process_query.side_effect = Exception("Agent Error")
        mock_agent_graph.return_value = mock_agent_instance

        with patch('builtins.print'):
            bot = ChatBot()

        with patch('builtins.print'):
            with pytest.raises(RuntimeError, match="Agent system failed"):
                bot.get_response("Test question")

    def test_response_time(self, bot):
        start_time = time.time()
        response = bot.get_response("Hello")
        end_time = time.time()

        response_time = end_time - start_time
        assert response is not None
        assert response_time < 15

    def test_conversation_history_initialization(self, bot):
        assert bot.conversation_history == []
        assert len(bot.get_history()) == 0

    def test_conversation_history_storage(self, bot):
        bot.get_response("Hello")
        history = bot.get_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert len(history[1]["content"]) > 0

    def test_conversation_history_multiple_messages(self, bot):
        bot.get_response("First message")
        bot.get_response("Second message")

        history = bot.get_history()
        assert len(history) == 4
        assert history[0]["content"] == "First message"
        assert history[2]["content"] == "Second message"

    def test_clear_history(self, bot):
        bot.get_response("Test message")
        assert len(bot.get_history()) == 2

        bot.clear_history()
        assert len(bot.get_history()) == 0
        assert bot.conversation_history == []

    def test_get_model_info(self, bot):
        model_info = bot.get_model_info()
        assert "CodeBot initialized with model:" in model_info
        assert "gpt-4o-mini" in model_info

    def test_rag_functionality(self, bot_with_rag):
        if not bot_with_rag.use_rag:
            pytest.skip("RAG not available")

        response = bot_with_rag.get_response("What is LangChain?")

        assert response is not None
        assert len(response) > 50
        assert "langchain" in response.lower() or "framework" in response.lower()

        assert "RAG" in bot_with_rag.model_info or "knowledge base" in bot_with_rag.model_info
