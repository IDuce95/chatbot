import pytest
import sys
import os
import time
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from chatbot import ChatBot


class TestChatBot:
    @pytest.fixture
    def bot(self):
        return ChatBot()

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
        assert bot.model_name == "gpt-4o-mini"
        assert bot.temperature == 0.1
        assert bot.max_tokens == 4000
        assert bot.system_prompt is not None
        assert len(bot.system_prompt) > 0

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
            bot = ChatBot()

        assert bot.model_name == "gpt-4o-mini"
        assert bot.temperature == 0.1
        assert bot.max_tokens == 4000
        assert bot.system_prompt == "You are a test assistant"

    @patch('chatbot.toml.load')
    @patch('chatbot.os.getenv')
    @patch('chatbot.openai.OpenAI')
    def test_get_response_success_unit(self, mock_openai, mock_getenv, mock_toml_load, mock_config):
        mock_toml_load.return_value = mock_config
        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with patch('builtins.print'):
            bot = ChatBot()

        response = bot.get_response("Test question")
        assert response == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('chatbot.toml.load')
    @patch('chatbot.os.getenv')
    @patch('chatbot.openai.OpenAI')
    def test_get_response_error_unit(self, mock_openai, mock_getenv, mock_toml_load, mock_config):
        mock_toml_load.return_value = mock_config
        mock_getenv.return_value = "test-api-key"

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        with patch('builtins.print'):
            bot = ChatBot()

        with patch('builtins.print'):
            response = bot.get_response("Test question")

        assert response is None

    def test_response_time(self, bot):
        start_time = time.time()
        response = bot.get_response("Hello")
        end_time = time.time()

        response_time = end_time - start_time
        assert response is not None
        assert response_time < 15
