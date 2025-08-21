import pytest
import sys
import os
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from chatbot import ChatBot


class TestChatBotUnit:

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

    @patch('chatbot.os.getenv')
    def test_missing_api_key_unit(self, mock_getenv):
        mock_getenv.return_value = None

        with pytest.raises(ValueError, match="Missing OpenAI API key"):
            ChatBot()

    @patch('chatbot.toml.load')
    def test_missing_config_file_unit(self, mock_toml_load):
        mock_toml_load.side_effect = FileNotFoundError("Config file not found")

        with pytest.raises(FileNotFoundError):
            ChatBot()

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

        with patch('builtins.print') as mock_print:
            response = bot.get_response("Test question")

        assert response is None
        mock_print.assert_called_with("❌ Error communicating with OpenAI: API Error")
