import pytest
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from chatbot import ChatBot


class TestChatBotBasic:
    @pytest.fixture
    def bot(self):
        return ChatBot()

    def test_chatbot_initialization(self, bot):
        assert bot.model_name == "gpt-4o-mini"
        assert bot.temperature == 0.1
        assert bot.max_tokens == 4000
        assert bot.system_prompt is not None
        assert len(bot.system_prompt) > 0

    def test_api_connection(self, bot):
        result = bot.test_connection()
        assert result is True

    def test_basic_response(self, bot):
        response = bot.get_response("What is Python?")
        assert response is not None
        assert len(response) > 10
        assert "python" in response.lower()


class TestChatBotPythonKnowledge:
    @pytest.fixture
    def bot(self):
        return ChatBot()

    @pytest.mark.parametrize("question,expected_keywords", [
        ("How does list comprehension work in Python?", ["list", "comprehension"]),
        ("What are decorators in Python?", ["decorator"]),
        ("Explain the difference between class methods and static methods", ["class", "static"]),
    ])
    def test_python_knowledge(self, bot, question, expected_keywords):
        response = bot.get_response(question)
        assert response is not None
        assert len(response) > 50

        response_lower = response.lower()
        for keyword in expected_keywords:
            assert keyword.lower() in response_lower

    @pytest.mark.parametrize("question", [
        "How to use pandas to read a CSV file?",
        "Show an example of using numpy for matrix operations",
        "How to create a simple Flask application?",
    ])
    def test_library_knowledge_with_code(self, bot, question):
        response = bot.get_response(question)
        assert response is not None
        assert ("import" in response or "def" in response)


class TestChatBotGeneralKnowledge:
    @pytest.fixture
    def bot(self):
        return ChatBot()

    @pytest.mark.parametrize("question", [
        "What are the advantages of remote work?",
        "Tell me about computer history",
    ])
    def test_general_questions(self, bot, question):
        response = bot.get_response(question)
        assert response is not None
        assert len(response) > 100


class TestChatBotPerformance:
    @pytest.fixture
    def bot(self):
        return ChatBot()

    def test_response_time(self, bot):
        start_time = time.time()
        response = bot.get_response("What is Python?")
        end_time = time.time()

        response_time = end_time - start_time
        assert response is not None
        assert response_time < 15

    def test_multiple_requests_performance(self, bot):
        questions = [
            "What is Python?",
            "How to create a function?",
            "What are lists?",
            "Explain for loops",
            "What is pandas?"
        ]

        total_time = 0
        successful_responses = 0

        for question in questions:
            start = time.time()
            response = bot.get_response(question)
            end = time.time()

            total_time += (end - start)
            if response:
                successful_responses += 1

        avg_time = total_time / len(questions)
        success_rate = (successful_responses / len(questions)) * 100

        # Multiple requests should have reasonable average time
        assert avg_time < 15
        assert success_rate >= 100


class TestChatBotErrorHandling:
    @pytest.fixture
    def bot(self):
        return ChatBot()

    def test_empty_message(self, bot):
        response = bot.get_response("")
        assert response is not None

    def test_very_long_message(self, bot):
        long_message = "What is Python? " * 100
        response = bot.get_response(long_message)
        assert response is not None

    def test_special_characters(self, bot):
        response = bot.get_response("What is Python? 🐍 @#$%^&*()")
        assert response is not None


class TestChatBotConfiguration:
    def test_missing_config_file(self):
        with pytest.raises(FileNotFoundError):
            ChatBot(config_path="nonexistent_config.toml")

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Missing OpenAI API key"):
            ChatBot()
