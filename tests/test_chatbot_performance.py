import pytest
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from chatbot import ChatBot


class TestChatBotPerformance:

    @pytest.fixture
    def bot(self):
        return ChatBot()

    @pytest.mark.slow
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

        print("\nPerformance Results:")
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {total_time:.2f}s")

        assert avg_time < 15
        assert success_rate >= 100

    @pytest.mark.slow
    def test_complex_questions_performance(self, bot):
        complex_questions = [
            "Explain object-oriented programming in Python with examples",
            "How to implement a RESTful API using FastAPI with authentication",
            "Describe the differences between pandas DataFrame and NumPy arrays"
        ]

        for question in complex_questions:
            start = time.time()
            response = bot.get_response(question)
            end = time.time()

            response_time = end - start

            assert response is not None
            assert len(response) > 100
            assert response_time < 20

    @pytest.mark.slow
    def test_concurrent_requests_simulation(self, bot):
        questions = ["What is Python?"] * 3

        start_time = time.time()
        responses = []

        for question in questions:
            response = bot.get_response(question)
            responses.append(response)

        total_time = time.time() - start_time

        assert len(responses) == 3
        assert all(r is not None for r in responses)
        assert total_time < 30
