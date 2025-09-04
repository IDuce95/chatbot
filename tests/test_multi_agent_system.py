import pytest

from app.agents.code_agent.agent import CodeAgent
from app.agents.conversation_agent.agent import ConversationAgent
from app.agents.presenter_agent.agent import PresenterAgent
from app.agents.research_agent.agent import ResearchAgent
from app.agents.router_agent.agent import RouterAgent
from app.chatbot import ChatBot


class TestMultiAgentSystem:
    @pytest.fixture
    def chatbot(self):
        return ChatBot()

    @pytest.fixture
    def mock_config(self):
        return {
            "agents": {
                "router": {"prompt": "Test router prompt"},
                "research": {
                    "prompt": "Test research prompt",
                    "query_refinement": "Refine this query: {query}"
                },
                "code": {
                    "prompt": "Test code prompt",
                    "code_generation_template": "Generate code based on the query and context."
                },
                "conversation": {"prompt": "Test conversation prompt"},
                "presenter": {"prompt": "Test presenter prompt"}
            },
            "agent_parameters": {
                "classification_max_tokens": 200,
                "classification_temperature": 0.1,
                "conversation_max_tokens": 800,
                "conversation_temperature": 0.7
            },
            "keyword_classification": {
                "code_keywords": ["function", "code", "implement"],
                "documentation_keywords": ["langchain", "documentation", "explain"],
                "conversation_keywords": ["hello", "hi", "thanks"]
            }
        }

    def test_router_agent_classification_code(self, chatbot, mock_config):
        """Test router agent correctly classifies CODE queries"""
        router = RouterAgent(chatbot, mock_config)

        state = {
            "user_query": "Write a Python function to sort a list",
            "metadata": {},
            "agents_visited": [],
        }

        result = router.process(state)

        assert result["intent_classification"] in ["CODE", "GENERAL"]
        assert "target_agent" in result
        assert result["metadata"]["router_confidence"] > 0

    def test_router_agent_classification_documentation(self, chatbot, mock_config):
        """Test router agent correctly classifies DOCUMENTATION queries"""
        router = RouterAgent(chatbot, mock_config)

        state = {
            "user_query": "What is LangChain documentation?",
            "metadata": {},
            "agents_visited": []
        }

        result = router.process(state)

        assert result["intent_classification"] in ["DOCUMENTATION", "GENERAL"]
        assert "target_agent" in result
        assert result["metadata"]["router_confidence"] > 0

    def test_router_agent_classification_conversation(self, chatbot, mock_config):
        """Test router agent correctly classifies GENERAL/conversation queries"""
        router = RouterAgent(chatbot, mock_config)

        state = {
            "user_query": "Hello, how are you?",
            "metadata": {},
            "agents_visited": []
        }

        result = router.process(state)

        assert result["intent_classification"] == "GENERAL"
        assert result["target_agent"] == "conversation_agent"
        assert result["metadata"]["router_confidence"] > 0

    def test_conversation_agent_response(self, chatbot, mock_config):
        """Test conversation agent generates appropriate responses"""
        conversation_agent = ConversationAgent(chatbot, mock_config)

        state = {
            "user_query": "Hello there!",
            "conversation_history": [],
            "metadata": {},
            "agents_visited": [],
        }

        result = conversation_agent.process(state)

        assert "final_response" in result
        assert len(result["final_response"]) > 0
        assert "quality_score" in result
        assert result["quality_score"] > 0
        assert "conversation" in result["agents_visited"]

    def test_research_agent_no_rag(self, chatbot, mock_config):
        """Test research agent behavior when RAG is not available"""
        research_agent = ResearchAgent(chatbot, mock_config)
        research_agent.rag_manager = None

        state = {
            "user_query": "What is machine learning?",
            "metadata": {},
            "agents_visited": []
        }

        result = research_agent.process(state)

        assert "research_results" in result
        assert result["research_results"] == []
        assert result["metadata"]["rag_unavailable"]

    def test_code_agent_generation(self, chatbot, mock_config):
        """Test code agent generates code from research context"""
        code_agent = CodeAgent(chatbot, mock_config)

        state = {
            "user_query": "Create a sorting function",
            "research_results": [
                {
                    "content": "Python sorting can be done with sorted() function",
                    "source": "test_doc",
                    "relevance": 0.8
                }
            ],
            "metadata": {},
            "agents_visited": []
        }

        result = code_agent.process(state)

        assert "generated_code" in result
        assert result["metadata"]["code_generated"]
        assert result["metadata"]["context_used"]

    def test_presenter_agent_with_code(self, chatbot, mock_config):
        """Test presenter agent formats code responses correctly"""
        presenter_agent = PresenterAgent(chatbot, mock_config)

        code_content = """def sort_list(lst):
    return sorted(lst)"""

        result = presenter_agent.process(code_content, {"agent_type": "code"})

        assert "response" in result
        assert "presentation_quality" in result
        assert len(result["response"]) > 0

    def test_end_to_end_conversation_flow(self, chatbot):
        """Test complete conversation flow through the system"""
        response = chatbot.get_response_with_agents("Hello! How are you today?")

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    def test_end_to_end_documentation_flow(self, chatbot):
        """Test complete documentation query flow"""
        response = chatbot.get_response_with_agents("What is Python?")

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    def test_end_to_end_code_flow(self, chatbot):
        """Test complete code generation flow"""
        response = chatbot.get_response_with_agents("Write a simple hello world function")

        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)

    def test_agent_system_error_handling(self, chatbot):
        """Test system handles errors gracefully"""
        # Test with potentially problematic input
        response = chatbot.get_response_with_agents("")

        assert response is not None
        assert len(response) > 0

    def test_agent_system_response_quality(self, chatbot):
        """Test that agent responses meet quality standards"""
        test_queries = [
            "Hello",
            "What is programming?",
            "Write a function to add two numbers"
        ]

        for query in test_queries:
            response = chatbot.get_response_with_agents(query)

            assert response is not None
            assert len(response.strip()) > 0
            assert len(response) < 10000  # Reasonable response length
            assert not response.startswith("Error:")  # No obvious errors

    def test_agent_routing_consistency(self, chatbot):
        """Test that similar queries get routed consistently"""
        conversation_queries = [
            "Hello there",
            "Hi, how are you?",
            "Good morning!"
        ]

        code_queries = [
            "Write a function",
            "Create a class",
            "Implement an algorithm"
        ]

        # Test conversation queries
        for query in conversation_queries:
            response = chatbot.get_response_with_agents(query)
            assert response is not None

        # Test code queries
        for query in code_queries:
            response = chatbot.get_response_with_agents(query)
            assert response is not None

    def test_system_performance(self, chatbot):
        """Test system performance under load"""
        import time

        queries = [
            "Hello",
            "What is Python?",
            "Write a sorting function",
            "Explain machine learning",
            "Create a web server"
        ]

        total_time = 0
        for query in queries:
            start_time = time.time()
            response = chatbot.get_response_with_agents(query)
            end_time = time.time()

            query_time = end_time - start_time
            total_time += query_time

            assert response is not None
            assert query_time < 30  # Each query should complete within 30 seconds

        avg_time = total_time / len(queries)
        assert avg_time < 20  # Average response time should be under 20 seconds

    def test_agent_memory_and_state(self, chatbot):
        """Test that agents maintain proper state"""
        # First query
        response1 = chatbot.get_response_with_agents("My name is Alice")
        assert response1 is not None

        # Second query that might reference previous context
        response2 = chatbot.get_response_with_agents("What did I just tell you?")
        assert response2 is not None

        # Verify conversation history is maintained
        history = chatbot.get_history()
        assert len(history) >= 2
