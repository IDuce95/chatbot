import time

import pytest
import requests

from app.config_utils import get_api_config


@pytest.mark.integration
class TestAPIIntegration:

    @pytest.fixture
    def api_config(self):
        return get_api_config()

    @pytest.fixture
    def api_base_url(self, api_config):
        return api_config["base_url"]

    def test_health_endpoint(self, api_base_url):
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "status" in health_data
        assert "agent_system" in health_data
        assert "rag_enabled" in health_data
        assert health_data["status"] == "healthy"

    def test_agents_info_endpoint(self, api_base_url):
        response = requests.get(f"{api_base_url}/agents/info")
        assert response.status_code == 200

        agents_data = response.json()
        assert "agent_system_active" in agents_data
        assert "total_agents" in agents_data
        assert "available_agents" in agents_data
        assert isinstance(agents_data["available_agents"], list)
        assert agents_data["total_agents"] > 0

    def test_chat_endpoint(self, api_base_url):
        test_message = "How to create a Python function?"

        start_time = time.time()
        response = requests.post(
            f"{api_base_url}/chat",
            json={"message": test_message},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 30

        chat_data = response.json()
        assert chat_data["success"] is True
        assert "rag_used" in chat_data
        assert "agents_used" in chat_data
        assert "intent" in chat_data
        assert "quality_score" in chat_data
        assert "response" in chat_data
        assert len(chat_data["response"]) > 0

    def test_metrics_endpoint(self, api_base_url):
        response = requests.get(f"{api_base_url}/metrics")
        assert response.status_code == 200

        metrics_data = response.json()
        assert "metrics" in metrics_data

        metrics = metrics_data["metrics"]
        assert "total_interactions" in metrics
        assert "rag_usage_rate" in metrics
        assert "avg_response_time" in metrics
        assert isinstance(metrics["total_interactions"], int)
        assert 0 <= metrics["rag_usage_rate"] <= 1
        assert metrics["avg_response_time"] >= 0

    def test_chat_endpoint_response_time(self, api_base_url):
        test_message = "Hello"

        start_time = time.time()
        response = requests.post(
            f"{api_base_url}/chat", json={"message": test_message}, timeout=15
        )
        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        assert elapsed_time < 15

    def test_chat_endpoint_empty_message(self, api_base_url):
        response = requests.post(
            f"{api_base_url}/chat",
            json={"message": ""},
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 200
        chat_data = response.json()
        assert chat_data["success"] is True

    @pytest.mark.parametrize("endpoint", ["/health", "/agents/info", "/metrics"])
    def test_get_endpoints_status(self, api_base_url, endpoint):
        response = requests.get(f"{api_base_url}{endpoint}")
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("application/json")
