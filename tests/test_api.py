from fastapi.testclient import TestClient

from app.api_server import app

client = TestClient(app)


class TestAPI:

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_info" in data
        assert "CodeBot" in data["model_info"]

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_chat_endpoint(self):
        test_message = {"message": "Hello"}
        response = client.post("/chat", json=test_message)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "success" in data
        assert data["success"] is True
        assert len(data["response"]) > 0

    def test_chat_empty_message(self):
        test_message = {"message": ""}
        response = client.post("/chat", json=test_message)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_history_endpoint(self):
        client.delete("/history")

        test_message = {"message": "Test message"}
        client.post("/chat", json=test_message)

        response = client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["history"]) == 2

    def test_clear_history_endpoint(self):
        test_message = {"message": "Test message"}
        client.post("/chat", json=test_message)

        response = client.delete("/history")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

        history_response = client.get("/history")
        history_data = history_response.json()
        assert history_data["count"] == 0
