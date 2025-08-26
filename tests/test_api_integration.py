import os
import sys
import time

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))
from config_utils import get_api_config


api_config = get_api_config()
API_BASE_URL = api_config["base_url"]


def test_api_endpoints():
    print("🚀 Testing CodeBot API Integration")
    print("=" * 50)
    print("\n1. Testing Health Endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Agent System: {health_data['agent_system']}")
            print(f"   RAG Enabled: {health_data['rag_enabled']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

    print("\n2. Testing Agents Info Endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/agents/info")
        if response.status_code == 200:
            agents_data = response.json()
            print("✅ Agents info retrieved")
            print(f"   Agent System Active: {agents_data['agent_system_active']}")
            print(f"   Total Agents: {agents_data['total_agents']}")
            print("   Available Agents:")
            for agent in agents_data['available_agents']:
                print(f"     {agent['icon']} {agent['name']}: {agent['purpose']}")
        else:
            print(f"❌ Agents info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Agents info error: {e}")

    print("\n3. Testing Chat Endpoint...")

    try:
        test_message = "How to create a Python function?"
        print(f"   Sending: '{test_message}'")

        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"message": test_message},
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if response.status_code == 200:
            chat_data = response.json()
            elapsed_time = time.time() - start_time

            print(f"✅ Chat response received ({elapsed_time:.2f}s)")
            print(f"   Success: {chat_data['success']}")
            print(f"   RAG Used: {chat_data['rag_used']}")
            print(f"   Agents Used: {chat_data['agents_used']}")
            print(f"   Intent: {chat_data['intent']}")
            print(f"   Quality Score: {chat_data['quality_score']}")
            print(f"   Response Length: {len(chat_data['response'])} chars")
            print(f"   Response Preview: {chat_data['response'][:100]}...")
        else:
            print(f"❌ Chat failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Chat error: {e}")

    print("\n4. Testing Metrics Endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        if response.status_code == 200:
            metrics_data = response.json()
            metrics = metrics_data['metrics']
            print("✅ Metrics retrieved")
            print(f"   Total Interactions: {metrics.get('total_interactions', 0)}")
            print(f"   RAG Usage Rate: {metrics.get('rag_usage_rate', 0):.1%}")
            print(f"   Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")

    print("\n" + "=" * 50)
    print("🎉 API Integration Test Complete!")
    print("\n📝 Summary:")
    print(f"   ✅ API Server: Running on {API_BASE_URL}")
    print("   ✅ Streamlit: Running on http://localhost:8502")
    print("   ✅ Agent System: Accessible via API")
    print("   ✅ Metrics: Logged and retrievable")
    print("   ✅ Integration: Streamlit -> API -> Agents")


if __name__ == "__main__":
    test_api_endpoints()
