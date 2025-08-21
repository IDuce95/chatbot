#!/bin/bash

echo "🧪 ChatBot Testing Suite"
echo "========================"

source venv/bin/activate

case "$1" in
    "unit")
        echo "🏃 Running unit tests (fast, no API calls)..."
        python -m pytest tests/test_chatbot_unit.py -v
        ;;
    "integration")
        echo "🔗 Running integration tests (with API calls)..."
        python -m pytest tests/test_chatbot.py -v -k "not slow"
        ;;
    "performance")
        echo "⚡ Running performance tests (slow)..."
        python -m pytest tests/test_chatbot_performance.py -v -m slow
        ;;
    "all")
        echo "🚀 Running all tests..."
        python -m pytest tests/ -v
        ;;
    "fast")
        echo "⚡ Running fast tests only..."
        python -m pytest tests/ -v -m "not slow"
        ;;
    *)
        echo "Usage: $0 {unit|integration|performance|all|fast}"
        echo ""
        echo "Options:"
        echo "  unit        - Unit tests (fast, mocked)"
        echo "  integration - Integration tests (API calls, medium speed)"
        echo "  performance - Performance tests (slow, marked as slow)"
        echo "  all         - All tests"
        echo "  fast        - All tests except slow ones"
        exit 1
        ;;
esac
