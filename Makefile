.PHONY: help install run test test-unit test-integration test-performance test-all test-fast clean setup activate check-env

VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

help:
	@echo "🤖 ChatBot - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  setup        - Create virtual environment and install dependencies"
	@echo "  install      - Install/update dependencies"
	@echo "  activate     - Show command to activate virtual environment"
	@echo "  clean        - Remove virtual environment and cache files"
	@echo ""
	@echo "Run Application:"
	@echo "  run          - Start the chatbot"
	@echo "  check-env    - Check environment configuration"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests except slow ones"
	@echo "  test-unit    - Run unit tests (fast, mocked)"
	@echo "  test-integration - Run integration tests (API calls)"
	@echo "  test-performance - Run performance tests (slow)"
	@echo "  test-all     - Run all tests including slow ones"
	@echo "  test-fast    - Run fast tests only"

setup: $(VENV_DIR)
	@echo "🏗️  Setting up project environment..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Setup complete!"

$(VENV_DIR):
	@echo "🐍 Creating virtual environment..."
	python3 -m venv $(VENV_DIR)

install:
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed!"

activate:
	@echo "To activate virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"

run:
	@echo "🚀 Starting ChatBot..."
	$(PYTHON) run_chatbot.py

check-env:
	@echo "🔍 Checking environment..."
	$(PYTHON) app/chatbot.py

test:
	@echo "🧪 Running tests (excluding slow ones)..."
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-unit:
	@echo "🏃 Running unit tests..."
	$(PYTHON) -m pytest tests/test_chatbot_unit.py -v

test-integration:
	@echo "🔗 Running integration tests..."
	$(PYTHON) -m pytest tests/test_chatbot.py -v -k "not slow"

test-performance:
	@echo "⚡ Running performance tests..."
	$(PYTHON) -m pytest tests/test_chatbot_performance.py -v -m slow

test-all:
	@echo "🚀 Running all tests..."
	$(PYTHON) -m pytest tests/ -v

test-fast:
	@echo "⚡ Running fast tests only..."
	$(PYTHON) -m pytest tests/ -v -m "not slow"

clean:
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"
