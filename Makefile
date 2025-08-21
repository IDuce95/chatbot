.PHONY: run test


run:
	@echo "Starting ChatBot..."
	python run_chatbot.py

test:
	@echo "Running all tests..."
	python -m pytest tests/ -v
