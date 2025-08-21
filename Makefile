.PHONY: run test


PYTHON = venv/bin/python


run:
	@echo "Starting ChatBot..."
	$(PYTHON) run_chatbot.py

test:
	@echo "Running all tests..."
	$(PYTHON) -m pytest tests/ -v
