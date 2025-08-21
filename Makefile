.PHONY: run test


run:
	python run_chatbot.py

test:
	python -m pytest tests/ -v
