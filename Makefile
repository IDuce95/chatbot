.PHONY: run test streamlit api


run:
	python run_chatbot.py

streamlit:
	python -m streamlit run streamlit_app.py

api:
	python -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest tests/ -v
