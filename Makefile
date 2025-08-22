.PHONY: run test streamlit


run:
	python run_chatbot.py

streamlit:
	python -m streamlit run streamlit_app.py

test:
	python -m pytest tests/ -v
