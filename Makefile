.PHONY: streamlit api docker test integration_test linting

streamlit:
	python -m streamlit run app/streamlit_app.py

api:
	python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

test:
	python -m pytest tests/ -v

integration_test:
	python -m pytest tests/ -m integration -v

linting:
	isort app/ tests/
	black app/ tests/