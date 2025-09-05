.PHONY: streamlit api docker test integration_test linting flake8

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

flake8:
	flake8 app/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 app/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics