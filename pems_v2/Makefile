.PHONY: help setup install install-dev test lint type-check format clean run test-extraction

help:
	@echo "Available commands:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make test-extraction - Test data extraction from InfluxDB"
	@echo "  make lint         - Run flake8 linter"
	@echo "  make type-check   - Run mypy type checker"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make run          - Run the PEMS application"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test-extraction:
	python3 test_data_extraction.py

test:
	pytest -v --cov=pems_v2 --cov-report=term-missing tests/

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

type-check:
	mypy --strict pems_v2/

format:
	black .
	isort .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

run:
	python main.py