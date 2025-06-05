.PHONY: help setup install install-dev test lint lint-check type-check format clean run test-extraction clean-analysis run-analysis run-relay-analysis

help:
	@echo "Available commands:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test-basic   - Test basic structure and imports"
	@echo "  make test-extraction - Test data extraction from InfluxDB"
	@echo "  make test-relay   - Test relay analysis functionality"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make lint         - Format code and run all linting (black, isort, flake8, mypy)"
	@echo "  make lint-check   - Check code quality without making changes"
	@echo "  make type-check   - Run mypy type checker only"
	@echo "  make format       - Format code with black and isort only"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-analysis - Remove analysis outputs and data files"
	@echo "  make run          - Run the PEMS application"
	@echo "  make run-analysis - Run 2-year thermal analysis pipeline"
	@echo "  make run-relay-analysis - Run corrected relay-based heating analysis"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test-basic:
	python3 tests/test_basic_structure.py

test-extraction:
	python3 tests/test_data_extraction.py

test-relay:
	python3 tests/test_relay_analysis.py

test:
	pytest -v --cov=pems_v2 --cov-report=term-missing tests/

lint:
	@echo "🔧 Formatting code with black..."
	black .
	@echo "🔧 Organizing imports with isort..."
	isort .
	@echo "🔍 Running flake8 linter..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	@echo "📝 Running mypy type checker..."
	-mypy --strict --explicit-package-bases . || echo "⚠️  Type checking found issues (expected during development)"
	@echo "✅ Linting complete!"

lint-check:
	@echo "🔍 Checking code formatting (no changes)..."
	black --check .
	isort --check-only .
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	mypy --strict --explicit-package-bases .

type-check:
	mypy --strict --explicit-package-bases .

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

clean-analysis:
	@echo "Cleaning analysis outputs..."
	rm -rf data/raw/*.parquet
	rm -rf data/processed/*.parquet
	rm -rf data/features/*.parquet
	rm -rf data/test_output/
	rm -rf analysis/results/*.json
	rm -rf analysis/reports/*.txt
	rm -rf analysis/figures/*.png
	rm -f analysis_*.log
	rm -f test_*.log
	@echo "Analysis outputs cleaned!"

run:
	python main.py

run-analysis:
	@echo "Running 2-year thermal analysis pipeline..."
	. venv/bin/activate && python run_2year_analysis.py

run-relay-analysis:
	@echo "Running corrected relay-based heating analysis..."
	. venv/bin/activate && python test_relay_analysis.py && python corrected_analysis_report.py