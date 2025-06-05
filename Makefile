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
	@echo "  make lint         - Format code and run all linting (black, isort, flake8)"
	@echo "  make lint-check   - Check code quality without making changes"
	@echo "  make format       - Format code with black and isort only"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-analysis - Remove analysis outputs and data files"
	@echo "  make run          - Run the PEMS application"
	@echo "  make run-analysis - Run 2-year thermal analysis pipeline"
	@echo "  make run-relay-analysis - Run corrected relay-based heating analysis"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r loxone_smart_home/requirements.txt
	. venv/bin/activate && pip install -r loxone_smart_home/requirements-dev.txt
	. venv/bin/activate && pip install -r pems_v2/requirements.txt
	. venv/bin/activate && pip install -r pems_v2/requirements-dev.txt

install:
	. venv/bin/activate && pip install -r loxone_smart_home/requirements.txt
	. venv/bin/activate && pip install -r pems_v2/requirements.txt

install-dev:
	. venv/bin/activate && pip install -r loxone_smart_home/requirements-dev.txt
	. venv/bin/activate && pip install -r pems_v2/requirements-dev.txt

test-basic:
	. venv/bin/activate && python3 pems_v2/tests/test_basic_structure.py

test-extraction:
	. venv/bin/activate && python3 pems_v2/tests/test_data_extraction.py

test-relay:
	. venv/bin/activate && python3 pems_v2/tests/test_relay_analysis.py

test:
	. venv/bin/activate && pytest -v --cov=pems_v2 --cov-report=term-missing pems_v2/tests/

lint:
	@echo "🔧 Formatting code with black..."
	. venv/bin/activate && black .
	@echo "🔧 Organizing imports with isort..."
	. venv/bin/activate && isort .
	@echo "🔍 Running flake8 linter..."
	. venv/bin/activate && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	. venv/bin/activate && flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --ignore=C901 --statistics
	@echo "✅ Linting complete!"

lint-check:
	@echo "🔍 Checking code formatting (no changes)..."
	. venv/bin/activate && black --check .
	. venv/bin/activate && isort --check-only .
	. venv/bin/activate && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	. venv/bin/activate && flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --ignore=C901 --statistics

format:
	. venv/bin/activate && black .
	. venv/bin/activate && isort .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

clean-analysis:
	@echo "Cleaning analysis outputs..."
	rm -rf pems_v2/data/raw/*.parquet
	rm -rf pems_v2/data/processed/*.parquet
	rm -rf pems_v2/data/features/*.parquet
	rm -rf pems_v2/data/test_output/
	rm -rf pems_v2/analysis/results/*.json
	rm -rf pems_v2/analysis/reports/*.txt
	rm -rf pems_v2/analysis/figures/*.png
	rm -f pems_v2/analysis_*.log
	rm -f pems_v2/test_*.log
	@echo "Analysis outputs cleaned!"

run:
	. venv/bin/activate && python loxone_smart_home/main.py

run-analysis:
	@echo "Running 2-year thermal analysis pipeline..."
	. venv/bin/activate && python pems_v2/run_2year_analysis.py

run-relay-analysis:
	@echo "Running corrected relay-based heating analysis..."
	. venv/bin/activate && python pems_v2/test_relay_analysis.py && python pems_v2/corrected_analysis_report.py