.PHONY: install install-dev lint type-check format test test-cov run clean setup-venv

# Python version
PYTHON := python3.11

# Virtual environment
VENV := venv
VENV_BIN := $(VENV)/bin

# Setup virtual environment
setup-venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip setuptools wheel

# Install production dependencies
install: setup-venv
	$(VENV_BIN)/pip install -r requirements.txt

# Install development dependencies
install-dev: install
	$(VENV_BIN)/pip install -r requirements-dev.txt
	$(VENV_BIN)/pre-commit install

# Run flake8 linter
lint:
	$(VENV_BIN)/flake8 .

# Run mypy type checker
type-check:
	$(VENV_BIN)/mypy .

# Format code with black and isort
format:
	$(VENV_BIN)/black .
	$(VENV_BIN)/isort .

# Run all checks
check: lint type-check

# Run tests
test:
	PYTHONPATH=. PYTHONWARNINGS="ignore::RuntimeWarning" $(VENV_BIN)/pytest

# Run tests with coverage
test-cov:
	PYTHONPATH=. PYTHONWARNINGS="ignore::RuntimeWarning" $(VENV_BIN)/pytest --cov=loxone_smart_home --cov-report=html --cov-report=term

# Run the application
run:
	$(VENV_BIN)/python main.py

# Run pre-commit on all files
pre-commit:
	$(VENV_BIN)/pre-commit run --all-files

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete

# Clean everything including venv
clean-all: clean
	rm -rf $(VENV)
