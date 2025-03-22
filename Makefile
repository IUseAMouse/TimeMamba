.PHONY: setup lockdeps installdeps format lint test clean run

# Variables
PYTHON := python3
UV := uv
PROJECT := mamba_forecast

# Default target
all: setup installdeps format lint test
build: setup installdeps

# Setup virtual environment
setup:
	@echo "Setting up virtual environment..."
	$(UV) venv

# Create a lockfile for dependencies
lockdeps:
	@echo "Locking dependencies..."
	$(UV) pip compile pyproject.toml
	$(UV) lock

# Install dependencies
installdeps:
	@echo "Installing dependencies..."
	$(UV) pip install -e .
	$(UV) pip install -r pyproject.toml

# Format code
format:
	@echo "Formatting code..."
	$(UV) run black src/ tests/
	$(UV) run isort src/ tests/

# Lint code
lint:
	@echo "Linting code..."
	$(UV) run flake8 src/ tests/test.py

# Run tests
test:
	@echo "Running tests..."
	$(UV) run pytest tests/test.py

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf models/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build

# Train the model on the collection of TSF files
train:
	@echo "Training ..."
	uv run python train.py