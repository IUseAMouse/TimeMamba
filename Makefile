.PHONY: setup lockdeps installdeps format lint test clean run

# Variables
PYTHON := python3
UV := uv
PROJECT := temporal_pyramid_moe

# Default target
all: setup installdeps format lint test

# Setup virtual environment
setup:
	@echo "Setting up virtual environment..."
	$(UV) venv

# Create a lockfile for dependencies
lockdeps:
	@echo "Locking dependencies..."
	$(UV) pip compile requirements.txt --output-file requirements.lock
	$(UV) pip compile requirements-dev.txt --output-file requirements-dev.lock

# Install dependencies
installdeps:
	@echo "Installing dependencies..."
	$(UV) pip install -e .
	$(UV) pip install -r requirements.txt
	$(UV) pip install -r requirements-dev.txt

# Format code
format:
	@echo "Formatting code..."
	$(UV) run black models/ lightning_module.py main.py tests/
	$(UV) run isort models/ lightning_module.py main.py tests/

# Lint code
lint:
	@echo "Linting code..."
	$(UV) run flake8 models/ lightning_module.py main.py tests/

# Run tests
test:
	@echo "Running tests..."
	$(UV) run pytest tests/

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

# Run the model with dummy data
run:
	@echo "Running model with dummy data..."
	$(UV) run python main.py

# Run with profiling
profile:
	@echo "Running with profiling..."
	$(UV) run python -m cProfile -o profile.out main.py
	$(UV) run python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumtime').print_stats(30)"

# Train the model
train:
	@echo "Training model..."
	$(UV) run python train.py

# Generate documentation
docs:
	@echo "Generating documentation..."
	$(UV) run sphinx-build -b html docs/source docs/build