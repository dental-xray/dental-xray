.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
install:
	@pip install -e .
	@pip install -r requirements.txt

install-api:
	@pip install -e .
	@pip install -r requirements-api.txt


install-model:
	@pip install -e .
	@pip install -r requirements-model.txt
	@pip uninstall torch torchvision torchaudio
	@pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

reinstall:
	@pip uninstall -y disease_recognition || :
	@pip install -e .

test:
	pytest tests/

check-egg-info:
	@echo "🔍 Checking egg-info..."
	@find . -name "*.egg-info" -type d | while read dir; do \
		echo "Found: $$dir"; \
		echo "Contents:"; \
		ls -la "$$dir"; \
		echo "\nPKG-INFO:"; \
		cat "$$dir/PKG-INFO" 2>/dev/null || echo "PKG-INFO not found"; \
		echo "\nTop level packages:"; \
		cat "$$dir/top_level.txt" 2>/dev/null || echo "top_level.txt not found"; \
		echo "\nDependencies:"; \
		cat "$$dir/requires.txt" 2>/dev/null || echo "No dependencies"; \
		echo "=" * 50; \
	done

clean-egg-info:
	@echo "🧹 Cleaning egg-info directories..."
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ egg-info directories removed"

default:
	@echo "Available commands:"
	@echo "  make install - Install dependencies"
	@echo "  make install-api - Install dependencies for API development"
	@echo "  make install-model - Install dependencies for model development"
	@echo "  make reinstall - Reinstall the package (disease-recognition)"
	@echo "  make test    - Run tests"
	@echo "  make check-egg-info - Check contents of egg-info directories"
	@echo "  make clean-egg-info - Remove all egg-info directories"
# 	@echo "  make lint    - Run linting"
