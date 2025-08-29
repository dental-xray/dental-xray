.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
install:
	@pip install -e .

install-api:
# 	@pip install -e .

install-model:
	@pip install -e .
	@pip uninstall torch torchvision torchaudio
	@pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

reinstall:
	@pip uninstall -y disease-recognition || :
	@pip install -e .

test:
	pytest tests/

default:  # この default ターゲットがデフォルトになる
	@echo "Available commands:"
	@echo "  make install - Install dependencies"
	@echo "  make install-api - Install dependencies for API development"
	@echo "  make install-model - Install dependencies for model development"
	@echo "  make reinstall - Reinstall the package (disease-recognition)"
	@echo "  make test    - Run tests"
# 	@echo "  make lint    - Run linting"
