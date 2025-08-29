.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
install:
	@pip install -e .

install-api:
# 	@pip install -e .

install-model:
# 	@pip install -e .

reinstall:
	@pip uninstall -y disease-recognition || :
	@pip install -e .

test:
	pytest tests/

default:  # この default ターゲットがデフォルトになる
	@echo "Available commands:"
	@echo "  make install - Install dependencies"
	@echo "  make install-api - Install dependencies for API"
	@echo "  make reinstall - Reinstall the package (disease-recognition)"
	@echo "  make test    - Run tests"
# 	@echo "  make lint    - Run linting"
