SHELL := /bin/bash
.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
install:
	@echo "Installing dependencies..."
	@pip install -e .
	@pip install -r requirements.txt

install-api:
	@echo "Installing dependencies for API development..."
	@pip install -e .
	@pip install -r requirements-api.txt --no-cache-dir


install-model:
	@echo "Installing dependencies for model development..."
	@pip install -e .
	@pip install -r requirements-model.txt
	@pip uninstall torch torchvision torchaudio
	@pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

reinstall:
	@echo "Reinstalling the package..."
	@pip uninstall -y disease_recognition || :
	@pip install -e .

run-frontend:
	@echo "Running the Streamlit frontend..."
	@streamlit run frontend/app.py

build-api-x86:
	@echo "Building Docker image for API (x86)..."
	@source .env && docker build -t $${GAR_IMAGE} -f api/Dockerfile .

build-api-amd64:
	@echo "Building Docker image for API (amd64)..."
	@source .env && docker build --platform linux/amd64 -t $${GAR_IMAGE} -f api/Dockerfile .

run-api:
	@echo "Running the API Docker container..."
	@source .env && docker run -it \
		--platform linux/amd64 \
		-p 8080:8080 \
		--name dental-api \
		--rm \
		$${GAR_IMAGE}

stop-api:
	@docker stop dental-api || true

push-api:
	@echo "Pushing Docker image to Google Artifact Registry..."
	@source .env && docker push ${GAR_IMAGE}

deploy-api:
	@echo "Deploying to Google Cloud Run..."
	@source .env && gcloud run deploy ${PROJECT_NAME} \
		--image ${GAR_IMAGE} \
		--region ${GCP_REGION} \
		--timeout=600 \
		--cpu=2 \
		--memory=2Gi

test:
	@echo "Running tests..."
	pytest tests/

check-egg-info:
	@echo "Checking for .egg-info directories..."
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
	@echo "Starting cleanup of egg-info directories..."
	@find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ egg-info directories removed"

default:
	@echo "Available commands:"
	@echo "  make install - Install dependencies"
	@echo "  make install-api - Install dependencies for API development"
	@echo "  make install-model - Install dependencies for model development"
	@echo "  make reinstall - Reinstall the package (disease-recognition)"
	@echo "  make run-frontend - Run the Streamlit frontend"
	@echo "  make build-api-x86 - Build Docker image for API (x86)"
	@echo "  make build-api-amd64 - Build Docker image for API (amd64)"
	@echo "  make run-api - Run the API Docker container"
	@echo "  make push-api - Push Docker image to Google Artifact Registry"
	@echo "  make deploy-api - Deploy to Google Cloud Run"
	@echo "  make test    - Run tests"
	@echo "  make check-egg-info - Check contents of egg-info directories"
	@echo "  make clean-egg-info - Remove all egg-info directories"
	@echo "  make help    - Show this help message"
