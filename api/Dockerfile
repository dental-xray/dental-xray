# ===============================
# Dockerfile for Dental Recognition API
# ===============================

# Use Ultralytics base image (CPU version for smaller size)
FROM ultralytics/ultralytics:latest-cpu

# Set working directory
WORKDIR /app

# Install required Python packages
# Includes fastapi, uvicorn, python-dotenv, and dill (needed by Ultralytics)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    python-multipart \
    google-cloud-storage \
    dill

# Copy your application code into container
COPY src ./src
COPY api ./api
COPY .env .

# Expose the API port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]

# Run with GCP credentials
# docker run -it -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/google-account.json -v /Users/zhangqi/code/Qi-Yaniv/disease-recognition-470203-b31fb290a999.json:/secrets/google-account.json   dental-recognition-api:lightv3
