# ===============================
# Dockerfile for Dental Recognition API
# ===============================

# Use Ultralytics base image (CPU version for smaller size)
FROM ultralytics/ultralytics:latest-cpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

# Copy your application code into container
COPY src ./src
COPY api ./api
COPY .env .
COPY models/default.pt ./models/default.pt
COPY Makefile .
COPY requirements-api.txt .
COPY setup.py .

RUN make install-api

# Expose the API port
EXPOSE 8080

# Run FastAPI using Uvicorn
CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8080"]
