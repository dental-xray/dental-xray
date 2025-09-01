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
    dill


# Copy your application code into container
COPY src ./src
COPY api ./api
#COPY best.pt .
COPY .env .

# Expose the API port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
