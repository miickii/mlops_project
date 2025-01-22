FROM python:3.11-slim AS base

# Install necessary dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the necessary files into the Docker image
COPY ./src/ src/
COPY ./models/fruits_model.ckpt models/
COPY ./data/processed/classes.json data/processed/
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir --verbose

# Set PYTHONPATH so the `mlops_project` module can be found
ENV PYTHONPATH="/app/src"

# Expose the application port
EXPOSE 8080

# Start the FastAPI app
CMD exec uvicorn mlops_project.api:app --port 8080 --host 0.0.0.0 --workers 1
