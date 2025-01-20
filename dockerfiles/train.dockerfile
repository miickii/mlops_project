# Base image
FROM python:3.11-slim AS base

# Install dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy necessary files
WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/processed data/processed

# Install Python packages
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Entry point
ENTRYPOINT ["python", "-u", "src/mlops_project/train_lightning.py"]
