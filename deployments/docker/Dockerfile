# Machine Intelligence Node - Dockerfile
# Defines a containerized environment for AI model execution.

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r deployments/docker/requirements.txt

# Expose API port
EXPOSE 8080

# Default command to start the AI model server
CMD ["bash", "deployments/docker/start_container.sh"]
