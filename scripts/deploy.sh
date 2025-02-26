#!/bin/bash

# Machine Intelligence Node - Deployment Script
# Automates AI model deployment using Docker and Kubernetes.
#
# Author: Machine Intelligence Node Development Team

# Default deployment parameters
DOCKER_IMAGE="minintel-node"
CONTAINER_NAME="minintel_container"
K8S_DEPLOYMENT="minintel-deployment"
LOG_DIR="logs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not found. Please install Docker and try again."
    exit 1
fi

# Build the Docker image
echo "[INFO] Building Docker image..."
docker build -t $DOCKER_IMAGE -f deployments/docker/Dockerfile .

if [ $? -ne 0 ]; then
    echo "[ERROR] Docker build failed."
    exit 1
fi

# Run the container
echo "[INFO] Starting AI model container..."
docker run -d --name $CONTAINER_NAME -p 8080:8080 $DOCKER_IMAGE

if [ $? -ne 0 ]; then
    echo "[ERROR] Docker container failed to start."
    exit 1
fi

echo "[INFO] AI model deployed successfully via Docker."

# Deploy to Kubernetes if available
if command -v kubectl &> /dev/null; then
    echo "[INFO] Kubernetes detected. Deploying to cluster..."

    kubectl apply -f deployments/k8s/deployment.yaml
    kubectl apply -f deployments/k8s/service.yaml

    echo "[INFO] Kubernetes deployment initiated."
else
    echo "[INFO] Kubernetes not detected. Skipping K8s deployment."
fi

# Log deployment status
echo "[INFO] Deployment complete. Logs available at $LOG_DIR."
