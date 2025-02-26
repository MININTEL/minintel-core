#!/bin/bash

# Machine Intelligence Node - Cleanup Script
# Automates removal of logs, Docker containers, and unnecessary dependencies.
#
# Author: Machine Intelligence Node Development Team

LOG_DIR="logs"
DOCKER_IMAGE="minintel-node"
CONTAINER_NAME="minintel_container"
K8S_DEPLOYMENT="minintel-deployment"

echo "[INFO] Cleaning up old logs..."
rm -rf $LOG_DIR/*.log
mkdir -p $LOG_DIR
echo "[INFO] Logs cleaned up."

# Stop and remove Docker container
if docker ps -q -f name=$CONTAINER_NAME &> /dev/null; then
    echo "[INFO] Stopping Docker container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    echo "[INFO] Docker container removed."
else
    echo "[INFO] No active Docker container found."
fi

# Remove unused Docker images
echo "[INFO] Pruning unused Docker images..."
docker image prune -f
echo "[INFO] Docker images cleaned up."

# Remove unused Docker volumes
echo "[INFO] Cleaning up Docker volumes..."
docker volume prune -f
echo "[INFO] Docker volumes removed."

# Kubernetes cleanup
if command -v kubectl &> /dev/null; then
    echo "[INFO] Cleaning up Kubernetes deployment..."
    kubectl delete deployment $K8S_DEPLOYMENT --ignore-not-found=true
    kubectl delete service $K8S_DEPLOYMENT --ignore-not-found=true
    echo "[INFO] Kubernetes resources cleaned up."
else
    echo "[INFO] Kubernetes not detected. Skipping cleanup."
fi

echo "[INFO] Cleanup complete."
