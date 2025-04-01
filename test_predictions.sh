#!/bin/bash

# Script to test model predictions
# Usage: ./test_predictions.sh [local|docker]

MODE=${1:-local}

echo "Running prediction tests in $MODE mode..."

if [ "$MODE" == "docker" ]; then
  # Check if container is running
  if [ -z "$(docker compose ps -q api 2>/dev/null)" ]; then
    echo "API container is not running. Starting it..."
    docker compose up -d
    echo "Waiting for container to start..."
    sleep 5
  fi
  
  # Run tests in Docker container
  docker compose exec api python test_model.py
else
  # Run tests locally
  cd app
  python test_model.py
fi
