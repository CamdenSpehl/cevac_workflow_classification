#!/bin/bash

# Script to manage the Work Order Classification API
# Usage: ./manage.sh [build|start|stop|restart|logs|test|train]

case "$1" in
  build)
    echo "Building Docker containers..."
    docker compose build
    ;;
    
  start)
    echo "Starting services..."
    docker compose up -d
    echo "Services started. API available at http://localhost:8000"
    echo "Documentation available at http://localhost:8000/docs"
    ;;
    
  stop)
    echo "Stopping services..."
    docker compose down
    ;;
    
  restart)
    echo "Restarting services..."
    docker compose down
    docker compose up -d
    echo "Services restarted. API available at http://localhost:8000"
    ;;
    
  logs)
    echo "Showing logs..."
    docker compose logs -f
    ;;
    
  test)
    echo "Running API tests..."
    # Ensure the API is running
    if [[ -z $(docker compose ps -q api) ]]; then
      echo "API is not running. Starting it..."
      docker compose up -d
      echo "Waiting for API to start..."
      sleep 5
    fi
    
    # Run the tests
    docker compose exec api python test_api.py
    ;;
    
  train)
    echo "Training model..."
    docker compose exec api python train_model.py --force
    ;;
    
  shell)
    echo "Opening shell in API container..."
    docker compose exec api bash
    ;;
    
  *)
    echo "Work Order Classification API Management Script"
    echo "Usage: $0 [command]"
    echo ""
    echo "Available commands:"
    echo "  build    - Build Docker containers"
    echo "  start    - Start services"
    echo "  stop     - Stop services"
    echo "  restart  - Restart services"
    echo "  logs     - Show logs"
    echo "  test     - Run API tests"
    echo "  train    - Train/retrain model"
    echo "  shell    - Open shell in API container"
    echo ""
    exit 1
    ;;
esac
