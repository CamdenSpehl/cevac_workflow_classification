# CEVAC Work Order Classification

This API provides a machine learning service for automatically classifying work orders to appropriate shops based on their descriptions and associated information. It uses natural language processing techniques to analyze work order descriptions and predict the most suitable shop assignment.

## Features

- **Intelligent Routing**: Automatically assigns work orders to the appropriate maintenance shops
- **REST API**: Simple HTTP interface for integration with existing systems
- **Batch Processing**: Support for processing multiple work orders in a single request
- **Containerized**: Easy deployment with Docker
- **Retraining Capability**: API endpoint for retraining the model with new data
- **Prediction Testing**: Tools to test and evaluate model predictions

## Getting Started

### Prerequisites

- Docker
- Docker Compose
- Work order CSV data for training

### Quick Start

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd cevac_workflow_classification
   ```

2. Start the service:
   ```bash
   ./manage.sh start
   ```

   This will:
   - Build the Docker image
   - Start the API server
   - Train the model if it doesn't exist already
   - Make the API available at http://localhost:8000

3. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## Using the API

### Predicting Shop Assignment for a Work Order

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "description": "BUILDING A - RM 100 - REPLACE HVAC FILTER",
  "building_desc": "ACADEMIC BUILDING"
}'
```

### Batch Predictions

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/batch' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "workorders": [
    {
      "description": "BUILDING B - RM 200 - LEAKING PIPE IN BATHROOM",
      "building_desc": "STUDENT CENTER"
    },
    {
      "description": "BUILDING C - ELEVATOR NOT WORKING",
      "building_desc": "SCIENCE BUILDING"
    }
  ]
}'
```

## Management Commands

The `manage.sh` script provides convenient commands for managing the service:

```bash
# Build the Docker image
./manage.sh build

# Start the service
./manage.sh start

# Stop the service
./manage.sh stop

# View logs
./manage.sh logs

# Run API tests
./manage.sh test

# Retrain the model
./manage.sh train

# Open a shell in the container
./manage.sh shell
```

## Testing Predictions

Use the `test_predictions.sh` script to test model predictions:

```bash
# Run tests in Docker container
./test_predictions.sh docker

# Run tests locally
./test_predictions.sh local
```

The tests will:
1. Display detailed model information (model type, number of shop classes, etc.)
2. Run predictions on sample work orders
3. Test with real data from your dataset and calculate accuracy

## Project Structure

```
cevac_workflow_classification/
├── app/                      # API application code
│   ├── main.py               # FastAPI application
│   ├── model.py              # ML model implementation
│   ├── train_model.py        # Model training script
│   ├── test_api.py           # API testing script
│   ├── test_model.py         # Model prediction testing
│   ├── requirements.txt      # Python dependencies
│   └── Dockerfile            # Docker configuration
├── data/                     # Data directory
│   └── workorder.csv         # Work order training data
├── models/                   # Trained model storage
├── old_model/                # Legacy model code
│   └── cevac_data_visualization.ipynb  # Original notebook
├── docker-compose.yml        # Docker Compose configuration
├── manage.sh                 # Management script
├── test_predictions.sh       # Prediction testing script
└── README.md                 # This file
```

## How It Works

The system uses a TF-IDF vectorizer to convert work order descriptions into numerical features, which are then fed into a logistic regression model to predict the appropriate shop. The model is trained on historical work orders that have already been assigned to shops.

### Technical Implementation:

1. **Text Processing**:
   - Custom stopwords list for filtering common words
   - Text cleaning with punctuation removal and lowercasing
   - TF-IDF vectorization to convert text to numerical features

2. **Machine Learning**:
   - Logistic Regression model for multi-class classification
   - Training on closed work orders with known shop assignments

3. **API**:
   - FastAPI framework for high-performance REST endpoints
   - Support for both single and batch predictions
   - Model information and statistics endpoint

## Development

### Local Development

To run the application locally without Docker:

1. Install dependencies:
   ```bash
   cd app
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

3. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```

4. Test predictions:
   ```bash
   python test_model.py
   ```

### Customizing the Model

The current implementation uses TF-IDF and logistic regression, but you can modify `model.py` to use different algorithms or features. Consider exploring:

- Different text vectorization techniques (Word2Vec, BERT embeddings)
- Different classification algorithms (Random Forest, SVM, neural networks)
- Additional features beyond the work order description

## Technical Details

- **Framework**: FastAPI
- **ML Libraries**: scikit-learn
- **Data Processing**: pandas
- **Containerization**: Docker

## License

[Add your license information here]

## Acknowledgments

- Clemson University Facilities Department
- [Add other acknowledgments as needed]
