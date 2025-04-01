import requests
import json
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"

def wait_for_api(timeout=30, interval=2):
    """Wait for the API to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_URL}/")
            if response.status_code == 200:
                logger.info("API is running")
                return True
        except requests.RequestException:
            pass
        
        logger.info(f"API not ready yet, waiting {interval} seconds...")
        time.sleep(interval)
    
    logger.error(f"API did not become available within {timeout} seconds")
    return False

def test_health_check():
    """Test the health check endpoint."""
    try:
        response = requests.get(f"{API_URL}/")
        logger.info(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    try:
        response = requests.get(f"{API_URL}/model/info")
        logger.info(f"Model info: {response.status_code}")
        if response.status_code == 200:
            logger.info(json.dumps(response.json(), indent=2))
            return True
        return False
    except Exception as e:
        logger.error(f"Model info check failed: {str(e)}")
        return False

def test_predict():
    """Test the prediction endpoint with a sample work order."""
    sample_data = {
        "description": "BUILDING XYZ - RM 100 - REPLACE HVAC FILTER",
        "building_desc": "CENTRAL ACADEMIC BUILDING"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=sample_data
        )
        
        logger.info(f"Prediction test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            logger.info(json.dumps(result, indent=2))
            return True
        else:
            logger.error(f"Prediction failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Prediction test failed: {str(e)}")
        return False

def test_batch_predict():
    """Test the batch prediction endpoint with sample work orders."""
    sample_data = {
        "workorders": [
            {
                "description": "BUILDING ABC - RM 200 - LEAKING PIPE IN BATHROOM",
                "building_desc": "STUDENT CENTER"
            },
            {
                "description": "BUILDING XYZ - ELEVATOR NOT WORKING",
                "building_desc": "SCIENCE BUILDING"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json=sample_data
        )
        
        logger.info(f"Batch prediction test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            logger.info(json.dumps(result, indent=2))
            return True
        else:
            logger.error(f"Batch prediction failed: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Batch prediction test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and return True if all pass."""
    logger.info("Starting API tests...")
    
    if not wait_for_api():
        return False
    
    results = {
        "health_check": test_health_check(),
        "model_info": test_model_info(),
        "predict": test_predict(),
        "batch_predict": test_batch_predict()
    }
    
    logger.info("\nTest Results:")
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    return all(results.values())

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
