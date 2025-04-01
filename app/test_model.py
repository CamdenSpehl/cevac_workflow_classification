import os
import sys
import json
import pandas as pd
import pickle
import logging
from tabulate import tabulate
import random
from model import WorkOrderClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MODEL_PATH = "../models/trained_model.pkl"
DATA_PATH = "../data/workorder.csv"

def test_model_info(model):
    """Test retrieving model information."""
    logger.info("Testing model information...")
    
    if not hasattr(model, 'model') or model.model is None:
        logger.error("Model not loaded")
        return False
    
    # Basic model info
    info = {
        "model_type": type(model.model).__name__,
        "classes": len(model.model.classes_),
        "feature_count": model.vectorizer.max_features if model.vectorizer else None,
        "available_shops": list(model.model.classes_)
    }
    
    # Print model info in a nice table
    headers = ["Property", "Value"]
    rows = [[k, v] for k, v in info.items() if k != "available_shops"]
    
    print("\n" + "="*60)
    print(" MODEL INFORMATION")
    print("="*60)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # Print available shops
    print("\n" + "="*60)
    print(" AVAILABLE SHOPS")
    print("="*60)
    shop_data = [{"Shop Code": shop} for shop in info["available_shops"]]
    print(tabulate(shop_data, headers="keys", tablefmt="grid"))
    
    return True

def test_single_prediction(model, description, building_desc=None):
    """Test predicting shop for a single work order."""
    enhanced_desc = description
    if building_desc:
        enhanced_desc = f"{building_desc} - {description}"
    
    prediction = model.predict(enhanced_desc)
    
    return {
        "description": description,
        "building_desc": building_desc,
        "predicted_shop": prediction["shop"],
        "confidence": prediction["confidence"]
    }

def test_batch_prediction(model, test_cases):
    """Test predicting shops for multiple work orders."""
    logger.info("Testing batch prediction with %d work orders...", len(test_cases))
    
    results = []
    for case in test_cases:
        result = test_single_prediction(
            model, 
            case.get("description"), 
            case.get("building_desc")
        )
        results.append(result)
    
    # Print results in a nice table
    print("\n" + "="*100)
    print(" BATCH PREDICTION RESULTS")
    print("="*100)
    
    # Format table rows
    headers = ["Description", "Building", "Predicted Shop", "Confidence"]
    rows = []
    for r in results:
        # Truncate description if it's too long
        desc = r["description"]
        if len(desc) > 40:
            desc = desc[:37] + "..."
        
        rows.append([
            desc,
            r["building_desc"] or "",
            r["predicted_shop"],
            f"{r['confidence']:.4f}"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    return results

def get_random_test_cases(data_path, num_cases=5):
    """Get random work orders from the dataset for testing."""
    try:
        df = pd.read_csv(data_path)
        
        # Filter to closed work orders with shop assignments
        closed_wos = df[df['WO_STATUS'] == 'CLOSED']
        closed_wos = closed_wos.dropna(subset=['WO_SHOP'])
        
        # Sample random work orders
        if len(closed_wos) < num_cases:
            num_cases = len(closed_wos)
            
        samples = closed_wos.sample(n=num_cases)
        
        test_cases = []
        for _, row in samples.iterrows():
            test_cases.append({
                "description": row['WO_DESC'],
                "building_desc": row.get('WO_BUILDING_DESC'),
                "actual_shop": row['WO_SHOP']  # Include actual shop for comparison
            })
        
        return test_cases
        
    except Exception as e:
        logger.error(f"Error getting test cases: {e}")
        return []

def custom_test_cases():
    """Provide custom test cases for testing."""
    return [
        {
            "description": "BUILDING REQUIRES AC FILTER REPLACEMENT",
            "building_desc": "ACADEMIC HALL"
        },
        {
            "description": "LEAKING PIPE IN BATHROOM NEEDS IMMEDIATE REPAIR",
            "building_desc": "STUDENT CENTER"
        },
        {
            "description": "ELEVATOR NOT WORKING PROPERLY",
            "building_desc": "SCIENCE BUILDING"
        },
        {
            "description": "LIGHT FIXTURES NEED REPLACEMENT",
            "building_desc": "LIBRARY"
        },
        {
            "description": "ANNUAL FIRE EXTINGUISHER INSPECTION",
            "building_desc": "ADMINISTRATIVE BUILDING"
        }
    ]

def test_with_real_data(model, test_cases):
    """Test the model with real data cases and compare with actual shop assignments."""
    logger.info("Testing with %d real data cases...", len(test_cases))
    
    results = []
    for case in test_cases:
        prediction = test_single_prediction(
            model, 
            case.get("description"), 
            case.get("building_desc")
        )
        
        results.append({
            **prediction,
            "actual_shop": case.get("actual_shop"),
            "correct": prediction["predicted_shop"] == case.get("actual_shop")
        })
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0
    
    # Print results in a nice table
    print("\n" + "="*120)
    print(f" REAL DATA TEST RESULTS (Accuracy: {accuracy:.2%})")
    print("="*120)
    
    # Format table rows
    headers = ["Description", "Building", "Predicted Shop", "Actual Shop", "Correct", "Confidence"]
    rows = []
    for r in results:
        # Truncate description if it's too long
        desc = r["description"]
        if len(desc) > 30:
            desc = desc[:27] + "..."
        
        rows.append([
            desc,
            r["building_desc"] or "",
            r["predicted_shop"],
            r["actual_shop"],
            "✓" if r["correct"] else "✗",
            f"{r['confidence']:.4f}"
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    return results, accuracy

def run_all_tests():
    """Run all test cases."""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
            return False
        
        # Load the model
        logger.info(f"Loading model from {MODEL_PATH}")
        model = WorkOrderClassifier()
        model.load(MODEL_PATH)
        
        # Test model info
        test_model_info(model)
        
        # Test with custom cases
        custom_cases = custom_test_cases()
        test_batch_prediction(model, custom_cases)
        
        # Test with real data
        real_cases = get_random_test_cases(DATA_PATH, num_cases=10)
        if real_cases:
            test_with_real_data(model, real_cases)
        
        logger.info("All tests completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
