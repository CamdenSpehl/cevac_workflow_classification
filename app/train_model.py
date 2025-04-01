import os
import sys
import pandas as pd
from model import WorkOrderClassifier
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_save_model(data_path, model_output_path, force=False):
    """Train the model and save it to the specified path."""
    try:
        # Check if model already exists
        if os.path.exists(model_output_path) and not force:
            logger.warning(f"Model already exists at {model_output_path}. Use --force to overwrite.")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        
        # Load data
        logger.info(f"Loading work order data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} work orders")
        
        # Filter to closed work orders with shop assignments
        closed_wos = df[df['WO_STATUS'] == 'CLOSED']
        closed_wos = closed_wos.dropna(subset=['WO_SHOP'])
        logger.info(f"Using {len(closed_wos)} closed work orders with assigned shops for training")
        
        # Train the model
        logger.info("Training the model...")
        classifier = WorkOrderClassifier()
        classifier.train(closed_wos)
        
        # Save the model
        logger.info(f"Saving model to {model_output_path}")
        classifier.save(model_output_path)
        
        logger.info("Model trained and saved successfully.")
        return True
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and save the work order classification model')
    parser.add_argument('--data', type=str, default='../data/workorder.csv', 
                        help='Path to the work order CSV data file')
    parser.add_argument('--output', type=str, default='../models/trained_model.pkl',
                       help='Path where the trained model will be saved')
    parser.add_argument('--force', action='store_true',
                       help='Force overwrite if model already exists')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Train and save the model
    success = train_and_save_model(args.data, args.output, args.force)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
