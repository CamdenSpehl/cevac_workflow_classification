from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
import pandas as pd
from typing import List, Optional
from model import WorkOrderClassifier
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Work Order Classification API",
    description="API for classifying work orders to appropriate shops",
    version="1.0.0"
)

# Path to the trained model
MODEL_PATH = os.getenv("MODEL_PATH", "../models/trained_model.pkl")

# Initialize model
classifier = WorkOrderClassifier()

# Request and response models
class WorkOrderRequest(BaseModel):
    description: str
    building: Optional[str] = None
    building_desc: Optional[str] = None
    region_desc: Optional[str] = None
    
class WorkOrderMultiRequest(BaseModel):
    workorders: List[WorkOrderRequest]

class ShopPrediction(BaseModel):
    shop: str
    shop_description: Optional[str] = None
    confidence: float

class WorkOrderPredictionResponse(BaseModel):
    description: str
    prediction: ShopPrediction

class WorkOrderMultiPredictionResponse(BaseModel):
    predictions: List[WorkOrderPredictionResponse]

# Shop descriptions mapping
SHOP_DESCRIPTIONS = {
    "MAINT_CONSTRUCTION": "MAINTENANCE CONSTRUCTION SHOP",
    "MAINT_PERIMETER": "MAINTENANCE PERIMETER AREA",
    "MAINT_DINING": "DINING AND RESIDENTIAL ELECTRICAL",
    "MAINT_HVAC": "MAINTENANCE HVAC SHOP",
    "UT_CHILLER": "UTILITY CHILLER SHOP",
    "UT_STEAM_PLANT": "UTILITY STEAM PLANT",
    # Add more mappings as needed
}

def get_model():
    """Dependency to get the model instance."""
    return classifier

@app.on_event("startup")
async def startup_event():
    """Load model when the application starts."""
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        classifier.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Continue with uninitialized model - will need to train first

@app.get("/")
async def root():
    """Root endpoint for API health check."""
    return {"status": "ok", "message": "Work Order Classification API is running"}

@app.post("/predict", response_model=WorkOrderPredictionResponse)
async def predict_shop(workorder: WorkOrderRequest, model: WorkOrderClassifier = Depends(get_model)):
    """Predict the shop for a single work order."""
    try:
        # Enhanced description with building info if available
        enhanced_desc = workorder.description
        if workorder.building_desc:
            enhanced_desc = f"{workorder.building_desc} - {enhanced_desc}"
            
        prediction = model.predict(enhanced_desc)
        shop_code = prediction["shop"]
        
        return WorkOrderPredictionResponse(
            description=workorder.description,
            prediction=ShopPrediction(
                shop=shop_code,
                shop_description=SHOP_DESCRIPTIONS.get(shop_code),
                confidence=prediction["confidence"]
            )
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predict/batch", response_model=WorkOrderMultiPredictionResponse)
async def predict_batch(workorders: WorkOrderMultiRequest, model: WorkOrderClassifier = Depends(get_model)):
    """Predict shops for multiple work orders in a batch."""
    predictions = []
    
    for wo in workorders.workorders:
        try:
            enhanced_desc = wo.description
            if wo.building_desc:
                enhanced_desc = f"{wo.building_desc} - {enhanced_desc}"
                
            prediction = model.predict(enhanced_desc)
            shop_code = prediction["shop"]
            
            predictions.append(
                WorkOrderPredictionResponse(
                    description=wo.description,
                    prediction=ShopPrediction(
                        shop=shop_code,
                        shop_description=SHOP_DESCRIPTIONS.get(shop_code),
                        confidence=prediction["confidence"]
                    )
                )
            )
        except Exception as e:
            logger.error(f"Error predicting shop for work order: {wo.description[:50]}... - {e}")
            # Continue with next workorder instead of failing the entire batch
    
    return WorkOrderMultiPredictionResponse(predictions=predictions)

@app.post("/train")
async def train_model(model: WorkOrderClassifier = Depends(get_model)):
    """Train the model using the available data."""
    try:
        import model as model_module
        
        data_path = "../data/workorder.csv"
        logger.info(f"Training model with data from {data_path}")
        
        # Train the model
        new_model = model_module.train_model_from_data(data_path)
        
        # Save the trained model
        new_model.save(MODEL_PATH)
        
        # Update the current model instance
        global classifier
        classifier = new_model
        
        return {"status": "success", "message": "Model trained and saved successfully"}
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.get("/model/info")
async def model_info(model: WorkOrderClassifier = Depends(get_model)):
    """Get information about the current model."""
    if not model.model:
        return {"status": "not_loaded", "message": "No model is currently loaded"}
    
    try:
        # Return basic model info
        return {
            "status": "loaded",
            "model_type": type(model.model).__name__,
            "classes": len(model.model.classes_),
            "feature_count": model.vectorizer.max_features if model.vectorizer else None,
            "available_shops": list(model.model.classes_)
        }
    except Exception as e:
        logger.error(f"Error retrieving model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

# Run the server with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
