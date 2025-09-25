"""
FastAPI ML Service Example
A production-ready ML inference service with monitoring and error handling.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator
import asyncio
import time
import logging
from typing import Optional, List
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_REQUESTS = Counter('ml_prediction_requests_total', 
                            'Total ML prediction requests', 
                            ['model_name', 'status'])

PREDICTION_LATENCY = Histogram('ml_prediction_latency_seconds',
                              'ML prediction latency')

app = FastAPI(title="ML Inference Service", version="1.0.0")

# Request/Response models
class PredictionRequest(BaseModel):
    text: str
    model_name: Optional[str] = "default"
    
    @validator('text')
    def validate_text(cls, v):
        if len(v) > 10000:
            raise ValueError('Text too long (max 10000 characters)')
        if len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        return v

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_used: str
    processing_time: float

# Mock ML Model (replace with actual model)
class MockMLModel:
    def __init__(self, name: str):
        self.name = name
        self.loaded = True
    
    async def predict(self, text: str) -> tuple:
        # Simulate model inference time
        await asyncio.sleep(0.1)
        
        # Mock prediction (replace with actual model inference)
        prediction = np.random.random()
        confidence = np.random.uniform(0.7, 0.99)
        
        return prediction, confidence

# Global model registry
models = {
    "default": MockMLModel("default"),
    "sentiment": MockMLModel("sentiment"),
}

@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    logger.info("Starting ML Inference Service...")
    
    # Warm up models
    for model_name, model in models.items():
        try:
            await model.predict("warmup")
            logger.info(f"Model {model_name} warmed up successfully")
        except Exception as e:
            logger.error(f"Failed to warm up model {model_name}: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "timestamp": time.time()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint with monitoring and error handling"""
    start_time = time.time()
    
    try:
        # Get model
        model = models.get(request.model_name)
        if not model:
            PREDICTION_REQUESTS.labels(
                model_name=request.model_name, 
                status='error'
            ).inc()
            raise HTTPException(
                status_code=400, 
                detail=f"Model {request.model_name} not found"
            )
        
        # Make prediction
        with PREDICTION_LATENCY.time():
            prediction, confidence = await model.predict(request.text)
        
        processing_time = time.time() - start_time
        
        # Record successful request
        PREDICTION_REQUESTS.labels(
            model_name=request.model_name, 
            status='success'
        ).inc()
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_used=request.model_name,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        PREDICTION_REQUESTS.labels(
            model_name=request.model_name, 
            status='error'
        ).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    if len(requests) > 100:
        raise HTTPException(
            status_code=400, 
            detail="Batch size too large (max 100)"
        )
    
    results = []
    for req in requests:
        try:
            result = await predict(req)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "input_text": req.text[:50] + "..." if len(req.text) > 50 else req.text
            })
    
    return {"results": results, "processed_count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)