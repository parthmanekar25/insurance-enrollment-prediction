"""
FastAPI application for insurance enrollment prediction.
"""
import sys
import os
from pathlib import Path

# Set up environment for XGBoost on macOS
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from api.schemas import EmployeeData, PredictionResponse, HealthResponse
from src.predict import PredictionEngine

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="REST API to predict employee insurance enrollment likelihood using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
prediction_engine = None
MODEL_LOADED = False


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global prediction_engine, MODEL_LOADED
    
    try:
        # Get absolute paths relative to project root
        project_root = Path(__file__).parent.parent
        model_path = project_root / 'models' / 'best_model.pkl'
        preprocessor_path = project_root / 'models' / 'preprocessor.pkl'
        
        prediction_engine = PredictionEngine(
            model_path=str(model_path),
            preprocessor_path=str(preprocessor_path)
        )
        MODEL_LOADED = True
        print("✓ Model loaded successfully on startup")
    except FileNotFoundError as e:
        MODEL_LOADED = False
        print(f"⚠ Warning: Model files not found. Please train the model first. {e}")
    except Exception as e:
        MODEL_LOADED = False
        print(f"✗ Error loading model: {e}")


@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint with API information.
    
    Returns:
        Dictionary with API status and basic information
    """
    return {
        "message": "Insurance Enrollment Prediction API",
        "status": "active",
        "version": "1.0.0",
        "docs": "/docs",
        "model_loaded": MODEL_LOADED
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status and model availability
    """
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_enrollment(employee: EmployeeData):
    """
    Predict insurance enrollment probability for an employee.
    
    Args:
        employee: Employee data with required features
    
    Returns:
        PredictionResponse with enrollment probability and confidence
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first."
        )
    
    try:
        # Convert Pydantic model to dictionary
        employee_dict = employee.dict()
        
        # Make prediction
        result = prediction_engine.predict(employee_dict)
        
        return PredictionResponse(
            enrolled_probability=result['enrolled_probability'],
            prediction=result['prediction'],
            confidence=result['confidence']
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict", tags=["Predictions"])
async def batch_predict(employees: list[EmployeeData]):
    """
    Make predictions for multiple employees.
    
    Args:
        employees: List of employee data records
    
    Returns:
        List of predictions for each employee
    
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first."
        )
    
    if len(employees) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty employee list provided"
        )
    
    if len(employees) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 employees per batch request"
        )
    
    try:
        predictions = []
        
        for employee in employees:
            employee_dict = employee.dict()
            result = prediction_engine.predict(employee_dict)
            predictions.append({
                **result,
                "employee_data": employee_dict
            })
        
        return {
            "count": len(predictions),
            "predictions": predictions
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/info", tags=["Information"])
async def api_info():
    """
    Get API information and available endpoints.
    
    Returns:
        Information about the API
    """
    return {
        "name": "Insurance Enrollment Prediction API",
        "version": "1.0.0",
        "description": "Machine learning REST API for predicting employee insurance enrollment",
        "model_status": "loaded" if MODEL_LOADED else "not_loaded",
        "endpoints": {
            "GET /": "Root endpoint with status",
            "GET /health": "Health check",
            "GET /info": "This endpoint",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)",
            "POST /predict": "Single prediction",
            "POST /batch-predict": "Batch predictions"
        }
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("INSURANCE ENROLLMENT PREDICTION API")
    print("="*60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("Documentation: http://localhost:8000/docs\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
