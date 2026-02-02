"""
API Schemas for Insurance Enrollment Prediction
"""
from pydantic import BaseModel, Field
from typing import Optional


class EmployeeData(BaseModel):
    """
    Schema for employee data input to prediction endpoint.
    
    Based on EDA analysis, includes only SIGNIFICANT predictors:
    ✅ KEEP: age (p=0.0000), salary (p=0.0000), employment_type (p=0.0000), has_dependents (corr=0.453)
    ❌ DROP: gender (p=0.5887), region (p=0.6147), marital_status (p=0.1942), tenure_years (p=0.4545)
    """
    
    age: int = Field(..., ge=18, le=100, description="Employee age in years (Significant predictor)")
    salary: float = Field(..., gt=0, description="Annual salary in dollars (Significant predictor)")
    employment_type: str = Field(..., description="Type of employment: Full-time, Part-time, or Contract (Significant predictor)")
    has_dependents: str = Field(..., description="Whether employee has dependents: Yes or No (Strongest predictor, corr=0.453)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 45,
                "salary": 75000,
                "employment_type": "Full-time",
                "has_dependents": "Yes"
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    enrolled_probability: float = Field(..., ge=0, le=1, description="Probability of enrollment (0-1)")
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0=Not Enrolled, 1=Enrolled)")
    confidence: str = Field(..., description="Confidence level of prediction (low, medium, high)")


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is successfully loaded")


class ErrorResponse(BaseModel):
    """Schema for error response."""
    
    detail: str = Field(..., description="Error message")
