"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union


# ============================================================================
# Predict Endpoint
# ============================================================================

class PredictRequest(BaseModel):
    """Request model for /api/predict endpoint."""
    features: Dict[str, Union[float, int, str]] = Field(
        ...,
        description="Customer features as key-value pairs (region can be string)",
        examples=[{"age": 35, "income": 45000, "loyalty_score": 0.3, "region": "US"}]
    )
    customer_id: Optional[str] = Field(
        None,
        description="Optional customer identifier"
    )


class PredictResponse(BaseModel):
    """Response model for /api/predict endpoint."""
    customer_id: str
    uplift_score: float = Field(..., description="Predicted CATE (treatment effect)")
    segment: str = Field(
        ..., 
        description="Behavior-Based Segment: Persuadable, Sleeping Dog, Neutral"
    )
    cate_percentile: float = Field(..., description="Percentile rank in training distribution")
    prob_treatment: Optional[float] = Field(None, description="Predicted outcome if treated")
    prob_control: Optional[float] = Field(None, description="Predicted outcome if control")


# ============================================================================
# Optimize Endpoint
# ============================================================================

class AllocationRequest(BaseModel):
    """Request model for /api/optimize/allocate endpoint."""
    budget_amount: int = Field(..., gt=0, description="Total marketing budget in dollars")
    cost_per_action: float = Field(default=10.0, gt=0, description="Cost to treat one customer")
    benefit_per_conversion: float = Field(default=100.0, gt=0, description="Revenue per incremental conversion")


class AllocationResponse(BaseModel):
    """Response model for /api/optimize/allocate endpoint."""
    total_customers_targeted: int
    expected_uplift: float
    projected_roi: float
    strategy: str
    optimal_threshold: float = Field(..., description="CATE threshold for targeting")


# ============================================================================
# Explain Endpoint
# ============================================================================

class FeatureImportance(BaseModel):
    """Single feature importance entry."""
    feature: str
    importance: float


class GlobalExplanationResponse(BaseModel):
    """Response model for /api/explain/global endpoint."""
    top_features: List[str]
    mean_abs_shap: List[float]
    feature_importance: List[FeatureImportance]


# ============================================================================
# Health Check
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    model_loaded: bool
    version: str
