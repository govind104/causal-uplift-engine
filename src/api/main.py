"""
FastAPI Application for Causal Uplift Engine

Endpoints:
- POST /api/predict - Real-time uplift scoring
- POST /api/optimize/allocate - Budget optimization
- GET /api/explain/global - Feature importance
- GET /health - Health check
"""

import warnings
# Suppress FutureWarnings from sklearn (caused by SHAP/EconML using deprecated params)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import os
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PredictRequest, PredictResponse,
    AllocationRequest, AllocationResponse,
    GlobalExplanationResponse, FeatureImportance,
    HealthResponse
)

# ============================================================================
# Global State
# ============================================================================

class ModelState:
    """Container for loaded model artifacts."""
    model = None
    feature_importance = None
    cate_train = None
    test_predictions = None


state = ModelState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    print("Loading model artifacts...")
    
    model_path = os.getenv("MODEL_PATH", "models/tlearner_model.pkl")
    importance_path = os.getenv("IMPORTANCE_PATH", "outputs/feature_importance.csv")
    predictions_path = os.getenv("PREDICTIONS_PATH", "outputs/test_predictions.parquet")
    
    try:
        # Load model
        if Path(model_path).exists():
            from src.models.t_learner import CausalUpliftModel
            state.model = CausalUpliftModel.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")
        
        # Load feature importance
        if Path(importance_path).exists():
            state.feature_importance = pd.read_csv(importance_path)
            print(f"Loaded feature importance from {importance_path}")
        
        # Load test predictions for optimization
        if Path(predictions_path).exists():
            state.test_predictions = pd.read_parquet(predictions_path)
            print(f"Loaded test predictions from {predictions_path}")
        
        print("Model artifacts loaded successfully!")
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")
    
    yield
    
    print("Shutting down...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Causal Uplift Engine API",
    description="Identify Persuadable customers using T-Learner causal inference",
    version="0.1.0",
    lifespan=lifespan
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        version="0.1.0"
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict_uplift(request: PredictRequest):
    """
    Predict uplift (CATE) for a single customer.
    
    Returns the predicted treatment effect and customer segment.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        df = pd.DataFrame([request.features])
        
        # One-hot encode region if present
        if 'region' in df.columns:
            df = pd.get_dummies(df, columns=['region'], drop_first=True)
        
        # Ensure all expected columns exist
        expected_cols = ['age', 'income', 'loyalty_score', 'region_EU', 'region_US']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns
        df = df[expected_cols]
        
        # Predict
        cate = state.model.predict(df.values)[0]
        segment = state.model.get_segment(cate)
        percentile = state.model.get_percentile(cate)
        
        # Get potential outcomes if available
        prob_control, prob_treatment = None, None
        try:
            mu0, mu1 = state.model.get_potential_outcomes(df.values)
            prob_control = float(mu0[0])
            prob_treatment = float(mu1[0])
        except Exception:
            pass # Keep None if not supported
        
        return PredictResponse(
            customer_id=request.customer_id or "unknown",
            uplift_score=float(cate),
            segment=segment,
            cate_percentile=float(percentile),
            prob_treatment=prob_treatment,
            prob_control=prob_control
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/api/optimize/allocate", response_model=AllocationResponse)
async def optimize_allocation(request: AllocationRequest):
    """
    Calculate optimal customer targeting strategy given budget constraints.
    
    Returns the number of customers to target and expected ROI.
    """
    if state.test_predictions is None:
        raise HTTPException(status_code=503, detail="Test predictions not loaded")
    
    try:
        df = state.test_predictions.copy()
        
        # Sort by predicted uplift (target highest first)
        df = df.sort_values('cate_predicted', ascending=False).reset_index(drop=True)
        
        # Calculate how many customers we can afford
        max_customers = int(request.budget_amount / request.cost_per_action)
        n_customers = min(max_customers, len(df))
        
        # Target top N customers
        targeted = df.head(n_customers)
        
        # Calculate expected uplift (sum of CATE * benefit)
        expected_conversions = targeted['cate_predicted'].sum()
        expected_revenue = expected_conversions * request.benefit_per_conversion
        total_cost = n_customers * request.cost_per_action
        
        # ROI calculation
        net_profit = expected_revenue - total_cost
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        
        # Get threshold CATE
        threshold = targeted['cate_predicted'].min() if n_customers > 0 else 0
        
        # Determine strategy description
        if n_customers < len(df) * 0.25:
            strategy = "Target top Persuadables only (high ROI focus)"
        elif n_customers < len(df) * 0.5:
            strategy = "Target Persuadables + some Sure Things"
        else:
            strategy = "Broad targeting (maximize reach)"
        
        return AllocationResponse(
            total_customers_targeted=n_customers,
            expected_uplift=float(expected_conversions),
            projected_roi=float(roi),
            strategy=strategy,
            optimal_threshold=float(threshold)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Optimization error: {str(e)}")


@app.get("/api/explain/global", response_model=GlobalExplanationResponse)
async def get_global_explanation():
    """
    Get global feature importance for uplift prediction.
    
    Returns top features driving treatment effect.
    """
    if state.feature_importance is None:
        raise HTTPException(status_code=503, detail="Feature importance not loaded")
    
    try:
        df = state.feature_importance.copy()
        
        # Build response
        top_features = df['feature'].tolist()
        mean_abs_shap = df['mean_abs_shap'].tolist()
        
        feature_importance = [
            FeatureImportance(feature=row['feature'], importance=row['mean_abs_shap'])
            for _, row in df.iterrows()
        ]
        
        return GlobalExplanationResponse(
            top_features=top_features,
            mean_abs_shap=mean_abs_shap,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {str(e)}")


# ============================================================================
# Run directly for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
