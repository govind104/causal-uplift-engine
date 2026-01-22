"""
T-Learner Implementation for Causal Uplift Estimation

Uses EconML's TLearner with XGBoost as base learners.
Includes SHAP-based explainability for uplift effects.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from econml.metalearners import TLearner
from xgboost import XGBRegressor
import shap


class CausalUpliftModel:
    """
    Wrapper for T-Learner causal inference model.
    
    Why T-Learner over S-Learner?
    - Avoids regularization bias where weak treatment effects get shrunk to zero
    - Better suited when treatment signal is subtle compared to main effects
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize T-Learner with XGBoost base models.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            random_state: Random seed for reproducibility
        """
        self.model = TLearner(
            models=XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                random_state=random_state,
                verbosity=0
            )
        )
        self.is_fitted = False
        self.feature_names = None
        self.cate_train = None
        
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        treatment: np.ndarray,
        feature_names: Optional[list] = None
    ) -> 'CausalUpliftModel':
        """
        Fit the T-Learner model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Outcome variable (binary: 0/1)
            treatment: Treatment indicator (binary: 0/1)
            feature_names: Optional list of feature names for SHAP
            
        Returns:
            self
        """
        print(f"Fitting T-Learner on {len(X):,} samples...")
        print(f"Treatment rate: {treatment.mean():.2%}")
        print(f"Outcome rate: {y.mean():.2%}")
        
        # Fit model
        self.model.fit(Y=y, T=treatment, X=X)
        self.is_fitted = True
        self.feature_names = feature_names
        
        # Store training CATE for percentile calculations
        self.cate_train = self.model.effect(X)
        
        print(f"Training CATE range: [{self.cate_train.min():.3f}, {self.cate_train.max():.3f}]")
        print(f"Mean CATE: {self.cate_train.mean():.3f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict CATE (uplift) for new samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of CATE predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.effect(X)

    def get_potential_outcomes(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict potential outcomes for control and treatment.
        
        Args:
           X: Feature matrix
           
        Returns:
           (prob_control, prob_treatment)
        """
        if not self.is_fitted:
             raise ValueError("Model not fitted. Call fit() first.")
        
        # EconML TLearner stores models in .models attribute
        # models[0] = control model, models[1] = treatment model
        if hasattr(self.model, 'models'):
             mu0 = self.model.models[0].predict(X)
             mu1 = self.model.models[1].predict(X)
             return mu0, mu1
        else:
             raise AttributeError("Underlying EconML model does not expose base learners.")
    
    def get_segment(self, cate_value: float) -> str:
        """
        Assign customer segment based on CATE thresholds (behavior-based).
        
        Uses absolute CATE values instead of percentiles for meaningful segments:
        - Persuadable (CATE > 0.05): Strong positive effect - TARGET these
        - Sleeping Dog (CATE < -0.05): Strong negative effect - AVOID these  
        - Neutral (|CATE| <= 0.05): No significant effect - IGNORE these
        
        This approach tells the true story: "Only X% are actually persuadable"
        rather than forcing equal 25% buckets.
        """
        if cate_value > 0.05:
            return "Persuadable"
        elif cate_value < -0.05:
            return "Sleeping Dog"
        else:
            return "Neutral"
    
    def get_percentile(self, cate_value: float) -> float:
        """Get percentile of a CATE value relative to training distribution."""
        if self.cate_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return (self.cate_train < cate_value).mean() * 100
    
    def compute_shap_values(
        self, 
        X: np.ndarray,
        sample_size: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute SHAP values for uplift explanation.
        
        Formula: SHAP(Uplift) = SHAP(Treatment Model) - SHAP(Control Model)
        
        Args:
            X: Feature matrix to explain
            sample_size: Number of samples for SHAP (for speed)
            
        Returns:
            Tuple of (uplift_shap_values, shap_treatment, shap_control)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Sample for speed if needed
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
        else:
            X_sample = X
        
        print(f"Computing SHAP values for {len(X_sample):,} samples...")
        
        # Get the underlying models - EconML stores as 'models' (list of [control, treatment])
        if hasattr(self.model, 'models_t'):
            control_model = self.model.models_t[0]
            treatment_model = self.model.models_t[1]
        elif hasattr(self.model, 'models'):
            control_model = self.model.models[0]
            treatment_model = self.model.models[1]
        else:
            raise AttributeError("Cannot find underlying models in TLearner object")
        
        # Use XGBoost's native predict for SHAP background data
        # This avoids compatibility issues between shap and xgboost versions
        try:
            # Try TreeExplainer first (faster)
            explainer_t = shap.TreeExplainer(treatment_model)
            shap_values_t = explainer_t.shap_values(X_sample)
            
            explainer_c = shap.TreeExplainer(control_model)
            shap_values_c = explainer_c.shap_values(X_sample)
        except (ValueError, TypeError) as e:
            print(f"TreeExplainer failed ({e}), falling back to Explainer...")
            # Fallback: Use general Explainer with background data
            background = X_sample[:min(100, len(X_sample))]
            
            explainer_t = shap.Explainer(treatment_model.predict, background)
            shap_values_t = explainer_t(X_sample).values
            
            explainer_c = shap.Explainer(control_model.predict, background)
            shap_values_c = explainer_c(X_sample).values
        
        # Uplift SHAP = Treatment SHAP - Control SHAP
        uplift_shap = shap_values_t - shap_values_c
        
        print(f"SHAP computation complete.")
        
        return uplift_shap, shap_values_t, shap_values_c
    
    def get_feature_importance(
        self, 
        X: np.ndarray,
        feature_names: Optional[list] = None,
        top_k: int = 10,
        use_shap: bool = True
    ) -> pd.DataFrame:
        """
        Get top features driving uplift.
        
        Uses SHAP if available, falls back to XGBoost native importance.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            top_k: Number of top features to return
            use_shap: Whether to try SHAP first
            
        Returns:
            DataFrame with feature names and importance values
        """
        names = feature_names or self.feature_names or [f"f{i}" for i in range(X.shape[1])]
        
        if use_shap:
            try:
                uplift_shap, _, _ = self.compute_shap_values(X)
                mean_abs_shap = np.abs(uplift_shap).mean(axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': names,
                    'mean_abs_shap': mean_abs_shap
                }).sort_values('mean_abs_shap', ascending=False).head(top_k)
                
                return importance_df
            except Exception as e:
                print(f"SHAP failed ({e}), using XGBoost native importance...")
        
        # Fallback: Use XGBoost native feature importance
        # Get the underlying models
        if hasattr(self.model, 'models_t'):
            control_model = self.model.models_t[0]
            treatment_model = self.model.models_t[1]
        elif hasattr(self.model, 'models'):
            control_model = self.model.models[0]
            treatment_model = self.model.models[1]
        else:
            raise AttributeError("Cannot find underlying models")
        
        # Get importance from both models (gain-based)
        importance_t = treatment_model.feature_importances_
        importance_c = control_model.feature_importances_
        
        # Uplift importance = difference in how features matter for treatment vs control
        # We use absolute sum to capture features that matter differently
        uplift_importance = np.abs(importance_t) + np.abs(importance_c)
        
        importance_df = pd.DataFrame({
            'feature': names,
            'importance_treatment': importance_t,
            'importance_control': importance_c,
            'mean_abs_shap': uplift_importance  # Keep same column name for compatibility
        }).sort_values('mean_abs_shap', ascending=False).head(top_k)
        
        return importance_df
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        true_uplift: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model against ground truth (only possible with synthetic data).
        
        Args:
            X_test: Test features
            true_uplift: Ground truth CATE values
            
        Returns:
            Dict with MSE, correlation, and other metrics
        """
        cate_pred = self.predict(X_test)
        
        mse = np.mean((cate_pred - true_uplift) ** 2)
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(cate_pred, true_uplift)[0, 1]
        
        # Ranking accuracy: do we correctly identify top 20% persuadables?
        true_top20_idx = set(np.argsort(true_uplift)[-int(len(true_uplift)*0.2):])
        pred_top20_idx = set(np.argsort(cate_pred)[-int(len(cate_pred)*0.2):])
        ranking_accuracy = len(true_top20_idx & pred_top20_idx) / len(true_top20_idx)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'correlation': correlation,
            'ranking_accuracy_top20': ranking_accuracy
        }
    
    def save(self, path: str = "models/tlearner_model.pkl"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'cate_train': self.cate_train,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str = "models/tlearner_model.pkl") -> 'CausalUpliftModel':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls()
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.cate_train = data['cate_train']
        instance.is_fitted = data['is_fitted']
        print(f"Model loaded from {path}")
        return instance


if __name__ == "__main__":
    from src.data.generator import load_dataset, get_train_test_split
    import os
    
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("Loading production data...")
    try:
        df = load_dataset("data/processed/causal_retail_1M.parquet")
    except (FileNotFoundError, OSError):
        print("Data not found. Running generator first...")
        from src.data.generator import generate_synthetic_data, save_dataset
        df = generate_synthetic_data(n_samples=1_000_000)
        save_dataset(df)
    
    # Split data
    (X_train, X_test, y_train, y_test, 
     treatment_train, treatment_test, true_uplift_test) = get_train_test_split(df)
    
    print(f"\nTraining production model on {len(X_train):,} samples...")
    # Using sufficient depth and estimators for the complex interaction
    model = CausalUpliftModel(n_estimators=100, max_depth=4)
    model.fit(X_train.values, y_train, treatment_train, 
              feature_names=list(X_train.columns))
    
    print("\nEvaluating...")
    metrics = model.evaluate(X_test.values, true_uplift_test)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"Top 20% Ranking Accuracy: {metrics['ranking_accuracy_top20']:.2%}")
    
    print("\nSaving Artifacts...")
    
    # 1. Save Model
    model.save("models/tlearner_model.pkl")
    
    # 2. Save Test Predictions (for Dashboard)
    print("Generating predictions for dashboard...")
    cate_pred = model.predict(X_test.values)
    
    # Create segment column
    segments = [model.get_segment(x) for x in cate_pred]
    
    # Recover customer_ids using the index from X_test (pandas preserves index)
    test_preds = pd.DataFrame({
        'customer_id': df.loc[X_test.index, 'customer_id'],
        'cate_predicted': cate_pred,
        'true_uplift': true_uplift_test,
        'segment': segments
    })
    
    # Add other useful columns for analysis
    test_preds['treatment'] = treatment_test
    test_preds['conversion'] = y_test
    
    test_preds.to_parquet("outputs/test_predictions.parquet", index=False)
    print(f"Saved {len(test_preds):,} predictions to outputs/test_predictions.parquet")
    
    # 3. Save Feature Importance
    print("Calculating feature importance...")
    importance = model.get_feature_importance(X_test.values, list(X_test.columns))
    importance.to_csv("outputs/feature_importance.csv", index=False)
    print("Saved outputs/feature_importance.csv")
    
    print("\n✅ Training pipeline complete. Dashboard is ready to load new data.")
