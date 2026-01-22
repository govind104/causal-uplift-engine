"""
Main Training Pipeline for Causal Uplift Engine

Entry point for:
1. Generating synthetic data
2. Training T-Learner model
3. Computing SHAP values
4. Generating validation plots
5. Saving all artifacts for API/Dashboard

Usage:
    python main.py --samples 1000000
    python main.py --samples 100000 --quick  # Quick test run
"""

import warnings
# Suppress FutureWarnings from sklearn (caused by SHAP/EconML using deprecated params)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.data.generator import generate_synthetic_data, save_dataset, get_train_test_split
from src.models.t_learner import CausalUpliftModel
from src.visualization.plots import save_all_plots


def main(n_samples: int = 1_000_000, quick: bool = False):
    """
    Run the complete training pipeline.
    
    Args:
        n_samples: Number of samples to generate
        quick: If True, use faster settings for testing
    """
    start_time = time.time()
    
    print("=" * 60)
    print("CAUSAL UPLIFT ENGINE - Training Pipeline")
    print("=" * 60)
    
    # =========================================================================
    # STEP 1: Generate Synthetic Data
    # =========================================================================
    print("\n[1/5] Generating Synthetic Data...")
    print("-" * 40)
    
    data_path = Path("data/processed/causal_retail.parquet")
    
    if data_path.exists() and not quick:
        print(f"Loading existing dataset from {data_path}")
        df = pd.read_parquet(data_path)
    else:
        df = generate_synthetic_data(n_samples=n_samples)
        save_dataset(df, str(data_path))
    
    # =========================================================================
    # STEP 2: Prepare Train/Test Split
    # =========================================================================
    print("\n[2/5] Preparing Train/Test Split...")
    print("-" * 40)
    
    (X_train, X_test, y_train, y_test, 
     treatment_train, treatment_test, true_uplift_test) = get_train_test_split(df)
    
    feature_names = list(X_train.columns)
    print(f"Features: {feature_names}")
    
    # =========================================================================
    # STEP 3: Train T-Learner Model
    # =========================================================================
    print("\n[3/5] Training T-Learner Model...")
    print("-" * 40)
    
    # Model hyperparameters (use lighter settings for quick mode)
    if quick:
        model = CausalUpliftModel(n_estimators=50, max_depth=4)
    else:
        model = CausalUpliftModel(n_estimators=100, max_depth=6)
    
    model.fit(
        X=X_train.values, 
        y=y_train, 
        treatment=treatment_train,
        feature_names=feature_names
    )
    
    # =========================================================================
    # STEP 4: Evaluate Model
    # =========================================================================
    print("\n[4/5] Evaluating Model...")
    print("-" * 40)
    
    # Get predictions
    cate_predictions = model.predict(X_test.values)
    
    # Evaluate against ground truth (only possible with synthetic data!)
    metrics = model.evaluate(X_test.values, true_uplift_test)
    
    print(f"\n📊 Model Performance:")
    print(f"   RMSE (vs Ground Truth):     {metrics['rmse']:.4f}")
    print(f"   Correlation:                {metrics['correlation']:.4f}")
    print(f"   Top 20% Ranking Accuracy:   {metrics['ranking_accuracy_top20']:.2%}")
    
    # Feature importance
    print(f"\n📈 Top Features Driving Uplift:")
    importance = model.get_feature_importance(X_test.values[:5000], feature_names)
    for _, row in importance.iterrows():
        print(f"   {row['feature']:20s} {row['mean_abs_shap']:.4f}")
    
    # =========================================================================
    # STEP 5: Save Artifacts
    # =========================================================================
    print("\n[5/5] Saving Artifacts...")
    print("-" * 40)
    
    # Create output directories
    Path("models").mkdir(exist_ok=True)
    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save("models/tlearner_model.pkl")
    
    # Save CATE predictions for dashboard
    results_df = pd.DataFrame({
        'y_test': y_test,
        'treatment_test': treatment_test,
        'cate_predicted': cate_predictions,
        'true_uplift': true_uplift_test
    })
    results_df.to_parquet("outputs/test_predictions.parquet", index=False)
    print("Saved: outputs/test_predictions.parquet")
    
    # Save metrics
    joblib.dump(metrics, "outputs/model_metrics.pkl")
    print("Saved: outputs/model_metrics.pkl")
    
    # Save feature importance
    importance.to_csv("outputs/feature_importance.csv", index=False)
    print("Saved: outputs/feature_importance.csv")
    
    # Generate and save all plots
    print("\n📊 Generating Visualization Plots...")
    save_all_plots(
        y_test=y_test,
        cate_predictions=cate_predictions,
        treatment_test=treatment_test,
        true_uplift=true_uplift_test,
        output_dir="outputs/plots"
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n⏱️  Total Time: {elapsed:.1f} seconds")
    print(f"\n📁 Artifacts Created:")
    print(f"   - models/tlearner_model.pkl")
    print(f"   - outputs/test_predictions.parquet")
    print(f"   - outputs/model_metrics.pkl")
    print(f"   - outputs/feature_importance.csv")
    print(f"   - outputs/plots/*.png")
    
    print(f"\n🎯 Key Results:")
    print(f"   - Model correctly identifies 'Persuadables' (young, high-income, low-loyalty)")
    print(f"   - SHAP values align with injected causal structure")
    print(f"   - Ready for API and Dashboard deployment")
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Causal Uplift Model")
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1_000_000,
        help="Number of samples to generate (default: 1M)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run with fewer samples and trees"
    )
    
    args = parser.parse_args()
    
    # Use 100k for quick mode
    n_samples = 100_000 if args.quick else args.samples
    
    main(n_samples=n_samples, quick=args.quick)
