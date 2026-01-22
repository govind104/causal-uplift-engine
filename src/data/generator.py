"""
Synthetic Causal Retail Dataset Generator

Generates a dataset with known ground truth treatment effects for:
1. Model validation (MSE of CATE)
2. Interpretable SHAP demonstrations
3. Scale testing (1M+ rows)

Target Profile: Quantile-based distribution (15% Persuadable, 10% Sleeping Dog, 75% Neutral)
"""

import numpy as np
import pandas as pd
import uuid
from pathlib import Path


def generate_synthetic_data(n_samples: int = 1_000_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic causal retail dataset with enforced distribution.
    
    The causal structure is Quantile-Based to ensure realistic business segments:
    - ~15% Persuadables (Strong Positive Lift)
    - ~10% Sleeping Dogs (Strong Negative Lift)
    - ~75% Neutrals (Zero Lift)
    
    Args:
        n_samples: Number of rows to generate (default 1M)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: customer_id, age, income, loyalty_score, 
                               region, treatment, conversion, true_uplift
    """
    np.random.seed(seed)
    
    # 1. Generate Covariates (The "Context")
    print(f"Generating {n_samples:,} samples...")
    
    data = pd.DataFrame({
        'customer_id': [str(uuid.uuid4()) for _ in range(n_samples)],
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples),
        'loyalty_score': np.random.beta(2, 5, n_samples),
        'region': np.random.choice(['US', 'EU', 'APAC'], n_samples)
    })
    
    # 2. Calculate Raw Propensity Score (Feature Correlation)
    # This ensures features still predict the outcome segment
    # Younger + Higher Income - Loyalty = Higher Score
    raw_score = (
        0.5 * (60 - data['age']) / 60 +
        0.3 * np.log1p(data['income']) / 12 -
        0.8 * data['loyalty_score']
    )
    
    # 3. Force Distribution using Percentiles (Quantile Squashing)
    p85 = np.percentile(raw_score, 85)
    p10 = np.percentile(raw_score, 10)
    
    # Initialize with Neutrals (Noise around 0)
    true_uplift = np.random.normal(0, 0.001, n_samples)
    
    # Assign Persuadables (Top 15%) -> Strong Positive
    mask_persuadable = raw_score > p85
    true_uplift[mask_persuadable] = np.random.uniform(0.05, 0.15, mask_persuadable.sum())
    
    # Assign Sleeping Dogs (Bottom 10%) -> Strong Negative
    mask_sleeping = raw_score < p10
    true_uplift[mask_sleeping] = np.random.uniform(-0.10, -0.02, mask_sleeping.sum())
    
    data['true_uplift'] = true_uplift
    
    # 4. Assign Treatment (Randomized Control Trial - 50/50 A/B Test)
    data['treatment'] = np.random.binomial(1, 0.5, n_samples)
    
    # 5. Generate Outcome (Conversion)
    # Baseline: loyal customers switch anyway
    baseline_prob = 0.1 + 0.5 * data['loyalty_score']
    
    # Treatment effect: add uplift only for treated users
    prob_conversion = baseline_prob + (data['true_uplift'] * data['treatment'])
    prob_conversion = np.clip(prob_conversion, 0, 1)
    
    data['conversion'] = np.random.binomial(1, prob_conversion)
    
    print(f"Generated dataset shape: {data.shape}")
    print(f"Treatment rate: {data['treatment'].mean():.2%}")
    print(f"Overall conversion rate: {data['conversion'].mean():.2%}")
    print(f"True uplift distribution:")
    print(f"  - Persuadables (> 0.02): {(data['true_uplift'] > 0.02).mean():.1%}")
    print(f"  - Sleeping Dogs (< -0.02): {(data['true_uplift'] < -0.02).mean():.1%}")
    print(f"  - Neutrals: {1 - (data['true_uplift'] > 0.02).mean() - (data['true_uplift'] < -0.02).mean():.1%}")
    
    return data


def save_dataset(df: pd.DataFrame, output_path: str = "data/processed/causal_retail_1M.parquet"):
    """Save dataset to parquet format for efficient I/O."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    print(f"File size: {path.stat().st_size / 1e6:.1f} MB")


def load_dataset(path: str = "data/processed/causal_retail_1M.parquet") -> pd.DataFrame:
    """Load dataset from parquet."""
    return pd.read_parquet(path)


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Split data into train/test sets.
    
    IMPORTANT: true_uplift is dropped from training but kept in test for validation.
    
    Returns:
        X_train, X_test, y_train, y_test, treatment_train, treatment_test, true_uplift_test
    """
    from sklearn.model_selection import train_test_split
    
    # Separate features, outcome, treatment, and ground truth
    feature_cols = ['age', 'income', 'loyalty_score', 'region']
    
    X = df[feature_cols].copy()
    y = df['conversion'].values
    treatment = df['treatment'].values
    true_uplift = df['true_uplift'].values
    
    # One-hot encode region
    X = pd.get_dummies(X, columns=['region'], drop_first=True)
    
    # Split
    (X_train, X_test, 
     y_train, y_test, 
     treatment_train, treatment_test,
     _, true_uplift_test) = train_test_split(
        X, y, treatment, true_uplift,
        test_size=test_size, 
        random_state=seed,
        stratify=treatment
    )
    
    print(f"Train size: {len(X_train):,}")
    print(f"Test size: {len(X_test):,}")
    
    return (X_train, X_test, y_train, y_test, 
            treatment_train, treatment_test, true_uplift_test)


if __name__ == "__main__":
    # Generate and save dataset
    df = generate_synthetic_data(n_samples=1_000_000)
    save_dataset(df)
    
    # Quick validation
    print("\n--- Sample Data ---")
    print(df.head())
    
    print("\n--- Summary Statistics ---")
    print(df[['age', 'income', 'loyalty_score', 'true_uplift']].describe())
