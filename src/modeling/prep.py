"""Data preparation and preprocessing for modeling"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def prepare_model_data(df: pd.DataFrame, target_col: str = "default"):
    """Prepare data for modeling"""
    print("\n" + "="*60)
    print("STEP 7: Preparing Model Data...")
    print("="*60)
    
    # Ensure target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target distribution %:\n{y.value_counts(normalize=True)}")
    
    return X, y, feature_cols


def create_preprocessing_pipeline(X_train: pd.DataFrame):
    """Create preprocessing pipeline with sparse output for memory efficiency"""
    X_train = X_train.copy()  # Avoid in-place modification
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Convert categorical columns to string for encoding
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
    
    # Use sparse output for memory efficiency (models accept sparse matrices)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_cols, categorical_cols

