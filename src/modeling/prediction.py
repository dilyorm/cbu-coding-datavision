"""Prediction utilities for judges' evaluation dataset"""
import pandas as pd
import numpy as np
from catboost import Pool
from src.features.cleaning import clean_data
from src.features.engineering import engineer_features
from src.features.missing_values import handle_missing_values
from .load_model import load_model_and_config


def predict_for_judges(raw_df, model, feature_cols, cat_feature_indices, 
                       imputation_method='median'):
    """Predict probabilities for judges' evaluation dataset
    
    This function runs the same cleaning, feature engineering, and missing value
    handling pipeline as training, then returns predicted probabilities.
    
    Args:
        raw_df: Raw DataFrame from judges (should have same structure as training data)
        model: Trained CatBoostClassifier
        feature_cols: List of feature column names used in training
        cat_feature_indices: List of categorical feature indices
        imputation_method: Imputation method to use (default: 'median')
        
    Returns:
        predictions: Array of predicted probabilities (probability of default)
    """
    # Step 1: Clean data (same as training pipeline)
    df_clean = clean_data(raw_df)
    
    # Step 2: Engineer features (same as training pipeline)
    df_feat = engineer_features(df_clean)
    
    # Step 3: Handle missing values (same as training pipeline)
    df_final = handle_missing_values(df_feat, imputation_method=imputation_method)
    
    # Step 4: Prepare features - ensure same columns and order as training
    # Get only the features that were used in training
    X_eval = df_final[feature_cols].copy()
    
    # Ensure categorical columns are strings (same as training)
    categorical_cols = [feature_cols[i] for i in cat_feature_indices if i < len(feature_cols)]
    for col in categorical_cols:
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype(str)
    
    # Create Pool object with categorical features
    eval_pool = Pool(X_eval, cat_features=cat_feature_indices)
    
    # Predict probabilities
    predictions = model.predict_proba(eval_pool)[:, 1]
    
    return predictions

