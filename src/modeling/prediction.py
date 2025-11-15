"""Prediction utilities for judges' evaluation dataset"""
import pandas as pd
import numpy as np
from catboost import Pool
from src.features.cleaning import clean_data
from src.features.engineering import engineer_features
from src.features.missing_values import handle_missing_values
from .load_model import load_model_and_config


def predict_for_judges(raw_df, model, feature_cols, cat_feature_indices, 
                       imputation_method='knn', id_column='customer_id'):
    """Predict probabilities for judges' evaluation dataset
    
    This function runs the same cleaning, feature engineering, and missing value
    handling pipeline as training, then returns predicted probabilities.
    
    Args:
        raw_df: Raw DataFrame from judges (should have same structure as training data)
        model: Trained CatBoostClassifier
        feature_cols: List of feature column names used in training
        cat_feature_indices: List of categorical feature indices
        imputation_method: Imputation method to use (default: 'knn')
        id_column: Name of ID column to preserve (default: 'customer_id')
        
    Returns:
        predictions: Array of predicted probabilities (probability of default)
        customer_ids: Array of customer IDs aligned with predictions (if available)
        valid_indices: Indices of rows that passed all cleaning steps
    """
    # Preserve customer ID before processing using a temporary column
    # Use a column name that won't be in the drop list
    temp_id_col = '__temp_prediction_id__'
    customer_ids_original = None
    if id_column in raw_df.columns:
        customer_ids_original = raw_df[id_column].values.copy()
        raw_df = raw_df.copy()
        raw_df[temp_id_col] = customer_ids_original
    else:
        raw_df = raw_df.copy()
    
    # Step 1: Clean data (same as training pipeline)
    # This may drop rows (duplicates, sanity checks)
    df_clean = clean_data(raw_df)
    
    # Step 2: Engineer features (same as training pipeline)
    df_feat = engineer_features(df_clean)
    
    # Extract customer IDs before missing value handling (which drops ID columns)
    if temp_id_col in df_feat.columns:
        customer_ids = df_feat[temp_id_col].values.copy()
        df_feat = df_feat.drop(columns=[temp_id_col])
    else:
        customer_ids = None
    
    # Step 3: Handle missing values (same as training pipeline)
    # Note: This will drop ID columns, but we've already extracted customer_ids
    df_final = handle_missing_values(df_feat, imputation_method=imputation_method)
    
    # customer_ids should align with df_final since missing value handling doesn't drop rows
    if customer_ids is not None and len(customer_ids) != len(df_final):
        # Rows were dropped during missing value handling (shouldn't happen, but handle it)
        print(f"Warning: Row count mismatch. Expected {len(customer_ids)}, got {len(df_final)}")
        customer_ids = customer_ids[:len(df_final)] if len(customer_ids) > len(df_final) else None
    
    valid_indices = np.arange(len(df_final))
    
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
    
    return predictions, customer_ids, valid_indices


def predict_for_judges_with_output(raw_df, model, feature_cols, cat_feature_indices,
                                   imputation_method='knn', id_column='customer_id',
                                   threshold=0.5, output_path=None):
    """Predict for judges and output results in required format
    
    Output format:
    - Column 1: Customer ID
    - Column 2: Probability of default (as percentage)
    - Column 3: Default prediction (0 or 1)
    
    Args:
        raw_df: Raw DataFrame from judges
        model: Trained CatBoostClassifier
        feature_cols: List of feature column names used in training
        cat_feature_indices: List of categorical feature indices
        imputation_method: Imputation method to use
        id_column: Name of ID column (default: 'customer_id')
        threshold: Classification threshold (default: 0.5)
        output_path: Optional path to save CSV file
        
    Returns:
        DataFrame with columns: customer_id, probability_pct, default_prediction
    """
    print("="*60)
    print("PREDICTING FOR JUDGES' EVALUATION DATASET")
    print("="*60)
    
    # Get predictions
    probabilities, customer_ids, valid_indices = predict_for_judges(
        raw_df, model, feature_cols, cat_feature_indices,
        imputation_method=imputation_method, id_column=id_column
    )
    
    # Convert probabilities to percentages
    probability_pct = probabilities * 100
    
    # Get binary predictions based on threshold
    default_predictions = (probabilities >= threshold).astype(int)
    
    # Create output DataFrame
    # If customer_ids is None, create sequential IDs
    if customer_ids is None:
        customer_ids = np.arange(1, len(probabilities) + 1)
        print(f"Warning: {id_column} not found, using sequential IDs")
    
    # Warn if some rows were dropped during cleaning
    if len(probabilities) < len(raw_df):
        dropped_count = len(raw_df) - len(probabilities)
        print(f"Warning: {dropped_count} rows were dropped during cleaning (invalid data)")
    
    results_df = pd.DataFrame({
        'customer_id': customer_ids,
        'probability_pct': probability_pct,
        'default': default_predictions
    })
    
    # Round probability to 2 decimal places
    results_df['probability_pct'] = results_df['probability_pct'].round(2)
    
    # Save to CSV if path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"  Total predictions: {len(results_df)}")
    print(f"  Predicted defaults: {default_predictions.sum()} ({default_predictions.sum()/len(default_predictions)*100:.2f}%)")
    print(f"  Average probability: {probability_pct.mean():.2f}%")
    print(f"  Min probability: {probability_pct.min():.2f}%")
    print(f"  Max probability: {probability_pct.max():.2f}%")
    
    return results_df


def predict_from_file(input_file, models_dir="models", output_file=None,
                     imputation_method='knn', id_column='customer_id', threshold=0.5):
    """Load model and predict from input file (convenience function)
    
    Args:
        input_file: Path to input CSV/Excel file with judge data
        models_dir: Directory containing saved model
        output_file: Output CSV file path (default: input_file with _predictions suffix)
        imputation_method: Imputation method to use
        id_column: Name of ID column
        threshold: Classification threshold
        
    Returns:
        DataFrame with predictions
    """
    # Load model and config
    print("Loading model and configuration...")
    model, feature_cols, cat_feature_indices = load_model_and_config(models_dir)
    
    # Load input data
    print(f"Loading input data from: {input_file}")
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        raw_df = pd.read_excel(input_file)
    else:
        raw_df = pd.read_csv(input_file)
    
    print(f"Loaded {len(raw_df)} records")
    
    # Generate output filename if not provided
    if output_file is None:
        if input_file.endswith('.csv'):
            output_file = input_file.replace('.csv', '_predictions.csv')
        elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            output_file = input_file.replace('.xlsx', '_predictions.csv').replace('.xls', '_predictions.csv')
        else:
            output_file = input_file + '_predictions.csv'
    
    # Make predictions
    results_df = predict_for_judges_with_output(
        raw_df, model, feature_cols, cat_feature_indices,
        imputation_method=imputation_method,
        id_column=id_column,
        threshold=threshold,
        output_path=output_file
    )
    
    return results_df

