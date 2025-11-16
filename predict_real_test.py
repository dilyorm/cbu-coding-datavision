"""Standalone prediction pipeline for real test data.

This script processes real test data through all feature engineering versions,
generates predictions from all trained models, and creates a weighted ensemble
prediction output in the format: customer_id, default
"""
import pandas as pd
import numpy as np
import pickle
import json
import torch
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.data.load_data import load_all_data
from src.data.clean_data import clean_data
from src.data.prepare_data import merge_data
from src.data.preprocessing import (
    load_preprocessing_pipeline,
    get_numeric_categorical_features
)
from src.features.feature_engineering import (
    create_temporal_features, create_interaction_features,
    create_aggregation_features, create_geographic_features, create_loan_features
)
from src.features.feature_engineering_v1_plus import engineer_features_v1_plus
from src.features.feature_engineering_v2 import engineer_features_v2
from src.features.feature_engineering_v3 import engineer_features_v3
from src.ensemble.ensemble import compute_weights, create_weighted_ensemble
from src.ensemble.create_meta_features import create_meta_features_from_predictions
from src.models.nn_models import SimpleMLP, TabMModel

import lightgbm as lgb
import xgboost as xgb
import catboost as cb


def load_test_data(data_dir='data_test'):
    """Load real test data from directory.
    
    Args:
        data_dir: Directory containing test data files
        
    Returns:
        Dictionary of DataFrames with loaded data
    """
    print(f"\n{'='*80}")
    print("LOADING TEST DATA")
    print(f"{'='*80}")
    
    data_dict = load_all_data(data_dir=data_dir)
    
    print(f"Loaded data sources:")
    for name, df in data_dict.items():
        print(f"  {name}: {df.shape}")
    
    return data_dict


def merge_test_data(data_dict):
    """Merge all test data sources on customer_id."""
    print("\nMerging test datasets...")
    
    # Start with application_metadata as base
    merged = data_dict['application_metadata'].copy()
    merged = merged.dropna(subset=['customer_id'])
    merged['customer_id'] = merged['customer_id'].astype('int64')
    
    # Merge other datasets
    datasets_to_merge = [
        'credit_history',
        'demographics',
        'financial_ratios',
        'geographic_data',
        'loan_details'
    ]
    
    for dataset_name in datasets_to_merge:
        if dataset_name in data_dict:
            df = data_dict[dataset_name].copy()
            df = df.dropna(subset=['customer_id'])
            df['customer_id'] = df['customer_id'].astype('int64')
            merged = merged.merge(
                df,
                on='customer_id',
                how='left',  # Left join for test data (may have missing)
                suffixes=('', f'_{dataset_name}')
            )
    
    print(f"Merged test dataset shape: {merged.shape}")
    return merged


def prepare_test_features(df, train_stats):
    """Apply basic feature engineering common to all versions."""
    print("\nApplying basic feature engineering...")
    
    df = create_temporal_features(df)
    df = create_interaction_features(df)
    df = create_aggregation_features(df)
    df = create_geographic_features(df)
    df = create_loan_features(df)
    
    return df


def process_version_features(df_base, version, train_stats):
    """Apply version-specific feature engineering.
    
    Args:
        df_base: DataFrame with basic features
        version: 'v1', 'v2', or 'v3'
        train_stats: Training statistics dict
        
    Returns:
        DataFrame with version-specific features
    """
    df = df_base.copy()
    
    if version == 'v1':
        df = engineer_features_v1_plus(df, train_stats=train_stats, train_value_counts=None)
    elif version == 'v2':
        df = engineer_features_v2(df, train_stats=train_stats, train_value_counts=None)
    elif version == 'v3':
        df = engineer_features_v3(df, train_stats=train_stats, train_value_counts=None)
    else:
        raise ValueError(f"Unknown version: {version}")
    
    return df


def load_preprocessing_and_transform(df, version):
    """Load preprocessing pipeline and transform test data.
    
    Args:
        df: DataFrame with features
        version: 'v1', 'v2', or 'v3'
        
    Returns:
        Transformed DataFrame and feature names
    """
    print(f"\nLoading preprocessing pipeline for {version}...")
    
    pipeline_path = Path(f'data/processed/{version}/preprocessing_pipeline.pkl')
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline not found: {pipeline_path}")
    
    preprocessor = load_preprocessing_pipeline(str(pipeline_path))
    
    # Load feature names
    feature_names_path = Path(f'data/processed/{version}/feature_names.txt')
    if feature_names_path.exists():
        with open(feature_names_path, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: try to get from preprocessor
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except:
            raise ValueError(f"Cannot determine feature names for {version}")
    
    # Separate features and customer_id
    X = df.drop(columns=['customer_id'], errors='ignore')
    
    # Identify numeric and categorical features
    target_col = 'default'
    numeric_features, categorical_features = get_numeric_categorical_features(
        df.drop(columns=[target_col], errors='ignore'), target_col
    )
    
    # Filter to only features present in current data
    numeric_features = [f for f in numeric_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Ensure categorical columns are strings
    X_clean = X.copy()
    for col in categorical_features:
        if col in X_clean.columns:
            X_clean[col] = X_clean[col].astype(str)
    
    # Transform
    print(f"Transforming test data ({X_clean.shape[0]} samples, {X_clean.shape[1]} features)...")
    X_transformed = preprocessor.transform(X_clean)
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X_transformed, columns=feature_names[:X_transformed.shape[1]])
    
    # Align features with training (add missing columns, remove extra)
    if len(feature_names) > X_transformed.shape[1]:
        # Add missing columns filled with zeros
        missing_cols = feature_names[X_transformed.shape[1]:]
        for col in missing_cols:
            X_df[col] = 0.0
        X_df = X_df[feature_names]
    elif len(feature_names) < X_transformed.shape[1]:
        # Remove extra columns
        X_df = X_df[feature_names]
    
    print(f"Transformed shape: {X_df.shape}")
    return X_df, feature_names


def load_tree_model(model_path, model_type):
    """Load tree-based model (LightGBM, XGBoost, CatBoost).
    
    Args:
        model_path: Path to model file (.pkl)
        model_type: 'lightgbm', 'xgboost', or 'catboost'
        
    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_nn_model(model_path, scaler_path, best_params_path, model_type):
    """Load neural network model (MLP or TabM).
    
    Args:
        model_path: Path to model file (.pth)
        scaler_path: Path to scaler file (.pkl)
        best_params_path: Path to best_params.json file
        model_type: 'mlp' or 'tabm'
        
    Returns:
        Tuple of (model, scaler)
    """
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load best parameters
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    model_params = best_params['model_params'].copy()
    
    # Load model state dict first to get actual input_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    
    # Extract actual input_dim from state dict (from input_block.0.weight shape)
    if 'input_block.0.weight' in state_dict:
        actual_input_dim = state_dict['input_block.0.weight'].shape[1]
    elif 'input_proj.weight' in state_dict:  # TabM uses input_proj
        actual_input_dim = state_dict['input_proj.weight'].shape[1]
    else:
        # Fallback to best_params
        actual_input_dim = model_params.get('input_dim')
        if actual_input_dim is None:
            raise ValueError(f"Cannot determine input_dim from state dict for {model_path}")
    
    # Update input_dim to match actual trained model
    model_params['input_dim'] = actual_input_dim
    
    # Create model with correct input_dim
    if model_type == 'mlp':
        model = SimpleMLP(**model_params)
    elif model_type == 'tabm':
        model = TabMModel(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, scaler


def predict_with_tree_model(model, X, model_type):
    """Generate predictions with tree-based model.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        model_type: 'lightgbm', 'xgboost', or 'catboost'
        
    Returns:
        Prediction probabilities
    """
    if model_type == 'lightgbm':
        return model.predict(X.values)
    elif model_type == 'xgboost':
        # XGBoost validates feature names, so pass DataFrame directly
        # This ensures feature names match what the model expects
        dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
        return model.predict(dtest)
    elif model_type == 'catboost':
        return model.predict_proba(X)[:, 1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_with_nn_model(model, scaler, X, device):
    """Generate predictions with neural network model.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted scaler
        X: Feature DataFrame (should already be filtered to selected features)
        device: PyTorch device
        
    Returns:
        Prediction probabilities
    """
    # Verify feature count matches
    expected_features = scaler.n_features_in_
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Feature mismatch: X has {X.shape[1]} features, "
            f"but scaler expects {expected_features} features. "
            f"Make sure selected features are loaded correctly."
        )
    
    # Scale features
    X_scaled = scaler.transform(X.values)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    # Predict (models already have sigmoid in output layer)
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        # Models already have sigmoid in output, so just extract values
        if outputs.dim() == 0:
            probs = outputs.item()
            probs = np.array([probs])
        else:
            probs = outputs.cpu().numpy()
    
    return probs


def load_all_models():
    """Load all trained models.
    
    Returns:
        Dictionary mapping model_key -> (model, model_type, scaler_path)
    """
    print(f"\n{'='*80}")
    print("LOADING ALL TRAINED MODELS")
    print(f"{'='*80}")
    
    models = {}
    versions = ['v1', 'v2', 'v3']
    tree_models = ['lightgbm', 'xgboost', 'catboost']
    nn_models = ['mlp', 'tabm']
    
    # Load base models
    for version in versions:
        for model_name in tree_models + nn_models:
            model_key = f'{version}_{model_name}'
            model_path = Path(f'data/models/{model_key}/model.pkl')
            
            if model_name in tree_models:
                if model_path.exists():
                    print(f"Loading {model_key}...")
                    model = load_tree_model(model_path, model_name)
                    models[model_key] = (model, model_name, None)
                else:
                    print(f"Warning: {model_key} not found, skipping")
            else:  # Neural network
                model_path_pth = Path(f'data/models/{model_key}/model.pth')
                scaler_path = Path(f'data/models/{model_key}/scaler.pkl')
                best_params_path = Path(f'data/models/{model_key}/best_params.json')
                
                if model_path_pth.exists() and scaler_path.exists() and best_params_path.exists():
                    print(f"Loading {model_key}...")
                    model, scaler = load_nn_model(
                        model_path_pth, scaler_path, best_params_path, model_name
                    )
                    models[model_key] = (model, model_name, scaler)
                else:
                    missing = []
                    if not model_path_pth.exists():
                        missing.append('model.pth')
                    if not scaler_path.exists():
                        missing.append('scaler.pkl')
                    if not best_params_path.exists():
                        missing.append('best_params.json')
                    print(f"Warning: {model_key} not found (missing: {', '.join(missing)}), skipping")
    
    # Load stacking models
    stacking_models = ['lightgbm', 'xgboost']
    for model_name in stacking_models:
        model_key = f'stacking_{model_name}'
        model_path = Path(f'data/models/stacking/{model_name}/model.pkl')
        
        if model_path.exists():
            print(f"Loading {model_key}...")
            model = load_tree_model(model_path, model_name)
            models[model_key] = (model, model_name, None)
        else:
            print(f"Warning: {model_key} not found, skipping")
    
    print(f"\nLoaded {len(models)} models")
    return models


def load_selected_features(version):
    """Load selected features for a specific version.
    
    Args:
        version: Version name ('v1', 'v2', 'v3')
        
    Returns:
        List of selected feature names, or None if not found
    """
    selected_features_path = Path(f'data/feature_selection/{version}/selected_features.txt')
    if selected_features_path.exists():
        with open(selected_features_path, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        return features
    return None


def generate_base_predictions(models, X_dict, device):
    """Generate predictions from base models only (not stacking).
    
    Args:
        models: Dictionary of loaded models
        X_dict: Dictionary mapping version -> transformed features DataFrame
        device: PyTorch device
        
    Returns:
        Dictionary mapping model_key -> predictions
    """
    print(f"\n{'='*80}")
    print("GENERATING BASE MODEL PREDICTIONS")
    print(f"{'='*80}")
    
    # Load selected features for each version
    selected_features = {}
    for version in ['v1', 'v2', 'v3']:
        features = load_selected_features(version)
        if features:
            selected_features[version] = features
            print(f"Loaded {len(features)} selected features for {version}")
    
    all_predictions = {}
    
    # Only process base models (not stacking)
    base_models = {k: v for k, v in models.items() if not k.startswith('stacking_')}
    
    for model_key, (model, model_type, scaler) in base_models.items():
        print(f"\nPredicting with {model_key}...")
        
        # Extract version from model_key
        version = model_key.split('_')[0]
        if version not in X_dict:
            print(f"Warning: Features for {version} not available, skipping {model_key}")
            continue
        
        X = X_dict[version].copy()
        
        # Filter to selected features if available
        if version in selected_features:
            available_features = [f for f in selected_features[version] if f in X.columns]
            if len(available_features) < len(selected_features[version]):
                missing = len(selected_features[version]) - len(available_features)
                print(f"  Warning: {missing} selected features not found in data")
            X = X[available_features]
            print(f"  Using {len(available_features)} selected features")
        
        try:
            if model_type in ['lightgbm', 'xgboost', 'catboost']:
                preds = predict_with_tree_model(model, X, model_type)
            elif model_type in ['mlp', 'tabm']:
                preds = predict_with_nn_model(model, scaler, X, device)
            else:
                print(f"Warning: Unknown model type {model_type}, skipping")
                continue
            
            all_predictions[model_key] = preds
            print(f"  {model_key}: {len(preds)} predictions, range [{preds.min():.6f}, {preds.max():.6f}]")
            
        except Exception as e:
            print(f"Error predicting with {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nGenerated predictions from {len(all_predictions)} base models")
    return all_predictions


def create_stacking_features(base_predictions, X_base_v2):
    """Create stacking features from base model predictions.
    
    Args:
        base_predictions: Dictionary mapping model_key -> predictions
        X_base_v2: Base v2 features DataFrame
        
    Returns:
        DataFrame with stacking features (base + meta), aligned with training features
    """
    print(f"\n{'='*80}")
    print("CREATING STACKING FEATURES")
    print(f"{'='*80}")
    
    # Check if selected features were used for stacking (from v2)
    selected_features_path = Path('data/feature_selection/v2/selected_features.txt')
    X_base_filtered = X_base_v2.copy()
    
    if selected_features_path.exists():
        print("Filtering base features to selected features...")
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f if line.strip()]
        
        available_selected = [f for f in selected_features if f in X_base_filtered.columns]
        if available_selected:
            print(f"  Using {len(available_selected)} selected features (from {len(selected_features)} total)")
            X_base_filtered = X_base_filtered[available_selected]
        else:
            print(f"  Warning: No selected features available, using all features")
    
    # Create meta features from base predictions
    print("Creating meta features from base model predictions...")
    meta_features = create_meta_features_from_predictions(base_predictions, X_base_filtered)
    
    print(f"Meta features shape: {meta_features.shape}")
    print(f"Meta features: {list(meta_features.columns[:10])}...")
    
    # Combine base features with meta features
    print("Combining base features with meta features...")
    X_stacking = pd.concat([X_base_filtered.reset_index(drop=True), meta_features.reset_index(drop=True)], axis=1)
    
    # Align with training stacking features if available
    stacking_feature_names_path = Path('data/stacking/feature_names.txt')
    if stacking_feature_names_path.exists():
        print("Aligning features with training stacking features...")
        with open(stacking_feature_names_path, 'r') as f:
            expected_features = [line.strip() for line in f if line.strip()]
        
        # Add missing columns (filled with zeros)
        missing_cols = [f for f in expected_features if f not in X_stacking.columns]
        if missing_cols:
            print(f"  Adding {len(missing_cols)} missing features (filled with zeros)")
            for col in missing_cols:
                X_stacking[col] = 0.0
        
        # Remove extra columns
        extra_cols = [f for f in X_stacking.columns if f not in expected_features]
        if extra_cols:
            print(f"  Removing {len(extra_cols)} extra features")
        
        # Reorder to match expected order
        X_stacking = X_stacking[expected_features]
    
    print(f"Stacking features shape: {X_stacking.shape}")
    print(f"  Base features: {X_base_filtered.shape[1]}")
    print(f"  Meta features: {len(meta_features.columns)}")
    print(f"  Total: {X_stacking.shape[1]}")
    
    return X_stacking


def generate_stacking_predictions(models, X_stacking, device):
    """Generate predictions from stacking models.
    
    Args:
        models: Dictionary of loaded models
        X_stacking: Stacking features DataFrame
        device: PyTorch device
        
    Returns:
        Dictionary mapping model_key -> predictions
    """
    print(f"\n{'='*80}")
    print("GENERATING STACKING MODEL PREDICTIONS")
    print(f"{'='*80}")
    
    stacking_predictions = {}
    
    # Only process stacking models
    stacking_models = {k: v for k, v in models.items() if k.startswith('stacking_')}
    
    for model_key, (model, model_type, scaler) in stacking_models.items():
        print(f"\nPredicting with {model_key}...")
        
        try:
            if model_type in ['lightgbm', 'xgboost']:
                preds = predict_with_tree_model(model, X_stacking, model_type)
            else:
                print(f"Warning: Unknown stacking model type {model_type}, skipping")
                continue
            
            stacking_predictions[model_key] = preds
            print(f"  {model_key}: {len(preds)} predictions, range [{preds.min():.6f}, {preds.max():.6f}]")
            
        except Exception as e:
            print(f"Error predicting with {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nGenerated predictions from {len(stacking_predictions)} stacking models")
    return stacking_predictions


def load_ensemble_weights():
    """Load ensemble weights from saved metrics.
    
    Returns:
        Dictionary mapping model_key -> weight
    """
    print(f"\nLoading ensemble weights...")
    
    metrics_path = Path('data/final_submission/ensemble_metrics.json')
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        if 'weights' in metrics:
            weights = metrics['weights']
            print(f"Loaded weights for {len(weights)} models")
            return weights
    
    # Fallback: load from individual model metrics and compute weights
    print("Computing weights from individual model metrics...")
    versions = ['v1', 'v2', 'v3']
    tree_models = ['lightgbm', 'xgboost', 'catboost']
    nn_models = ['mlp', 'tabm']
    stacking_models = ['lightgbm', 'xgboost']
    
    cv_metrics = {}
    
    # Load base model metrics
    for version in versions:
        for model_name in tree_models + nn_models:
            model_key = f'{version}_{model_name}'
            metrics_file = Path(f'data/models/{model_key}/metrics.json')
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    cv_metrics[model_key] = metrics.get('cv_metrics', {}).get('auc_roc', 0.5)
    
    # Load stacking model metrics
    for model_name in stacking_models:
        model_key = f'stacking_{model_name}'
        metrics_file = Path(f'data/models/stacking/{model_name}/metrics.json')
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                cv_metrics[model_key] = metrics.get('cv_metrics', {}).get('auc_roc', 0.5)
    
    if not cv_metrics:
        raise ValueError("No model metrics found! Cannot compute ensemble weights.")
    
    weights = compute_weights(cv_metrics, method='performance')
    print(f"Computed weights for {len(weights)} models")
    return weights


def create_ensemble_predictions(all_predictions, weights):
    """Create weighted ensemble predictions.
    
    Args:
        all_predictions: Dictionary mapping model_key -> predictions
        weights: Dictionary mapping model_key -> weight
        
    Returns:
        Ensemble predictions array
    """
    print(f"\n{'='*80}")
    print("CREATING WEIGHTED ENSEMBLE")
    print(f"{'='*80}")
    
    # Get common models
    common_keys = set(all_predictions.keys()) & set(weights.keys())
    
    if not common_keys:
        raise ValueError("No common models found for ensemble!")
    
    print(f"Using {len(common_keys)} models for ensemble")
    
    # Normalize weights to sum to 1 for available models
    available_weights = {k: weights[k] for k in common_keys}
    total_weight = sum(available_weights.values())
    normalized_weights = {k: v / total_weight for k, v in available_weights.items()}
    
    # Display weights
    print("\nEnsemble weights:")
    sorted_weights = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)
    for model_key, weight in sorted_weights[:10]:  # Show top 10
        print(f"  {model_key:30s}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Compute weighted average
    ensemble_preds = np.zeros(len(list(all_predictions.values())[0]))
    
    for model_key in common_keys:
        weight = normalized_weights[model_key]
        ensemble_preds += weight * all_predictions[model_key]
    
    print(f"\nEnsemble predictions range: [{ensemble_preds.min():.6f}, {ensemble_preds.max():.6f}]")
    
    return ensemble_preds


def save_submission(customer_ids, predictions, output_path):
    """Save final submission file.
    
    Args:
        customer_ids: Array of customer IDs
        predictions: Array of prediction probabilities
        output_path: Path to save submission file
    """
    print(f"\n{'='*80}")
    print("SAVING SUBMISSION FILE")
    print(f"{'='*80}")
    
    submission = pd.DataFrame({
        'customer_id': customer_ids,
        'default': predictions
    })
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Customer IDs: {submission['customer_id'].min()} to {submission['customer_id'].max()}")
    print(f"  Prediction range: [{submission['default'].min():.6f}, {submission['default'].max():.6f}]")
    print(f"  Mean prediction: {submission['default'].mean():.6f}")
    print(f"  Std prediction: {submission['default'].std():.6f}")


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description='Generate predictions for real test data')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing test data files (default: data_test)')
    parser.add_argument('--output', type=str, default='predictions/submission_real_test.csv',
                       help='Output path for submission file (default: predictions/submission_real_test.csv)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("REAL TEST DATA PREDICTION PIPELINE")
    print("="*80)
    
    # Load training statistics (needed for feature engineering)
    print("\nLoading training statistics...")
    train_stats = {}
    
    # Load training data to compute statistics
    # We need the original training data before preprocessing to compute proper statistics
    try:
        print("Loading training data to compute statistics...")
        train_data_dict = load_all_data(data_dir='data')
        train_df = merge_data(train_data_dict)
        train_df = clean_data(train_df)
        train_df_base = prepare_test_features(train_df, {})
        
        # Compute statistics for zscore features
        zscore_columns = [
            'credit_score', 'annual_income', 'monthly_income',
            'debt_to_income_ratio', 'credit_utilization', 'account_age',
            'loan_amount', 'total_debt_amount'
        ]
        
        for col in zscore_columns:
            if col in train_df_base.columns:
                train_stats[col] = {
                    'mean': train_df_base[col].mean(),
                    'std': train_df_base[col].std()
                }
        
        print(f"Loaded training statistics for {len(train_stats)} columns")
    except Exception as e:
        print(f"Warning: Could not load training statistics: {e}")
        print("Feature engineering will compute statistics from test data (less ideal)")
        train_stats = {}
    
    # Load test data
    data_dict = load_test_data(args.data_dir)
    
    # Merge test data
    df_test = merge_test_data(data_dict)
    
    # Clean test data
    print("\nCleaning test data...")
    df_test = clean_data(df_test)
    
    # Extract customer IDs
    customer_ids = df_test['customer_id'].values
    
    # Prepare basic features
    df_test_base = prepare_test_features(df_test, train_stats)
    
    # Process all feature versions
    print(f"\n{'='*80}")
    print("FEATURE ENGINEERING - ALL VERSIONS")
    print(f"{'='*80}")
    
    X_dict = {}
    versions = ['v1', 'v2', 'v3']
    
    for version in versions:
        print(f"\nProcessing {version} features...")
        df_version = process_version_features(df_test_base, version, train_stats)
        X_transformed, feature_names = load_preprocessing_and_transform(df_version, version)
        X_dict[version] = X_transformed
    
    # Load all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    models = load_all_models()
    
    if not models:
        raise ValueError("No models loaded! Ensure models are trained and saved.")
    
    # Generate base model predictions
    base_predictions = generate_base_predictions(models, X_dict, device)
    
    if not base_predictions:
        raise ValueError("No base predictions generated! Check model loading and feature alignment.")
    
    # Create stacking features from base predictions
    X_stacking = create_stacking_features(base_predictions, X_dict['v2'])
    
    # Generate stacking model predictions
    stacking_predictions = generate_stacking_predictions(models, X_stacking, device)
    
    # Combine all predictions
    all_predictions = {**base_predictions, **stacking_predictions}
    
    if not all_predictions:
        raise ValueError("No predictions generated! Check model loading and feature alignment.")
    
    # Load ensemble weights
    weights = load_ensemble_weights()
    
    # Create ensemble predictions
    ensemble_preds = create_ensemble_predictions(all_predictions, weights)
    
    # Save submission
    save_submission(customer_ids, ensemble_preds, args.output)
    
    print("\n" + "="*80)
    print("PREDICTION PIPELINE COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

