"""Create meta features from OOF predictions of all base models."""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_oof_predictions():
    """Load all OOF predictions from base models across all feature versions."""
    versions = ['v1', 'v2', 'v3']
    tree_models = ['lightgbm', 'xgboost', 'catboost']
    nn_models = ['mlp', 'tabm']
    
    all_oof = {}
    
    for version in versions:
        for model_name in tree_models + nn_models:
            model_key = f'{version}_{model_name}'
            oof_file = Path(f'data/models/{model_key}/oof_predictions.csv')
            
            if oof_file.exists():
                oof_df = pd.read_csv(oof_file)
                all_oof[model_key] = oof_df['oof_pred_proba'].values
                print(f"Loaded OOF predictions: {model_key} ({len(all_oof[model_key])} samples)")
            else:
                print(f"Warning: OOF predictions not found for {model_key} (skipping)")
    
    return all_oof


def create_meta_features_from_predictions(all_preds_dict, X_base):
    """Create meta features from predictions (OOF for train, val/test predictions for val/test).
    
    Args:
        all_preds_dict: Dict mapping model_key -> predictions array
        X_base: Base features DataFrame to align with
    
    Returns:
        DataFrame with meta features added
    """
    expected_length = len(X_base)
    
    # Validate all predictions have correct length
    for model_key, preds in all_preds_dict.items():
        if len(preds) != expected_length:
            raise ValueError(
                f"Length mismatch for {model_key}: "
                f"expected {expected_length}, got {len(preds)}"
            )
    
    meta_features = pd.DataFrame(index=X_base.index)
    
    # Group predictions by model type (across feature versions)
    model_types = {
        'lightgbm': [k for k in all_preds_dict.keys() if 'lightgbm' in k],
        'xgboost': [k for k in all_preds_dict.keys() if 'xgboost' in k],
        'catboost': [k for k in all_preds_dict.keys() if 'catboost' in k],
        'mlp': [k for k in all_preds_dict.keys() if 'mlp' in k],
        'tabm': [k for k in all_preds_dict.keys() if 'tabm' in k],
    }
    
    # For each model type, create aggregated meta features
    for model_type, model_keys in model_types.items():
        if not model_keys:
            continue
        
        # Collect predictions for this model type
        preds_list = [all_preds_dict[k] for k in model_keys if k in all_preds_dict]
        
        if not preds_list:
            continue
        
        # Stack predictions (each row is predictions from different feature versions)
        preds_array = np.column_stack(preds_list)
        
        # Create aggregated features
        meta_features[f'{model_type}_pred_mean'] = preds_array.mean(axis=1)
        meta_features[f'{model_type}_pred_std'] = preds_array.std(axis=1)
        meta_features[f'{model_type}_pred_min'] = preds_array.min(axis=1)
        meta_features[f'{model_type}_pred_max'] = preds_array.max(axis=1)
        meta_features[f'{model_type}_pred_last'] = preds_array[:, -1]  # Last feature version
        
        # Individual predictions from each feature version
        for idx, model_key in enumerate(model_keys):
            if model_key in all_preds_dict:
                meta_features[f'{model_key}_pred'] = all_preds_dict[model_key]
    
    # Overall statistics across all models
    if all_preds_dict:
        all_preds = np.column_stack([all_preds_dict[k] for k in all_preds_dict.keys()])
        meta_features['all_models_pred_mean'] = all_preds.mean(axis=1)
        meta_features['all_models_pred_std'] = all_preds.std(axis=1)
        meta_features['all_models_pred_min'] = all_preds.min(axis=1)
        meta_features['all_models_pred_max'] = all_preds.max(axis=1)
    
    return meta_features


def main():
    """Create meta features and save stacking dataset."""
    print("=" * 80)
    print("CREATING META FEATURES FROM OOF PREDICTIONS")
    print("=" * 80)
    
    # Load base features (v2) - we'll combine with meta features
    print("\nLoading base features (v2)...")
    data_dir_v2 = Path('data/processed/v2')
    X_train_base = pd.read_csv(data_dir_v2 / 'X_train.csv')
    X_test_base = pd.read_csv(data_dir_v2 / 'X_test.csv')
    
    y_train = pd.read_csv(data_dir_v2 / 'y_train.csv')['default']
    y_test = pd.read_csv(data_dir_v2 / 'y_test.csv')['default']
    
    print(f"Base features shape: Train={X_train_base.shape}, Test={X_test_base.shape}")
    
    # Optionally filter to selected features if available
    selected_features_file = Path('data/feature_selection/v2/selected_features.txt')
    if selected_features_file.exists():
        print(f"\nLoading selected features for v2...")
        with open(selected_features_file, 'r') as f:
            selected_features = [line.strip() for line in f if line.strip()]
        
        available_features = [f for f in selected_features if f in X_train_base.columns]
        if available_features:
            print(f"  Using {len(available_features)} selected features (from {len(selected_features)} total)")
            X_train_base = X_train_base[available_features]
            X_test_base = X_test_base[available_features]
        else:
            print(f"  Warning: No selected features available, using all features")
    
    # Load OOF predictions
    print("\nLoading OOF predictions from all models...")
    all_oof = load_oof_predictions()
    
    if not all_oof:
        raise ValueError("No OOF predictions found! Run train_all_feature_sets.py first.")
    
    print(f"\nLoaded OOF predictions from {len(all_oof)} models")
    
    # Validate OOF prediction lengths match training set
    expected_train_len = len(X_train_base)
    for model_key, oof_preds in all_oof.items():
        if len(oof_preds) != expected_train_len:
            raise ValueError(
                f"OOF prediction length mismatch for {model_key}: "
                f"expected {expected_train_len}, got {len(oof_preds)}"
            )
    
    # Create meta features for train set (using OOF predictions)
    print("\nCreating meta features for training set...")
    meta_train = create_meta_features_from_predictions(all_oof, X_train_base)
    
    # For test set, we need to load test predictions (not OOF)
    print("\nLoading test predictions...")
    versions = ['v1', 'v2', 'v3']
    tree_models = ['lightgbm', 'xgboost', 'catboost']
    nn_models = ['mlp', 'tabm']
    
    test_preds = {}
    
    for version in versions:
        for model_name in tree_models + nn_models:
            model_key = f'{version}_{model_name}'
            test_file = Path(f'data/models/{model_key}/test_predictions.csv')
            
            if test_file.exists():
                test_df = pd.read_csv(test_file)
                test_preds[model_key] = test_df['pred_proba'].values
            else:
                print(f"Warning: Test predictions not found for {model_key}")
    
    # Validate prediction lengths
    expected_test_len = len(X_test_base)
    
    for model_key, test_preds_array in test_preds.items():
        if len(test_preds_array) != expected_test_len:
            raise ValueError(
                f"Test prediction length mismatch for {model_key}: "
                f"expected {expected_test_len}, got {len(test_preds_array)}"
            )
    
    # Create meta features for test set (using test predictions, not OOF)
    print("Creating meta features for test set...")
    if not test_preds:
        raise ValueError("No test predictions found! Run train_all_feature_sets.py first.")
    meta_test = create_meta_features_from_predictions(test_preds, X_test_base)
    
    # Combine base features with meta features
    print("\nCombining base features with meta features...")
    X_train_stacking = pd.concat([X_train_base.reset_index(drop=True), meta_train.reset_index(drop=True)], axis=1)
    X_test_stacking = pd.concat([X_test_base.reset_index(drop=True), meta_test.reset_index(drop=True)], axis=1)
    
    print(f"Stacking dataset shape:")
    print(f"  Train: {X_train_stacking.shape}")
    print(f"  Test: {X_test_stacking.shape}")
    print(f"  Meta features added: {len(meta_train.columns)}")
    
    # Save stacking dataset
    output_dir = Path('data/stacking')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving stacking dataset to {output_dir}...")
    X_train_stacking.to_csv(output_dir / 'X_train.csv', index=False)
    X_test_stacking.to_csv(output_dir / 'X_test.csv', index=False)
    
    y_train.to_frame().to_csv(output_dir / 'y_train.csv', index=False)
    y_test.to_frame().to_csv(output_dir / 'y_test.csv', index=False)
    
    # Save feature names
    with open(output_dir / 'feature_names.txt', 'w') as f:
        for name in X_train_stacking.columns:
            f.write(f"{name}\n")
    
    print("\n" + "=" * 80)
    print("META FEATURES CREATION COMPLETE")
    print("=" * 80)
    print(f"Stacking dataset saved to: {output_dir}/")
    print(f"  Total features: {X_train_stacking.shape[1]} (base: {X_train_base.shape[1]}, meta: {len(meta_train.columns)})")


if __name__ == '__main__':
    main()

