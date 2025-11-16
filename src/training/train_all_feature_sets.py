"""Train all models on all feature engineering versions."""
import pandas as pd
import numpy as np
import torch
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.training.model_training import (
    train_lightgbm_cv, train_xgboost_cv, train_catboost_cv,
    train_lightgbm_final, train_xgboost_final, train_catboost_final,
    evaluate_model
)
from src.training.nn_training import train_nn_cv, train_nn_final, predict_nn
from src.models.nn_models import SimpleMLP, TabMModel
from src.training.optuna_optimization import optimize_lightgbm, optimize_xgboost, optimize_catboost
from src.training.nn_optimization import optimize_mlp, optimize_tabm
import xgboost as xgb


def load_data_for_version(version):
    """Load preprocessed data for a specific feature version.
    
    Also loads version-specific selected features if available.
    """
    data_dir = Path(f'data/processed/{version}')
    features_dir = Path(f'data/feature_selection/{version}')
    
    print(f"\nLoading {version} data...")
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv')['default']
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv')['default']
    
    print(f"Loaded {version} data:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Load selected features if available
    selected_features_file = features_dir / 'selected_features.txt'
    if selected_features_file.exists():
        print(f"Loading selected features for {version}...")
        with open(selected_features_file, 'r') as f:
            selected_features = [line.strip() for line in f if line.strip()]
        
        # Filter to available selected features
        available_features = [f for f in selected_features if f in X_train.columns]
        missing_features = [f for f in selected_features if f not in X_train.columns]
        
        if missing_features:
            print(f"  Warning: {len(missing_features)} selected features not found in {version} data")
        
        if available_features:
            print(f"  Using {len(available_features)} selected features (from {len(selected_features)} total)")
            X_train = X_train[available_features]
            X_test = X_test[available_features]
        else:
            print(f"  Warning: No selected features available for {version}, using all features")
    else:
        print(f"  No selected features file found for {version}, using all features")
    
    return X_train, y_train, X_test, y_test


def load_best_params_from_v1(model_name):
    """Load best hyperparameters from v1 or v2 training (to reuse for other versions)."""
    # Try v2 first (default), then v1
    for version in ['v2', 'v1']:
        params_file = Path(f'data/models/{version}_{model_name}/best_params.json')
        if params_file.exists():
            with open(params_file, 'r') as f:
                return json.load(f)
    return None


def train_tree_model_on_version(
    model_name, version, X_train, y_train, X_test, y_test,
    n_folds=5, use_existing_params=True
):
    """Train a tree-based model (LGBM/XGB/CatBoost) on a feature version."""
    model_key = model_name.lower()
    output_dir = Path(f'data/models/{version}_{model_key}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already completed
    metrics_file = output_dir / 'metrics.json'
    if metrics_file.exists():
        print(f"  {model_name} on {version} already completed, skipping...")
        return
    
    print(f"\n  Training {model_name} on {version}...")
    
    # Load best params from v2/v1 if available, otherwise optimize
    if use_existing_params:
        best_params = load_best_params_from_v1(model_key)
        if best_params:
            print(f"    Using best params from previous training")
            study = None
        else:
            print(f"    Optimizing hyperparameters...")
            n_trials = 10
            if model_key == 'lightgbm':
                study, best_params = optimize_lightgbm(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
            elif model_key == 'xgboost':
                study, best_params = optimize_xgboost(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
            elif model_key == 'catboost':
                study, best_params = optimize_catboost(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
    else:
        print(f"    Optimizing hyperparameters...")
        n_trials = 10
        if model_key == 'lightgbm':
            study, best_params = optimize_lightgbm(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
        elif model_key == 'xgboost':
            study, best_params = optimize_xgboost(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
        elif model_key == 'catboost':
            study, best_params = optimize_catboost(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
    
    # Get CV metrics and OOF predictions first (needed for final training)
    print(f"    Computing CV metrics...")
    if model_key == 'lightgbm':
        cv_metrics, _, oof_predictions = train_lightgbm_cv(X_train, y_train, best_params, n_folds=n_folds)
    elif model_key == 'xgboost':
        cv_metrics, _, oof_predictions = train_xgboost_cv(X_train, y_train, best_params, n_folds=n_folds)
    elif model_key == 'catboost':
        cv_metrics, _, oof_predictions = train_catboost_cv(X_train, y_train, best_params, n_folds=n_folds)
    
    # Train final model
    print(f"    Training final model...")
    if model_key == 'lightgbm':
        model, _ = train_lightgbm_final(X_train, y_train, best_params, verbose=False)
        y_test_pred_proba = model.predict(X_test.values)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        y_train_pred_proba = model.predict(X_train.values)
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    elif model_key == 'xgboost':
        model, _ = train_xgboost_final(X_train, y_train, best_params, verbose=False)
        dtest = xgb.DMatrix(X_test)
        dtrain = xgb.DMatrix(X_train)
        y_test_pred_proba = model.predict(dtest)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        y_train_pred_proba = model.predict(dtrain)
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    elif model_key == 'catboost':
        model, _ = train_catboost_final(X_train, y_train, best_params, verbose=False)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = model.predict(X_test)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_train_pred = model.predict(X_train)
    
    # Evaluate
    test_metrics = evaluate_model(y_test.values, y_test_pred, y_test_pred_proba)
    
    # Save results
    with open(output_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2, default=str)
    
    metrics = {
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    if study:
        with open(output_dir / 'optuna_study.pkl', 'wb') as f:
            pickle.dump(study, f)
    
    # Save predictions
    pd.DataFrame({
        'pred': y_train_pred,
        'pred_proba': y_train_pred_proba
    }).to_csv(output_dir / 'train_predictions.csv', index=False)
    
    pd.DataFrame({
        'pred': y_test_pred,
        'pred_proba': y_test_pred_proba
    }).to_csv(output_dir / 'test_predictions.csv', index=False)
    
    pd.DataFrame({
        'oof_pred_proba': oof_predictions
    }).to_csv(output_dir / 'oof_predictions.csv', index=False)
    
    print(f"    {model_name} on {version} complete: CV AUC={cv_metrics['auc_roc']:.6f}, Test AUC={test_metrics['auc_roc']:.6f}")


def train_nn_model_on_version(
    model_name, version, X_train, y_train, X_test, y_test,
    n_folds=5, device=None, use_existing_params=True
):
    """Train a neural network model (MLP/TabM) on a feature version."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_key = model_name.lower()
    output_dir = Path(f'data/models/{version}_{model_key}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already completed
    metrics_file = output_dir / 'metrics.json'
    if metrics_file.exists():
        print(f"  {model_name} on {version} already completed, skipping...")
        return
    
    print(f"\n  Training {model_name} on {version}...")
    
    # Load best params from v2/v1 if available, otherwise optimize
    if use_existing_params:
        best_params = load_best_params_from_v1(model_key)
        if best_params:
            print(f"    Using best params from previous training")
            study = None
        else:
            print(f"    Optimizing hyperparameters...")
            n_trials = 5
            if model_key == 'mlp':
                study, best_params = optimize_mlp(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
            elif model_key == 'tabm':
                study, best_params = optimize_tabm(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
    else:
        print(f"    Optimizing hyperparameters...")
        n_trials = 5
        if model_key == 'mlp':
            study, best_params = optimize_mlp(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
        elif model_key == 'tabm':
            study, best_params = optimize_tabm(X_train, y_train, n_trials=n_trials, n_folds=n_folds, verbose=False)
    
    # MLP or TabM
    model_class = SimpleMLP if model_key == 'mlp' else TabMModel
    
    # Update input_dim to match actual number of features
    actual_input_dim = X_train.shape[1]
    model_params = best_params['model_params'].copy()
    model_params['input_dim'] = actual_input_dim
    
    # CV metrics and OOF predictions first
    print(f"    Computing CV metrics...")
    cv_metrics, _, oof_predictions = train_nn_cv(
        model_class,
        X_train, y_train,
        model_params,  # Use updated model_params with correct input_dim
        best_params['training_params'],
        n_folds=n_folds,
        device=device
    )
    
    print(f"    Training final model...")
    model, _, scaler = train_nn_final(
        model_class,
        X_train, y_train,
        model_params,
        best_params['training_params'],
        device=device
    )
    
    y_test_pred, y_test_pred_proba = predict_nn(model, X_test, scaler, device=device)
    y_train_pred, y_train_pred_proba = predict_nn(model, X_train, scaler, device=device)
    
    test_metrics = evaluate_model(y_test.values, y_test_pred, y_test_pred_proba)
    
    # Save results
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), output_dir / 'model.pth')
    else:
        with open(output_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2, default=str)
    
    metrics = {
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    if study:
        with open(output_dir / 'optuna_study.pkl', 'wb') as f:
            pickle.dump(study, f)
    
    # Save predictions
    pd.DataFrame({
        'pred': y_train_pred,
        'pred_proba': y_train_pred_proba
    }).to_csv(output_dir / 'train_predictions.csv', index=False)
    
    pd.DataFrame({
        'pred': y_test_pred,
        'pred_proba': y_test_pred_proba
    }).to_csv(output_dir / 'test_predictions.csv', index=False)
    
    pd.DataFrame({
        'oof_pred_proba': oof_predictions
    }).to_csv(output_dir / 'oof_predictions.csv', index=False)
    
    print(f"    {model_name} on {version} complete: CV AUC={cv_metrics['auc_roc']:.6f}, Test AUC={test_metrics['auc_roc']:.6f}")


def main():
    """Main pipeline to train all models on all feature versions."""
    print("=" * 80)
    print("TRAINING ALL MODELS ON ALL FEATURE SETS")
    print("=" * 80)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Neural networks will use: GPU")
    else:
        print(f"\nGPU not available - neural networks will use CPU")
    
    # Check GPU for tree models
    try:
        from src.training.optuna_optimization import (
            GPU_AVAILABLE, GPU_NAME,
            LIGHTGBM_GPU_AVAILABLE, XGBOOST_GPU_AVAILABLE, CATBOOST_GPU_AVAILABLE
        )
        if GPU_AVAILABLE:
            print(f"  LightGBM will use: {'GPU' if LIGHTGBM_GPU_AVAILABLE else 'CPU'}")
            print(f"  XGBoost will use: {'GPU' if XGBOOST_GPU_AVAILABLE else 'CPU'}")
            print(f"  CatBoost will use: {'GPU' if CATBOOST_GPU_AVAILABLE else 'CPU'}")
    except ImportError:
        pass
    
    versions = ['v1', 'v2', 'v3']
    tree_models = ['LightGBM', 'XGBoost', 'CatBoost']
    nn_models = ['MLP', 'TabM']
    
    n_folds = 5
    
    for version in versions:
        print("\n" + "=" * 80)
        print(f"PROCESSING {version.upper()}")
        print("=" * 80)
        
        # Load data for this version
        X_train, y_train, X_test, y_test = load_data_for_version(version)
        
        # Train tree models
        print(f"\nTraining tree models on {version}...")
        for model_name in tree_models:
            train_tree_model_on_version(
                model_name, version,
                X_train, y_train, X_test, y_test,
                n_folds=n_folds, use_existing_params=True
            )
        
        # Train NN models
        print(f"\nTraining neural network models on {version}...")
        for model_name in nn_models:
            train_nn_model_on_version(
                model_name, version,
                X_train, y_train, X_test, y_test,
                n_folds=n_folds, device=device, use_existing_params=True
            )
    
    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE")
    print("=" * 80)
    print(f"Results saved to data/models/[version]_[model]/")


if __name__ == '__main__':
    main()

