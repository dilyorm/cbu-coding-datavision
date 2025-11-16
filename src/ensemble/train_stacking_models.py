"""Train Level-2 stacking models on meta features."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from src.training.model_training import (
    train_lightgbm_cv, train_xgboost_cv,
    train_lightgbm_final, train_xgboost_final,
    evaluate_model
)


def load_stacking_data():
    """Load stacking dataset with meta features."""
    data_dir = Path('data/stacking')
    
    print("Loading stacking dataset...")
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv')['default']
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv')['default']
    
    print(f"Loaded stacking data:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def train_stacking_model(model_name, X_train, y_train, X_test, y_test, n_folds=5):
    """Train a stacking model (LGBM or XGB) on meta features."""
    output_dir = Path('data/models/stacking')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already completed
    metrics_file = model_dir / 'metrics.json'
    if metrics_file.exists():
        print(f"  {model_name} stacking model already completed, skipping...")
        return
    
    print(f"\nTraining {model_name} stacking model...")
    
    # Simple hyperparameters for stacking (prevent overfitting)
    if model_name == 'lightgbm':
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'max_depth': 5,
            'verbose': -1,
            'seed': 42
        }
        
        # Train with CV to get OOF predictions
        print("  Computing CV metrics...")
        cv_metrics, _, oof_predictions = train_lightgbm_cv(
            X_train, y_train, params, n_folds=n_folds, verbose=False
        )
        
        # Train final model
        print("  Training final model...")
        model, _ = train_lightgbm_final(
            X_train, y_train, params, verbose=False
        )
        
        y_test_pred_proba = model.predict(X_test.values)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        y_train_pred_proba = model.predict(X_train.values)
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        
    elif model_name == 'xgboost':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'gamma': 0.1,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        # Train with CV to get OOF predictions
        print("  Computing CV metrics...")
        cv_metrics, _, oof_predictions = train_xgboost_cv(
            X_train, y_train, params, n_folds=n_folds, verbose=False
        )
        
        # Train final model
        print("  Training final model...")
        model, _ = train_xgboost_final(
            X_train, y_train, params, verbose=False
        )
        
        dtest = xgb.DMatrix(X_test)
        dtrain = xgb.DMatrix(X_train)
        
        y_test_pred_proba = model.predict(dtest)
        y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
        y_train_pred_proba = model.predict(dtrain)
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    
    # Evaluate
    test_metrics = evaluate_model(y_test.values, y_test_pred, y_test_pred_proba)
    
    # Save results
    with open(model_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(model_dir / 'best_params.json', 'w') as f:
        json.dump(params, f, indent=2, default=str)
    
    metrics = {
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics
    }
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save predictions
    pd.DataFrame({
        'pred': y_train_pred,
        'pred_proba': y_train_pred_proba
    }).to_csv(model_dir / 'train_predictions.csv', index=False)
    
    pd.DataFrame({
        'pred': y_test_pred,
        'pred_proba': y_test_pred_proba
    }).to_csv(model_dir / 'test_predictions.csv', index=False)
    
    pd.DataFrame({
        'oof_pred_proba': oof_predictions
    }).to_csv(model_dir / 'oof_predictions.csv', index=False)
    
    print(f"  {model_name} stacking complete: CV AUC={cv_metrics['auc_roc']:.6f}, Test AUC={test_metrics['auc_roc']:.6f}")


def main():
    """Train Level-2 stacking models."""
    print("=" * 80)
    print("TRAINING STACKING MODELS (LEVEL-2)")
    print("=" * 80)
    
    # Load stacking data
    X_train, y_train, X_test, y_test = load_stacking_data()
    
    # Train stacking models (simple architectures to prevent overfitting)
    stacking_models = ['lightgbm', 'xgboost']
    n_folds = 5
    
    for model_name in stacking_models:
        train_stacking_model(
            model_name,
            X_train, y_train, X_test, y_test,
            n_folds=n_folds
        )
    
    print("\n" + "=" * 80)
    print("STACKING MODELS TRAINING COMPLETE")
    print("=" * 80)
    print(f"Results saved to: data/models/stacking/")


if __name__ == '__main__':
    main()

