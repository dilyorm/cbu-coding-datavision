"""CatBoost model training with raw categorical features"""
import catboost as cb
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from .tuning import tune_catboost, tune_catboost_cv


def train_catboost_raw(X_train, y_train, X_valid, y_valid, categorical_cols, 
                       tune_hyperparams=False, model_config=None):
    """Train CatBoost with raw categorical features using Pool (no one-hot encoding)"""
    # Identify categorical columns
    cat_cols = [col for col in categorical_cols if col in X_train.columns]
    
    # Ensure they're strings
    X_train_cat = X_train.copy()
    X_valid_cat = X_valid.copy()
    for c in cat_cols:
        X_train_cat[c] = X_train_cat[c].astype(str)
        X_valid_cat[c] = X_valid_cat[c].astype(str)
    
    cat_feature_indices = [X_train_cat.columns.get_loc(c) for c in cat_cols if c in X_train_cat.columns]
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # Use default config if not provided
    if model_config is None:
        from config.default_config import model_config
    
    # Create Pool objects for CatBoost
    train_pool = Pool(X_train_cat, y_train, cat_features=cat_feature_indices)
    valid_pool = Pool(X_valid_cat, y_valid, cat_features=cat_feature_indices)
    
    if tune_hyperparams:
        print("   Tuning hyperparameters (this may take a while)...")
        # Get n_trials from training config
        from config.default_config import training
        n_trials = training.catboost_n_trials
        best_params, best_score = tune_catboost(
            X_train_cat, y_train, X_valid_cat, y_valid, 
            categorical_cols, n_trials=n_trials
        )
        print(f"   Best validation AUC during tuning: {best_score:.4f}")
        cat_model = cb.CatBoostClassifier(**best_params)
    else:
        # Default parameters
        cat_model = cb.CatBoostClassifier(
            iterations=model_config.catboost_iterations,
            learning_rate=model_config.catboost_learning_rate,
            depth=model_config.catboost_depth,
            l2_leaf_reg=model_config.catboost_l2_leaf_reg,
            loss_function='Logloss',
            eval_metric='AUC',
            class_weights=[1, pos_weight],
            random_seed=42,
            verbose=False
        )
    
    cat_model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=model_config.early_stopping_rounds
    )
    
    cat_pred = cat_model.predict_proba(valid_pool)[:, 1]
    cat_auc = roc_auc_score(y_valid, cat_pred)
    
    return cat_model, cat_auc, X_train_cat, X_valid_cat


def train_catboost_model_cv(X_train, y_train, categorical_cols, 
                             tune_hyperparams=False, model_config=None, 
                             n_splits=5, random_state=42):
    """Train CatBoost model using StratifiedKFold CV"""
    print("\n" + "="*60)
    print("STEP 9: Training CatBoost Model with Cross-Validation...")
    print("="*60)
    
    # Identify categorical columns
    cat_cols = [col for col in categorical_cols if col in X_train.columns]
    
    # Ensure categorical columns are strings
    X_train_cat = X_train.copy()
    for c in cat_cols:
        X_train_cat[c] = X_train_cat[c].astype(str)
    
    cat_feature_indices = [X_train_cat.columns.get_loc(c) for c in cat_cols if c in X_train_cat.columns]
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # Use default config if not provided
    if model_config is None:
        from config.default_config import model_config
    
    # Setup CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = []
    fold_models = []
    
    print(f"   Using {n_splits}-fold StratifiedKFold CV")
    
    # Hyperparameter tuning with CV
    if tune_hyperparams:
        print("   Tuning hyperparameters with CV (this may take a while)...")
        from config.default_config import training
        n_trials = getattr(training, 'catboost_n_trials', 50)
        best_params, best_cv_score = tune_catboost_cv(
            X_train_cat, y_train, cat_feature_indices, 
            n_splits=n_splits, n_trials=n_trials, random_state=random_state
        )
        print(f"   Best CV AUC during tuning: {best_cv_score:.4f}")
        model_params = best_params
    else:
        # Default parameters
        model_params = {
            'iterations': model_config.catboost_iterations,
            'learning_rate': model_config.catboost_learning_rate,
            'depth': model_config.catboost_depth,
            'l2_leaf_reg': model_config.catboost_l2_leaf_reg,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'class_weights': [1, pos_weight],
            'random_seed': random_state,
            'verbose': False
        }
    
    # Train models on each fold
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train_cat, y_train), 1):
        X_fold_train = X_train_cat.iloc[train_idx]
        X_fold_valid = X_train_cat.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]
        
        # Create Pool objects
        train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_feature_indices)
        valid_pool = Pool(X_fold_valid, y_fold_valid, cat_features=cat_feature_indices)
        
        # Train model
        fold_model = cb.CatBoostClassifier(**model_params)
        fold_model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=model_config.early_stopping_rounds
        )
        
        # Evaluate
        fold_pred = fold_model.predict_proba(valid_pool)[:, 1]
        fold_auc = roc_auc_score(y_fold_valid, fold_pred)
        cv_scores.append(fold_auc)
        fold_models.append(fold_model)
        
        print(f"   Fold {fold_idx}/{n_splits}: AUC = {fold_auc:.4f}")
    
    # Average CV score
    mean_cv_auc = np.mean(cv_scores)
    std_cv_auc = np.std(cv_scores)
    print(f"\n   Mean CV AUC: {mean_cv_auc:.4f} (+/- {std_cv_auc:.4f})")
    
    # Train final model on all training data
    print("\n   Training final model on all training data...")
    train_pool_all = Pool(X_train_cat, y_train, cat_features=cat_feature_indices)
    final_model = cb.CatBoostClassifier(**model_params)
    final_model.fit(
        train_pool_all,
        early_stopping_rounds=model_config.early_stopping_rounds
    )
    
    results = {
        'CatBoost': {
            'cv_auc_mean': mean_cv_auc,
            'cv_auc_std': std_cv_auc,
            'cv_scores': cv_scores
        }
    }
    
    models = {
        'CatBoost': final_model,
        'CatBoost_X_train': X_train_cat,
        'CatBoost_fold_models': fold_models
    }
    
    return models, results


def train_catboost_model(X_train, y_train, X_valid, y_valid, categorical_cols,
                          tune_hyperparams=False, model_config=None):
    """Train optimized CatBoost model (legacy function for backward compatibility)"""
    print("\n" + "="*60)
    print("STEP 9: Training CatBoost Model (Raw Categoricals)...")
    print("="*60)
    
    model, auc, X_train_cat, X_valid_cat = train_catboost_raw(
        X_train, y_train, X_valid, y_valid, categorical_cols,
        tune_hyperparams=tune_hyperparams,
        model_config=model_config
    )
    
    print(f"   Validation AUC: {auc:.4f}")
    
    results = {
        'CatBoost': {
            'auc': auc,
            'predictions': model.predict_proba(X_valid_cat)[:, 1]
        }
    }
    
    models = {
        'CatBoost': model,
        'CatBoost_X_train': X_train_cat,
        'CatBoost_X_valid': X_valid_cat
    }
    
    return models, results


def train_final_catboost(X_train, y_train, cat_feature_indices, best_params):
    """Train final CatBoost model on all training data with best parameters
    
    Args:
        X_train: Training feature DataFrame
        y_train: Training target Series
        cat_feature_indices: List of categorical feature indices
        best_params: Best hyperparameters from tuning
        
    Returns:
        final_model: Trained CatBoostClassifier
    """
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(train_pool)
    return final_model
