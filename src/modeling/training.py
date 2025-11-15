"""CatBoost model training with raw categorical features"""
import catboost as cb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from .tuning import tune_catboost


def train_catboost_raw(X_train, y_train, X_valid, y_valid, categorical_cols, 
                       tune_hyperparams=False, model_config=None):
    """Train CatBoost with raw categorical features (no one-hot encoding)"""
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
        X_train_cat, y_train,
        eval_set=(X_valid_cat, y_valid),
        cat_features=cat_feature_indices,
        early_stopping_rounds=model_config.early_stopping_rounds
    )
    
    cat_pred = cat_model.predict_proba(X_valid_cat)[:, 1]
    cat_auc = roc_auc_score(y_valid, cat_pred)
    
    return cat_model, cat_auc, X_train_cat, X_valid_cat


def train_catboost_model(X_train, y_train, X_valid, y_valid, categorical_cols,
                          tune_hyperparams=False, model_config=None):
    """Train optimized CatBoost model"""
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
