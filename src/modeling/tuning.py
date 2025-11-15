"""Hyperparameter tuning functions"""
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import catboost as cb
from catboost import Pool
from .gpu_utils import get_task_type


def tune_lightgbm(X_train, y_train, X_valid, y_valid, n_trials=20):
    """Hyperparameter tuning for LightGBM using Optuna"""
    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'objective': 'binary',
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        pred = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, pred)
    
    study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({
        'n_estimators': 1000,
        'objective': 'binary',
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    })
    
    return best_params, study.best_value


def tune_catboost_cv(X, y, cat_feature_indices, n_splits=5, n_trials=50, random_state=42, use_gpu=True):
    """Hyperparameter tuning for CatBoost using Optuna with StratifiedKFold CV
    
    Args:
        X: Feature DataFrame
        y: Target Series
        cat_feature_indices: List of categorical feature indices
        n_splits: Number of CV folds (default: 5)
        n_trials: Number of Optuna trials (default: 50)
        random_state: Random state for reproducibility
        
    Returns:
        best_params: Best hyperparameters found
        best_cv_auc: Best mean CV AUC score
    """
    # Setup CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Get task type (GPU or CPU)
    task_type = get_task_type(use_gpu=use_gpu)
    if task_type == "GPU":
        print(f"   Using GPU for training (faster)")
        print(f"   Note: Using Logloss as eval_metric (AUC not supported on GPU, will compute separately)")
        eval_metric = "Logloss"  # GPU doesn't support AUC as eval_metric
    else:
        print(f"   Using CPU for training (GPU not available)")
        eval_metric = "AUC"  # CPU supports AUC
    
    def objective(trial):
        params = {
            "loss_function": "Logloss",
            "eval_metric": eval_metric,  # Use Logloss for GPU, AUC for CPU
            "iterations": 5000,  # rely on early stopping
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 20.0),
            "border_count": trial.suggest_int("border_count", 64, 254),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 15),
            "task_type": task_type,  # Use GPU if available
            "verbose": False,
            "random_seed": random_state,
        }
        
        # Add bootstrap-specific parameters
        if params["bootstrap_type"] == "Bayesian":
            # bagging_temperature is only available for Bayesian bootstrap
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
        elif params["bootstrap_type"] == "Bernoulli":
            # subsample is required for Bernoulli bootstrap
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        elif params["bootstrap_type"] == "MVS":
            # MVS (Minimal Variance Sampling) parameters
            params["mvs_reg"] = trial.suggest_float("mvs_reg", 20.0, 50.0)
        
        # Cross-validation
        fold_scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_fold_train = X.iloc[train_idx]
            X_fold_valid = X.iloc[valid_idx]
            y_fold_train = y.iloc[train_idx]
            y_fold_valid = y.iloc[valid_idx]
            
            # Create Pool objects with cat_features
            train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_feature_indices)
            valid_pool = Pool(X_fold_valid, y_fold_valid, cat_features=cat_feature_indices)
            
            # Train CatBoostClassifier
            model = cb.CatBoostClassifier(**params)
            model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=200,
                use_best_model=True
            )
            
            # Compute ROC-AUC on validation fold
            pred = model.predict_proba(valid_pool)[:, 1]
            fold_auc = roc_auc_score(y_fold_valid, pred)
            fold_scores.append(fold_auc)
        
        # Return mean ROC-AUC across folds
        return np.mean(fold_scores)
    
    study = optuna.create_study(direction="maximize", study_name='catboost_cv_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_cv_auc = study.best_value
    
    # Add fixed fields (preserve bootstrap-specific params if they exist)
    fixed_params = {
        "loss_function": "Logloss",
        "eval_metric": eval_metric,  # Use appropriate metric for GPU/CPU
        "iterations": 5000,
        "task_type": task_type,  # Preserve GPU/CPU setting
        "verbose": False,
        "random_seed": random_state,
    }
    # Only update with fixed params that aren't already in best_params
    for key, value in fixed_params.items():
        if key not in best_params:
            best_params[key] = value
    
    return best_params, best_cv_auc


def tune_catboost(X_train, y_train, X_valid, y_valid, categorical_cols, n_trials=20):
    """Hyperparameter tuning for CatBoost using Optuna with Pool (legacy function)"""
    # Prepare categorical features
    cat_feature_indices = [X_train.columns.get_loc(c) for c in categorical_cols if c in X_train.columns]
    
    # Create Pool objects
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_feature_indices)
    
    def objective(trial):
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 10.0),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "class_weights": [1, pos_weight],
            "random_seed": 42,
            "verbose": False
        }
        
        model = cb.CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=50
        )
        pred = model.predict_proba(valid_pool)[:, 1]
        return roc_auc_score(y_valid, pred)
    
    study = optuna.create_study(direction="maximize", study_name='catboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    best_params.update({
        "iterations": 1000,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "class_weights": [1, pos_weight],
        "random_seed": 42,
        "verbose": False
    })
    
    return best_params, study.best_value

