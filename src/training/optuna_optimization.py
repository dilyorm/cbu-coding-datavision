"""Optuna hyperparameter optimization objectives for LightGBM, XGBoost, and CatBoost."""
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_NAME = None
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = None

# Check GPU support for tree models
LIGHTGBM_GPU_AVAILABLE = False
XGBOOST_GPU_AVAILABLE = False
CATBOOST_GPU_AVAILABLE = False

if GPU_AVAILABLE:
    try:
        # Test LightGBM GPU
        test_data = lgb.Dataset([[1, 2], [3, 4]], label=[0, 1])
        test_params = {'objective': 'binary', 'device': 'gpu', 'verbose': -1}
        try:
            lgb.train(test_params, test_data, num_boost_round=1, callbacks=[lgb.log_evaluation(0)])
            LIGHTGBM_GPU_AVAILABLE = True
        except Exception:
            pass
    except Exception:
        pass
    
    try:
        # Test XGBoost GPU
        test_data = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
        test_params = {'objective': 'binary:logistic', 'tree_method': 'gpu_hist'}
        try:
            xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
            XGBOOST_GPU_AVAILABLE = True
        except Exception:
            pass
    except Exception:
        pass
    
    # CatBoost GPU is usually available if GPU is available
    CATBOOST_GPU_AVAILABLE = GPU_AVAILABLE


def optimize_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[optuna.Study, dict]:
    """Optimize LightGBM hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        n_estimators: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (Optuna study, best parameters)
    """
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'device': 'gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu',
            'verbose': -1,
            'seed': random_state
        }
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = model.predict(X_fold_val)
            auc = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(auc)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='maximize', study_name='lightgbm_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    
    best_params = study.best_params.copy()
    best_params['objective'] = 'binary'
    best_params['metric'] = 'auc'
    best_params['boosting_type'] = 'gbdt'
    best_params['device'] = 'gpu' if LIGHTGBM_GPU_AVAILABLE else 'cpu'
    best_params['verbose'] = -1
    best_params['seed'] = random_state
    
    return study, best_params


def optimize_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[optuna.Study, dict]:
    """Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        n_estimators: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (Optuna study, best parameters)
    """
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'tree_method': 'gpu_hist' if XGBOOST_GPU_AVAILABLE else 'hist',
            'random_state': random_state
        }
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
            dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, 'eval')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False
            )
            
            y_pred = model.predict(dval)
            auc = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(auc)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='maximize', study_name='xgboost_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    
    best_params = study.best_params.copy()
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['tree_method'] = 'gpu_hist' if XGBOOST_GPU_AVAILABLE else 'hist'
    best_params['random_state'] = random_state
    
    return study, best_params


def optimize_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[optuna.Study, dict]:
    """Optimize CatBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        n_estimators: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (Optuna study, best parameters)
    """
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'task_type': 'GPU' if CATBOOST_GPU_AVAILABLE else 'CPU',
            'iterations': n_estimators,
            'early_stopping_rounds': early_stopping_rounds,
            'verbose': False,
            'random_seed': random_state,
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss'  # Use Logloss for GPU, calculate AUC manually
        }
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = cb.CatBoostClassifier(**params)
            
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                use_best_model=True
            )
            
            y_pred = model.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, y_pred)
            cv_scores.append(auc)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='maximize', study_name='catboost_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    
    best_params = study.best_params.copy()
    best_params['iterations'] = n_estimators
    best_params['early_stopping_rounds'] = early_stopping_rounds
    best_params['task_type'] = 'GPU' if CATBOOST_GPU_AVAILABLE else 'CPU'
    best_params['verbose'] = False
    best_params['random_seed'] = random_state
    best_params['loss_function'] = 'Logloss'
    best_params['eval_metric'] = 'Logloss'  # Use Logloss for GPU compatibility
    
    return study, best_params

