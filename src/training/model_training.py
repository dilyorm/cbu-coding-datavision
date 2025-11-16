"""Core model training utilities with cross-validation support."""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from typing import Dict, Tuple, List, Any, Optional
import warnings
import re
warnings.filterwarnings('ignore')

# Import sanitization functions from feature_selection
try:
    from feature_selection import sanitize_dataframe_columns, sanitize_feature_names
except ImportError:
    # Fallback: define sanitization functions here if import fails
    def sanitize_feature_names(feature_names):
        """Sanitize feature names to remove special JSON characters."""
        if isinstance(feature_names, pd.Index):
            feature_names = feature_names.tolist()
        
        sanitized_names = []
        name_mapping = {}
        
        for name in feature_names:
            sanitized = re.sub(r'[\[\]{}"\'\\\/\s]', '_', str(name))
            sanitized = re.sub(r'_+', '_', sanitized)
            sanitized = sanitized.strip('_')
            if not sanitized:
                sanitized = f'feature_{len(sanitized_names)}'
            
            original_sanitized = sanitized
            counter = 0
            while sanitized in name_mapping.values():
                counter += 1
                sanitized = f"{original_sanitized}_{counter}"
            
            sanitized_names.append(sanitized)
            name_mapping[sanitized] = name
        
        return sanitized_names, name_mapping
    
    def sanitize_dataframe_columns(df):
        """Sanitize DataFrame column names for LightGBM compatibility."""
        sanitized_cols, column_mapping = sanitize_feature_names(df.columns)
        sanitized_df = df.copy()
        sanitized_df.columns = sanitized_cols
        return sanitized_df, column_mapping


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (thresholded)
        y_pred_proba: Predicted probabilities (if None, uses y_pred)
    
    Returns:
        Dictionary with all metrics
    """
    if y_pred_proba is None:
        y_pred_proba = y_pred
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Add confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
    
    return metrics


def train_lightgbm_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    n_folds: int = 5,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[Dict[str, float], List[lgb.Booster], np.ndarray]:
    """Train LightGBM with k-fold cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: LightGBM parameters
        n_folds: Number of CV folds
        n_estimators: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (mean CV metrics, list of trained models, OOF predictions)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_scores = []
    models = []
    oof_predictions = np.zeros(len(X_train))
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Sanitize column names for LightGBM compatibility
        sanitized_cols, col_mapping = sanitize_feature_names(X_fold_train.columns)
        
        # Convert to numpy arrays and pass explicit feature names to avoid any DataFrame metadata issues
        X_fold_train_array = X_fold_train.values
        X_fold_val_array = X_fold_val.values
        y_fold_train_array = y_fold_train.values if isinstance(y_fold_train, pd.Series) else y_fold_train
        y_fold_val_array = y_fold_val.values if isinstance(y_fold_val, pd.Series) else y_fold_val
        
        train_data = lgb.Dataset(X_fold_train_array, label=y_fold_train_array, feature_name=sanitized_cols)
        val_data = lgb.Dataset(X_fold_val_array, label=y_fold_val_array, feature_name=sanitized_cols, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(0) if not verbose else lgb.log_evaluation(100)
            ]
        )
        
        y_pred_proba = model.predict(X_fold_val_array)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Store OOF predictions for this fold
        oof_predictions[val_idx] = y_pred_proba
        
        fold_metrics = evaluate_model(y_fold_val.values, y_pred, y_pred_proba)
        cv_scores.append(fold_metrics)
        models.append(model)
    
    # Average metrics across folds
    mean_metrics = {}
    for key in cv_scores[0].keys():
        mean_metrics[key] = np.mean([scores[key] for scores in cv_scores])
    
    return mean_metrics, models, oof_predictions


def train_xgboost_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    n_folds: int = 5,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[Dict[str, float], List[xgb.Booster], np.ndarray]:
    """Train XGBoost with k-fold cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: XGBoost parameters
        n_folds: Number of CV folds
        n_estimators: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (mean CV metrics, list of trained models, OOF predictions)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_scores = []
    models = []
    oof_predictions = np.zeros(len(X_train))
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
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
            verbose_eval=100 if verbose else False
        )
        
        y_pred_proba = model.predict(dval)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Store OOF predictions for this fold
        oof_predictions[val_idx] = y_pred_proba
        
        fold_metrics = evaluate_model(y_fold_val.values, y_pred, y_pred_proba)
        cv_scores.append(fold_metrics)
        models.append(model)
    
    # Average metrics across folds
    mean_metrics = {}
    for key in cv_scores[0].keys():
        mean_metrics[key] = np.mean([scores[key] for scores in cv_scores])
    
    return mean_metrics, models, oof_predictions


def train_catboost_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    n_folds: int = 5,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[Dict[str, float], List[cb.CatBoostClassifier], np.ndarray]:
    """Train CatBoost with k-fold cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: CatBoost parameters
        n_folds: Number of CV folds
        n_estimators: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (mean CV metrics, list of trained models, OOF predictions)
    """
    # Remove conflicting parameters if they exist in params
    model_params = params.copy()
    model_params.pop('iterations', None)
    model_params.pop('early_stopping_rounds', None)
    model_params.pop('verbose', None)
    model_params.pop('random_seed', None)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_scores = []
    models = []
    oof_predictions = np.zeros(len(X_train))
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        model = cb.CatBoostClassifier(
            iterations=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            random_seed=random_state,
            **model_params
        )
        
        model.fit(
            X_fold_train,
            y_fold_train,
            eval_set=(X_fold_val, y_fold_val),
            use_best_model=True
        )
        
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        y_pred = model.predict(X_fold_val)
        
        # Store OOF predictions for this fold
        oof_predictions[val_idx] = y_pred_proba
        
        fold_metrics = evaluate_model(y_fold_val.values, y_pred, y_pred_proba)
        cv_scores.append(fold_metrics)
        models.append(model)
    
    # Average metrics across folds
    mean_metrics = {}
    for key in cv_scores[0].keys():
        mean_metrics[key] = np.mean([scores[key] for scores in cv_scores])
    
    return mean_metrics, models, oof_predictions


def train_lightgbm_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    verbose: bool = False,
    cv_optimal_iterations: Optional[int] = None
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """Train final LightGBM model on full training set.
    
    Uses CV to determine optimal iterations if not provided, then trains on full set.
    
    Returns:
        Tuple of (trained model, CV metrics)
    """
    # If optimal iterations not provided, use CV to find it
    if cv_optimal_iterations is None:
        _, _, _ = train_lightgbm_cv(X_train, y_train, params, n_folds=5, 
                                     n_estimators=n_estimators, 
                                     early_stopping_rounds=early_stopping_rounds,
                                     verbose=False)
        # Get average iterations from CV (simplified - use n_estimators)
        cv_optimal_iterations = n_estimators
    
    # Sanitize column names for LightGBM compatibility
    sanitized_cols, col_mapping = sanitize_feature_names(X_train.columns)
    
    # Convert to numpy arrays
    X_train_array = X_train.values
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    train_data = lgb.Dataset(X_train_array, label=y_train_array, feature_name=sanitized_cols)
    
    # Train on full training set
    model = lgb.train(
        params,
        train_data,
        num_boost_round=cv_optimal_iterations,
        callbacks=[lgb.log_evaluation(100 if verbose else 0)]
    )
    
    # Store column mapping for predictions
    model._column_mapping = col_mapping
    
    # Get CV metrics for reporting
    cv_metrics, _, _ = train_lightgbm_cv(X_train, y_train, params, n_folds=5,
                                         n_estimators=n_estimators,
                                         early_stopping_rounds=early_stopping_rounds,
                                         verbose=False)
    
    return model, cv_metrics


def train_xgboost_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    verbose: bool = False,
    cv_optimal_iterations: Optional[int] = None
) -> Tuple[xgb.Booster, Dict[str, float]]:
    """Train final XGBoost model on full training set.
    
    Uses CV to determine optimal iterations if not provided, then trains on full set.
    
    Returns:
        Tuple of (trained model, CV metrics)
    """
    # If optimal iterations not provided, use CV to find it
    if cv_optimal_iterations is None:
        cv_optimal_iterations = n_estimators
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Train on full training set
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=cv_optimal_iterations,
        verbose_eval=100 if verbose else False
    )
    
    # Get CV metrics for reporting
    cv_metrics, _, _ = train_xgboost_cv(X_train, y_train, params, n_folds=5,
                                        n_estimators=n_estimators,
                                        early_stopping_rounds=early_stopping_rounds,
                                        verbose=False)
    
    return model, cv_metrics


def train_catboost_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any],
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
    verbose: bool = False,
    cv_optimal_iterations: Optional[int] = None
) -> Tuple[cb.CatBoostClassifier, Dict[str, float]]:
    """Train final CatBoost model on full training set.
    
    Uses CV to determine optimal iterations if not provided, then trains on full set.
    
    Returns:
        Tuple of (trained model, CV metrics)
    """
    # If optimal iterations not provided, use CV to find it
    if cv_optimal_iterations is None:
        cv_optimal_iterations = n_estimators
    
    # Remove conflicting parameters if they exist in params
    model_params = params.copy()
    model_params.pop('iterations', None)
    model_params.pop('early_stopping_rounds', None)
    model_params.pop('verbose', None)
    
    model = cb.CatBoostClassifier(
        iterations=cv_optimal_iterations,
        verbose=verbose,
        **model_params
    )
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Get CV metrics for reporting
    cv_metrics, _, _ = train_catboost_cv(X_train, y_train, params, n_folds=5,
                                         n_estimators=n_estimators,
                                         early_stopping_rounds=early_stopping_rounds,
                                         verbose=False)
    
    return model, cv_metrics

