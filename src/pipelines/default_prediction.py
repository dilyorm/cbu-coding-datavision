"""Default prediction pipeline orchestration - CatBoost only with Optuna tuning"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from config.default_config import paths, training, model_config
from src.data.merging import merge_all_data
from src.features.cleaning import clean_data
from src.features.engineering import engineer_features
from src.features.missing_values import handle_missing_values
from src.modeling.prep import prepare_model_data, prepare_catboost_data
from src.modeling.training import train_final_catboost
from src.modeling.tuning import tune_catboost_cv
from src.modeling.evaluation import evaluate_model, find_best_threshold
from src.data.io_utils import save_artifacts
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from catboost import Pool


def evaluate_final_model(model, X_test, y_test, cat_feature_indices, model_name="CatBoost"):
    """Evaluate final CatBoost model on test set using Pool
    
    Args:
        model: Trained CatBoostClassifier
        X_test: Test feature DataFrame
        y_test: Test target Series
        cat_feature_indices: List of categorical feature indices
        model_name: Name of model for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Prepare test data - ensure categorical columns are strings
    X_test_cat = X_test.copy()
    categorical_cols = [X_test.columns[i] for i in cat_feature_indices if i < len(X_test.columns)]
    for col in categorical_cols:
        if col in X_test_cat.columns:
            X_test_cat[col] = X_test_cat[col].astype(str)
    
    # Create Pool object with categorical features
    test_pool = Pool(X_test_cat, y_test, cat_features=cat_feature_indices)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Compute metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Test Results:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {'auc': auc, 'pr_auc': pr_auc, 'predictions': y_pred_proba}


def run_default_pipeline():
    """Run the complete default prediction pipeline with CatBoost only
    
    Training flow:
    1. Split df_final into train_df and test_df once (small test set)
    2. From train_df, create X_train_full, y_train_full, cat_idx using prepare_catboost_data
    3. Run tune_catboost_cv to get best_params
    4. Train final_model using train_final_catboost
    5. Evaluate final_model on held-out test_df
    """
    print("="*60)
    print("CATBOOST DEFAULT PREDICTION MODEL PIPELINE")
    print("="*60)
    
    # Step 1-3: Load and merge data
    df = merge_all_data(paths)
    
    # Step 4: Clean data
    df_clean = clean_data(df)
    
    # Step 5: Engineer features (includes log transforms and bucketed features for CatBoost)
    df_feat = engineer_features(df_clean)
    
    # Step 6: Handle missing values
    df_final = handle_missing_values(df_feat, imputation_method=training.imputation_method)
    
    # Step 7: Split df_final into train_df and test_df once (small test set)
    print("\n" + "="*60)
    print("STEP 7: Creating Train/Test Split...")
    print("="*60)
    test_size = getattr(training, 'holdout_test_size', 0.15)  # Default 15%
    # Split using indices to get DataFrames
    train_idx, test_idx = train_test_split(
        df_final.index, test_size=test_size, stratify=df_final['default'], random_state=training.random_state
    )
    train_df = df_final.loc[train_idx].copy()
    test_df = df_final.loc[test_idx].copy()
    print(f"Training set: {train_df.shape[0]}, Test set: {test_df.shape[0]}")
    print(f"Test set will only be used for final evaluation")
    
    # Step 8: Prepare CatBoost data from train_df
    print("\n" + "="*60)
    print("STEP 8: Preparing CatBoost Data...")
    print("="*60)
    X_train_full, y_train_full, cat_feature_indices = prepare_catboost_data(train_df, target_col="default")
    print(f"Training features: {X_train_full.shape[1]}")
    print(f"Categorical feature indices: {len(cat_feature_indices)}")
    
    # Ensure categorical columns are strings
    categorical_cols = [X_train_full.columns[i] for i in cat_feature_indices if i < len(X_train_full.columns)]
    X_train_full_cat = X_train_full.copy()
    for col in categorical_cols:
        if col in X_train_full_cat.columns:
            X_train_full_cat[col] = X_train_full_cat[col].astype(str)
    
    # Step 9: Run Optuna tuning with CV
    print("\n" + "="*60)
    print("STEP 9: Hyperparameter Tuning with Optuna and CV...")
    print("="*60)
    n_trials = getattr(training, 'catboost_n_trials', 50)
    n_splits = getattr(training, 'cv_n_splits', 5)
    best_params, best_cv_auc = tune_catboost_cv(
        X_train_full_cat, y_train_full, cat_feature_indices,
        n_splits=n_splits, n_trials=n_trials, random_state=training.random_state
    )
    print(f"Best CV AUC: {best_cv_auc:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Step 10: Train final model on all training data
    print("\n" + "="*60)
    print("STEP 10: Training Final CatBoost Model...")
    print("="*60)
    final_model = train_final_catboost(
        X_train_full_cat, y_train_full, cat_feature_indices, best_params
    )
    print("Final model trained on all training data")
    
    # Step 11: Evaluate final model on held-out test set
    print("\n" + "="*60)
    print("STEP 11: Final Test Set Evaluation (Holdout)...")
    print("="*60)
    
    # Prepare test data
    X_test, y_test, _ = prepare_catboost_data(test_df, target_col="default")
    
    # Evaluate using Pool
    test_results = evaluate_final_model(
        final_model, X_test, y_test, cat_feature_indices, 'CatBoost'
    )
    
    # Step 12: Save artifacts
    feature_cols = X_train_full_cat.columns.tolist()
    feature_names_list = feature_cols
    results = {'CatBoost': {'cv_auc_mean': best_cv_auc, 'cv_auc_std': 0.0}}  # CV std not available from tuning
    
    save_artifacts(
        final_model, feature_cols, cat_feature_indices, feature_names_list,
        results, df_final, paths, test_results=test_results
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Final CatBoost CV AUC: {best_cv_auc:.4f}")
    print(f"Final CatBoost Test AUC: {test_results['auc']:.4f}")
    print(f"Final CatBoost Test PR-AUC: {test_results['pr_auc']:.4f}")
    
    return {
        'model': final_model,
        'best_params': best_params,
        'cv_auc': best_cv_auc,
        'test_results': test_results,
        'feature_cols': feature_cols,
        'cat_feature_indices': cat_feature_indices
    }


if __name__ == "__main__":
    run_default_pipeline()
