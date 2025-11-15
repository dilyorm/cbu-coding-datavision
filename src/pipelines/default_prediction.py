"""Default prediction pipeline orchestration - CatBoost only"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from config.default_config import paths, training, model_config
from src.data.merging import merge_all_data
from src.features.cleaning import clean_data
from src.features.engineering import engineer_features
from src.features.missing_values import handle_missing_values
from src.modeling.prep import prepare_model_data, create_preprocessing_pipeline
from src.modeling.training import train_catboost_model
from src.modeling.evaluation import evaluate_model, find_best_threshold
from src.data.io_utils import save_artifacts
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix


def run_default_pipeline():
    """Run the complete default prediction pipeline with CatBoost only"""
    print("="*60)
    print("CATBOOST DEFAULT PREDICTION MODEL PIPELINE")
    print("="*60)
    
    # Step 1-3: Load and merge data
    df = merge_all_data(paths)
    
    # Step 4: Clean data
    df_clean = clean_data(df)
    
    # Step 5: Engineer features
    df_feat = engineer_features(df_clean)
    
    # Step 6: Handle missing values
    df_final = handle_missing_values(df_feat, imputation_method=training.imputation_method)
    
    # Step 7: Prepare model data
    X, y, feature_cols = prepare_model_data(df_final)
    
    # Step 8: Train/Validation/Test Split
    print("\n" + "="*60)
    print("STEP 8: Train/Validation/Test Split...")
    print("="*60)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=training.test_size, stratify=y, random_state=training.random_state
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=training.valid_size, stratify=y_temp, random_state=training.random_state
    )
    print(f"Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")
    
    # Get categorical columns (needed for CatBoost)
    # Note: We don't need the preprocessor for CatBoost, but we keep it for compatibility
    _, _, categorical_cols = create_preprocessing_pipeline(X_train)
    
    # Step 9: Train CatBoost model
    models, results = train_catboost_model(
        X_train, y_train, X_valid, y_valid, categorical_cols,
        tune_hyperparams=training.tune_hyperparams,
        model_config=model_config
    )
    
    # Step 10: Find optimal threshold
    print("\n" + "="*60)
    print("STEP 10: Optimizing Classification Threshold...")
    print("="*60)
    thresholds = {}
    
    cat_valid_scores = results['CatBoost']['predictions']
    best_thr, bp, br, bf1 = find_best_threshold(y_valid, cat_valid_scores, beta=1.0)
    thresholds['CatBoost'] = best_thr
    print(f"CatBoost: Best threshold={best_thr:.4f}, Precision={bp:.4f}, Recall={br:.4f}, F1={bf1:.4f}")
    
    # Step 11: Evaluate on test set
    print("\n" + "="*60)
    print("STEP 11: Test Set Evaluation...")
    print("="*60)
    
    # Prepare test data for CatBoost (raw categoricals)
    X_test_cat = X_test.copy()
    for c in categorical_cols:
        if c in X_test_cat.columns:
            X_test_cat[c] = X_test_cat[c].astype(str)
    
    # CatBoost evaluation
    test_results = {}
    test_results['CatBoost'] = evaluate_model(
        models['CatBoost'], X_test, y_test, None, 'CatBoost',
        best_threshold=thresholds['CatBoost'],
        use_raw_catboost=True,
        X_test_raw=X_test_cat
    )
    
    # Step 12: Save artifacts
    save_artifacts(
        models, results, None, thresholds, None, 
        feature_cols, categorical_cols, df_final, paths
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Final CatBoost Test AUC: {test_results['CatBoost']['auc']:.4f}")
    print(f"Final CatBoost Test PR-AUC: {test_results['CatBoost']['pr_auc']:.4f}")
    
    return {
        'model': models['CatBoost'],
        'results': results,
        'test_results': test_results,
        'threshold': thresholds['CatBoost']
    }
