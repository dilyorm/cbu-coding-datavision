"""Test script to compare baseline vs improved CatBoost models"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.merging import merge_all_data
from src.features.cleaning import clean_data
from src.features.engineering import engineer_features
from src.features.missing_values import handle_missing_values
from src.modeling.prep import prepare_catboost_data
from src.modeling.model_comparison import compare_models
from sklearn.model_selection import train_test_split
from config.default_config import paths, training

import pandas as pd


def main():
    """Main function to test model improvements"""
    print("="*80)
    print("TESTING MODEL IMPROVEMENTS")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load and prepare data")
    print("  2. Split into train/test sets")
    print("  3. Train baseline model (original narrow search space)")
    print("  4. Train improved model (expanded search space + new features)")
    print("  5. Compare results")
    print("="*80)
    
    # Load and prepare data
    print("\n[Step 1] Loading and preparing data...")
    print("   Loading raw data...")
    df_merged = merge_all_data(paths)
    
    if df_merged is None or len(df_merged) == 0:
        print("ERROR: Failed to load data")
        return
    
    print("   Cleaning data...")
    df_clean = clean_data(df_merged)
    
    print("   Engineering features...")
    df_feat = engineer_features(df_clean)
    
    print("   Handling missing values...")
    imputation_method = getattr(training, 'imputation_method', 'knn')
    df_final = handle_missing_values(df_feat, imputation_method=imputation_method)
    
    print(f"   Final dataset: {len(df_final)} rows with {len(df_final.columns)} features")
    
    # Split into train/test
    print("\n[Step 2] Splitting data into train/test sets...")
    test_size = getattr(training, 'holdout_test_size', 0.15)
    train_idx, test_idx = train_test_split(
        df_final.index, 
        test_size=test_size, 
        stratify=df_final['default'], 
        random_state=training.random_state
    )
    train_df = df_final.loc[train_idx].copy()
    test_df = df_final.loc[test_idx].copy()
    
    print(f"   Train set: {len(train_df)} rows")
    print(f"   Test set:  {len(test_df)} rows")
    
    # Prepare CatBoost data
    print("\n[Step 3] Preparing CatBoost data...")
    X_train, y_train, cat_feature_indices = prepare_catboost_data(train_df, target_col="default")
    X_test, y_test, _ = prepare_catboost_data(test_df, target_col="default")
    
    # Ensure categorical columns are strings
    categorical_cols = [X_train.columns[i] for i in cat_feature_indices if i < len(X_train.columns)]
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()
    for col in categorical_cols:
        if col in X_train_cat.columns:
            X_train_cat[col] = X_train_cat[col].astype(str)
        if col in X_test_cat.columns:
            X_test_cat[col] = X_test_cat[col].astype(str)
    
    print(f"   Features: {len(X_train_cat.columns)}")
    print(f"   Categorical features: {len(cat_feature_indices)}")
    
    # Compare models
    print("\n[Step 4] Comparing models...")
    n_trials = getattr(training, 'catboost_n_trials', 50)
    n_splits = getattr(training, 'cv_n_splits', 5)
    
    use_gpu = getattr(training, 'use_gpu', True)
    results, baseline_model, improved_model = compare_models(
        X_train_cat, y_train, X_test_cat, y_test, cat_feature_indices,
        n_trials=n_trials, n_splits=n_splits, 
        random_state=training.random_state,
        output_dir=paths.datas_dir,
        use_gpu=use_gpu
    )
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    
    # Final verdict
    cv_improvement = results['improvement']['cv_auc_delta']
    if cv_improvement > 0.001:  # Significant improvement
        print(f"\n✅ SUCCESS: Model improved by {cv_improvement:.5f} AUC")
        print("   The improvements are working!")
    elif cv_improvement < -0.001:  # Significant regression
        print(f"\n❌ REGRESSION: Model decreased by {abs(cv_improvement):.5f} AUC")
        print("   Consider reverting some changes")
    else:
        print(f"\n➡️  NO SIGNIFICANT CHANGE: Improvement within noise margin")
        print("   May need more trials or different improvements")
    
    return results


if __name__ == "__main__":
    results = main()

