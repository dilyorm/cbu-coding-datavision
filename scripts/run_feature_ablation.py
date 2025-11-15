"""Standalone script to run feature ablation analysis"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.default_config import paths, training
from src.data.merging import merge_all_data
from src.features.cleaning import clean_data
from src.features.engineering import engineer_features
from src.features.missing_values import handle_missing_values
from src.modeling.prep import prepare_catboost_data
from src.modeling.feature_ablation import run_feature_ablation_analysis


def main():
    """Run feature ablation analysis to identify features that hurt AUC"""
    print("="*60)
    print("FEATURE ABLATION ANALYSIS")
    print("="*60)
    print("This will test each feature by training models with/without it")
    print("to identify features that negatively impact AUC")
    print("="*60)
    
    # Step 1-3: Load and merge data
    print("\nLoading and preparing data...")
    df = merge_all_data(paths)
    
    # Step 4: Clean data
    df_clean = clean_data(df)
    
    # Step 5: Engineer features
    df_feat = engineer_features(df_clean)
    
    # Step 6: Handle missing values
    df_final = handle_missing_values(df_feat, imputation_method=training.imputation_method)
    
    # Step 7: Prepare CatBoost data (use all data for ablation analysis)
    print("\nPreparing data for analysis...")
    X_train_full, y_train_full, cat_feature_indices = prepare_catboost_data(df_final, target_col="default")
    print(f"Total features: {X_train_full.shape[1]}")
    print(f"Categorical feature indices: {len(cat_feature_indices)}")
    
    # Ensure categorical columns are strings
    categorical_cols = [X_train_full.columns[i] for i in cat_feature_indices if i < len(X_train_full.columns)]
    X_train_full_cat = X_train_full.copy()
    for col in categorical_cols:
        if col in X_train_full_cat.columns:
            X_train_full_cat[col] = X_train_full_cat[col].astype(str)
    
    # Run feature ablation analysis
    print("\n" + "="*60)
    print("Starting Feature Ablation Analysis...")
    print("="*60)
    print("This may take a while as it trains models for each feature...")
    
    ablation_results = run_feature_ablation_analysis(
        X_train_full_cat, y_train_full, cat_feature_indices,
        output_path=f'{paths.datas_dir}/feature_ablation_analysis.csv',
        n_splits=training.cv_n_splits,
        random_state=training.random_state
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Results saved to: {paths.datas_dir}/feature_ablation_analysis.csv")
    print("\nTop features that HURT performance (consider removing):")
    negative_impact = ablation_results[ablation_results['auc_difference'] > 0.001].head(10)
    if len(negative_impact) > 0:
        for _, row in negative_impact.iterrows():
            print(f"  {row['feature']:30s} : Removing improves AUC by +{row['auc_difference']:.4f}")
    else:
        print("  None found")
    
    print("\nTop features that HELP performance (keep these):")
    positive_impact = ablation_results[ablation_results['auc_difference'] < -0.001].head(10)
    if len(positive_impact) > 0:
        for _, row in positive_impact.iterrows():
            print(f"  {row['feature']:30s} : Removing decreases AUC by {row['auc_difference']:.4f}")
    else:
        print("  None found")
    
    return ablation_results


if __name__ == "__main__":
    main()

