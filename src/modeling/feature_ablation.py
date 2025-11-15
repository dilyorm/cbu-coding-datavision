"""Feature ablation analysis to test impact of each feature on model performance"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
from .gpu_utils import get_task_type
import warnings
warnings.filterwarnings('ignore')


def analyze_feature_impact(X_train, y_train, cat_feature_indices, 
                          n_splits=5, random_state=42, n_trials_per_feature=10, use_gpu=True):
    """Analyze impact of each feature on AUC by training models with/without each feature
    
    For each feature:
    1. Train model with all features (baseline)
    2. Train model without that feature
    3. Compare AUC scores
    
    Args:
        X_train: Training feature DataFrame
        y_train: Training target Series
        cat_feature_indices: List of categorical feature indices (for full model)
        n_splits: Number of CV folds
        random_state: Random state for reproducibility
        n_trials_per_feature: Number of models to train per feature (for stability)
        
    Returns:
        DataFrame with feature impact analysis
    """
    print("\n" + "="*60)
    print("FEATURE ABLATION ANALYSIS")
    print("="*60)
    print("Testing impact of each feature on AUC...")
    print(f"Using {n_splits}-fold CV for each test")
    
    # Setup CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Get task type and eval_metric
    task_type = get_task_type(use_gpu=use_gpu)
    eval_metric = "Logloss" if task_type == "GPU" else "AUC"
    
    # Calculate baseline AUC with all features
    print("\n" + "-"*60)
    print("Step 1: Calculating baseline AUC (all features)...")
    print("-"*60)
    baseline_scores = []
    for train_idx, valid_idx in skf.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]
        
        train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_feature_indices)
        valid_pool = Pool(X_fold_valid, y_fold_valid, cat_features=cat_feature_indices)
        
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            loss_function='Logloss',
            eval_metric=eval_metric,
            task_type=task_type,
            random_seed=random_state,
            verbose=False
        )
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)
        pred = model.predict_proba(valid_pool)[:, 1]
        auc = roc_auc_score(y_fold_valid, pred)
        baseline_scores.append(auc)
    
    baseline_auc = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)
    print(f"Baseline AUC: {baseline_auc:.4f} (+/- {baseline_std:.4f})")
    
    # Test each feature
    print("\n" + "-"*60)
    print("Step 2: Testing each feature (removing one at a time)...")
    print("-"*60)
    
    feature_names = X_train.columns.tolist()
    results = []
    
    for i, feature_name in enumerate(feature_names, 1):
        print(f"\n[{i}/{len(feature_names)}] Testing feature: {feature_name}")
        
        # Create dataset without this feature
        X_train_ablated = X_train.drop(columns=[feature_name])
        
        # Update categorical feature indices (adjust indices after removing a feature)
        original_feature_idx = feature_names.index(feature_name)
        cat_feature_indices_ablated = []
        for idx in cat_feature_indices:
            if idx < original_feature_idx:
                # Index before removed feature stays the same
                cat_feature_indices_ablated.append(idx)
            elif idx > original_feature_idx:
                # Index after removed feature needs to be decremented
                cat_feature_indices_ablated.append(idx - 1)
            # If idx == original_feature_idx, skip it (removed feature was categorical)
        
        # Train models without this feature
        ablated_scores = []
        for train_idx, valid_idx in skf.split(X_train_ablated, y_train):
            X_fold_train = X_train_ablated.iloc[train_idx]
            X_fold_valid = X_train_ablated.iloc[valid_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_valid = y_train.iloc[valid_idx]
            
            train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_feature_indices_ablated)
            valid_pool = Pool(X_fold_valid, y_fold_valid, cat_features=cat_feature_indices_ablated)
            
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                loss_function='Logloss',
                eval_metric=eval_metric,
                task_type=task_type,
                random_seed=random_state,
                verbose=False
            )
            model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)
            pred = model.predict_proba(valid_pool)[:, 1]
            auc = roc_auc_score(y_fold_valid, pred)
            ablated_scores.append(auc)
        
        ablated_auc = np.mean(ablated_scores)
        ablated_std = np.std(ablated_scores)
        auc_diff = ablated_auc - baseline_auc
        auc_diff_pct = (auc_diff / baseline_auc) * 100 if baseline_auc > 0 else 0
        
        results.append({
            'feature': feature_name,
            'baseline_auc': baseline_auc,
            'ablated_auc': ablated_auc,
            'auc_difference': auc_diff,
            'auc_difference_pct': auc_diff_pct,
            'impact': 'positive' if auc_diff < 0 else 'negative' if auc_diff > 0 else 'neutral',
            'is_categorical': original_feature_idx in cat_feature_indices
        })
        
        impact_symbol = "✓" if auc_diff < 0 else "✗" if auc_diff > 0.001 else "="
        print(f"  Without {feature_name}: AUC = {ablated_auc:.4f} ({auc_diff:+.4f}, {auc_diff_pct:+.2f}%) {impact_symbol}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('auc_difference', ascending=False)
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE IMPACT SUMMARY")
    print("="*60)
    
    negative_impact = results_df[results_df['auc_difference'] > 0.001]
    positive_impact = results_df[results_df['auc_difference'] < -0.001]
    neutral = results_df[(results_df['auc_difference'] >= -0.001) & (results_df['auc_difference'] <= 0.001)]
    
    print(f"\nFeatures that HURT performance (removing improves AUC):")
    if len(negative_impact) > 0:
        for _, row in negative_impact.head(10).iterrows():
            print(f"  {row['feature']:30s} : +{row['auc_difference']:.4f} ({row['auc_difference_pct']:+.2f}%)")
    else:
        print("  None found")
    
    print(f"\nFeatures that HELP performance (removing decreases AUC):")
    if len(positive_impact) > 0:
        for _, row in positive_impact.head(10).iterrows():
            print(f"  {row['feature']:30s} : {row['auc_difference']:.4f} ({row['auc_difference_pct']:+.2f}%)")
    else:
        print("  None found")
    
    print(f"\nNeutral features (minimal impact): {len(neutral)}")
    print(f"Total features tested: {len(results_df)}")
    
    return results_df


def run_feature_ablation_analysis(X_train, y_train, cat_feature_indices, 
                                  output_path='datas/feature_ablation_analysis.csv',
                                  n_splits=5, random_state=42, use_gpu=True):
    """Run feature ablation analysis and save results
    
    Args:
        X_train: Training feature DataFrame
        y_train: Training target Series
        cat_feature_indices: List of categorical feature indices
        output_path: Path to save results CSV
        n_splits: Number of CV folds
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with feature impact analysis
    """
    results_df = analyze_feature_impact(
        X_train, y_train, cat_feature_indices,
        n_splits=n_splits, random_state=random_state
    )
    
    # Save results
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results_df

