import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import json
from pathlib import Path
from tqdm import tqdm
import warnings
import re
warnings.filterwarnings('ignore')


# LightGBM parameters for feature selection (fast gbdt mode)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',  # Fast for experiments
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 50,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}


def sanitize_feature_names(feature_names):
    """Sanitize feature names to remove special JSON characters that LightGBM doesn't support.
    
    LightGBM doesn't support special JSON characters like [, ], {, }, ", ', etc. in feature names.
    This function replaces them with underscores.
    
    Args:
        feature_names: List of feature names or pandas Index
    
    Returns:
        sanitized_names: List of sanitized feature names
        name_mapping: Dict mapping sanitized -> original names
    """
    if isinstance(feature_names, pd.Index):
        feature_names = feature_names.tolist()
    
    # Characters that cause issues in LightGBM JSON serialization
    # Replace with underscore
    sanitized_names = []
    name_mapping = {}
    
    for name in feature_names:
        # Replace special JSON characters with underscore
        sanitized = re.sub(r'[\[\]{}"\'\\\/\s]', '_', str(name))
        # Replace multiple consecutive underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it's not empty
        if not sanitized:
            sanitized = f'feature_{len(sanitized_names)}'
        
        # Handle duplicates by appending index
        original_sanitized = sanitized
        counter = 0
        while sanitized in name_mapping.values():
            counter += 1
            sanitized = f"{original_sanitized}_{counter}"
        
        sanitized_names.append(sanitized)
        name_mapping[sanitized] = name
    
    return sanitized_names, name_mapping


def sanitize_dataframe_columns(df):
    """Sanitize DataFrame column names for LightGBM compatibility.
    
    Args:
        df: DataFrame with potentially problematic column names
    
    Returns:
        sanitized_df: DataFrame with sanitized column names
        column_mapping: Dict mapping sanitized -> original column names
    """
    sanitized_cols, column_mapping = sanitize_feature_names(df.columns)
    # Create a completely new DataFrame to avoid any internal references
    sanitized_df = pd.DataFrame(df.values.copy(), columns=sanitized_cols, index=df.index)
    return sanitized_df, column_mapping


def train_baseline_lgbm(X_train, y_train, params=None, n_estimators=500, cv_folds=5):
    """Train baseline LightGBM model for feature importance extraction using CV.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target
        params: LightGBM parameters (defaults to LGB_PARAMS)
        n_estimators: Number of boosting rounds
        cv_folds: Number of CV folds for evaluation
    
    Returns:
        Trained model (trained on full data) and CV metrics
    """
    if params is None:
        params = LGB_PARAMS.copy()
    
    # Sanitize column names for LightGBM compatibility
    X_train_sanitized, train_col_mapping = sanitize_dataframe_columns(X_train)
    
    # Perform cross-validation for metrics
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_aucs = []
    cv_loglosses = []
    
    for train_idx, val_idx in skf.split(X_train_sanitized, y_train):
        X_train_fold = X_train_sanitized.iloc[train_idx]
        X_val_fold = X_train_sanitized.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        model_fold = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        y_pred = model_fold.predict(X_val_fold)
        cv_aucs.append(roc_auc_score(y_val_fold, y_pred))
        cv_loglosses.append(log_loss(y_val_fold, y_pred))
    
    # Train final model on full training set
    train_data = lgb.Dataset(X_train_sanitized, label=y_train)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # Store column mapping in model for later use
    model._column_mapping = train_col_mapping
    
    avg_auc = np.mean(cv_aucs)
    avg_logloss = np.mean(cv_loglosses)
    
    return model, {'auc': avg_auc, 'log_loss': avg_logloss}


def zero_importance_filter(X_train, y_train, feature_names, 
                          n_estimators=500, importance_type='gain'):
    """Remove features with exactly zero importance.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target
        feature_names: List of feature names
        n_estimators: Number of boosting rounds
        importance_type: 'gain' or 'split'
    
    Returns:
        selected_features: List of feature names with non-zero importance
        importances: Dict of {feature: importance}
        metrics: Dict with model metrics
    """
    print(f"\n{'='*80}")
    print("STAGE 1: Zero Importance Filtering")
    print(f"{'='*80}")
    print(f"Starting with {len(feature_names)} features")
    
    # Train model
    print("Training baseline LightGBM model...")
    model, metrics = train_baseline_lgbm(X_train, y_train, n_estimators=n_estimators)
    
    print(f"Baseline AUC: {metrics['auc']:.6f}, Log Loss: {metrics['log_loss']:.6f}")
    
    # Get feature importances
    importances = model.feature_importance(importance_type=importance_type)
    
    # Map sanitized feature names back to original names
    if hasattr(model, '_column_mapping'):
        sanitized_feature_names = list(model._column_mapping.keys())
        # Create mapping from original feature names to importances
        feature_importance_dict = {}
        for orig_name in feature_names:
            # Find corresponding sanitized name
            sanitized_name = None
            for sanitized, orig in model._column_mapping.items():
                if orig == orig_name:
                    sanitized_name = sanitized
                    break
            
            if sanitized_name and sanitized_name in sanitized_feature_names:
                idx = sanitized_feature_names.index(sanitized_name)
                feature_importance_dict[orig_name] = importances[idx]
            else:
                feature_importance_dict[orig_name] = 0.0
    else:
        # Fallback: direct mapping (shouldn't happen if sanitization worked)
        feature_importance_dict = dict(zip(feature_names, importances))
    
    # Filter zero importance features using mapped importances
    selected_features = [f for f in feature_names if feature_importance_dict.get(f, 0) > 0]
    zero_features = [f for f in feature_names if feature_importance_dict.get(f, 0) == 0]
    
    print(f"\nRemoved {len(zero_features)} features with zero importance")
    print(f"Remaining features: {len(selected_features)} ({len(selected_features)/len(feature_names)*100:.1f}%)")
    
    return selected_features, feature_importance_dict, metrics


def permutation_importance_filter(X_train, y_train, feature_names,
                                  n_repeats=5, threshold=0.0, n_estimators=300,
                                  cv_folds=5, random_seeds=[42, 206]):
    """Remove features with negative permutation importance (averaged over seeds).
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target
        feature_names: List of feature names
        n_repeats: Number of permutation repeats per feature
        threshold: Minimum importance threshold (default 0.0 = remove negative)
        n_estimators: Number of boosting rounds
        cv_folds: Number of CV folds for robust estimates
        random_seeds: List of seeds for averaging
    
    Returns:
        selected_features: List of selected feature names
        importance_scores: Dict of {feature: mean_importance}
        metrics: Dict with model metrics
    """
    print(f"\n{'='*80}")
    print("STAGE 2: Permutation Importance Filtering")
    print(f"{'='*80}")
    print(f"Starting with {len(feature_names)} features")
    print(f"Using {cv_folds}-fold CV with {len(random_seeds)} seeds")
    
    # Baseline score
    print("\nComputing baseline score...")
    baseline_model, baseline_metrics = train_baseline_lgbm(
        X_train, y_train, n_estimators=n_estimators
    )
    baseline_score = baseline_metrics['auc']
    print(f"Baseline AUC: {baseline_score:.6f}")
    
    # Compute permutation importance with CV
    print(f"\nComputing permutation importance ({n_repeats} repeats per feature)...")
    feature_importance_scores = {f: [] for f in feature_names}
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for seed in random_seeds:
        print(f"  Seed {seed}...")
        params = LGB_PARAMS.copy()
        params['seed'] = seed
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # Sanitize column names for LightGBM
            X_train_fold_sanitized, _ = sanitize_dataframe_columns(X_train_fold)
            X_val_fold_sanitized, _ = sanitize_dataframe_columns(X_val_fold)
            X_val_fold_sanitized = X_val_fold_sanitized[X_train_fold_sanitized.columns]
            
            # Train baseline for this fold
            train_data = lgb.Dataset(X_train_fold_sanitized, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold_sanitized, label=y_val_fold, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=n_estimators,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
            )
            baseline_pred = model.predict(X_val_fold_sanitized)
            baseline_auc = roc_auc_score(y_val_fold, baseline_pred)
            
            # Compute mapping once before loop
            _, col_mapping = sanitize_feature_names(X_train_fold.columns)
            reverse_mapping = {v: k for k, v in col_mapping.items()}  # original -> sanitized
            
            # Permute each feature
            for feat_idx, feat_name in enumerate(tqdm(feature_names, desc=f"    Fold {fold_idx+1}", leave=False)):
                # Find corresponding sanitized column name
                sanitized_feat_name = reverse_mapping.get(feat_name)
                
                if sanitized_feat_name is None or sanitized_feat_name not in X_val_fold_sanitized.columns:
                    continue
                
                X_val_permuted = X_val_fold_sanitized.copy()
                X_val_permuted[sanitized_feat_name] = np.random.permutation(X_val_permuted[sanitized_feat_name].values)
                
                permuted_pred = model.predict(X_val_permuted)
                permuted_auc = roc_auc_score(y_val_fold, permuted_pred)
                
                # Importance = baseline - permuted (positive = important)
                importance = baseline_auc - permuted_auc
                feature_importance_scores[feat_name].append(importance)
    
    # Average importance across all repeats and seeds
    mean_importances = {f: np.mean(scores) for f, scores in feature_importance_scores.items()}
    
    # Filter features
    selected_features = [f for f, imp in mean_importances.items() if imp > threshold]
    removed_features = [f for f, imp in mean_importances.items() if imp <= threshold]
    
    print(f"\nRemoved {len(removed_features)} features with importance <= {threshold}")
    print(f"Remaining features: {len(selected_features)} ({len(selected_features)/len(feature_names)*100:.1f}%)")
    
    # Retrain with selected features
    if len(selected_features) < len(feature_names):
        print("\nRetraining with selected features...")
        X_train_selected = X_train[selected_features]
        final_model, final_metrics = train_baseline_lgbm(
            X_train_selected, y_train, n_estimators=n_estimators
        )
        print(f"Final AUC: {final_metrics['auc']:.6f}, Log Loss: {final_metrics['log_loss']:.6f}")
    else:
        final_metrics = baseline_metrics
    
    return selected_features, mean_importances, final_metrics


def forward_feature_selection(X_train, y_train, feature_groups,
                              threshold=0.0, n_estimators=300):
    """Forward feature selection by feature groups.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target
        feature_groups: Dict of {group_name: [feature_names]}
        threshold: Minimum AUC improvement to keep group (default 0.0003)
        n_estimators: Number of boosting rounds
    
    Returns:
        selected_features: List of selected feature names
        selection_history: List of selection steps with metrics
    """
    print(f"\n{'='*80}")
    print("STAGE 3: Forward Feature Selection")
    print(f"{'='*80}")
    
    selected_features = []
    selection_history = []
    
    # Start with first group (usually original features)
    group_names = list(feature_groups.keys())
    if len(group_names) == 0:
        return selected_features, selection_history
    
    # Baseline with first group
    first_group = group_names[0]
    selected_features = feature_groups[first_group].copy()
    print(f"\nStarting with group '{first_group}': {len(selected_features)} features")
    
    X_train_current = X_train[selected_features]
    baseline_model, baseline_metrics = train_baseline_lgbm(
        X_train_current, y_train, n_estimators=n_estimators
    )
    baseline_auc = baseline_metrics['auc']
    print(f"Baseline AUC: {baseline_auc:.6f}")
    
    selection_history.append({
        'group': first_group,
        'features_added': len(selected_features),
        'total_features': len(selected_features),
        'auc': baseline_auc,
        'log_loss': baseline_metrics['log_loss']
    })
    
    # Add remaining groups incrementally
    for group_name in group_names[1:]:
        group_features = feature_groups[group_name]
        # Only consider features that exist in X_train
        group_features = [f for f in group_features if f in X_train.columns]
        
        if len(group_features) == 0:
            continue
        
        print(f"\nTesting group '{group_name}': {len(group_features)} features")
        
        # Try adding this group
        candidate_features = selected_features + group_features
        X_train_candidate = X_train[candidate_features]
        
        candidate_model, candidate_metrics = train_baseline_lgbm(
            X_train_candidate, y_train, n_estimators=n_estimators
        )
        candidate_auc = candidate_metrics['auc']
        improvement = candidate_auc - baseline_auc
        
        print(f"  AUC: {candidate_auc:.6f} (improvement: {improvement:+.6f})")
        
        if improvement > threshold:
            selected_features = candidate_features
            baseline_auc = candidate_auc
            baseline_metrics = candidate_metrics
            print(f"  ✓ Kept group '{group_name}'")
            selection_history.append({
                'group': group_name,
                'features_added': len(group_features),
                'total_features': len(selected_features),
                'auc': candidate_auc,
                'log_loss': candidate_metrics['log_loss'],
                'improvement': improvement
            })
        else:
            print(f"  ✗ Rejected group '{group_name}' (improvement < {threshold})")
            selection_history.append({
                'group': group_name,
                'features_added': 0,
                'total_features': len(selected_features),
                'auc': baseline_auc,
                'improvement': improvement,
                'status': 'rejected'
            })
    
    print(f"\nFinal feature count: {len(selected_features)}")
    return selected_features, selection_history


# Commented out - stepped_permutation_selection
# def stepped_permutation_selection(X_train, y_train, feature_names,
#                                   drop_pct=0.2, max_iterations=10, n_estimators=300, cv_folds=3):
#     """Iteratively drop worst performing features based on permutation importance using CV.
#     
#     Args:
#         X_train: Training features DataFrame
#         y_train: Training target
#         feature_names: List of feature names
#         drop_pct: Percentage of worst features to drop each iteration (default 0.2 = 20%)
#         max_iterations: Maximum number of iterations
#         n_estimators: Number of boosting rounds
#         cv_folds: Number of CV folds for evaluation
#     
#     Returns:
#         selected_features: List of selected feature names
#         iteration_history: List of metrics for each iteration
#     """
#     print(f"\n{'='*80}")
#     print("STAGE 2B: Stepped Permutation Selection")
#     print(f"{'='*80}")
#     print(f"Starting with {len(feature_names)} features")
#     print(f"Dropping {drop_pct*100:.0f}% worst features per iteration")
#     
#     current_features = feature_names.copy()
#     iteration_history = []
#     skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
#     
#     for iteration in range(max_iterations):
#         print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
#         print(f"Current features: {len(current_features)}")
#         
#         # Train baseline and compute CV metrics
#         X_train_current = X_train[current_features]
#         baseline_model, baseline_metrics = train_baseline_lgbm(
#             X_train_current, y_train, n_estimators=n_estimators, cv_folds=cv_folds
#         )
#         baseline_auc = baseline_metrics['auc']
#         print(f"Baseline CV AUC: {baseline_auc:.6f}")
#         
#         # Quick permutation importance using CV
#         print("Computing permutation importance...")
#         importances = {f: [] for f in current_features}
#         
#         for train_idx, val_idx in skf.split(X_train_current, y_train):
#             X_train_fold = X_train_current.iloc[train_idx]
#             X_val_fold = X_train_current.iloc[val_idx]
#             y_train_fold = y_train.iloc[train_idx]
#             y_val_fold = y_train.iloc[val_idx]
#             
#             # Train model for this fold
#             X_train_fold_sanitized, _ = sanitize_dataframe_columns(X_train_fold)
#             X_val_fold_sanitized, _ = sanitize_dataframe_columns(X_val_fold)
#             X_val_fold_sanitized = X_val_fold_sanitized[X_train_fold_sanitized.columns]
#             
#             train_data = lgb.Dataset(X_train_fold_sanitized, label=y_train_fold)
#             val_data = lgb.Dataset(X_val_fold_sanitized, label=y_val_fold, reference=train_data)
#             model_fold = lgb.train(
#                 LGB_PARAMS,
#                 train_data,
#                 num_boost_round=n_estimators,
#                 valid_sets=[val_data],
#                 callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
#             )
#             
#             baseline_pred = model_fold.predict(X_val_fold_sanitized)
#             baseline_auc_fold = roc_auc_score(y_val_fold, baseline_pred)
#             
#             # Get column mapping
#             _, col_mapping = sanitize_feature_names(X_train_fold.columns)
#             reverse_mapping = {v: k for k, v in col_mapping.items()}
#             
#             for feat_name in current_features:
#                 sanitized_name = reverse_mapping.get(feat_name)
#                 if sanitized_name is None or sanitized_name not in X_val_fold_sanitized.columns:
#                     continue
#                 
#                 X_val_permuted = X_val_fold_sanitized.copy()
#                 X_val_permuted[sanitized_name] = np.random.permutation(X_val_permuted[sanitized_name].values)
#                 
#                 permuted_pred = model_fold.predict(X_val_permuted)
#                 permuted_auc = roc_auc_score(y_val_fold, permuted_pred)
#                 importances[feat_name].append(baseline_auc_fold - permuted_auc)
#         
#         # Average importances across folds
#         mean_importances = {f: np.mean(scores) if scores else 0.0 for f, scores in importances.items()}
#         
#         # Sort by importance
#         sorted_features = sorted(mean_importances.items(), key=lambda x: x[1])
#         n_to_drop = max(1, int(len(current_features) * drop_pct))
#         features_to_drop = [f for f, _ in sorted_features[:n_to_drop]]
#         
#         # Only drop features with negative importance
#         negative_features = [f for f, imp in sorted_features if imp < 0]
#         if len(negative_features) == 0:
#             print("No negative importance features found. Stopping.")
#             break
#         
#         n_to_drop = min(n_to_drop, len(negative_features))
#         features_to_drop = negative_features[:n_to_drop]
#         
#         current_features = [f for f in current_features if f not in features_to_drop]
#         
#         print(f"Dropped {len(features_to_drop)} features (all negative importance)")
#         print(f"Remaining: {len(current_features)} features")
#         
#         iteration_history.append({
#             'iteration': iteration + 1,
#             'features': len(current_features),
#             'auc': baseline_auc,
#             'log_loss': baseline_metrics['log_loss'],
#             'dropped': len(features_to_drop)
#         })
#         
#         if len(current_features) == 0:
#             print("No features remaining. Stopping.")
#             break
#     
#     print(f"\nFinal feature count: {len(current_features)}")
#     return current_features, iteration_history

