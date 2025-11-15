"""Model comparison framework to test different improvements"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
import json
from datetime import datetime
import os

from .prep import prepare_catboost_data
from .tuning import tune_catboost_cv
from .training import train_final_catboost
from .gpu_utils import get_task_type


def evaluate_model_cv(model, X, y, cat_feature_indices, n_splits=5, random_state=42):
    """Evaluate model using StratifiedKFold CV"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    
    for train_idx, valid_idx in skf.split(X, y):
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]
        
        train_pool = Pool(X_fold_train, y_fold_train, cat_features=cat_feature_indices)
        valid_pool = Pool(X_fold_valid, y_fold_valid, cat_features=cat_feature_indices)
        
        # Train on fold
        fold_model = CatBoostClassifier(**model.get_params())
        fold_model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=200, use_best_model=True, verbose=False)
        
        pred = fold_model.predict_proba(valid_pool)[:, 1]
        fold_auc = roc_auc_score(y_fold_valid, pred)
        fold_scores.append(fold_auc)
    
    return {
        'mean_auc': np.mean(fold_scores),
        'std_auc': np.std(fold_scores),
        'fold_scores': fold_scores
    }


def train_baseline_model(X_train, y_train, cat_feature_indices, random_state=42, use_gpu=True):
    """Train baseline model with original narrow search space"""
    # Get task type (GPU or CPU)
    task_type = get_task_type(use_gpu=use_gpu)
    # Use Logloss for GPU (AUC not supported), AUC for CPU
    eval_metric = "Logloss" if task_type == "GPU" else "AUC"
    
    # Baseline parameters (original narrow search space - conservative)
    baseline_params = {
        "loss_function": "Logloss",
        "eval_metric": eval_metric,
        "iterations": 5000,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "border_count": 128,
        "auto_class_weights": None,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
        "one_hot_max_size": 10,
        "task_type": task_type,
        "verbose": False,
        "random_seed": random_state,
    }
    
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    model = CatBoostClassifier(**baseline_params)
    model.fit(train_pool, verbose=False)
    
    return model, baseline_params


def train_improved_model(X_train, y_train, cat_feature_indices, n_trials=50, n_splits=5, random_state=42, use_gpu=True):
    """Train improved model with expanded search space"""
    best_params, best_cv_auc = tune_catboost_cv(
        X_train, y_train, cat_feature_indices,
        n_splits=n_splits, n_trials=n_trials, random_state=random_state, use_gpu=use_gpu
    )
    
    model = train_final_catboost(X_train, y_train, cat_feature_indices, best_params)
    
    return model, best_params, best_cv_auc


def compare_models(X_train, y_train, X_test, y_test, cat_feature_indices, 
                   n_trials=50, n_splits=5, random_state=42, output_dir="datas", use_gpu=True):
    """Compare baseline vs improved models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Baseline vs Improved")
    print("="*80)
    
    results = {
        'baseline': {},
        'improved': {},
        'improvement': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Train and evaluate baseline model
    print("\n[1/3] Training baseline model...")
    baseline_model, baseline_params = train_baseline_model(
        X_train, y_train, cat_feature_indices, random_state, use_gpu=use_gpu
    )
    
    print("   Evaluating baseline with CV...")
    baseline_cv = evaluate_model_cv(
        baseline_model, X_train, y_train, cat_feature_indices, n_splits, random_state
    )
    
    test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)
    baseline_test_pred = baseline_model.predict_proba(test_pool)[:, 1]
    baseline_test_auc = roc_auc_score(y_test, baseline_test_pred)
    baseline_test_pr = average_precision_score(y_test, baseline_test_pred)
    
    results['baseline'] = {
        'cv_auc_mean': baseline_cv['mean_auc'],
        'cv_auc_std': baseline_cv['std_auc'],
        'test_auc': baseline_test_auc,
        'test_pr_auc': baseline_test_pr,
        'params': baseline_params
    }
    
    print(f"   Baseline CV AUC: {baseline_cv['mean_auc']:.5f} (+/- {baseline_cv['std_auc']:.5f})")
    print(f"   Baseline Test AUC: {baseline_test_auc:.5f}")
    
    # 2. Train and evaluate improved model
    print("\n[2/3] Training improved model with expanded search space...")
    improved_model, improved_params, improved_cv_auc = train_improved_model(
        X_train, y_train, cat_feature_indices, n_trials, n_splits, random_state, use_gpu=use_gpu
    )
    
    print("   Evaluating improved with CV...")
    improved_cv = evaluate_model_cv(
        improved_model, X_train, y_train, cat_feature_indices, n_splits, random_state
    )
    
    improved_test_pred = improved_model.predict_proba(test_pool)[:, 1]
    improved_test_auc = roc_auc_score(y_test, improved_test_pred)
    improved_test_pr = average_precision_score(y_test, improved_test_pred)
    
    results['improved'] = {
        'cv_auc_mean': improved_cv['mean_auc'],
        'cv_auc_std': improved_cv['std_auc'],
        'tuning_cv_auc': improved_cv_auc,
        'test_auc': improved_test_auc,
        'test_pr_auc': improved_test_pr,
        'params': improved_params
    }
    
    print(f"   Improved CV AUC: {improved_cv['mean_auc']:.5f} (+/- {improved_cv['std_auc']:.5f})")
    print(f"   Improved Test AUC: {improved_test_auc:.5f}")
    
    # 3. Calculate improvements
    cv_improvement = improved_cv['mean_auc'] - baseline_cv['mean_auc']
    test_improvement = improved_test_auc - baseline_test_auc
    pr_improvement = improved_test_pr - baseline_test_pr
    
    results['improvement'] = {
        'cv_auc_delta': cv_improvement,
        'cv_auc_pct': (cv_improvement / baseline_cv['mean_auc']) * 100,
        'test_auc_delta': test_improvement,
        'test_auc_pct': (test_improvement / baseline_test_auc) * 100,
        'test_pr_delta': pr_improvement,
        'test_pr_pct': (pr_improvement / baseline_test_pr) * 100
    }
    
    # 4. Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nBaseline Model:")
    print(f"  CV AUC:     {baseline_cv['mean_auc']:.5f} (+/- {baseline_cv['std_auc']:.5f})")
    print(f"  Test AUC:   {baseline_test_auc:.5f}")
    print(f"  Test PR-AUC: {baseline_test_pr:.5f}")
    
    print(f"\nImproved Model:")
    print(f"  CV AUC:     {improved_cv['mean_auc']:.5f} (+/- {improved_cv['std_auc']:.5f})")
    print(f"  Test AUC:   {improved_test_auc:.5f}")
    print(f"  Test PR-AUC: {improved_test_pr:.5f}")
    
    print(f"\nImprovements:")
    print(f"  CV AUC:     +{cv_improvement:.5f} ({results['improvement']['cv_auc_pct']:+.2f}%)")
    print(f"  Test AUC:   +{test_improvement:.5f} ({results['improvement']['test_auc_pct']:+.2f}%)")
    print(f"  Test PR-AUC: +{pr_improvement:.5f} ({results['improvement']['test_pr_pct']:+.2f}%)")
    
    if cv_improvement > 0:
        print(f"\n✅ IMPROVEMENT: CV AUC increased by {cv_improvement:.5f}")
    elif cv_improvement < 0:
        print(f"\n❌ REGRESSION: CV AUC decreased by {abs(cv_improvement):.5f}")
    else:
        print(f"\n➡️  NO CHANGE: CV AUC unchanged")
    
    # 5. Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_file}")
    
    return results, baseline_model, improved_model

