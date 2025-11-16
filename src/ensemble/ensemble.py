"""Weighted ensemble of all models based on validation performance."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


def load_all_predictions():
    """Load all predictions from base models and stacking models.
    
    Uses CV metrics (from OOF predictions) for weighting instead of validation metrics.
    """
    versions = ['v1', 'v2', 'v3']
    tree_models = ['lightgbm', 'xgboost', 'catboost']
    nn_models = ['mlp', 'tabm']
    stacking_models = ['lightgbm', 'xgboost']
    
    all_cv_preds = {}  # OOF predictions for weighting
    all_test_preds = {}
    all_cv_metrics = {}
    
    # Load base model predictions
    for version in versions:
        for model_name in tree_models + nn_models:
            model_key = f'{version}_{model_name}'
            oof_file = Path(f'data/models/{model_key}/oof_predictions.csv')
            test_file = Path(f'data/models/{model_key}/test_predictions.csv')
            metrics_file = Path(f'data/models/{model_key}/metrics.json')
            
            if oof_file.exists() and test_file.exists():
                oof_df = pd.read_csv(oof_file)
                test_df = pd.read_csv(test_file)
                all_cv_preds[model_key] = oof_df['oof_pred_proba'].values
                all_test_preds[model_key] = test_df['pred_proba'].values
                
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        all_cv_metrics[model_key] = metrics.get('cv_metrics', {}).get('auc_roc', 0.5)
                else:
                    all_cv_metrics[model_key] = 0.5
    
    # Load stacking model predictions
    for model_name in stacking_models:
        model_key = f'stacking_{model_name}'
        oof_file = Path(f'data/models/stacking/{model_name}/oof_predictions.csv')
        test_file = Path(f'data/models/stacking/{model_name}/test_predictions.csv')
        metrics_file = Path(f'data/models/stacking/{model_name}/metrics.json')
        
        if oof_file.exists() and test_file.exists():
            oof_df = pd.read_csv(oof_file)
            test_df = pd.read_csv(test_file)
            all_cv_preds[model_key] = oof_df['oof_pred_proba'].values
            all_test_preds[model_key] = test_df['pred_proba'].values
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    all_cv_metrics[model_key] = metrics.get('cv_metrics', {}).get('auc_roc', 0.5)
            else:
                all_cv_metrics[model_key] = 0.5
    
    return all_cv_preds, all_test_preds, all_cv_metrics


def compute_weights(cv_metrics, method='performance'):
    """Compute ensemble weights based on CV performance.
    
    Args:
        cv_metrics: Dict mapping model_key -> CV AUC-ROC
        method: 'performance' (weight by AUC-ROC) or 'equal' (equal weights)
    
    Returns:
        Dict mapping model_key -> weight
    """
    if method == 'equal':
        n_models = len(cv_metrics)
        return {k: 1.0 / n_models for k in cv_metrics.keys()}
    
    # Performance-based weighting: weight = (auc - 0.5) / sum(all auc - 0.5)
    # This gives more weight to better models and zero weight to models at random chance
    adjusted_scores = {k: max(0, v - 0.5) for k, v in cv_metrics.items()}
    total_score = sum(adjusted_scores.values())
    
    if total_score == 0:
        # Fallback to equal weights if all models are at random chance
        n_models = len(cv_metrics)
        return {k: 1.0 / n_models for k in cv_metrics.keys()}
    
    weights = {k: v / total_score for k, v in adjusted_scores.items()}
    return weights


def create_weighted_ensemble(cv_preds_dict, test_preds_dict, weights):
    """Create weighted ensemble predictions.
    
    Args:
        cv_preds_dict: Dict mapping model_key -> CV (OOF) predictions (for reference, not used in final ensemble)
        test_preds_dict: Dict mapping model_key -> test predictions
        weights: Dict mapping model_key -> weight
    
    Returns:
        Tuple of (ensemble_cv_preds, ensemble_test_preds)
    """
    # Get model keys that exist in both dicts
    common_keys = set(cv_preds_dict.keys()) & set(test_preds_dict.keys()) & set(weights.keys())
    
    if not common_keys:
        raise ValueError("No common models found for ensemble!")
    
    # Normalize weights to sum to 1 for available models
    available_weights = {k: weights[k] for k in common_keys}
    total_weight = sum(available_weights.values())
    normalized_weights = {k: v / total_weight for k, v in available_weights.items()}
    
    # Compute weighted average
    ensemble_cv = np.zeros(len(list(cv_preds_dict.values())[0]))
    ensemble_test = np.zeros(len(list(test_preds_dict.values())[0]))
    
    for model_key in common_keys:
        weight = normalized_weights[model_key]
        ensemble_cv += weight * cv_preds_dict[model_key]
        ensemble_test += weight * test_preds_dict[model_key]
    
    return ensemble_cv, ensemble_test


def main():
    """Create weighted ensemble of all models."""
    print("=" * 80)
    print("WEIGHTED ENSEMBLE CREATION")
    print("=" * 80)
    
    # Load all predictions
    print("\nLoading predictions from all models...")
    all_cv_preds, all_test_preds, all_cv_metrics = load_all_predictions()
    
    if not all_cv_preds:
        raise ValueError("No predictions found! Run train_all_feature_sets.py and train_stacking_models.py first.")
    
    print(f"Loaded predictions from {len(all_cv_preds)} models")
    
    # Display model performance
    print("\nModel CV Performance:")
    print("-" * 80)
    sorted_models = sorted(all_cv_metrics.items(), key=lambda x: x[1], reverse=True)
    for model_key, auc in sorted_models:
        print(f"  {model_key:30s}: {auc:.6f}")
    
    # Compute weights
    print("\nComputing ensemble weights...")
    weights = compute_weights(all_cv_metrics, method='performance')
    
    print("\nEnsemble Weights:")
    print("-" * 80)
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for model_key, weight in sorted_weights:
        print(f"  {model_key:30s}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Create weighted ensemble
    print("\nCreating weighted ensemble predictions...")
    ensemble_cv, ensemble_test = create_weighted_ensemble(
        all_cv_preds, all_test_preds, weights
    )
    
    # Load true labels for evaluation
    data_dir = Path('data/processed/v2')
    y_train = pd.read_csv(data_dir / 'y_train.csv')['default']
    y_test = pd.read_csv(data_dir / 'y_test.csv')['default']
    
    # Evaluate ensemble
    from src.training.model_training import evaluate_model
    
    ensemble_cv_pred = (ensemble_cv >= 0.5).astype(int)
    ensemble_test_pred = (ensemble_test >= 0.5).astype(int)
    
    cv_metrics = evaluate_model(y_train.values, ensemble_cv_pred, ensemble_cv)
    test_metrics = evaluate_model(y_test.values, ensemble_test_pred, ensemble_test)
    
    print("\nEnsemble Performance:")
    print("-" * 80)
    print(f"CV AUC-ROC: {cv_metrics['auc_roc']:.6f}")
    print(f"CV Log Loss: {cv_metrics['log_loss']:.6f}")
    print(f"CV Accuracy: {cv_metrics['accuracy']:.6f}")
    print(f"Test AUC-ROC: {test_metrics['auc_roc']:.6f}")
    print(f"Test Log Loss: {test_metrics['log_loss']:.6f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.6f}")
    
    # Save ensemble results
    output_dir = Path('data/final_submission')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving ensemble results to {output_dir}...")
    
    # Save predictions
    pd.DataFrame({
        'pred': ensemble_cv_pred,
        'pred_proba': ensemble_cv
    }).to_csv(output_dir / 'ensemble_cv_predictions.csv', index=False)
    
    pd.DataFrame({
        'pred': ensemble_test_pred,
        'pred_proba': ensemble_test
    }).to_csv(output_dir / 'ensemble_test_predictions.csv', index=False)
    
    # Save metrics
    ensemble_metrics = {
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics,
        'weights': weights,
        'model_performance': all_cv_metrics
    }
    with open(output_dir / 'ensemble_metrics.json', 'w') as f:
        json.dump(ensemble_metrics, f, indent=2, default=str)
    
    # Save summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ENSEMBLE SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Total Models in Ensemble: {len(weights)}")
    report_lines.append("")
    report_lines.append("Model Weights:")
    for model_key, weight in sorted_weights:
        report_lines.append(f"  {model_key:30s}: {weight:.4f} ({weight*100:.2f}%)")
    report_lines.append("")
    report_lines.append("Ensemble Performance:")
    report_lines.append(f"  CV AUC-ROC: {cv_metrics['auc_roc']:.6f}")
    report_lines.append(f"  CV Log Loss: {cv_metrics['log_loss']:.6f}")
    report_lines.append(f"  Test AUC-ROC: {test_metrics['auc_roc']:.6f}")
    report_lines.append(f"  Test Log Loss: {test_metrics['log_loss']:.6f}")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    with open(output_dir / 'ensemble_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nEnsemble results saved to: {output_dir}/")


if __name__ == '__main__':
    main()

