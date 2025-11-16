"""Orchestrate full pipeline and generate final submission predictions."""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.ensemble.ensemble import load_all_predictions, compute_weights, create_weighted_ensemble
from src.training.model_training import evaluate_model


def generate_submission_file(ensemble_test_preds, output_path):
    """Generate submission file with final predictions.
    
    Args:
        ensemble_test_preds: Final ensemble test predictions (probabilities)
        output_path: Path to save submission file
    """
    # Load customer IDs for test set
    customer_ids_file = Path('data/processed/v2/customer_ids_test.csv')
    if customer_ids_file.exists():
        customer_ids = pd.read_csv(customer_ids_file)['customer_id'].values
    else:
        # Fallback: use index
        customer_ids = np.arange(len(ensemble_test_preds))
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'customer_id': customer_ids,
        'default_probability': ensemble_test_preds
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Prediction range: [{submission['default_probability'].min():.6f}, {submission['default_probability'].max():.6f}]")


def compare_all_models():
    """Compare performance of all individual models and ensemble."""
    from src.ensemble.ensemble import load_all_predictions
    
    print("=" * 80)
    print("MODEL COMPARISON REPORT")
    print("=" * 80)
    
    # Load data
    data_dir = Path('data/processed/v2')
    y_train = pd.read_csv(data_dir / 'y_train.csv')['default']
    y_test = pd.read_csv(data_dir / 'y_test.csv')['default']
    
    # Load all predictions
    all_cv_preds, all_test_preds, all_cv_metrics = load_all_predictions()
    
    print("\nIndividual Model Performance:")
    print("-" * 80)
    print(f"{'Model':<35s} {'CV AUC':<12s} {'Test AUC':<12s}")
    print("-" * 80)
    
    model_results = []
    for model_key in sorted(all_cv_preds.keys()):
        cv_preds = all_cv_preds[model_key]
        test_preds = all_test_preds[model_key]
        
        cv_pred = (cv_preds >= 0.5).astype(int)
        test_pred = (test_preds >= 0.5).astype(int)
        
        cv_metrics = evaluate_model(y_train.values, cv_pred, cv_preds)
        test_metrics = evaluate_model(y_test.values, test_pred, test_preds)
        
        model_results.append({
            'model': model_key,
            'cv_auc': cv_metrics['auc_roc'],
            'test_auc': test_metrics['auc_roc']
        })
        
        print(f"{model_key:<35s} {cv_metrics['auc_roc']:>11.6f} {test_metrics['auc_roc']:>11.6f}")
    
    # Ensemble performance
    weights = compute_weights(all_cv_metrics, method='performance')
    ensemble_cv, ensemble_test = create_weighted_ensemble(
        all_cv_preds, all_test_preds, weights
    )
    
    ensemble_cv_pred = (ensemble_cv >= 0.5).astype(int)
    ensemble_test_pred = (ensemble_test >= 0.5).astype(int)
    
    ensemble_cv_metrics = evaluate_model(y_train.values, ensemble_cv_pred, ensemble_cv)
    ensemble_test_metrics = evaluate_model(y_test.values, ensemble_test_pred, ensemble_test)
    
    print("-" * 80)
    print(f"{'ENSEMBLE (Weighted)':<35s} {ensemble_cv_metrics['auc_roc']:>11.6f} {ensemble_test_metrics['auc_roc']:>11.6f}")
    print("=" * 80)
    
    # Find best individual model
    best_model = max(model_results, key=lambda x: x['test_auc'])
    print(f"\nBest Individual Model: {best_model['model']} (Test AUC: {best_model['test_auc']:.6f})")
    print(f"Ensemble Improvement: {ensemble_test_metrics['auc_roc'] - best_model['test_auc']:.6f}")
    
    return model_results, ensemble_cv_metrics, ensemble_test_metrics


def main():
    """Main pipeline to generate final predictions."""
    print("=" * 80)
    print("FINAL PREDICTIONS GENERATION")
    print("=" * 80)
    
    # Compare all models
    model_results, ensemble_cv_metrics, ensemble_test_metrics = compare_all_models()
    
    # Load ensemble predictions
    print("\n" + "=" * 80)
    print("GENERATING FINAL ENSEMBLE PREDICTIONS")
    print("=" * 80)
    
    all_cv_preds, all_test_preds, all_cv_metrics = load_all_predictions()
    weights = compute_weights(all_cv_metrics, method='performance')
    ensemble_cv, ensemble_test = create_weighted_ensemble(
        all_cv_preds, all_test_preds, weights
    )
    
    # Generate submission file
    output_dir = Path('data/final_submission')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    submission_file = output_dir / 'submission.csv'
    generate_submission_file(ensemble_test, submission_file)
    
    # Save comprehensive report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FINAL PREDICTIONS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Ensemble Performance:")
    report_lines.append(f"  CV AUC-ROC: {ensemble_cv_metrics['auc_roc']:.6f}")
    report_lines.append(f"  CV Log Loss: {ensemble_cv_metrics['log_loss']:.6f}")
    report_lines.append(f"  CV Accuracy: {ensemble_cv_metrics['accuracy']:.6f}")
    report_lines.append(f"  Test AUC-ROC: {ensemble_test_metrics['auc_roc']:.6f}")
    report_lines.append(f"  Test Log Loss: {ensemble_test_metrics['log_loss']:.6f}")
    report_lines.append(f"  Test Accuracy: {ensemble_test_metrics['accuracy']:.6f}")
    report_lines.append("")
    report_lines.append("Individual Model Performance:")
    for result in sorted(model_results, key=lambda x: x['test_auc'], reverse=True):
        report_lines.append(f"  {result['model']:30s}: CV AUC={result['cv_auc']:.6f}, Test AUC={result['test_auc']:.6f}")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    with open(output_dir / 'final_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nFinal submission saved to: {submission_file}")
    print(f"Full report saved to: {output_dir}/final_report.txt")


if __name__ == '__main__':
    main()

