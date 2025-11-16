"""Main script to train neural network models with Optuna hyperparameter optimization."""
import pandas as pd
import numpy as np
import torch
import pickle
import json
from pathlib import Path
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
import warnings
warnings.filterwarnings('ignore')

from src.models.nn_models import SimpleMLP, TabMModel
from src.training.nn_training import (
    train_nn_cv, train_nn_final, predict_nn, evaluate_model
)
from src.training.nn_optimization import (
    optimize_mlp, optimize_tabm
)


def load_data_and_features():
    """Load preprocessed data and selected features."""
    data_dir = Path('data/processed')
    features_dir = Path('data/feature_selection')
    
    print("Loading preprocessed data...")
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv')['default']
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv')['default']
    
    print(f"Loaded data:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Load selected features
    print("\nLoading selected features...")
    with open(features_dir / 'selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    print(f"Selected features: {len(selected_features)}")
    
    # Filter to selected features
    available_features = [f for f in selected_features if f in X_train.columns]
    missing_features = [f for f in selected_features if f not in X_train.columns]
    
    if missing_features:
        print(f"Warning: {len(missing_features)} selected features not found in data")
    
    X_train = X_train[available_features]
    X_test = X_test[available_features]
    
    print(f"Using {len(available_features)} available features")
    
    return X_train, y_train, X_test, y_test, available_features


def save_model_results(
    model_name: str,
    model: object,
    scaler: object,
    best_params: dict,
    cv_metrics: dict,
    test_metrics: dict,
    study: optuna.Study,
    output_dir: Path
):
    """Save model results including model, metrics, and visualizations."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if hasattr(model, 'state_dict'):
        torch.save(model.state_dict(), model_dir / 'model.pth')
    
    # Save scaler
    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save hyperparameters
    with open(model_dir / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2, default=str)
    
    # Save metrics
    metrics = {
        'cv_metrics': cv_metrics,
        'test_metrics': test_metrics
    }
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save Optuna study
    with open(model_dir / 'optuna_study.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    # Generate and save visualizations
    try:
        fig1 = plot_optimization_history(study)
        fig1.write_html(str(model_dir / 'optimization_history.html'))
        
        fig2 = plot_param_importances(study)
        fig2.write_html(str(model_dir / 'param_importances.html'))
        
        fig3 = plot_parallel_coordinate(study)
        fig3.write_html(str(model_dir / 'parallel_coordinate.html'))
    except Exception as e:
        print(f"Warning: Could not generate Optuna visualizations: {e}")


def save_predictions(
    model_name: str,
    y_train_pred: np.ndarray,
    y_train_pred_proba: np.ndarray,
    y_test_pred: np.ndarray,
    y_test_pred_proba: np.ndarray,
    output_dir: Path,
    oof_predictions: np.ndarray = None
):
    """Save predictions for train and test splits."""
    model_dir = output_dir / model_name
    
    # Train predictions
    pd.DataFrame({
        'pred': y_train_pred,
        'pred_proba': y_train_pred_proba
    }).to_csv(model_dir / 'train_predictions.csv', index=False)
    
    # Test predictions
    pd.DataFrame({
        'pred': y_test_pred,
        'pred_proba': y_test_pred_proba
    }).to_csv(model_dir / 'test_predictions.csv', index=False)
    
    # OOF predictions (if provided)
    if oof_predictions is not None:
        pd.DataFrame({
            'oof_pred_proba': oof_predictions
        }).to_csv(model_dir / 'oof_predictions.csv', index=False)


def generate_training_report(
    results: dict,
    output_dir: Path
) -> str:
    """Generate comprehensive training report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("NEURAL NETWORK MODEL TRAINING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for model_name in ['mlp', 'tabm']:
        if model_name not in results:
            continue
        
        r = results[model_name]
        report_lines.append(f"{model_name.upper()} RESULTS")
        report_lines.append("-" * 80)
        report_lines.append(f"Best CV AUC-ROC: {r['cv_metrics']['auc_roc']:.6f}")
        report_lines.append(f"Best CV Log Loss: {r['cv_metrics']['log_loss']:.6f}")
        report_lines.append("")
        report_lines.append("Test Metrics:")
        report_lines.append(f"  AUC-ROC: {r['test_metrics']['auc_roc']:.6f}")
        report_lines.append(f"  Log Loss: {r['test_metrics']['log_loss']:.6f}")
        report_lines.append(f"  Accuracy: {r['test_metrics']['accuracy']:.6f}")
        report_lines.append(f"  Precision: {r['test_metrics']['precision']:.6f}")
        report_lines.append(f"  Recall: {r['test_metrics']['recall']:.6f}")
        report_lines.append(f"  F1 Score: {r['test_metrics']['f1']:.6f}")
        report_lines.append("")
        report_lines.append("Best Hyperparameters:")
        for param, value in r['best_params'].items():
            if isinstance(value, dict):
                for sub_param, sub_value in value.items():
                    report_lines.append(f"  {param}.{sub_param}: {sub_value}")
            else:
                report_lines.append(f"  {param}: {value}")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open(output_dir / 'nn_training_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text


def load_existing_nn_results(output_dir: Path) -> dict:
    """Load existing NN model results if available."""
    results = {}
    
    for model_name in ['mlp', 'tabm']:
        model_dir = output_dir / model_name
        metrics_file = model_dir / 'metrics.json'
        
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Load best params
            params_file = model_dir / 'best_params.json'
            best_params = {}
            if params_file.exists():
                with open(params_file, 'r') as f:
                    best_params = json.load(f)
            
            results[model_name] = {
                'best_params': best_params,
                'cv_metrics': metrics.get('cv_metrics', {}),
                'test_metrics': metrics.get('test_metrics', {}),
                'completed': True
            }
            print(f"Found existing results for {model_name}")
    
    return results


def load_or_create_study(model_name, output_dir, n_trials, X_train, y_train, n_folds):
    """Load existing Optuna study or create new one."""
    import pickle
    from nn_optimization import optimize_mlp, optimize_tabm
    
    study_file = output_dir / model_name / 'optuna_study.pkl'
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if study exists
    if study_file.exists():
        print(f"Found existing Optuna study for {model_name}, loading...")
        with open(study_file, 'rb') as f:
            study = pickle.load(f)
        
        completed_trials = len([t for t in study.trials if t.state.name == 'COMPLETE'])
        print(f"  Existing study has {completed_trials} completed trials")
        
        if completed_trials >= n_trials:
            print(f"  Study already has {completed_trials} trials (target: {n_trials}), using best params")
            # Extract best params in the format expected by training code
            best_params_raw = study.best_params
            if model_name == 'mlp':
                best_params = {
                    'model_params': {
                        'input_dim': X_train.shape[1],
                        'hidden_dim': best_params_raw['hidden_dim'],
                        'num_layers': best_params_raw['num_layers'],
                        'dropout_rate': best_params_raw['dropout_rate'],
                        'use_residual': True
                    },
                    'training_params': {
                        'learning_rate': best_params_raw['learning_rate'],
                        'batch_size': best_params_raw['batch_size'],
                        'weight_decay': best_params_raw['weight_decay'],
                        'max_epochs': 200,
                        'early_stopping_patience': 20
                    }
                }
            elif model_name == 'tabm':
                best_params = {
                    'model_params': {
                        'input_dim': X_train.shape[1],
                        'hidden_dim': best_params_raw['hidden_dim'],
                        'num_layers': best_params_raw['num_layers'],
                        'multiplicative_units': best_params_raw['multiplicative_units'],
                        'dropout_rate': best_params_raw['dropout_rate']
                    },
                    'training_params': {
                        'learning_rate': best_params_raw['learning_rate'],
                        'batch_size': best_params_raw['batch_size'],
                        'weight_decay': best_params_raw['weight_decay'],
                        'max_epochs': 200,
                        'early_stopping_patience': 20
                    }
                }
            
            return study, best_params
        else:
            # Study exists but incomplete - continue optimization
            remaining_trials = n_trials - completed_trials
            print(f"  Resuming optimization with {remaining_trials} more trials...")
            print(f"  Note: This will create a new study. Previous {completed_trials} trials will be lost.")
            print(f"  To avoid this, manually merge studies or wait for completion.")
            
            # Create new study or continue with remaining trials
            print(f"Running Optuna optimization ({remaining_trials} trials, {n_folds}-fold CV)...")
            if model_name == 'mlp':
                study, best_params = optimize_mlp(
                    X_train, y_train,
                    n_trials=remaining_trials,
                    n_folds=n_folds,
                    verbose=True
                )
            elif model_name == 'tabm':
                study, best_params = optimize_tabm(
                    X_train, y_train,
                    n_trials=remaining_trials,
                    n_folds=n_folds,
                    verbose=True
                )
            
            return study, best_params
    
    # Create new study
    print(f"Running Optuna optimization ({n_trials} trials, {n_folds}-fold CV)...")
    if model_name == 'mlp':
        study, best_params = optimize_mlp(
            X_train, y_train,
            n_trials=n_trials,
            n_folds=n_folds,
            verbose=True
        )
    elif model_name == 'tabm':
        study, best_params = optimize_tabm(
            X_train, y_train,
            n_trials=n_trials,
            n_folds=n_folds,
            verbose=True
        )
    
    return study, best_params


def main():
    """Main neural network training pipeline."""
    print("=" * 80)
    print("NEURAL NETWORK MODEL TRAINING WITH OPTUNA OPTIMIZATION")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, y_train, X_test, y_test, selected_features = load_data_and_features()
    
    # Create output directory
    output_dir = Path('data/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected features used
    with open(output_dir / 'nn_selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    # Try to load existing results
    results = load_existing_nn_results(output_dir)
    
    n_trials = 5
    n_folds = 5
    
    # Train MLP
    if 'mlp' in results and results['mlp'].get('completed', False):
        print("\n" + "=" * 80)
        print("SKIPPING MLP (already completed)")
        print("=" * 80)
        study_mlp = None
        best_params_mlp = results['mlp']['best_params']
        cv_metrics_mlp = results['mlp']['cv_metrics']
        test_metrics_mlp = results['mlp']['test_metrics']
    else:
        print("\n" + "=" * 80)
        print("TRAINING MLP")
        print("=" * 80)
        
        study_mlp, best_params_mlp = load_or_create_study(
            'mlp', output_dir, n_trials, X_train, y_train, n_folds
        )
        
        if study_mlp:
            print(f"\nBest CV AUC-ROC: {study_mlp.best_value:.6f}")
        else:
            print(f"\nUsing best params from existing study")
        
        print("Best parameters:")
        for param, value in best_params_mlp.items():
            if isinstance(value, dict):
                for sub_param, sub_value in value.items():
                    print(f"  {param}.{sub_param}: {sub_value}")
            else:
                print(f"  {param}: {value}")
        
        print("\nTraining final model with best hyperparameters...")
        model_mlp, cv_metrics_mlp, scaler_mlp = train_nn_final(
            SimpleMLP,
            X_train, y_train,
            best_params_mlp['model_params'],
            best_params_mlp['training_params'],
            device=device
        )
        
        # Get predictions
        y_test_pred_mlp, y_test_pred_proba_mlp = predict_nn(
            model_mlp, X_test, scaler_mlp, device=device
        )
        
        # Evaluate on test set
        test_metrics_mlp = evaluate_model(y_test.values, y_test_pred_mlp, y_test_pred_proba_mlp)
        
        # Get OOF predictions
        print("\nComputing CV metrics with best hyperparameters...")
        _, _, oof_predictions_mlp = train_nn_cv(
            SimpleMLP,
            X_train, y_train,
            best_params_mlp['model_params'],
            best_params_mlp['training_params'],
            n_folds=n_folds,
            device=device
        )
        
        results['mlp'] = {
            'best_params': best_params_mlp,
            'cv_metrics': cv_metrics_mlp,
            'test_metrics': test_metrics_mlp,
            'study': study_mlp,
            'model': model_mlp,
            'scaler': scaler_mlp
        }
        
        # Save MLP results
        y_train_pred_mlp, y_train_pred_proba_mlp = predict_nn(
            model_mlp, X_train, scaler_mlp, device=device
        )
        
        save_model_results(
            'mlp', model_mlp, scaler_mlp, best_params_mlp,
            cv_metrics_mlp, test_metrics_mlp,
            study_mlp, output_dir
        )
        save_predictions(
            'mlp',
            y_train_pred_mlp, y_train_pred_proba_mlp,
            y_test_pred_mlp, y_test_pred_proba_mlp,
            output_dir,
            oof_predictions=oof_predictions_mlp
        )
    
    # Train TabM
    if 'tabm' in results and results['tabm'].get('completed', False):
        print("\n" + "=" * 80)
        print("SKIPPING TABM (already completed)")
        print("=" * 80)
        study_tabm = None
        best_params_tabm = results['tabm']['best_params']
        cv_metrics_tabm = results['tabm']['cv_metrics']
        test_metrics_tabm = results['tabm']['test_metrics']
    else:
        print("\n" + "=" * 80)
        print("TRAINING TABM")
        print("=" * 80)
        
        study_tabm, best_params_tabm = load_or_create_study(
            'tabm', output_dir, n_trials, X_train, y_train, n_folds
        )
        
        if study_tabm:
            print(f"\nBest CV AUC-ROC: {study_tabm.best_value:.6f}")
        else:
            print(f"\nUsing best params from existing study")
        
        print("Best parameters:")
        for param, value in best_params_tabm.items():
            if isinstance(value, dict):
                for sub_param, sub_value in value.items():
                    print(f"  {param}.{sub_param}: {sub_value}")
            else:
                print(f"  {param}: {value}")
        
        print("\nTraining final model with best hyperparameters...")
        model_tabm, cv_metrics_tabm, scaler_tabm = train_nn_final(
            TabMModel,
            X_train, y_train,
            best_params_tabm['model_params'],
            best_params_tabm['training_params'],
            device=device
        )
        
        # Get predictions
        y_test_pred_tabm, y_test_pred_proba_tabm = predict_nn(
            model_tabm, X_test, scaler_tabm, device=device
        )
        
        # Evaluate on test set
        test_metrics_tabm = evaluate_model(y_test.values, y_test_pred_tabm, y_test_pred_proba_tabm)
        
        # Get OOF predictions
        print("\nComputing CV metrics with best hyperparameters...")
        _, _, oof_predictions_tabm = train_nn_cv(
            TabMModel,
            X_train, y_train,
            best_params_tabm['model_params'],
            best_params_tabm['training_params'],
            n_folds=n_folds,
            device=device
        )
        
        results['tabm'] = {
            'best_params': best_params_tabm,
            'cv_metrics': cv_metrics_tabm,
            'test_metrics': test_metrics_tabm,
            'study': study_tabm,
            'model': model_tabm,
            'scaler': scaler_tabm
        }
        
        # Save TabM results
        y_train_pred_tabm, y_train_pred_proba_tabm = predict_nn(
            model_tabm, X_train, scaler_tabm, device=device
        )
        
        save_model_results(
            'tabm', model_tabm, scaler_tabm, best_params_tabm,
            cv_metrics_tabm, test_metrics_tabm,
            study_tabm, output_dir
        )
        save_predictions(
            'tabm',
            y_train_pred_tabm, y_train_pred_proba_tabm,
            y_test_pred_tabm, y_test_pred_proba_tabm,
            output_dir,
            oof_predictions=oof_predictions_tabm
        )
    
    # Generate and print report
    report = generate_training_report(results, output_dir)
    print("\n" + report)
    
    print("\n" + "=" * 80)
    print("NEURAL NETWORK TRAINING COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}/")
    print(f"  - Models: {output_dir}/[mlp|tabm]/model.pth")
    print(f"  - Hyperparameters: {output_dir}/[model]/best_params.json")
    print(f"  - Metrics: {output_dir}/[model]/metrics.json")
    print(f"  - Predictions: {output_dir}/[model]/[train|test]_predictions.csv")
    print(f"  - Visualizations: {output_dir}/[model]/optimization_history.html")


if __name__ == '__main__':
    main()

