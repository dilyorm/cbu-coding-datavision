"""Main script to train models with Optuna hyperparameter optimization."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
import warnings
warnings.filterwarnings('ignore')

from src.training.model_training import (
    train_lightgbm_cv, train_xgboost_cv, train_catboost_cv,
    train_lightgbm_final, train_xgboost_final, train_catboost_final,
    evaluate_model
)
from src.training.optuna_optimization import (
    optimize_lightgbm, optimize_xgboost, optimize_catboost
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
    with open(model_dir / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
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
    report_lines.append("MODEL TRAINING REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for model_name in ['lightgbm', 'xgboost', 'catboost']:
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
            if param not in ['objective', 'metric', 'eval_metric', 'boosting_type', 
                           'tree_method', 'verbose', 'seed', 'random_seed', 
                           'random_state', 'loss_function', 'iterations', 
                           'early_stopping_rounds']:
                report_lines.append(f"  {param}: {value}")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open(output_dir / 'training_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text


def load_existing_results(output_dir: Path) -> dict:
    """Load existing results if available."""
    results = {}
    
    for model_name in ['lightgbm', 'xgboost', 'catboost']:
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


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("MODEL TRAINING WITH OPTUNA OPTIMIZATION")
    print("=" * 80)
    
    # Check GPU availability for tree models
    from src.training.optuna_optimization import (
        GPU_AVAILABLE, GPU_NAME,
        LIGHTGBM_GPU_AVAILABLE, XGBOOST_GPU_AVAILABLE, CATBOOST_GPU_AVAILABLE
    )
    if GPU_AVAILABLE:
        print(f"\nGPU detected: {GPU_NAME}")
        print(f"  LightGBM will use: {'GPU' if LIGHTGBM_GPU_AVAILABLE else 'CPU'}")
        print(f"  XGBoost will use: {'GPU' if XGBOOST_GPU_AVAILABLE else 'CPU'}")
        print(f"  CatBoost will use: {'GPU' if CATBOOST_GPU_AVAILABLE else 'CPU'}")
    else:
        print("\nGPU not available - all tree models will use CPU")
    
    # Load data
    X_train, y_train, X_test, y_test, selected_features = load_data_and_features()
    
    # Create output directory
    output_dir = Path('data/models')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected features used
    with open(output_dir / 'selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    # Try to load existing results
    results = load_existing_results(output_dir)
    
    n_trials = 10
    n_folds = 5
    
    # Train LightGBM
    if 'lightgbm' in results and results['lightgbm'].get('completed', False):
        print("\n" + "=" * 80)
        print("SKIPPING LIGHTGBM (already completed)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TRAINING LIGHTGBM")
        print("=" * 80)
        
        print(f"\nRunning Optuna optimization ({n_trials} trials, {n_folds}-fold CV)...")
        study_lgb, best_params_lgb = optimize_lightgbm(
            X_train, y_train,
            n_trials=n_trials,
            n_folds=n_folds,
            verbose=True
        )
        
        print(f"\nBest CV AUC-ROC: {study_lgb.best_value:.6f}")
        print("Best parameters:")
        for param, value in best_params_lgb.items():
            if param not in ['objective', 'metric', 'boosting_type', 'verbose', 'seed']:
                print(f"  {param}: {value}")
        
        print("\nTraining final model with best hyperparameters...")
        model_lgb, cv_metrics_lgb = train_lightgbm_final(
            X_train, y_train,
            best_params_lgb,
            verbose=True
        )
        
        # Get predictions
        y_test_pred_proba_lgb = model_lgb.predict(X_test)
        y_test_pred_lgb = (y_test_pred_proba_lgb >= 0.5).astype(int)
        
        # Evaluate on test set
        test_metrics_lgb = evaluate_model(y_test.values, y_test_pred_lgb, y_test_pred_proba_lgb)
        
        # Get OOF predictions
        print("\nComputing CV metrics with best hyperparameters...")
        _, _, oof_predictions_lgb = train_lightgbm_cv(
            X_train, y_train,
            best_params_lgb,
            n_folds=n_folds
        )
        
        results['lightgbm'] = {
            'best_params': best_params_lgb,
            'cv_metrics': cv_metrics_lgb,
            'test_metrics': test_metrics_lgb,
            'study': study_lgb,
            'model': model_lgb
        }
        
        # Save LightGBM results
        y_train_pred_proba_lgb = model_lgb.predict(X_train)
        y_train_pred_lgb = (y_train_pred_proba_lgb >= 0.5).astype(int)
        
        save_model_results(
            'lightgbm', model_lgb, best_params_lgb,
            cv_metrics_lgb, test_metrics_lgb,
            study_lgb, output_dir
        )
        save_predictions(
            'lightgbm',
            y_train_pred_lgb, y_train_pred_proba_lgb,
            y_test_pred_lgb, y_test_pred_proba_lgb,
            output_dir,
            oof_predictions=oof_predictions_lgb
        )
    
    # Train XGBoost
    if 'xgboost' in results and results['xgboost'].get('completed', False):
        print("\n" + "=" * 80)
        print("SKIPPING XGBOOST (already completed)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("TRAINING XGBOOST")
        print("=" * 80)
        
        print(f"\nRunning Optuna optimization ({n_trials} trials, {n_folds}-fold CV)...")
        study_xgb, best_params_xgb = optimize_xgboost(
            X_train, y_train,
            n_trials=n_trials,
            n_folds=n_folds,
            verbose=True
        )
        
        print(f"\nBest CV AUC-ROC: {study_xgb.best_value:.6f}")
        print("Best parameters:")
        for param, value in best_params_xgb.items():
            if param not in ['objective', 'eval_metric', 'tree_method', 'random_state']:
                print(f"  {param}: {value}")
        
        print("\nTraining final model with best hyperparameters...")
        model_xgb, cv_metrics_xgb = train_xgboost_final(
            X_train, y_train,
            best_params_xgb,
            verbose=True
        )
        
        # Get predictions
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test)
        
        y_test_pred_proba_xgb = model_xgb.predict(dtest)
        y_test_pred_xgb = (y_test_pred_proba_xgb >= 0.5).astype(int)
        
        # Evaluate on test set
        test_metrics_xgb = evaluate_model(y_test.values, y_test_pred_xgb, y_test_pred_proba_xgb)
        
        # Get OOF predictions
        print("\nComputing CV metrics with best hyperparameters...")
        _, _, oof_predictions_xgb = train_xgboost_cv(
            X_train, y_train,
            best_params_xgb,
            n_folds=n_folds
        )
        
        results['xgboost'] = {
            'best_params': best_params_xgb,
            'cv_metrics': cv_metrics_xgb,
            'test_metrics': test_metrics_xgb,
            'study': study_xgb,
            'model': model_xgb
        }
        
        # Save XGBoost results
        dtrain = xgb.DMatrix(X_train)
        y_train_pred_proba_xgb = model_xgb.predict(dtrain)
        y_train_pred_xgb = (y_train_pred_proba_xgb >= 0.5).astype(int)
        
        save_model_results(
            'xgboost', model_xgb, best_params_xgb,
            cv_metrics_xgb, test_metrics_xgb,
            study_xgb, output_dir
        )
        save_predictions(
            'xgboost',
            y_train_pred_xgb, y_train_pred_proba_xgb,
            y_test_pred_xgb, y_test_pred_proba_xgb,
            output_dir,
            oof_predictions=oof_predictions_xgb
        )
    
    # Train CatBoost
    if 'catboost' in results and results['catboost'].get('completed', False):
        print("\n" + "=" * 80)
        print("SKIPPING CATBOOST (already completed)")
        print("=" * 80)
        print("Loading existing CatBoost results...")
        best_params_cb = results['catboost']['best_params']
        test_metrics_cb = results['catboost']['test_metrics']
        cv_metrics_cb = results['catboost']['cv_metrics']
        
        # Load model if available
        model_file = output_dir / 'catboost' / 'model.pkl'
        if model_file.exists():
            import pickle
            with open(model_file, 'rb') as f:
                model_cb = pickle.load(f)
        else:
            print("Warning: Model file not found, will need to retrain")
            model_cb = None
    else:
        print("\n" + "=" * 80)
        print("TRAINING CATBOOST")
        print("=" * 80)
        
        # Check if optimization study exists
        study_file = output_dir / 'catboost' / 'optuna_study.pkl'
        if study_file.exists():
            print("Found existing Optuna study, loading...")
            import pickle
            with open(study_file, 'rb') as f:
                study_cb = pickle.load(f)
            
            # Get best params from study
            best_params_cb = study_cb.best_params.copy()
            best_params_cb['iterations'] = 1000
            best_params_cb['early_stopping_rounds'] = 50
            best_params_cb['verbose'] = False
            best_params_cb['random_seed'] = 42
            best_params_cb['loss_function'] = 'Logloss'
            best_params_cb['eval_metric'] = 'Logloss'  # Use Logloss for GPU compatibility
            
            print(f"Best CV AUC-ROC from existing study: {study_cb.best_value:.6f}")
        else:
            print(f"\nRunning Optuna optimization ({n_trials} trials, {n_folds}-fold CV)...")
            study_cb, best_params_cb = optimize_catboost(
                X_train, y_train,
                n_trials=n_trials,
                n_folds=n_folds,
                verbose=True
            )
            
            print(f"\nBest CV AUC-ROC: {study_cb.best_value:.6f}")
        
        print("Best parameters:")
        for param, value in best_params_cb.items():
            if param not in ['iterations', 'early_stopping_rounds', 'verbose', 
                            'random_seed', 'loss_function', 'eval_metric']:
                print(f"  {param}: {value}")
        
        print("\nTraining final model with best hyperparameters...")
        model_cb, cv_metrics_cb = train_catboost_final(
            X_train, y_train,
            best_params_cb,
            verbose=True
        )
        
        # Get predictions
        y_test_pred_proba_cb = model_cb.predict_proba(X_test)[:, 1]
        y_test_pred_cb = model_cb.predict(X_test)
        
        # Evaluate on test set
        test_metrics_cb = evaluate_model(y_test.values, y_test_pred_cb, y_test_pred_proba_cb)
        
        # Get OOF predictions
        print("\nComputing CV metrics with best hyperparameters...")
        _, _, oof_predictions_cb = train_catboost_cv(
            X_train, y_train,
            best_params_cb,
            n_folds=n_folds
        )
        
        results['catboost'] = {
            'best_params': best_params_cb,
            'cv_metrics': cv_metrics_cb,
            'test_metrics': test_metrics_cb,
            'study': study_cb if 'study_cb' in locals() else None,
            'model': model_cb
        }
        
        # Save CatBoost results
        y_train_pred_proba_cb = model_cb.predict_proba(X_train)[:, 1]
        y_train_pred_cb = model_cb.predict(X_train)
        
        save_model_results(
            'catboost', model_cb, best_params_cb,
            cv_metrics_cb, test_metrics_cb,
            study_cb if 'study_cb' in locals() else None, output_dir
        )
        save_predictions(
            'catboost',
            y_train_pred_cb, y_train_pred_proba_cb,
            y_test_pred_cb, y_test_pred_proba_cb,
            output_dir,
            oof_predictions=oof_predictions_cb
        )
    
    # Generate and print report
    report = generate_training_report(results, output_dir)
    print("\n" + report)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {output_dir}/")
    print(f"  - Models: {output_dir}/[lightgbm|xgboost|catboost]/model.pkl")
    print(f"  - Hyperparameters: {output_dir}/[model]/best_params.json")
    print(f"  - Metrics: {output_dir}/[model]/metrics.json")
    print(f"  - Predictions: {output_dir}/[model]/[train|test]_predictions.csv")
    print(f"  - Visualizations: {output_dir}/[model]/optimization_history.html")


if __name__ == '__main__':
    main()

