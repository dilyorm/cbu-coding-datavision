"""Optuna hyperparameter optimization for neural network models."""
import optuna
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from src.models.nn_models import SimpleMLP, TabMModel
from src.training.nn_training import train_nn_cv, create_dataloaders, NNDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def optimize_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[optuna.Study, dict]:
    """Optimize MLP hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (Optuna study, best parameters)
    """
    input_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def objective(trial):
        # Model parameters
        hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # Training parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        model_params = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout_rate': dropout_rate,
            'use_residual': True
        }
        
        training_params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'max_epochs': 200,
            'early_stopping_patience': 20
        }
        
        try:
            cv_metrics, _, _ = train_nn_cv(
                SimpleMLP,
                X_train,
                y_train,
                model_params,
                training_params,
                n_folds=n_folds,
                random_state=random_state,
                device=device
            )
            return cv_metrics['auc_roc']
        except Exception as e:
            if verbose:
                print(f"Trial failed: {e}")
            return 0.0
    
    study = optuna.create_study(direction='maximize', study_name='mlp_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    
    best_params = study.best_params.copy()
    
    # Separate model and training params
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': best_params['hidden_dim'],
        'num_layers': best_params['num_layers'],
        'dropout_rate': best_params['dropout_rate'],
        'use_residual': True
    }
    
    training_params = {
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'weight_decay': best_params['weight_decay'],
        'max_epochs': 200,
        'early_stopping_patience': 20
    }
    
    return study, {'model_params': model_params, 'training_params': training_params}


def optimize_tabm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 100,
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[optuna.Study, dict]:
    """Optimize TabM hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        n_folds: Number of CV folds
        random_state: Random seed
        verbose: Whether to print progress
    
    Returns:
        Tuple of (Optuna study, best parameters)
    """
    input_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def objective(trial):
        # Model parameters
        hidden_dim = trial.suggest_int('hidden_dim', 64, 512)
        num_layers = trial.suggest_int('num_layers', 2, 6)
        multiplicative_units = trial.suggest_int('multiplicative_units', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        
        # Training parameters
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        model_params = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'multiplicative_units': multiplicative_units,
            'dropout_rate': dropout_rate
        }
        
        training_params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'max_epochs': 200,
            'early_stopping_patience': 20
        }
        
        try:
            cv_metrics, _, _ = train_nn_cv(
                TabMModel,
                X_train,
                y_train,
                model_params,
                training_params,
                n_folds=n_folds,
                random_state=random_state,
                device=device
            )
            return cv_metrics['auc_roc']
        except Exception as e:
            if verbose:
                print(f"Trial failed: {e}")
            return 0.0
    
    study = optuna.create_study(direction='maximize', study_name='tabm_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    
    best_params = study.best_params.copy()
    
    # Separate model and training params
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': best_params['hidden_dim'],
        'num_layers': best_params['num_layers'],
        'multiplicative_units': best_params['multiplicative_units'],
        'dropout_rate': best_params['dropout_rate']
    }
    
    training_params = {
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'weight_decay': best_params['weight_decay'],
        'max_epochs': 200,
        'early_stopping_patience': 20
    }
    
    return study, {'model_params': model_params, 'training_params': training_params}


