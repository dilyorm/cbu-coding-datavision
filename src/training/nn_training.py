"""Neural network training utilities with cross-validation support."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

from src.training.model_training import evaluate_model


class NNDataset(Dataset):
    """PyTorch Dataset for tabular data."""
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    batch_size: int = 512,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation."""
    train_dataset = NNDataset(X_train, y_train)
    val_dataset = NNDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
            
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, np.array(all_preds), np.array(all_probs), np.array(all_labels)


def train_nn_cv(
    model_class: nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict,
    training_params: Dict,
    n_folds: int = 5,
    random_state: int = 42,
    device: Optional[torch.device] = None
) -> Tuple[Dict[str, float], List[nn.Module], np.ndarray]:
    """Train neural network with k-fold cross-validation.
    
    Args:
        model_class: PyTorch model class
        X_train: Training features
        y_train: Training target
        model_params: Parameters for model initialization
        training_params: Training hyperparameters (lr, batch_size, epochs, etc.)
        n_folds: Number of CV folds
        random_state: Random seed
        device: PyTorch device (auto-detect if None)
    
    Returns:
        Tuple of (mean CV metrics, list of trained models, OOF predictions)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_scores = []
    models = []
    oof_predictions = np.zeros(len(X_train))
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Normalize features (fit on train, transform both)
        scaler = StandardScaler()
        X_fold_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_fold_train),
            columns=X_fold_train.columns,
            index=X_fold_train.index
        )
        X_fold_val_scaled = pd.DataFrame(
            scaler.transform(X_fold_val),
            columns=X_fold_val.columns,
            index=X_fold_val.index
        )
        
        # Create model
        model = model_class(**model_params).to(device)
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            X_fold_train_scaled, y_fold_train,
            X_fold_val_scaled, y_fold_val,
            batch_size=training_params['batch_size']
        )
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params.get('weight_decay', 0.0)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_epochs = training_params.get('max_epochs', 200)
        early_stopping_patience = training_params.get('early_stopping_patience', 20)
        
        for epoch in range(max_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_probs, val_labels = validate_epoch(
                model, val_loader, criterion, device
            )
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate and get OOF predictions
        val_loss, val_preds, val_probs, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Store OOF predictions for this fold
        oof_predictions[val_idx] = val_probs
        
        fold_metrics = evaluate_model(val_labels, val_preds, val_probs)
        cv_scores.append(fold_metrics)
        models.append(model)
    
    # Average metrics across folds
    mean_metrics = {}
    for key in cv_scores[0].keys():
        mean_metrics[key] = np.mean([scores[key] for scores in cv_scores])
    
    return mean_metrics, models, oof_predictions


def train_nn_final(
    model_class: nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict,
    training_params: Dict,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, float], StandardScaler]:
    """Train final neural network model on full training set.
    
    Uses CV to determine optimal training, then trains on full set.
    
    Returns:
        Tuple of (trained model, CV metrics, fitted scaler)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Create model
    model = model_class(**model_params).to(device)
    
    # Create data loader for full training set
    from torch.utils.data import TensorDataset, DataLoader
    X_tensor = torch.FloatTensor(X_train_scaled.values).to(device)
    y_tensor = torch.FloatTensor(y_train.values).to(device)
    train_dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_params['learning_rate'],
        weight_decay=training_params.get('weight_decay', 0.0)
    )
    
    # Training loop - train for fixed epochs (use CV to determine optimal)
    max_epochs = training_params.get('max_epochs', 200)
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Get CV metrics for reporting
    cv_metrics, _, _ = train_nn_cv(
        model_class, X_train, y_train,
        model_params, training_params,
        n_folds=5, device=device
    )
    
    return model, cv_metrics, scaler


def predict_nn(
    model: nn.Module,
    X: pd.DataFrame,
    scaler: StandardScaler,
    batch_size: int = 512,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with neural network model.
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize features
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Create dataset and dataloader
    dataset = NNDataset(X_scaled, pd.Series([0] * len(X_scaled)))  # Dummy y
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            probs = outputs.cpu().numpy()
            all_probs.extend(probs)
    
    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)
    
    return preds, probs

