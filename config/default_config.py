"""Configuration file for default prediction pipeline"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Paths:
    """File paths for data sources and output directories"""
    geographic_data: str = "geographic_data.xml"
    financial_ratios: str = "financial_ratios.jsonl"
    demographics: str = "demographics.csv"
    application_metadata: str = "application_metadata.csv"
    loan_details: str = "loan_details.xlsx"
    credit_history: str = "credit_history.parquet"
    models_dir: str = "models"
    datas_dir: str = "datas"


@dataclass
class TrainingConfig:
    """Training and model configuration"""
    test_size: float = 0.3
    valid_size: float = 0.5
    random_state: int = 42
    tune_hyperparams: bool = True  # Enable tuning by default for CatBoost
    catboost_n_trials: int = 30  # More trials for better optimization
    imputation_method: str = 'knn'  # 'knn', 'iterative', 'median'
    cv_n_splits: int = 5


@dataclass
class ModelConfig:
    """Model-specific hyperparameters"""
    # LightGBM defaults
    lgbm_n_estimators: int = 1000
    lgbm_learning_rate: float = 0.03
    lgbm_max_depth: int = -1
    lgbm_num_leaves: int = 63
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    
    # XGBoost defaults
    xgb_n_estimators: int = 1000
    xgb_learning_rate: float = 0.03
    xgb_max_depth: int = 6
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # CatBoost defaults
    catboost_iterations: int = 1000
    catboost_learning_rate: float = 0.03
    catboost_depth: int = 6
    catboost_l2_leaf_reg: float = 3
    
    # Early stopping
    early_stopping_rounds: int = 50


# Global config instances
paths = Paths()
training = TrainingConfig()
model_config = ModelConfig()

