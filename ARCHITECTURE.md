# Project Architecture

This document describes the modular architecture of the default prediction pipeline.

**Note: This pipeline uses CatBoost exclusively for optimal performance with categorical features.**

## Directory Structure

```
project_root/
├── config/
│   ├── __init__.py
│   └── default_config.py          # Configuration (paths, hyperparameters)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py             # Data loading functions
│   │   ├── merging.py              # Data merging and aggregation
│   │   └── io_utils.py             # Save/load utilities
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── cleaning.py             # Data cleaning functions
│   │   ├── engineering.py           # Feature engineering
│   │   └── missing_values.py       # Missing value handling
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── prep.py                 # Data preparation and preprocessing
│   │   ├── training.py             # Model training
│   │   ├── tuning.py               # Hyperparameter tuning
│   │   ├── ensemble.py             # Ensemble creation
│   │   └── evaluation.py           # Model evaluation
│   │
│   ├── utils/
│   │   └── __init__.py
│   │
│   └── pipelines/
│       ├── __init__.py
│       └── default_prediction.py   # Main pipeline orchestration
│
└── scripts/
    └── train_default.py            # Entry point script
```

## Module Responsibilities

### Config Layer (`config/`)
- **default_config.py**: Centralized configuration
  - `Paths`: File paths for data sources and output directories
  - `TrainingConfig`: Training parameters (split sizes, random state, etc.)
  - `ModelConfig`: Model-specific hyperparameters

### Data Layer (`src/data/`)
- **loaders.py**: Functions to load raw data from various sources
  - `load_geographic_data()`
  - `load_financial_ratios()`
  - `load_demographics()`
  - `load_application_metadata()`
  - `load_loan_details()`
  - `load_credit_history()`

- **merging.py**: Data merging and aggregation
  - `merge_all_data()`: Orchestrates loading and merging all datasets

- **io_utils.py**: I/O utilities
  - `save_artifacts()`: Saves models, preprocessors, and metadata

### Feature Layer (`src/features/`)
- **cleaning.py**: Data cleaning
  - `clean_currency()`: Clean currency strings
  - `clean_data()`: Systematic data cleaning

- **engineering.py**: Feature engineering
  - `aggregate_credit_features()`: Aggregate credit history
  - `engineer_features()`: Create derived features

- **missing_values.py**: Missing value handling
  - `handle_missing_values()`: Advanced imputation (KNN, Iterative, Median)

### Modeling Layer (`src/modeling/`)
- **prep.py**: Data preparation
  - `prepare_model_data()`: Separate features and target
  - `create_preprocessing_pipeline()`: Create preprocessing pipeline

- **training.py**: CatBoost model training
  - `train_catboost_model()`: Train optimized CatBoost with raw categorical features
  - `train_catboost_raw()`: Core CatBoost training function

- **tuning.py**: Hyperparameter tuning
  - `tune_catboost()`: Optuna-based CatBoost tuning (optimized for categorical features)

- **evaluation.py**: Model evaluation
  - `find_best_threshold()`: Optimize classification threshold
  - `evaluate_model()`: Evaluate model on test set

### Pipeline Layer (`src/pipelines/`)
- **default_prediction.py**: Main pipeline orchestration
  - `run_default_pipeline()`: End-to-end pipeline execution

## Usage

### Running the Pipeline

```bash
python scripts/train_default.py
```

### Modifying Configuration

Edit `config/default_config.py` to change:
- Data file paths
- Training/test split sizes
- Hyperparameters
- Imputation method

### Adding New Features

1. Add feature engineering logic to `src/features/engineering.py`
2. Update `engineer_features()` function

### Adding New Models

1. Add training logic to `src/modeling/training.py`
2. Update `train_models()` function
3. Add to ensemble in `src/modeling/ensemble.py`

## Benefits of This Architecture

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Testability**: Functions can be tested independently
3. **Maintainability**: Easy to locate and modify specific functionality
4. **Reusability**: Functions can be reused across different pipelines
5. **Reviewability**: Code reviewers can focus on specific modules
6. **Configurability**: All configuration in one place

## Migration Notes

The original `model_pipeline_advanced.py` has been refactored into this modular structure. The functionality remains the same, but the code is now organized for better maintainability.

