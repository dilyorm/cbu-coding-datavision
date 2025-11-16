# Default Prediction ML Pipeline

A comprehensive machine learning pipeline for predicting loan defaults using ensemble stacking with multiple feature engineering approaches and diverse model architectures.

## Project Overview

- **Multiple Feature Engineering Pipelines**: 3 different feature sets (v1, v2, v3)
- **Diverse Base Models**: LightGBM, XGBoost, CatBoost, MLP, TabM
- **Out-of-Fold (OOF) Predictions**: Used as meta-features for stacking
- **Level-2 Stacking Models**: Trained on meta-features
- **Weighted Ensemble**: Final predictions from all models

## Project Structure

```
.
├── data/
│   ├── processed/          # Preprocessed data (v1, v2, v3 feature sets)
│   ├── feature_selection/  # Feature selection results
│   ├── models/             # Trained models and predictions
│   ├── stacking/           # Stacking dataset with meta features
│   └── final_submission/   # Final ensemble predictions
├── main.py                 # Main workflow orchestrator (run complete pipeline)
├── predict_real_test.py    # Standalone prediction pipeline for real test data
├── src/
│   ├── data/               # Data processing modules
│   │   ├── load_data.py            # Data loading utilities
│   │   ├── clean_data.py           # Data cleaning functions
│   │   ├── preprocessing.py        # Preprocessing pipeline
│   │   ├── data_insights.py        # Data analysis and insights
│   │   ├── prepare_data.py         # Single feature set preparation
│   │   └── prepare_data_multi.py  # Multiple feature sets preparation
│   ├── features/           # Feature engineering modules
│   │   ├── feature_engineering.py      # Base feature engineering (v1)
│   │   ├── feature_engineering_v2.py  # Aggregation-heavy features (v2)
│   │   ├── feature_engineering_v3.py  # Domain-specific simple features (v3)
│   │   ├── feature_selection.py       # Feature selection functions
│   │   └── run_feature_selection.py   # Feature selection orchestration
│   ├── models/             # Model definitions
│   │   └── nn_models.py    # Neural network architectures
│   ├── training/           # Training utilities and scripts
│   │   ├── model_training.py        # Tree model training functions
│   │   ├── nn_training.py           # Neural network training functions
│   │   ├── optuna_optimization.py   # Optuna optimization for tree models
│   │   ├── nn_optimization.py       # Optuna optimization for NN models
│   │   ├── train_models.py          # Train tree models (single feature set)
│   │   ├── train_nn_models.py       # Train NN models (single feature set)
│   │   └── train_all_feature_sets.py # Train all models on all feature sets
│   └── ensemble/           # Stacking and ensemble modules
│       ├── create_meta_features.py  # Create meta features from OOF predictions
│       ├── train_stacking_models.py # Train Level-2 stacking models
│       ├── ensemble.py              # Weighted ensemble creation
│       └── generate_final_predictions.py # Final submission generation
└── requirements.txt        # Python dependencies
```

## Installation

1. **Clone the repository** (or navigate to project directory)

2. **Create a virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Workflow

The complete pipeline consists of several phases:

### Phase 1: Data Preparation

#### 1.1 Generate Multiple Feature Sets

```bash
python -m src.data.prepare_data_multi
```

This script:
- Loads raw data from `data/` directory
- Cleans and merges datasets
- Applies 3 different feature engineering pipelines:
  - **v1**: Comprehensive features (temporal, interaction, aggregation, geographic, loan-specific, rank, advanced ratios, binned categoricals, population statistics)
  - **v2**: Aggregation-heavy features (customer/loan aggregations, interaction aggregations)
  - **v3**: Domain-specific simple features (simple ratios, binary flags, categorical counts)
- Preprocesses each feature set (standardization, one-hot encoding)
- Saves processed data to `data/processed/v1/`, `data/processed/v2/`, `data/processed/v3/`

**Outputs:**
- `data/processed/v[1-3]/X_train.csv`, `X_val.csv`, `X_test.csv`
- `data/processed/v[1-3]/y_train.csv`, `y_val.csv`, `y_test.csv`
- `data/processed/v[1-3]/preprocessing_pipeline.pkl`
- `data/processed/v[1-3]/feature_names.txt`
- `data/processed/v[1-3]/data_report.txt`

#### 1.2 Feature Selection (Optional)

```bash
python -m src.features.run_feature_selection
```

This script:
- Loads preprocessed data
- Performs feature selection using:
  - Zero importance filtering
  - Permutation importance filtering
  - Stepped permutation selection
  - Forward feature selection (optional)
- Saves selected features to `data/feature_selection/selected_features.txt`

**Note:** Feature selection is performed on v1 feature set. Selected features are then applied to all feature sets.

### Phase 2: Base Model Training

#### 2.1 Train Models on Single Feature Set (v1)

**Tree Models:**
```bash
python -m src.training.train_models
```

**Neural Network Models:**
```bash
python -m src.training.train_nn_models
```

These scripts:
- Load preprocessed data and selected features
- Perform Optuna hyperparameter optimization (5-fold CV)
- Train final models with best hyperparameters
- Generate Out-of-Fold (OOF) predictions
- Save models, metrics, and predictions

**Outputs:**
- `data/models/[model_name]/model.pkl` or `model.pth`
- `data/models/[model_name]/best_params.json`
- `data/models/[model_name]/metrics.json`
- `data/models/[model_name]/oof_predictions.csv`
- `data/models/[model_name]/[train|val|test]_predictions.csv`
- `data/models/[model_name]/optuna_study.pkl`

#### 2.2 Train All Models on All Feature Sets

```bash
python -m src.training.train_all_feature_sets
```

This script:
- Trains all 5 models (LightGBM, XGBoost, CatBoost, MLP, TabM) on all 3 feature sets
- Reuses best hyperparameters from v1 training (or optimizes if not available)
- Generates OOF predictions for each model-feature combination
- Saves results to `data/models/[version]_[model]/`

**Outputs:**
- 15 model directories (3 feature sets × 5 models)
- Each directory contains: model, metrics, predictions, OOF predictions

**Resume Capability:**
- Script automatically skips already completed models
- Can resume from interruptions (KeyboardInterrupt, errors)
- Checks for existing Optuna studies and continues optimization

### Phase 3: Meta Features and Stacking

#### 3.1 Create Meta Features

```bash
python -m src.ensemble.create_meta_features
```

This script:
- Loads OOF predictions from all base models across all feature sets
- Creates aggregated meta features:
  - Per model type: mean, std, min, max, last predictions
  - Individual predictions from each model-feature combination
  - Overall statistics across all models
- Combines base features (v1) with meta features
- Saves stacking dataset to `data/stacking/`

**Outputs:**
- `data/stacking/X_train.csv`, `X_val.csv`, `X_test.csv`
- `data/stacking/y_train.csv`, `y_val.csv`, `y_test.csv`
- `data/stacking/feature_names.txt`

#### 3.2 Train Stacking Models

```bash
python -m src.ensemble.train_stacking_models
```

This script:
- Trains Level-2 models (LightGBM, XGBoost) on meta features
- Uses simple hyperparameters to prevent overfitting
- Generates OOF predictions for stacking models
- Saves results to `data/models/stacking/[model]/`

**Outputs:**
- `data/models/stacking/[model]/model.pkl`
- `data/models/stacking/[model]/metrics.json`
- `data/models/stacking/[model]/oof_predictions.csv`
- `data/models/stacking/[model]/[train|val|test]_predictions.csv`

### Phase 4: Ensemble and Final Predictions

#### 4.1 Create Weighted Ensemble

```bash
python -m src.ensemble.ensemble
```

This script:
- Loads predictions from all base models and stacking models
- Computes ensemble weights based on validation AUC-ROC performance
- Creates weighted ensemble predictions
- Evaluates ensemble performance
- Saves ensemble results to `data/final_submission/`

**Outputs:**
- `data/final_submission/ensemble_val_predictions.csv`
- `data/final_submission/ensemble_test_predictions.csv`
- `data/final_submission/ensemble_metrics.json`
- `data/final_submission/ensemble_report.txt`

#### 4.2 Generate Final Submission

```bash
python -m src.ensemble.generate_final_predictions
```

This script:
- Compares performance of all individual models and ensemble
- Generates final submission file
- Creates comprehensive performance report

**Outputs:**
- `data/final_submission/submission.csv`
- `data/final_submission/final_report.txt`

### Phase 5: Real Test Data Prediction

#### 5.1 Generate Predictions for Real Test Data

```bash
python predict_real_test.py --data-dir data_test --output predictions/submission_real_test.csv
```

This standalone script:
- Loads real test data from a specified directory (default: `data_test/`)
- Applies the same data cleaning and feature engineering pipelines used in training
- Generates predictions from all trained models (15 base + 2 stacking)
- Creates weighted ensemble predictions using saved ensemble weights
- Outputs final predictions in the required format: `customer_id, default`

**Arguments:**
- `--data-dir`: Directory containing test data files (default: `data_test`)
- `--output`: Output path for submission file (default: `predictions/submission_real_test.csv`)

**Test Data Requirements:**

Place your test data files in the specified directory with the same structure as training data:

```
data_test/
├── application_metadata.csv
├── credit_history.parquet
├── demographics.csv
├── financial_ratios.jsonl
├── geographic_data.xml
└── loan_details.xlsx
```

**Note:** The test data should NOT include the `default` column (target variable).

**What the Script Does:**

1. **Data Loading**: Loads all test data files and merges them on `customer_id`
2. **Data Cleaning**: Applies the same cleaning pipeline as training data
3. **Feature Engineering**: Generates features for all 3 versions (v1, v2, v3)
4. **Preprocessing**: Loads saved preprocessing pipelines and transforms test data
5. **Model Loading**: Loads all 17 trained models:
   - 15 base models (v1/v2/v3 × lightgbm/xgboost/catboost/mlp/tabm)
   - 2 stacking models (lightgbm/xgboost)
6. **Prediction Generation**:
   - Generates predictions from all base models
   - Creates stacking meta-features from base predictions
   - Generates predictions from stacking models
7. **Ensemble**: Creates weighted ensemble using saved weights from training
8. **Output**: Saves final predictions as CSV with columns: `customer_id, default`

**Outputs:**
- `predictions/submission_real_test.csv` (or specified output path)

**Example Usage:**

```bash
# Use default data directory and output path
python predict_real_test.py

# Specify custom data directory and output path
python predict_real_test.py --data-dir /path/to/test/data --output /path/to/output/submission.csv
```

**Important Notes:**

- Ensure all models are trained and saved before running this script
- The script automatically handles feature alignment with training data
- Missing features are filled with zeros, extra features are removed
- Stacking features are automatically created from base model predictions
- Ensemble weights are loaded from `data/final_submission/ensemble_metrics.json`

## Complete Workflow Example

### Option 1: Run Complete Workflow (Recommended)

Execute all steps in sequence with a single command:

```bash
python main.py
```

This will run the complete pipeline:
1. Data preparation (multiple feature sets)
2. Feature selection (optional)
3. Model training (all models on all feature sets)
4. Meta features creation
5. Stacking models training
6. Ensemble creation
7. Final predictions generation

**Skip Optional Steps:**
```bash
# Skip feature selection (optional step)
python main.py --skip-feature-selection

# Resume from a specific step (e.g., skip data prep and training)
python main.py --skip-data-prep --skip-training
```

**See all options:**
```bash
python main.py --help
```

### Option 2: Run Steps Individually

If you prefer to run each step manually:

```bash
# Step 1: Prepare multiple feature sets
python -m src.data.prepare_data_multi

# Step 2: (Optional) Feature selection
python -m src.features.run_feature_selection

# Step 3: Train all models on all feature sets
python -m src.training.train_all_feature_sets

# Step 4: Create meta features
python -m src.ensemble.create_meta_features

# Step 5: Train stacking models
python -m src.ensemble.train_stacking_models

# Step 6: Create weighted ensemble
python -m src.ensemble.ensemble

# Step 7: Generate final submission
python -m src.ensemble.generate_final_predictions
```

## Key Features

### Resume Capability

All training scripts support resuming from interruptions:
- **Completed models**: Automatically skipped
- **Incomplete Optuna studies**: Resumed with remaining trials
- **Partial training**: Continues from last checkpoint

### Model Architectures

**Tree Models:**
- LightGBM: Gradient boosting with leaf-wise tree growth
- XGBoost: Gradient boosting with regularization
- CatBoost: Gradient boosting with categorical handling

**Neural Networks:**
- MLP: Multi-layer perceptron with BatchNorm, Dropout, residual connections
- TabM: Tabular model with multiplicative interactions and feature embeddings

### Evaluation Metrics

All models are evaluated using:
- **AUC-ROC** (primary metric)
- Log Loss
- Accuracy
- Precision
- Recall
- F1 Score

### Hyperparameter Optimization

- **Framework**: Optuna
- **Strategy**: K-fold cross-validation (5 folds)
- **Trials**: 10 for tree models, 5 for neural networks
- **Metric**: AUC-ROC (maximized)

## Data Requirements

### Training Data

Place your raw training data files in the `data/` directory with the following structure:

```
data/
├── application_metadata.csv
├── credit_history.parquet
├── demographics.csv
├── financial_ratios.jsonl
├── geographic_data.xml
└── loan_details.xlsx
```

**Required columns:**
- `customer_id`: Customer identifier (will be normalized)
- `default`: Target variable (binary: 0/1)

### Test Data

For generating predictions on new test data, place files in a separate directory (e.g., `data_test/`) with the same structure:

```
data_test/
├── application_metadata.csv
├── credit_history.parquet
├── demographics.csv
├── financial_ratios.jsonl
├── geographic_data.xml
└── loan_details.xlsx
```

**Required columns:**
- `customer_id`: Customer identifier (will be normalized)
- **Note:** Test data should NOT include the `default` column

## Output Structure

```
data/
├── processed/
│   ├── v1/          # Feature set v1
│   ├── v2/          # Feature set v2
│   └── v3/          # Feature set v3
├── feature_selection/
│   └── selected_features.txt
├── models/
│   ├── lightgbm/    # v1 LightGBM
│   ├── xgboost/     # v1 XGBoost
│   ├── catboost/    # v1 CatBoost
│   ├── mlp/         # v1 MLP
│   ├── tabm/        # v1 TabM
│   ├── v2_lightgbm/ # v2 LightGBM
│   ├── ...          # All other combinations
│   └── stacking/    # Level-2 stacking models
├── stacking/         # Stacking dataset
└── final_submission/ # Final predictions
```

## Troubleshooting

### Prediction Pipeline Issues

**Error: "No models loaded!"**
- Ensure all models are trained first by running the complete training pipeline
- Check that model files exist in `data/models/` directories

**Error: "Feature mismatch"**
- The script automatically handles feature alignment
- Missing features are filled with zeros
- Extra features are removed
- If issues persist, ensure preprocessing pipelines are saved correctly

**Error: "Stacking features mismatch"**
- Ensure base model predictions are generated successfully
- Check that `data/stacking/feature_names.txt` exists
- Verify that meta-features are created correctly

### Resume Training After Interruption

Simply re-run the training script:
```bash
python -m src.training.train_all_feature_sets
```

The script will automatically:
- Skip completed models
- Resume incomplete Optuna studies
- Continue from where it left off

### Memory Issues

If you encounter memory issues:
- Reduce `n_trials` in optimization scripts
- Reduce `n_folds` (though 5 is recommended)
- Process feature sets sequentially instead of in parallel

## Performance Tips

1. **Use GPU**: Neural network training benefits significantly from GPU acceleration
2. **Parallel Processing**: Optuna can use multiple workers (set `n_jobs` parameter)
3. **Early Stopping**: All models use early stopping to prevent overfitting
4. **Feature Selection**: Reduces training time and memory usage

## Model Performance Tracking

Each model saves comprehensive metrics:
- Cross-validation metrics (5-fold CV)
- Validation set metrics
- Test set metrics
- Best hyperparameters
- Optuna optimization history

View metrics:
```python
import json
with open('data/models/lightgbm/metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f"AUC-ROC: {metrics['test_metrics']['auc_roc']:.6f}")
```

## Quick Reference

### Training Pipeline
```bash
# Complete training pipeline
python main.py

# Or step by step:
python -m src.data.prepare_data_multi          # Prepare data
python -m src.training.train_all_feature_sets  # Train models
python -m src.ensemble.create_meta_features    # Create meta features
python -m src.ensemble.train_stacking_models   # Train stacking
python -m src.ensemble.ensemble                # Create ensemble
```

### Prediction Pipeline
```bash
# Generate predictions for real test data
python predict_real_test.py --data-dir data_test --output predictions/submission.csv
```

## License

[Add your license here]

## Contact

[Add contact information here]

