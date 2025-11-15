# Default Prediction Model

A comprehensive machine learning pipeline for predicting loan defaults using advanced feature engineering, multiple model ensembling, and proper validation strategies.

## 🚀 Features

- **Advanced Feature Engineering**: Aggregated credit history, financial ratios, demographic features, and geographic indicators
- **Multiple Models**: LightGBM, XGBoost, CatBoost, and Logistic Regression
- **Ensemble Method**: Weighted ensemble combining all models for optimal performance
- **Proper Validation**: Stratified train/validation/test splits with early stopping
- **Production Ready**: Organized file structure with models and data separation

## 📁 Project Structure

```
.
├── models/                  # Trained models and preprocessing objects
│   ├── default_prediction_model_advanced.pkl
│   ├── all_models_advanced.pkl
│   ├── preprocessor_advanced.pkl
│   ├── feature_columns_advanced.pkl
│   └── ensemble_weights.pkl
├── datas/                   # Data files
│   ├── cleaned_data_advanced.csv
│   ├── feature_importance_advanced.csv
│   └── example_input.csv
├── model_pipeline_advanced.py  # Main training pipeline
├── predict_advanced.py         # Core prediction functions
├── predict_set.py              # User-friendly prediction script
└── requirements.txt            # Python dependencies
```

## 📊 Data Sources

The model uses the following data sources:

- `application_metadata.csv` - Application metadata and target variable
- `loan_details.xlsx` - Loan-specific information
- `credit_history.parquet` - Credit history data
- `demographics.csv` - Customer demographics
- `financial_ratios.jsonl` - Financial ratios and metrics
- `geographic_data.xml` - Geographic and regional economic data

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/dilyorm/cbu-coding-datavision.git
cd cbu-coding-datavision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Training the Model

Train the model using the advanced pipeline:

```bash
python model_pipeline_advanced.py
```

This will:
- Load and merge all data sources
- Clean and engineer features
- Train multiple models (LightGBM, XGBoost, CatBoost, Logistic Regression)
- Create an ensemble model
- Save all models to `models/` folder
- Save processed data to `datas/` folder

### Making Predictions

#### Single Prediction

```python
from predict_advanced import predict_default_advanced

result = predict_default_advanced({
    'customer_id': 10000,
    'age': 41,
    'credit_score': 696,
    'annual_income': 61800,
    'loan_amount': 50000,
    'debt_to_income_ratio': 0.258
})

print(f"Default Probability: {result['default_probability']:.4f}")
print(f"Risk Level: {result['risk_level']}")
```

#### Batch Prediction

```python
from predict_advanced import predict_batch_advanced

results = predict_batch_advanced(
    'input_data.csv',
    output_file='datas/predictions.csv'
)
```

#### Using the Prediction Script

```bash
# Single prediction example
python predict_set.py single

# Batch prediction
python predict_set.py batch input.csv datas/output.csv
```

## 📈 Model Performance

The ensemble model achieves:
- **ROC-AUC**: ~0.81
- **PR-AUC**: Optimized for imbalanced dataset
- **Best Single Model**: LightGBM (typically)

### Top Important Features

1. Credit Score
2. Monthly Free Cash Flow
3. Age
4. Debt Service Ratio
5. Debt to Income Ratio

## 🔧 Configuration

### Hyperparameter Tuning

To enable hyperparameter tuning with Optuna, modify `model_pipeline_advanced.py`:

```python
# Line 711: Change to True
models, results = train_models(X_train, y_train, X_valid, y_valid, preprocessor, tune_hyperparams=True)
```

### Model Selection

The pipeline automatically selects the best model based on validation AUC. You can also use:
- **Ensemble**: Best performance (default)
- **Single Model**: Faster predictions

## 📝 Requirements

See `requirements.txt` for full list. Key dependencies:

- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- optuna (for hyperparameter tuning)
- pyarrow (for parquet files)
- openpyxl (for Excel files)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Dilyor M**

- GitHub: [@dilyorm](https://github.com/dilyorm)

## 🙏 Acknowledgments

- Built following best practices for machine learning pipelines
- Implements proper validation and ensemble techniques
- Production-ready code structure

