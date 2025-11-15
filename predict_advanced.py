import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


def preprocess_new_data(data, preprocessor, models_dir='models'):
    """Preprocess new data using saved preprocessor"""
    df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    
    # Get feature columns from preprocessor (if available)
    try:
        feature_cols = joblib.load(f'{models_dir}/feature_columns_advanced.pkl')
        # Ensure all required columns exist
        missing_cols = set(feature_cols) - set(df.columns)
        for col in missing_cols:
            df[col] = np.nan
        # Select only feature columns
        df = df[feature_cols]
    except:
        pass  # Will use whatever columns are in the data
    
    # Preprocess using the saved preprocessor
    X = preprocessor.transform(df)
    
    return X


def predict_default_advanced(data, use_ensemble=True, models_dir='models'):
    """Predict default probability using advanced models"""
    # Load saved models and preprocessor
    preprocessor = joblib.load(f'{models_dir}/preprocessor_advanced.pkl')
    
    # Preprocess data
    X = preprocess_new_data(data, preprocessor, models_dir)
    
    if use_ensemble:
        # Load all models and ensemble weights
        all_models = joblib.load(f'{models_dir}/all_models_advanced.pkl')
        ensemble_weights = joblib.load(f'{models_dir}/ensemble_weights.pkl')
        
        # Get predictions from all models
        predictions = {}
        for name, model in all_models.items():
            predictions[name] = model.predict_proba(X)[0][1]
        
        # Weighted ensemble
        probability = (
            ensemble_weights[0] * predictions['LightGBM'] +
            ensemble_weights[1] * predictions['XGBoost'] +
            ensemble_weights[2] * predictions['CatBoost'] +
            ensemble_weights[3] * predictions['LogisticRegression']
        )
        
        # Get prediction from best single model
        prediction = all_models['LightGBM'].predict(X)[0]
    else:
        # Use best single model
        model = joblib.load(f'{models_dir}/default_prediction_model_advanced.pkl')
        probability = model.predict_proba(X)[0][1]
        prediction = model.predict(X)[0]
    
    return {
        'default_probability': float(probability),
        'predicted_default': int(prediction),
        'risk_level': 'High' if probability > 0.5 else 'Medium' if probability > 0.3 else 'Low'
    }


def predict_batch_advanced(data_file, use_ensemble=True, output_file='datas/predictions_advanced.csv', models_dir='models'):
    """Predict defaults for a batch of records from CSV file"""
    df = pd.read_csv(data_file)
    
    # Load saved models and preprocessor
    preprocessor = joblib.load(f'{models_dir}/preprocessor_advanced.pkl')
    
    # Preprocess data
    X = preprocess_new_data(df, preprocessor, models_dir)
    
    if use_ensemble:
        # Load all models and ensemble weights
        all_models = joblib.load(f'{models_dir}/all_models_advanced.pkl')
        ensemble_weights = joblib.load(f'{models_dir}/ensemble_weights.pkl')
        
        # Get predictions from all models
        predictions_dict = {}
        for name, model in all_models.items():
            predictions_dict[name] = model.predict_proba(X)[:, 1]
        
        # Weighted ensemble
        probabilities = (
            ensemble_weights[0] * predictions_dict['LightGBM'] +
            ensemble_weights[1] * predictions_dict['XGBoost'] +
            ensemble_weights[2] * predictions_dict['CatBoost'] +
            ensemble_weights[3] * predictions_dict['LogisticRegression']
        )
        
        predictions = all_models['LightGBM'].predict(X)
    else:
        # Use best single model
        model = joblib.load(f'{models_dir}/default_prediction_model_advanced.pkl')
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
    
    # Add predictions to dataframe
    df['default_probability'] = probabilities
    df['predicted_default'] = predictions
    df['risk_level'] = df['default_probability'].apply(
        lambda x: 'High' if x > 0.5 else 'Medium' if x > 0.3 else 'Low'
    )
    
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Batch prediction from CSV file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions_advanced.csv'
        use_ensemble = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True
        
        print(f"Processing {input_file}...")
        results = predict_batch_advanced(input_file, use_ensemble=use_ensemble, output_file=output_file)
        print(f"Predictions saved to {output_file}")
        print(f"\nSummary:")
        print(results[['default_probability', 'predicted_default', 'risk_level']].describe())
    else:
        # Example single prediction
        print("Example: Predicting for a single customer...")
        example_data = {
            'customer_id': 10000,
            'age': 41,
            'annual_income': 61800,
            'credit_score': 696,
            'debt_to_income_ratio': 0.258,
            'loan_amount': 50000,
            'interest_rate': 5.5
        }
        result = predict_default_advanced(example_data)
        print(f"\nPrediction Result:")
        print(f"  Default Probability: {result['default_probability']:.4f}")
        print(f"  Predicted Default: {result['predicted_default']}")
        print(f"  Risk Level: {result['risk_level']}")

