"""Model loading utilities for prediction"""
import joblib
import json
from catboost import CatBoostClassifier


def load_model_and_config(models_dir="models"):
    """Load trained CatBoost model and configuration
    
    Args:
        models_dir: Directory containing saved model artifacts
        
    Returns:
        model: Trained CatBoostClassifier
        feature_cols: List of feature column names
        cat_feature_indices: List of categorical feature indices
    """
    # Try loading from CatBoost native format first
    try:
        model = CatBoostClassifier()
        model.load_model(f'{models_dir}/default_prediction_model_advanced.cbm')
        print("Loaded model from CatBoost native format (.cbm)")
    except:
        # Fallback to joblib
        model = joblib.load(f'{models_dir}/default_prediction_model_advanced.pkl')
        print("Loaded model from joblib format (.pkl)")
    
    # Load configuration
    try:
        with open(f'{models_dir}/model_config_advanced.json', 'r') as f:
            config = json.load(f)
        feature_cols = config['feature_cols']
        cat_feature_indices = config['cat_feature_indices']
        print(f"Loaded configuration: {len(feature_cols)} features, {len(cat_feature_indices)} categorical")
    except:
        # Fallback to individual files
        feature_cols = joblib.load(f'{models_dir}/feature_columns_advanced.pkl')
        cat_feature_indices = joblib.load(f'{models_dir}/categorical_feature_indices_advanced.pkl')
        print(f"Loaded configuration from individual files")
    
    return model, feature_cols, cat_feature_indices

