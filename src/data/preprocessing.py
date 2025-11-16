import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

# Import advanced preprocessing
try:
    from src.data.preprocessing_advanced import (
        fit_advanced_preprocessing_pipeline,
        save_preprocessing_pipeline as save_advanced_pipeline,
        load_preprocessing_pipeline as load_advanced_pipeline,
        get_feature_names_advanced
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False


def get_numeric_categorical_features(df, target_col='default'):
    """Identify numeric and categorical feature columns."""
    # Exclude target and ID columns
    exclude_cols = [target_col, 'customer_id', 'application_id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate numeric and categorical
    numeric_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in [np.int64, np.float64]:
            # Check if it's actually categorical (low cardinality integers)
            if df[col].dtype == np.int64:
                unique_vals = df[col].nunique()
                if unique_vals <= 20 and df[col].min() >= 0:
                    # Likely categorical, but keep as numeric if it makes sense
                    # Check if values are sequential starting from 0/1
                    if unique_vals == df[col].max() - df[col].min() + 1:
                        # Sequential integers - could be categorical
                        # For now, treat as numeric unless explicitly categorical
                        numeric_features.append(col)
                    else:
                        numeric_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    return numeric_features, categorical_features


def create_preprocessing_pipeline(numeric_features, categorical_features):
    """Create sklearn preprocessing pipeline."""
    # Numeric pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def fit_preprocessing_pipeline(X_train, numeric_features, categorical_features, y_train=None, use_advanced=True):
    """Fit preprocessing pipeline on training data.
    
    Args:
        X_train: Training features
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        y_train: Target variable (required for advanced encoding with target encoding)
        use_advanced: If True, use advanced encoding strategies (ordinal, target, frequency)
                     If False, use simple one-hot encoding for all categoricals
    
    Returns:
        Fitted preprocessor
    """
    # Use advanced preprocessing if available and requested
    if use_advanced and ADVANCED_AVAILABLE and y_train is not None:
        print("Using advanced preprocessing with multiple encoding strategies...")
        return fit_advanced_preprocessing_pipeline(X_train, y_train, numeric_features, categorical_features)
    
    # Fallback to simple one-hot encoding
    print("Using simple preprocessing with one-hot encoding...")
    # Ensure all categorical columns are strings (OneHotEncoder requires uniform types)
    X_train_clean = X_train.copy()
    for col in categorical_features:
        if col in X_train_clean.columns:
            X_train_clean[col] = X_train_clean[col].astype(str)
    
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    preprocessor.fit(X_train_clean)
    return preprocessor


def save_preprocessing_pipeline(preprocessor, file_path='data/processed/preprocessing_pipeline.pkl'):
    """Save fitted preprocessing pipeline."""
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(preprocessor, f)


def load_preprocessing_pipeline(file_path='data/processed/preprocessing_pipeline.pkl'):
    """Load fitted preprocessing pipeline."""
    with open(file_path, 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor


def get_feature_names(preprocessor, numeric_features, categorical_features, X_train):
    """Get feature names after preprocessing."""
    # Check if this is an advanced preprocessor
    if hasattr(preprocessor, '_feature_metadata') and ADVANCED_AVAILABLE:
        try:
            feature_names = get_feature_names_advanced(preprocessor)
            if feature_names is not None:
                return feature_names
        except Exception:
            pass  # Fall through to standard method
    
    # Use sklearn's get_feature_names_out if available (sklearn >= 1.0)
    try:
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except (AttributeError, Exception):
        # Fallback for older sklearn versions or if get_feature_names_out fails
        # Get numeric feature names
        numeric_names = list(numeric_features)
        
        # Get categorical feature names from onehot encoder
        categorical_names = []
        if categorical_features and hasattr(X_train, 'columns'):
            try:
                cat_transformer = preprocessor.named_transformers_.get('cat')
                if cat_transformer is not None:
                    onehot = cat_transformer.named_steps.get('onehot')
                    if onehot is not None:
                        for i, col in enumerate(categorical_features):
                            if col in X_train.columns and i < len(onehot.categories_):
                                categories = onehot.categories_[i]
                                for cat in categories[1:]:  # Skip first due to drop='first'
                                    categorical_names.append(f'{col}_{cat}')
            except (KeyError, AttributeError, IndexError):
                pass  # If we can't extract names, just use numeric features
        
        # Combine all feature names
        all_feature_names = numeric_names + categorical_names
        return all_feature_names

