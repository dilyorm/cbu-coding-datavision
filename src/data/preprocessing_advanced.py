"""Advanced preprocessing with multiple encoding strategies."""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from pathlib import Path


# Define ordinal feature mappings
ORDINAL_MAPPINGS = {
    'education': ['High School', 'Some College', 'Bachelor', 'Graduate', 'Advanced'],
    'credit_score_category': ['poor', 'fair', 'good', 'excellent'],
    'income_category': ['very_low', 'low', 'medium', 'high', 'very_high'],
    'loan_amount_category': ['small', 'medium', 'large', 'very_large', 'huge'],
    'credit_score_bin': ['poor', 'fair', 'good', 'excellent'],  # Alternative name
}


class TargetEncoder:
    """Target encoding with smoothing and regularization."""
    
    def __init__(self, smoothing=1.0, min_samples_leaf=1):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encodings = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """Fit target encoder on training data."""
        self.global_mean = y.mean()
        
        for col in X.columns:
            # Calculate target mean for each category
            agg = pd.DataFrame({
                'target': y,
                'category': X[col].astype(str)
            }).groupby('category')['target'].agg(['mean', 'count'])
            
            # Apply smoothing
            smoothing_factor = 1 / (1 + np.exp(-(agg['count'] - self.min_samples_leaf) / self.smoothing))
            agg['smoothed_mean'] = (
                self.global_mean * (1 - smoothing_factor) + 
                agg['mean'] * smoothing_factor
            )
            
            self.encodings[col] = agg['smoothed_mean'].to_dict()
        
        return self
    
    def transform(self, X):
        """Transform data using fitted encodings."""
        X_encoded = X.copy()
        
        for col in X.columns:
            if col in self.encodings:
                X_encoded[col] = X[col].astype(str).map(self.encodings[col]).fillna(self.global_mean)
            else:
                X_encoded[col] = self.global_mean
        
        return X_encoded
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class FrequencyEncoder:
    """Frequency/count encoding for categorical features."""
    
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.encodings = {}
        self.total_counts = {}
        
    def fit(self, X):
        """Fit frequency encoder on training data."""
        for col in X.columns:
            value_counts = X[col].astype(str).value_counts()
            self.encodings[col] = value_counts.to_dict()
            self.total_counts[col] = len(X) if self.normalize else 1
        
        return self
    
    def transform(self, X):
        """Transform data using fitted frequencies."""
        X_encoded = X.copy()
        
        for col in X.columns:
            if col in self.encodings:
                frequencies = X[col].astype(str).map(self.encodings[col]).fillna(0)
                if self.normalize:
                    frequencies = frequencies / self.total_counts[col]
                X_encoded[col] = frequencies
            else:
                X_encoded[col] = 0
        
        return X_encoded
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class PreFittedTargetEncoder(BaseEstimator, TransformerMixin):
    """Wrapper for pre-fitted target encoder that can be pickled."""
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.feature_names_ = None
    
    def set_feature_names(self, feature_names):
        """Set feature names after initialization."""
        self.feature_names_ = feature_names
        return self
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform data using pre-fitted encoder."""
        # Convert numpy array to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.feature_names_)
        return self.encoder.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is None:
            input_features = self.feature_names_ if self.feature_names_ else []
        return np.array([f'{col}' for col in input_features], dtype=object)


class PreFittedFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Wrapper for pre-fitted frequency encoder that can be pickled."""
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.feature_names_ = None
    
    def set_feature_names(self, feature_names):
        """Set feature names after initialization."""
        self.feature_names_ = feature_names
        return self
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform data using pre-fitted encoder."""
        # Convert numpy array to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X = pd.DataFrame(X, columns=self.feature_names_)
        return self.encoder.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        if input_features is None:
            input_features = self.feature_names_ if self.feature_names_ else []
        return np.array([f'{col}' for col in input_features], dtype=object)


def categorize_features(df, categorical_features):
    """Categorize features by encoding strategy.
    
    Returns:
        ordinal_features: List of ordinal features with their mappings
        onehot_features: List of nominal features for one-hot encoding (low cardinality)
        target_features: List of features for target encoding (medium cardinality)
        frequency_features: List of features for frequency encoding
    """
    ordinal_features = []
    onehot_features = []
    target_features = []
    frequency_features = []
    
    for col in categorical_features:
        if col not in df.columns:
            continue
            
        # Check if it's ordinal
        if col in ORDINAL_MAPPINGS or any(k in col for k in ['category', 'score_bin']):
            ordinal_features.append(col)
            continue
        
        # Get cardinality
        unique_count = df[col].nunique()
        
        # Low cardinality (<=10): one-hot encoding
        if unique_count <= 10:
            onehot_features.append(col)
        # Medium cardinality (11-30): target encoding or frequency
        elif unique_count <= 30:
            target_features.append(col)
        # High cardinality (>30): frequency encoding
        else:
            frequency_features.append(col)
    
    return ordinal_features, onehot_features, target_features, frequency_features


def get_ordinal_categories(col_name, df):
    """Get ordered categories for ordinal feature."""
    # Check if we have a predefined mapping
    for key, mapping in ORDINAL_MAPPINGS.items():
        if key in col_name.lower():
            # Filter to only categories that exist in the data
            existing_cats = set(df[col_name].astype(str).unique())
            ordered_cats = [cat for cat in mapping if cat in existing_cats]
            # Add any categories not in our mapping
            unmapped = sorted(existing_cats - set(ordered_cats))
            return ordered_cats + unmapped
    
    # If no mapping, return sorted unique values
    return sorted(df[col_name].astype(str).unique())


def create_advanced_preprocessing_pipeline(
    numeric_features, 
    ordinal_features,
    onehot_features, 
    target_features,
    frequency_features,
    df_train,
    y_train=None
):
    """Create preprocessing pipeline with multiple encoding strategies.
    
    Args:
        numeric_features: List of numeric feature names
        ordinal_features: List of ordinal feature names
        onehot_features: List of nominal features for one-hot encoding
        target_features: List of features for target encoding
        frequency_features: List of features for frequency encoding
        df_train: Training dataframe (for fitting encoders)
        y_train: Target variable (required for target encoding)
    
    Returns:
        ColumnTransformer with all encoders
    """
    transformers = []
    
    # 1. Numeric features
    if numeric_features:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    # 2. Ordinal features
    if ordinal_features:
        ordinal_transformers = []
        for col in ordinal_features:
            categories = [get_ordinal_categories(col, df_train)]
            ordinal_pipe = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(
                    categories=categories,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ))
            ])
            transformers.append((f'ord_{col}', ordinal_pipe, [col]))
    
    # 3. One-hot encoded features (low cardinality)
    if onehot_features:
        onehot_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='if_binary'  # Drop first category only if binary
            ))
        ])
        transformers.append(('cat_onehot', onehot_transformer, onehot_features))
    
    # 4. Target encoded features (medium cardinality)
    if target_features and y_train is not None:
        # Fit target encoder
        target_encoder = TargetEncoder(smoothing=10.0, min_samples_leaf=5)
        target_encoder.fit(df_train[target_features], y_train)
        
        # Create pipeline with pre-fitted target encoder (using module-level class)
        prefitted_target = PreFittedTargetEncoder(target_encoder)
        prefitted_target.set_feature_names(target_features)
        
        target_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_encode', prefitted_target)
        ])
        transformers.append(('cat_target', target_transformer, target_features))
    
    # 5. Frequency encoded features (high cardinality)
    if frequency_features:
        # Fit frequency encoder
        freq_encoder = FrequencyEncoder(normalize=True)
        freq_encoder.fit(df_train[frequency_features])
        
        # Create pipeline with pre-fitted frequency encoder (using module-level class)
        prefitted_freq = PreFittedFrequencyEncoder(freq_encoder)
        prefitted_freq.set_feature_names(frequency_features)
        
        freq_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('freq_encode', prefitted_freq)
        ])
        transformers.append(('cat_freq', freq_transformer, frequency_features))
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        verbose_feature_names_out=True
    )
    
    return preprocessor


def fit_advanced_preprocessing_pipeline(X_train, y_train, numeric_features, categorical_features):
    """Fit advanced preprocessing pipeline with multiple encoding strategies."""
    # Ensure categorical columns are strings
    X_train_clean = X_train.copy()
    for col in categorical_features:
        if col in X_train_clean.columns:
            X_train_clean[col] = X_train_clean[col].astype(str)
    
    # Categorize features by encoding strategy
    ordinal_features, onehot_features, target_features, frequency_features = \
        categorize_features(X_train_clean, categorical_features)
    
    print(f"\nEncoding strategy:")
    print(f"  Ordinal encoding: {len(ordinal_features)} features")
    if ordinal_features:
        print(f"    {ordinal_features}")
    print(f"  One-hot encoding: {len(onehot_features)} features")
    if onehot_features:
        print(f"    {onehot_features}")
    print(f"  Target encoding: {len(target_features)} features")
    if target_features:
        print(f"    {target_features}")
    print(f"  Frequency encoding: {len(frequency_features)} features")
    if frequency_features:
        print(f"    {frequency_features}")
    
    # Create and fit preprocessor
    preprocessor = create_advanced_preprocessing_pipeline(
        numeric_features,
        ordinal_features,
        onehot_features,
        target_features,
        frequency_features,
        X_train_clean,
        y_train
    )
    
    preprocessor.fit(X_train_clean)
    
    # Store metadata for feature name extraction
    preprocessor._feature_metadata = {
        'numeric': numeric_features,
        'ordinal': ordinal_features,
        'onehot': onehot_features,
        'target': target_features,
        'frequency': frequency_features
    }
    
    return preprocessor


def save_preprocessing_pipeline(preprocessor, file_path='data/processed/preprocessing_pipeline_advanced.pkl'):
    """Save fitted preprocessing pipeline."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessing pipeline to {file_path}")


def load_preprocessing_pipeline(file_path='data/processed/preprocessing_pipeline_advanced.pkl'):
    """Load fitted preprocessing pipeline."""
    with open(file_path, 'rb') as f:
        preprocessor = pickle.load(f)
    return preprocessor


def get_feature_names_advanced(preprocessor):
    """Get feature names after advanced preprocessing."""
    # First try sklearn's built-in method
    try:
        feature_names = preprocessor.get_feature_names_out()
        return list(feature_names)
    except Exception:
        # If that fails, manually construct from metadata
        pass
    
    # Manual construction from metadata
    if not hasattr(preprocessor, '_feature_metadata'):
        return None
    
    metadata = preprocessor._feature_metadata
    feature_names = []
    
    # 1. Numeric features (from 'num' transformer)
    if 'numeric' in metadata and metadata['numeric']:
        feature_names.extend([f'num__{col}' for col in metadata['numeric']])
    
    # 2. Ordinal features (from 'ord_*' transformers)
    if 'ordinal' in metadata and metadata['ordinal']:
        for col in metadata['ordinal']:
            feature_names.append(f'ord_{col}__{col}')
    
    # 3. One-hot encoded features (from 'cat_onehot' transformer)
    if 'onehot' in metadata and metadata['onehot']:
        try:
            onehot_transformer = preprocessor.named_transformers_.get('cat_onehot')
            if onehot_transformer is not None:
                onehot_encoder = onehot_transformer.named_steps.get('onehot')
                if onehot_encoder is not None:
                    for i, col in enumerate(metadata['onehot']):
                        if i < len(onehot_encoder.categories_):
                            categories = onehot_encoder.categories_[i]
                            # Handle drop='if_binary' - only drop first if binary
                            drop_first = len(categories) == 2
                            start_idx = 1 if drop_first else 0
                            for cat in categories[start_idx:]:
                                # Clean category name for feature name
                                cat_str = str(cat).replace(' ', '_').replace('/', '_')
                                feature_names.append(f'cat_onehot__{col}_{cat_str}')
        except (AttributeError, KeyError, IndexError) as e:
            # Fallback: estimate based on unique values (conservative estimate)
            # This is a fallback, actual count will be verified in prepare_data.py
            for col in metadata['onehot']:
                # Add placeholder - actual count will be determined from transform
                feature_names.append(f'cat_onehot__{col}_placeholder')
    
    # 4. Target encoded features (from 'cat_target' transformer)
    if 'target' in metadata and metadata['target']:
        for col in metadata['target']:
            feature_names.append(f'cat_target__{col}')
    
    # 5. Frequency encoded features (from 'cat_freq' transformer)
    if 'frequency' in metadata and metadata['frequency']:
        for col in metadata['frequency']:
            feature_names.append(f'cat_freq__{col}')
    
    # Verify the count matches actual transformation by checking output shape
    # This is more reliable than trying to count from transformers
    try:
        # Get actual output shape by transforming a small sample
        # We need the original data structure, so we'll check if we can get it from metadata
        # If feature count doesn't match, we'll use the actual transform output shape
        pass  # We'll handle this in the calling function if needed
    except:
        pass
    
    return feature_names if feature_names else None

