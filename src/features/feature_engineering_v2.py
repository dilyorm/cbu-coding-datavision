"""Feature engineering v2: Aggregation-heavy approach with statistical transformations."""
import pandas as pd
import numpy as np


def create_aggregation_heavy_features(df):
    """Create extensive aggregation features grouped by various categories."""
    df = df.copy()
    
    # Group by state aggregations
    if 'state' in df.columns:
        numeric_cols = ['annual_income', 'credit_score', 'debt_to_income_ratio', 
                       'loan_amount', 'total_debt_amount', 'credit_utilization']
        
        for col in numeric_cols:
            if col in df.columns:
                # Mean, median, std, min, max by state
                df[f'{col}_state_mean'] = df.groupby('state')[col].transform('mean')
                df[f'{col}_state_median'] = df.groupby('state')[col].transform('median')
                df[f'{col}_state_std'] = df.groupby('state')[col].transform('std')
                df[f'{col}_state_min'] = df.groupby('state')[col].transform('min')
                df[f'{col}_state_max'] = df.groupby('state')[col].transform('max')
                
                # Percentile within state
                df[f'{col}_state_p25'] = df.groupby('state')[col].transform(lambda x: x.quantile(0.25))
                df[f'{col}_state_p75'] = df.groupby('state')[col].transform(lambda x: x.quantile(0.75))
                
                # Distance from state statistics
                df[f'{col}_dist_from_state_mean'] = df[col] - df[f'{col}_state_mean']
                df[f'{col}_dist_from_state_median'] = df[col] - df[f'{col}_state_median']
    
    # Group by employment type aggregations
    if 'employment_type' in df.columns:
        numeric_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score']
        
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_emp_mean'] = df.groupby('employment_type')[col].transform('mean')
                df[f'{col}_emp_median'] = df.groupby('employment_type')[col].transform('median')
                df[f'{col}_emp_std'] = df.groupby('employment_type')[col].transform('std')
                df[f'{col}_dist_from_emp_mean'] = df[col] - df[f'{col}_emp_mean']
    
    # Group by credit score bin aggregations
    if 'credit_score' in df.columns:
        df['credit_score_bin_v2'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 850],
            labels=['poor', 'fair', 'good', 'excellent']
        ).astype(str)
        df['credit_score_bin_v2'] = df['credit_score_bin_v2'].replace('nan', np.nan)
        
        if df['credit_score_bin_v2'].notna().sum() > 0:
            numeric_cols = ['annual_income', 'loan_amount', 'debt_to_income_ratio']
            for col in numeric_cols:
                if col in df.columns:
                    df[f'{col}_credit_bin_mean'] = df.groupby('credit_score_bin_v2')[col].transform('mean')
                    df[f'{col}_credit_bin_median'] = df.groupby('credit_score_bin_v2')[col].transform('median')
    
    # Group by loan amount bins
    if 'loan_amount' in df.columns:
        df['loan_amount_bin'] = pd.qcut(
            df['loan_amount'].rank(method='first'),
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        ).astype(str)
        df['loan_amount_bin'] = df['loan_amount_bin'].replace('nan', np.nan)
        
        if df['loan_amount_bin'].notna().sum() > 0:
            numeric_cols = ['credit_score', 'annual_income', 'debt_to_income_ratio']
            for col in numeric_cols:
                if col in df.columns:
                    df[f'{col}_loan_bin_mean'] = df.groupby('loan_amount_bin')[col].transform('mean')
    
    return df


def create_statistical_transformations(df):
    """Create log, sqrt, and power transformations of numeric features."""
    df = df.copy()
    
    # Columns suitable for transformations (positive values)
    transform_cols = [
        'annual_income', 'monthly_income', 'loan_amount', 'total_debt_amount',
        'total_credit_limit', 'revolving_balance', 'available_credit',
        'monthly_payment', 'credit_score'
    ]
    
    for col in transform_cols:
        if col in df.columns:
            # Log transformation (add small constant to handle zeros)
            df[f'{col}_log'] = np.log1p(df[col].fillna(0))
            
            # Square root transformation
            df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0).fillna(0))
            
            # Square transformation
            df[f'{col}_squared'] = (df[col].fillna(0) ** 2)
            
            # Cube root transformation
            df[f'{col}_cbrt'] = np.cbrt(df[col].clip(lower=0).fillna(0))
    
    return df


def create_aggressive_binning(df):
    """Create more aggressive binning for high-cardinality categoricals."""
    df = df.copy()
    
    # Binning numeric features into more bins
    if 'credit_score' in df.columns:
        df['credit_score_bin_10'] = pd.qcut(
            df['credit_score'].rank(method='first'),
            q=10,
            labels=[f'bin_{i}' for i in range(10)],
            duplicates='drop'
        ).astype(str)
        df['credit_score_bin_10'] = df['credit_score_bin_10'].replace('nan', np.nan)
    
    if 'annual_income' in df.columns:
        df['income_bin_10'] = pd.qcut(
            df['annual_income'].rank(method='first'),
            q=10,
            labels=[f'income_bin_{i}' for i in range(10)],
            duplicates='drop'
        ).astype(str)
        df['income_bin_10'] = df['income_bin_10'].replace('nan', np.nan)
    
    if 'debt_to_income_ratio' in df.columns:
        df['dti_bin_10'] = pd.qcut(
            df['debt_to_income_ratio'].rank(method='first'),
            q=10,
            labels=[f'dti_bin_{i}' for i in range(10)],
            duplicates='drop'
        ).astype(str)
        df['dti_bin_10'] = df['dti_bin_10'].replace('nan', np.nan)
    
    # Interaction bins
    if 'credit_score' in df.columns and 'annual_income' in df.columns:
        df['credit_income_interaction'] = (
            pd.qcut(df['credit_score'].rank(method='first'), q=5, duplicates='drop').astype(str) + '_' +
            pd.qcut(df['annual_income'].rank(method='first'), q=5, duplicates='drop').astype(str)
        )
        df['credit_income_interaction'] = df['credit_income_interaction'].replace('nan_nan', np.nan)
    
    return df


def create_categorical_interactions(df):
    """Create interaction features between categorical variables."""
    df = df.copy()
    
    # State x Employment type
    if 'state' in df.columns and 'employment_type' in df.columns:
        df['state_employment_interaction'] = (
            df['state'].astype(str) + '_' + df['employment_type'].astype(str)
        )
        df['state_employment_interaction'] = df['state_employment_interaction'].replace('nan_nan', np.nan)
    
    # State x Credit score bin
    if 'state' in df.columns and 'credit_score_bin_v2' in df.columns:
        df['state_credit_bin_interaction'] = (
            df['state'].astype(str) + '_' + df['credit_score_bin_v2'].astype(str)
        )
        df['state_credit_bin_interaction'] = df['state_credit_bin_interaction'].replace('nan_nan', np.nan)
    
    return df


def engineer_features_v2(df, train_stats=None, train_value_counts=None):
    """Main feature engineering function for v2 (aggregation-heavy approach).
    
    Args:
        df: DataFrame to engineer features for
        train_stats: Dict with statistics from training set (for consistency)
        train_value_counts: Dict with value_counts from training set (for consistency)
    
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Basic temporal features
    if 'account_open_year' in df.columns:
        current_year = 2025
        df['account_age'] = current_year - df['account_open_year']
    
    if 'application_hour' in df.columns:
        df['application_hour_sin'] = np.sin(2 * np.pi * df['application_hour'] / 24)
        df['application_hour_cos'] = np.cos(2 * np.pi * df['application_hour'] / 24)
    
    if 'application_day_of_week' in df.columns:
        df['is_weekend'] = (df['application_day_of_week'] >= 5).astype(int)
    
    # Aggregation-heavy features
    df = create_aggregation_heavy_features(df)
    
    # Statistical transformations
    df = create_statistical_transformations(df)
    
    # Removed aggressive binning to preserve essential values
    
    # Categorical interactions
    df = create_categorical_interactions(df)
    
    # Basic ratio features
    if 'annual_income' in df.columns and 'regional_median_income' in df.columns:
        df['income_to_regional_median_ratio'] = (
            df['annual_income'] / df['regional_median_income'].replace(0, np.nan)
        )
        df['income_to_regional_median_ratio'] = df['income_to_regional_median_ratio'].replace(
            [np.inf, -np.inf], np.nan
        )
    
    if 'existing_monthly_debt' in df.columns and 'total_credit_limit' in df.columns:
        df['debt_to_credit_limit_ratio'] = (
            df['existing_monthly_debt'] / df['total_credit_limit'].replace(0, np.nan)
        )
        df['debt_to_credit_limit_ratio'] = df['debt_to_credit_limit_ratio'].replace(
            [np.inf, -np.inf], np.nan
        )
    
    return df

