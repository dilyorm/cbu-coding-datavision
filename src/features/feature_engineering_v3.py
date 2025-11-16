"""Feature engineering v3: Domain-specific simple features (minimal feature engineering)."""
import pandas as pd
import numpy as np


def create_domain_specific_features(df):
    """Create domain-specific financial features."""
    df = df.copy()
    
    # Loan-to-income ratio
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df['loan_to_income_ratio'] = (
            df['loan_amount'] / (df['annual_income'] / 12).replace(0, np.nan)
        )
        df['loan_to_income_ratio'] = df['loan_to_income_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Debt metrics
    if 'existing_monthly_debt' in df.columns and 'monthly_income' in df.columns:
        df['debt_to_income_ratio'] = (
            df['existing_monthly_debt'] / df['monthly_income'].replace(0, np.nan)
        )
        df['debt_to_income_ratio'] = df['debt_to_income_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Credit utilization
    if 'revolving_balance' in df.columns and 'total_credit_limit' in df.columns:
        df['credit_utilization'] = (
            df['revolving_balance'] / df['total_credit_limit'].replace(0, np.nan)
        )
        df['credit_utilization'] = df['credit_utilization'].replace([np.inf, -np.inf], np.nan)
    
    # Available credit
    if 'total_credit_limit' in df.columns and 'revolving_balance' in df.columns:
        df['available_credit'] = df['total_credit_limit'] - df['revolving_balance']
        df['available_credit'] = df['available_credit'].clip(lower=0)
    
    # Payment capacity
    if 'monthly_income' in df.columns and 'monthly_payment' in df.columns:
        df['payment_capacity'] = df['monthly_income'] - df['monthly_payment']
        df['payment_capacity_ratio'] = (
            df['payment_capacity'] / df['monthly_income'].replace(0, np.nan)
        )
        df['payment_capacity_ratio'] = df['payment_capacity_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Credit score categories (simple bins)
    if 'credit_score' in df.columns:
        df['credit_score_category'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 850],
            labels=['poor', 'fair', 'good', 'excellent']
        ).astype(str)
        df['credit_score_category'] = df['credit_score_category'].replace('nan', np.nan)
    
    # Income categories
    if 'annual_income' in df.columns:
        df['income_category'] = pd.cut(
            df['annual_income'],
            bins=[0, 30000, 50000, 75000, 100000, np.inf],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        ).astype(str)
        df['income_category'] = df['income_category'].replace('nan', np.nan)
    
    # Loan amount categories
    if 'loan_amount' in df.columns:
        df['loan_amount_category'] = pd.cut(
            df['loan_amount'],
            bins=[0, 10000, 25000, 50000, 100000, np.inf],
            labels=['small', 'medium', 'large', 'very_large', 'huge']
        ).astype(str)
        df['loan_amount_category'] = df['loan_amount_category'].replace('nan', np.nan)
    
    return df


def create_simple_temporal_features(df):
    """Create simple temporal features."""
    df = df.copy()
    
    # Account age
    if 'account_open_year' in df.columns:
        current_year = 2025
        df['account_age'] = current_year - df['account_open_year']
    
    # Weekend indicator
    if 'application_day_of_week' in df.columns:
        df['is_weekend'] = (df['application_day_of_week'] >= 5).astype(int)
    
    return df


def create_simple_geographic_features(df):
    """Create simple geographic features."""
    df = df.copy()
    
    # High unemployment region
    if 'regional_unemployment_rate' in df.columns:
        df['high_unemployment_region'] = (
            df['regional_unemployment_rate'] > 5.0
        ).astype(int)
    
    # High cost area
    if 'cost_of_living_index' in df.columns:
        df['high_cost_area'] = (df['cost_of_living_index'] > 100).astype(int)
    
    return df


def engineer_features_v3(df, train_stats=None, train_value_counts=None):
    """Main feature engineering function for v3 (simple, domain-specific approach).
    
    Args:
        df: DataFrame to engineer features for
        train_stats: Dict with statistics from training set (for consistency, not used in v3)
        train_value_counts: Dict with value_counts from training set (for consistency, not used in v3)
    
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Domain-specific financial features
    df = create_domain_specific_features(df)
    
    # Simple temporal features
    df = create_simple_temporal_features(df)
    
    # Simple geographic features
    df = create_simple_geographic_features(df)
    
    # Basic indicators
    if 'loan_term' in df.columns:
        df['is_revolving_credit'] = (df['loan_term'] == 0).astype(int)
    
    if 'loan_to_value_ratio' in df.columns:
        df['high_ltv_loan'] = (df['loan_to_value_ratio'] > 0.8).astype(int)
    
    return df

