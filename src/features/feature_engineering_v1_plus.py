"""Feature Engineering V1+ (Enhanced): V1 with selective improvements."""
import pandas as pd
import numpy as np

# Import all v1 functions
from src.features.feature_engineering import (
    create_temporal_features,
    create_interaction_features,
    create_aggregation_features,
    create_geographic_features,
    create_loan_features,
    create_rank_features,
    create_advanced_ratio_features,
    create_population_statistics
)


def create_log_sqrt_transforms(df):
    """Create log and sqrt transformations for skewed features.
    
    Improvement from V2: Selective statistical transformations.
    """
    df = df.copy()
    
    # Log transforms for highly skewed features (right-skewed)
    log_features = [
        'loan_amount',
        'annual_income', 
        'total_debt_amount',
        'total_credit_limit',
        'monthly_income'
    ]
    
    for col in log_features:
        if col in df.columns:
            # Add 1 to handle zeros, use log1p for numerical stability
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0).fillna(0))
    
    # Sqrt transforms for moderately skewed features
    sqrt_features = [
        'debt_to_income_ratio',
        'credit_utilization',
        'num_credit_accounts',
        'oldest_credit_line_age',
        'employment_length'
    ]
    
    for col in sqrt_features:
        if col in df.columns:
            df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0).fillna(0))
    
    return df


def create_selective_group_features(df):
    """Create selective group statistics (only mean and distance).
    
    Improvement from V2: Selective aggregations to avoid feature explosion.
    Only creates the most valuable group statistics.
    """
    df = df.copy()
    
    # State-level patterns (major geographic patterns)
    if 'state' in df.columns:
        key_cols = ['credit_score', 'annual_income', 'debt_to_income_ratio']
        
        for col in key_cols:
            if col in df.columns:
                # Mean by state
                df[f'{col}_state_mean'] = df.groupby('state')[col].transform('mean')
                # Distance from state mean (relative position)
                df[f'{col}_dist_from_state_mean'] = df[col] - df[f'{col}_state_mean']
                # Normalize by state std (if not constant)
                state_std = df.groupby('state')[col].transform('std')
                df[f'{col}_state_zscore'] = np.where(
                    state_std > 0,
                    (df[col] - df[f'{col}_state_mean']) / state_std,
                    0
                )
    
    # Employment type patterns (occupation-based patterns)
    if 'employment_type' in df.columns:
        emp_cols = ['annual_income', 'credit_score', 'debt_to_income_ratio']
        
        for col in emp_cols:
            if col in df.columns:
                df[f'{col}_emp_mean'] = df.groupby('employment_type')[col].transform('mean')
                df[f'{col}_vs_emp_mean'] = df[col] - df[f'{col}_emp_mean']
    
    return df


def create_ordinal_categories(df):
    """Create ordinal categorical features.
    
    Improvement from V3: Income and loan amount categories.
    These work well with ordinal encoding.
    """
    df = df.copy()
    
    # Income categories (ordinal)
    if 'annual_income' in df.columns:
        df['income_category'] = pd.cut(
            df['annual_income'],
            bins=[0, 30000, 50000, 75000, 100000, np.inf],
            labels=['very_low', 'low', 'medium', 'high', 'very_high']
        ).astype(str)
        df['income_category'] = df['income_category'].replace('nan', np.nan)
    
    # Loan amount categories (ordinal)
    if 'loan_amount' in df.columns:
        df['loan_amount_category'] = pd.cut(
            df['loan_amount'],
            bins=[0, 10000, 25000, 50000, 100000, np.inf],
            labels=['small', 'medium', 'large', 'very_large', 'huge']
        ).astype(str)
        df['loan_amount_category'] = df['loan_amount_category'].replace('nan', np.nan)
    
    return df


def create_polynomial_features(df):
    """Create polynomial features for key numeric pairs.
    
    New improvement: Capture non-linear relationships.
    """
    df = df.copy()
    
    # Key pairs with known non-linear relationships
    pairs = [
        ('credit_score', 'debt_to_income_ratio'),  # Credit quality × debt burden
        ('annual_income', 'loan_amount'),           # Income × loan size
        ('credit_utilization', 'num_credit_accounts'),  # Utilization × account count
        ('age', 'employment_length'),               # Age × job stability
    ]
    
    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            # Squared terms
            df[f'{col1}_squared'] = df[col1] ** 2
            
            # Interaction term
            df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
    
    return df


def create_advanced_temporal_features(df):
    """Create advanced temporal features.
    
    New improvement: Better time-of-day and day-of-week patterns.
    """
    df = df.copy()
    
    # Time of day categories
    if 'application_hour' in df.columns:
        # Business hours indicator
        df['business_hours'] = (
            (df['application_hour'] >= 9) & 
            (df['application_hour'] <= 17)
        ).astype(int)
        
        # Late night application (risk indicator)
        df['late_night_application'] = (
            (df['application_hour'] >= 22) | 
            (df['application_hour'] <= 5)
        ).astype(int)
    
    # Day of week patterns
    if 'application_day_of_week' in df.columns:
        # Mid-week indicator (Tuesday-Thursday)
        df['mid_week'] = df['application_day_of_week'].isin([1, 2, 3]).astype(int)
        
        # Monday rush (beginning of week)
        df['is_monday'] = (df['application_day_of_week'] == 0).astype(int)
    
    # Account age patterns
    if 'account_age' in df.columns:
        df['new_account'] = (df['account_age'] <= 1).astype(int)
        df['mature_account'] = (df['account_age'] >= 5).astype(int)
        df['account_age_squared'] = df['account_age'] ** 2
    
    return df


def create_risk_scores(df):
    """Create composite risk score features.
    
    New improvement: Interpretable risk indicators.
    """
    df = df.copy()
    
    # Credit Risk Score (0-100 scale, higher = more risky)
    risk_components = []
    weights = []
    
    if 'credit_score' in df.columns:
        # Normalize and invert (higher credit score = lower risk)
        credit_risk = 1 - ((df['credit_score'] - 300) / (850 - 300)).clip(0, 1)
        risk_components.append(credit_risk)
        weights.append(0.30)  # 30% weight
    
    if 'debt_to_income_ratio' in df.columns:
        # Normalize DTI (cap at 2.0)
        dti_risk = (df['debt_to_income_ratio'] / 2.0).clip(0, 1)
        risk_components.append(dti_risk)
        weights.append(0.25)  # 25% weight
    
    if 'credit_utilization' in df.columns:
        risk_components.append(df['credit_utilization'].clip(0, 1))
        weights.append(0.20)  # 20% weight
    
    if 'num_delinquencies_2yrs' in df.columns:
        # Normalize delinquencies (cap at 2)
        delinq_risk = (df['num_delinquencies_2yrs'] / 2).clip(0, 1)
        risk_components.append(delinq_risk)
        weights.append(0.15)  # 15% weight
    
    if 'num_inquiries_6mo' in df.columns:
        # Normalize inquiries (cap at 5)
        inquiry_risk = (df['num_inquiries_6mo'] / 5).clip(0, 1)
        risk_components.append(inquiry_risk)
        weights.append(0.10)  # 10% weight
    
    # Calculate weighted risk score
    if risk_components:
        weights_array = np.array(weights)
        weights_array = weights_array / weights_array.sum()  # Normalize
        
        risk_matrix = np.column_stack([comp.fillna(0.5) for comp in risk_components])
        df['composite_risk_score'] = (risk_matrix * weights_array).sum(axis=1) * 100
        df['composite_risk_score'] = df['composite_risk_score'].round(2)
    
    # Financial Stability Score (0-100 scale, higher = more stable)
    stability_components = []
    
    if 'employment_length' in df.columns:
        stability_components.append((df['employment_length'] / 10).clip(0, 1))
    
    if 'oldest_credit_line_age' in df.columns:
        stability_components.append((df['oldest_credit_line_age'] / 20).clip(0, 1))
    
    if 'account_age' in df.columns:
        stability_components.append((df['account_age'] / 10).clip(0, 1))
    
    if stability_components:
        stability_matrix = np.column_stack([comp.fillna(0.5) for comp in stability_components])
        df['financial_stability_score'] = (stability_matrix.mean(axis=1) * 100).round(2)
    
    return df


def create_repayment_features(df):
    """Create enhanced repayment ability features.
    
    New improvement: More detailed payment capacity metrics.
    """
    df = df.copy()
    
    # Total monthly obligations
    if 'monthly_payment' in df.columns and 'existing_monthly_debt' in df.columns:
        df['total_monthly_obligations'] = df['monthly_payment'] + df['existing_monthly_debt']
        
        if 'monthly_income' in df.columns:
            # Total debt burden
            df['total_obligation_ratio'] = (
                df['total_monthly_obligations'] / 
                df['monthly_income'].replace(0, np.nan)
            )
            df['total_obligation_ratio'] = df['total_obligation_ratio'].replace(
                [np.inf, -np.inf], np.nan
            )
            
            # Free cash flow after all obligations
            df['free_cash_flow'] = df['monthly_income'] - df['total_monthly_obligations']
            df['free_cash_flow_pct'] = (
                df['free_cash_flow'] / df['monthly_income'].replace(0, np.nan) * 100
            )
            
            # Payment stress indicator
            df['high_payment_stress'] = (df['total_obligation_ratio'] > 0.5).astype(int)
    
    # Total cost of loan
    if 'loan_term' in df.columns and 'monthly_payment' in df.columns:
        df['total_loan_cost'] = df['monthly_payment'] * df['loan_term']
        
        if 'loan_amount' in df.columns:
            df['total_interest_paid'] = (df['total_loan_cost'] - df['loan_amount']).clip(lower=0)
            df['effective_interest_multiplier'] = (
                df['total_loan_cost'] / df['loan_amount'].replace(0, np.nan)
            )
            df['effective_interest_multiplier'] = df['effective_interest_multiplier'].replace(
                [np.inf, -np.inf], np.nan
            )
    
    return df


def engineer_features_v1_plus(df, train_stats=None, train_value_counts=None):
    """Main feature engineering function for V1+ (Enhanced).
    
    Combines all V1 features with selective improvements from V2, V3, and new features.
    
    Args:
        df: DataFrame to engineer features for
        train_stats: Dict with statistics from training set (for rank/zscore features)
        train_value_counts: Dict with value_counts from training set (for consistency, not used)
    
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    print("Applying V1+ feature engineering...")
    
    # V1 BASE FEATURES (6,000-7,000 features)
    print("  - V1 base features...")
    df = create_temporal_features(df)
    df = create_interaction_features(df)
    df = create_aggregation_features(df)
    df = create_geographic_features(df)
    df = create_loan_features(df)
    df = create_rank_features(df, train_stats)
    df = create_advanced_ratio_features(df)
    # Removed create_binned_categorical_features to preserve essential values
    df = create_population_statistics(df, train_stats)
    
    # ENHANCEMENT 1: Log/sqrt transforms (+50-80 features)
    print("  - Log/sqrt transforms...")
    df = create_log_sqrt_transforms(df)
    
    # ENHANCEMENT 2: Selective group features (+100-150 features)
    print("  - Selective group statistics...")
    df = create_selective_group_features(df)
    
    # ENHANCEMENT 3: Ordinal categories (+2 categorical features)
    print("  - Ordinal categories...")
    df = create_ordinal_categories(df)
    
    # ENHANCEMENT 4: Polynomial features (+30-50 features)
    print("  - Polynomial features...")
    df = create_polynomial_features(df)
    
    # ENHANCEMENT 5: Advanced temporal features (+8-12 features)
    print("  - Advanced temporal features...")
    df = create_advanced_temporal_features(df)
    
    # ENHANCEMENT 6: Risk scores (+4-6 features)
    print("  - Composite risk scores...")
    df = create_risk_scores(df)
    
    # ENHANCEMENT 7: Repayment features (+10-15 features)
    print("  - Enhanced repayment features...")
    df = create_repayment_features(df)
    
    print(f"V1+ feature engineering complete: {df.shape[1]} total columns")
    
    return df

