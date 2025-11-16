import pandas as pd
import numpy as np


def create_temporal_features(df):
    """Create temporal features from date/time columns."""
    df = df.copy()
    
    # Account age
    if 'account_open_year' in df.columns:
        current_year = 2025
        df['account_age'] = current_year - df['account_open_year']
    
    # Cyclical encoding for application_hour
    if 'application_hour' in df.columns:
        df['application_hour_sin'] = np.sin(2 * np.pi * df['application_hour'] / 24)
        df['application_hour_cos'] = np.cos(2 * np.pi * df['application_hour'] / 24)
    
    # Weekend indicator
    if 'application_day_of_week' in df.columns:
        df['is_weekend'] = (df['application_day_of_week'] >= 5).astype(int)
    
    return df


def create_interaction_features(df):
    """Create interaction and ratio features."""
    df = df.copy()
    
    # Income to regional median ratio
    if 'annual_income' in df.columns and 'regional_median_income' in df.columns:
        df['income_to_regional_median_ratio'] = (
            df['annual_income'] / df['regional_median_income']
        )
        # Handle division by zero
        df['income_to_regional_median_ratio'] = df['income_to_regional_median_ratio'].replace(
            [np.inf, -np.inf], np.nan
        )
    
    # Debt to credit limit ratio
    if 'existing_monthly_debt' in df.columns and 'total_credit_limit' in df.columns:
        df['debt_to_credit_limit_ratio'] = (
            df['existing_monthly_debt'] / df['total_credit_limit']
        )
        df['debt_to_credit_limit_ratio'] = df['debt_to_credit_limit_ratio'].replace(
            [np.inf, -np.inf], np.nan
        )
    
    # Credit score bins
    if 'credit_score' in df.columns:
        df['credit_score_bin'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 850],
            labels=['poor', 'fair', 'good', 'excellent']
        )
        # Convert to string for categorical encoding
        df['credit_score_bin'] = df['credit_score_bin'].astype(str)
        # Replace NaN strings
        df['credit_score_bin'] = df['credit_score_bin'].replace('nan', np.nan)
    
    return df


def create_aggregation_features(df):
    """Create aggregated features."""
    df = df.copy()
    
    # Total inquiries
    if 'num_inquiries_6mo' in df.columns and 'recent_inquiry_count' in df.columns:
        df['total_inquiries'] = df['num_inquiries_6mo'] + df['recent_inquiry_count']
    
    # Engagement score
    engagement_cols = []
    if 'num_login_sessions' in df.columns:
        engagement_cols.append('num_login_sessions')
    if 'has_mobile_app' in df.columns:
        engagement_cols.append('has_mobile_app')
    if 'paperless_billing' in df.columns:
        engagement_cols.append('paperless_billing')
    
    if engagement_cols:
        df['engagement_score'] = df[engagement_cols].sum(axis=1)
    
    return df


def create_geographic_features(df):
    """Create geographic risk features."""
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


def create_loan_features(df):
    """Create loan-specific features."""
    df = df.copy()
    
    # Revolving credit indicator
    if 'loan_term' in df.columns:
        df['is_revolving_credit'] = (df['loan_term'] == 0).astype(int)
    
    # High LTV loan
    if 'loan_to_value_ratio' in df.columns:
        df['high_ltv_loan'] = (df['loan_to_value_ratio'] > 0.8).astype(int)
    
    return df


def create_rank_features(df, train_stats=None):
    """Create percentile rank features for numeric columns.
    
    Args:
        df: DataFrame to create ranks for
        train_stats: Dict with 'mean' and 'std' for each column (from training set)
                    If None, computes from df itself (use only for training set)
    
    Returns:
        DataFrame with rank features added
    """
    df = df.copy()
    
    # Columns to create rank features for
    rank_columns = [
        'credit_score', 'annual_income', 'monthly_income', 
        'debt_to_income_ratio', 'credit_utilization', 'account_age',
        'loan_amount', 'total_debt_amount', 'total_credit_limit',
        'revolving_balance', 'available_credit', 'monthly_payment',
        'payment_to_income_ratio', 'loan_to_annual_income'
    ]
    
    # Overall percentile ranks
    for col in rank_columns:
        if col in df.columns:
            # Use percentileofscore for percentile ranking (0-100 scale, normalize to 0-1)
            valid_mask = df[col].notna()
            if valid_mask.sum() > 0:
                ranks = np.full(len(df), np.nan)
                ranks[valid_mask] = pd.Series(df.loc[valid_mask, col]).rank(pct=True).values
                df[f'{col}_rank_pct'] = ranks
    
    # Group-specific ranks (within state, employment_type, etc.)
    if 'state' in df.columns:
        for col in ['annual_income', 'credit_score', 'debt_to_income_ratio']:
            if col in df.columns:
                df[f'{col}_rank_pct_by_state'] = df.groupby('state')[col].transform(
                    lambda x: x.rank(pct=True) if x.notna().sum() > 0 else x
                )
    
    if 'employment_type' in df.columns:
        for col in ['annual_income', 'debt_to_income_ratio']:
            if col in df.columns:
                df[f'{col}_rank_pct_by_employment'] = df.groupby('employment_type')[col].transform(
                    lambda x: x.rank(pct=True) if x.notna().sum() > 0 else x
                )
    
    return df


def create_advanced_ratio_features(df):
    """Create advanced ratio and interaction features."""
    df = df.copy()
    
    # Regional cost-adjusted income
    if all(col in df.columns for col in ['annual_income', 'regional_median_income', 'cost_of_living_index']):
        cost_adjusted_denom = df['regional_median_income'] * (df['cost_of_living_index'] / 100)
        df['income_to_regional_cost_adjusted'] = df['annual_income'] / cost_adjusted_denom.replace(0, np.nan)
        df['income_to_regional_cost_adjusted'] = df['income_to_regional_cost_adjusted'].replace(
            [np.inf, -np.inf], np.nan
        )
    
    # Available credit to income ratio
    if all(col in df.columns for col in ['available_credit', 'annual_income']):
        df['available_credit_to_income_ratio'] = (
            df['available_credit'] / (df['annual_income'] / 12).replace(0, np.nan)
        )
        df['available_credit_to_income_ratio'] = df['available_credit_to_income_ratio'].replace(
            [np.inf, -np.inf], np.nan
        )
    
    # Payment to available credit ratio
    if all(col in df.columns for col in ['monthly_payment', 'available_credit']):
        df['payment_to_available_credit_ratio'] = (
            df['monthly_payment'] / df['available_credit'].replace(0, np.nan)
        )
        df['payment_to_available_credit_ratio'] = df['payment_to_available_credit_ratio'].replace(
            [np.inf, -np.inf], np.nan
        )
    
    # Debt burden score (composite of multiple debt ratios)
    debt_components = []
    if 'debt_to_income_ratio' in df.columns:
        debt_components.append(df['debt_to_income_ratio'])
    if 'debt_service_ratio' in df.columns:
        debt_components.append(df['debt_service_ratio'])
    if 'payment_to_income_ratio' in df.columns:
        debt_components.append(df['payment_to_income_ratio'])
    if 'credit_utilization' in df.columns:
        debt_components.append(df['credit_utilization'])
    
    if debt_components:
        # Normalize each component to 0-1 scale, then average
        normalized_components = []
        for comp in debt_components:
            valid_mask = comp.notna()
            if valid_mask.sum() > 0:
                comp_min = comp[valid_mask].min()
                comp_max = comp[valid_mask].max()
                if comp_max > comp_min:
                    normalized = (comp - comp_min) / (comp_max - comp_min)
                    normalized_components.append(normalized)
        
        if normalized_components:
            df['debt_burden_score'] = pd.concat(normalized_components, axis=1).mean(axis=1)
    
    # Credit depth score (composite of account age and diversity)
    depth_components = []
    if 'account_age' in df.columns:
        depth_components.append(df['account_age'])
    if 'account_diversity_index' in df.columns:
        depth_components.append(df['account_diversity_index'])
    if 'oldest_account_age_months' in df.columns:
        depth_components.append(df['oldest_account_age_months'])
    
    if depth_components:
        normalized_components = []
        for comp in depth_components:
            valid_mask = comp.notna()
            if valid_mask.sum() > 0:
                comp_min = comp[valid_mask].min()
                comp_max = comp[valid_mask].max()
                if comp_max > comp_min:
                    normalized = (comp - comp_min) / (comp_max - comp_min)
                    normalized_components.append(normalized)
        
        if normalized_components:
            df['credit_depth_score'] = pd.concat(normalized_components, axis=1).mean(axis=1)
    
    return df


def create_binned_categorical_features(df, train_value_counts=None):
    """Bin high-cardinality categoricals by frequency.
    
    Args:
        df: DataFrame to bin
        train_value_counts: Dict of {col: value_counts} from training set
                          If None, computes from df (use only for training set)
    
    Returns:
        DataFrame with binned categorical features added
    """
    df = df.copy()
    
    # Binning strategy: keep top N frequent values, bin rest as "OTHER"
    binning_config = {
        'loan_officer_id': {'top_n': 20, 'other_label': 'OTHER_OFFICER'},
        'marketing_campaign': {'top_n': 10, 'other_label': 'OTHER_CAMPAIGN'}
    }
    
    for col, config in binning_config.items():
        if col in df.columns:
            if train_value_counts is not None and col in train_value_counts:
                # Use training set value counts
                top_values = set(train_value_counts[col].head(config['top_n']).index)
            else:
                # Compute from current df (training set only)
                value_counts = df[col].value_counts()
                top_values = set(value_counts.head(config['top_n']).index)
            
            # Create binned column - ensure string type
            df[f'{col}_binned'] = df[col].apply(
                lambda x: str(x) if pd.notna(x) and x in top_values else config['other_label']
            )
            # Ensure it's string type
            df[f'{col}_binned'] = df[f'{col}_binned'].astype(str)
    
    return df


def create_population_statistics(df, train_stats=None):
    """Create z-scores and population comparison features.
    
    Args:
        df: DataFrame to create statistics for
        train_stats: Dict with 'mean' and 'std' for each column (from training set)
                    If None, computes from df itself (use only for training set)
    
    Returns:
        DataFrame with population statistics features added
    """
    df = df.copy()
    
    # Columns for z-scores
    zscore_columns = [
        'credit_score', 'annual_income', 'monthly_income',
        'debt_to_income_ratio', 'credit_utilization', 'account_age',
        'loan_amount', 'total_debt_amount'
    ]
    
    # Compute or use provided statistics
    if train_stats is None:
        train_stats = {}
        for col in zscore_columns:
            if col in df.columns:
                train_stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
    
    # Create z-scores
    for col in zscore_columns:
        if col in df.columns and col in train_stats:
            mean_val = train_stats[col]['mean']
            std_val = train_stats[col]['std']
            if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
                df[f'{col}_zscore'] = df[f'{col}_zscore'].replace([np.inf, -np.inf], np.nan)
    
    # Create quartile bins
    quartile_columns = [
        'credit_score', 'annual_income', 'debt_to_income_ratio',
        'credit_utilization', 'account_age'
    ]
    
    for col in quartile_columns:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                quartiles = valid_values.quantile([0.25, 0.5, 0.75])
                df[f'{col}_quartile'] = pd.cut(
                    df[col],
                    bins=[-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                    labels=['Q1', 'Q2', 'Q3', 'Q4']
                )
                df[f'{col}_quartile'] = df[f'{col}_quartile'].astype(str)
                df[f'{col}_quartile'] = df[f'{col}_quartile'].replace('nan', np.nan)
    
    # Distance from group medians
    if 'state' in df.columns:
        for col in ['annual_income', 'credit_score']:
            if col in df.columns:
                state_medians = df.groupby('state')[col].transform('median')
                df[f'{col}_dist_from_state_median'] = df[col] - state_medians
    
    if 'employment_type' in df.columns:
        for col in ['annual_income', 'debt_to_income_ratio']:
            if col in df.columns:
                emp_medians = df.groupby('employment_type')[col].transform('median')
                df[f'{col}_dist_from_emp_median'] = df[col] - emp_medians
    
    return df


def engineer_features(df, train_stats=None, train_value_counts=None):
    """Main feature engineering function that applies all feature creation steps.
    
    Args:
        df: DataFrame to engineer features for
        train_stats: Dict with statistics from training set (for rank/zscore features)
        train_value_counts: Dict with value_counts from training set (for consistency, not used)
    
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Original feature engineering
    df = create_temporal_features(df)
    df = create_interaction_features(df)
    df = create_aggregation_features(df)
    df = create_geographic_features(df)
    df = create_loan_features(df)
    
    # Enhanced feature engineering
    df = create_rank_features(df, train_stats)
    df = create_advanced_ratio_features(df)
    # Removed create_binned_categorical_features to preserve essential values
    df = create_population_statistics(df, train_stats)
    
    return df

