"""Feature engineering functions"""
import pandas as pd
import numpy as np


def aggregate_credit_features(credit_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit history features per customer"""
    if credit_df.empty:
        return pd.DataFrame()
    
    # Create aggregated features
    agg_dict = {
        'credit_score': ['mean', 'min', 'max'],
        'num_credit_accounts': ['sum', 'mean', 'max'],
        'oldest_credit_line_age': ['max', 'mean'],
        'oldest_account_age_months': ['max', 'mean'],
        'total_credit_limit': ['sum', 'mean', 'max'],
        'num_delinquencies_2yrs': ['sum', 'max', 'mean'],
        'num_inquiries_6mo': ['sum', 'max', 'mean'],
        'recent_inquiry_count': ['sum', 'max', 'mean'],
        'num_public_records': ['sum', 'max'],
        'num_collections': ['sum', 'max'],
        'account_diversity_index': ['mean', 'max', 'min']
    }
    
    # Filter to columns that exist
    available_cols = {k: v for k, v in agg_dict.items() if k in credit_df.columns}
    
    if not available_cols:
        return pd.DataFrame()
    
    credit_agg = credit_df.groupby('customer_id').agg(available_cols)
    
    # Flatten MultiIndex columns
    credit_agg.columns = ['credit_' + '_'.join(col).strip('_') for col in credit_agg.columns.values]
    credit_agg = credit_agg.reset_index()
    
    # Add ratio features
    if 'credit_total_credit_limit_sum' in credit_agg.columns and 'credit_num_credit_accounts_sum' in credit_agg.columns:
        credit_agg['credit_avg_limit_per_account'] = (
            credit_agg['credit_total_credit_limit_sum'] / 
            (credit_agg['credit_num_credit_accounts_sum'] + 1)
        )
    
    # Add flags
    if 'credit_num_delinquencies_2yrs_sum' in credit_agg.columns:
        credit_agg['credit_ever_delinquent'] = (credit_agg['credit_num_delinquencies_2yrs_sum'] > 0).astype(int)
    
    if 'credit_num_collections_sum' in credit_agg.columns:
        credit_agg['credit_ever_collection'] = (credit_agg['credit_num_collections_sum'] > 0).astype(int)
    
    return credit_agg


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering"""
    print("\n" + "="*60)
    print("STEP 5: Feature Engineering...")
    print("="*60)
    
    df_feat = df.copy()
    
    # 1. Financial ratios
    if 'annual_income' in df_feat.columns and 'loan_amount' in df_feat.columns:
        df_feat['loan_to_annual_income_ratio'] = df_feat['loan_amount'] / (df_feat['annual_income'] + 1e-3)
        df_feat['income_to_loan_ratio'] = df_feat['annual_income'] / (df_feat['loan_amount'] + 1e-3)
    
    if 'monthly_income' in df_feat.columns and 'loan_amount' in df_feat.columns:
        df_feat['loan_to_monthly_income_ratio'] = df_feat['loan_amount'] / (df_feat['monthly_income'] * 12 + 1e-3)
    
    if 'total_debt_amount' in df_feat.columns and 'annual_income' in df_feat.columns:
        df_feat['total_debt_to_income'] = df_feat['total_debt_amount'] / (df_feat['annual_income'] + 1e-3)
    
    if 'monthly_free_cash_flow' in df_feat.columns and 'monthly_payment' in df_feat.columns:
        df_feat['cash_flow_to_payment_ratio'] = df_feat['monthly_free_cash_flow'] / (df_feat['monthly_payment'] + 1e-3)
    
    # 2. Credit utilization features
    if 'revolving_balance' in df_feat.columns and 'available_credit' in df_feat.columns:
        df_feat['total_revolving_credit'] = df_feat['revolving_balance'] + df_feat['available_credit']
        df_feat['revolving_utilization'] = df_feat['revolving_balance'] / (df_feat['total_revolving_credit'] + 1e-3)
    
    if 'credit_total_credit_limit_sum' in df_feat.columns and 'revolving_balance' in df_feat.columns:
        df_feat['overall_credit_utilization'] = df_feat['revolving_balance'] / (df_feat['credit_total_credit_limit_sum'] + 1e-3)
    
    # 3. Age features
    if 'age' in df_feat.columns:
        df_feat['age_band'] = pd.cut(df_feat['age'], bins=[0, 25, 35, 45, 55, 65, 100], 
                                     labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], right=False)
        df_feat['age_band'] = df_feat['age_band'].astype(str)
        df_feat['is_young'] = (df_feat['age'] < 30).astype(int)
        df_feat['is_senior'] = (df_feat['age'] >= 55).astype(int)
    
    # 4. Credit score features
    if 'credit_score' in df_feat.columns:
        df_feat['credit_score_category'] = pd.cut(df_feat['credit_score'],
                                                  bins=[0, 580, 670, 740, 850],
                                                  labels=['Poor', 'Fair', 'Good', 'Excellent'])
        df_feat['credit_score_category'] = df_feat['credit_score_category'].astype(str)
        df_feat['credit_score_poor'] = (df_feat['credit_score'] < 580).astype(int)
        df_feat['credit_score_excellent'] = (df_feat['credit_score'] >= 740).astype(int)
    
    # 5. Employment features
    if 'employment_length' in df_feat.columns:
        df_feat['employment_stable'] = (df_feat['employment_length'] >= 5).astype(int)
        df_feat['employment_new'] = (df_feat['employment_length'] < 1).astype(int)
    
    # 6. Loan features
    if 'loan_term' in df_feat.columns and 'loan_amount' in df_feat.columns:
        df_feat['monthly_payment_estimate'] = df_feat['loan_amount'] / (df_feat['loan_term'] + 1e-3)
    
    if 'interest_rate' in df_feat.columns:
        df_feat['high_interest'] = (df_feat['interest_rate'] > 10).astype(int)
        df_feat['low_interest'] = (df_feat['interest_rate'] < 5).astype(int)
    
    # 7. Demographic flags
    if 'marital_status' in df_feat.columns:
        df_feat['is_married'] = (df_feat['marital_status'].str.lower() == 'married').astype(int)
    
    if 'num_dependents' in df_feat.columns:
        df_feat['has_dependents'] = (df_feat['num_dependents'] > 0).astype(int)
        df_feat['many_dependents'] = (df_feat['num_dependents'] >= 3).astype(int)
    
    # 8. Geographic features
    if 'regional_unemployment_rate' in df_feat.columns:
        df_feat['high_unemployment'] = (df_feat['regional_unemployment_rate'] > 6).astype(int)
    
    if 'regional_median_income' in df_feat.columns and 'annual_income' in df_feat.columns:
        df_feat['income_vs_regional'] = df_feat['annual_income'] / (df_feat['regional_median_income'] + 1e-3)
    
    # 9. Application features
    if 'application_hour' in df_feat.columns:
        df_feat['application_hour_bin'] = pd.cut(df_feat['application_hour'], 
                                                  bins=[0, 6, 12, 18, 24], 
                                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
        df_feat['application_hour_bin'] = df_feat['application_hour_bin'].astype(str)
    
    if 'num_customer_service_calls' in df_feat.columns:
        df_feat['frequent_service_calls'] = (df_feat['num_customer_service_calls'] > 3).astype(int)
    
    # 10. Log transforms for skewed positive numeric features (for tree models)
    log_transform_cols = ['loan_amount', 'annual_income', 'total_debt_amount', 
                         'monthly_income', 'revolving_balance']
    # Also check for credit_total_credit_limit_sum (aggregated feature)
    if 'credit_total_credit_limit_sum' in df_feat.columns:
        log_transform_cols.append('credit_total_credit_limit_sum')
    
    for col in log_transform_cols:
        if col in df_feat.columns:
            # Only apply log transform if values are positive
            if (df_feat[col] > 0).any():
                df_feat[f'log_{col}'] = np.log1p(df_feat[col].fillna(0))
    
    # 11. Bucketed categorical versions of important numeric fields (for CatBoost)
    # These will be treated as categoricals by CatBoost
    bucket_features = [
        ('loan_amount', 10),
        ('debt_to_income_ratio', 10),
        ('age', 10),
        ('credit_score', 10),
        ('annual_income', 10),
        ('total_debt_amount', 10),
        ('interest_rate', 5),
        ('employment_length', 5)
    ]
    
    for col, q in bucket_features:
        if col in df_feat.columns:
            try:
                # Use qcut to create quantile-based buckets
                df_feat[f'{col}_bucket'] = pd.qcut(
                    df_feat[col], 
                    q=q, 
                    duplicates='drop',
                    labels=False
                )
                # Convert to string so CatBoost treats it as categorical
                df_feat[f'{col}_bucket'] = df_feat[f'{col}_bucket'].astype(str)
                # Fill any NaN values from qcut
                df_feat[f'{col}_bucket'] = df_feat[f'{col}_bucket'].fillna('Unknown')
            except (ValueError, TypeError):
                # Skip if qcut fails (e.g., too many duplicates)
                pass
    
    print(f"Created {len(df_feat.columns)} features (from {len(df.columns)})")
    return df_feat

