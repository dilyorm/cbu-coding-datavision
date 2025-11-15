"""Data merging and aggregation functions"""
import pandas as pd
from .loaders import (
    load_geographic_data,
    load_financial_ratios,
    load_demographics,
    load_application_metadata,
    load_loan_details,
    load_credit_history,
)
from src.features.engineering import aggregate_credit_features


def merge_all_data(paths) -> pd.DataFrame:
    """Load and merge all datasets with proper aggregations"""
    print("="*60)
    print("STEP 1: Loading all data files...")
    print("="*60)
    
    geo_df = load_geographic_data(paths.geographic_data)
    financial_df = load_financial_ratios(paths.financial_ratios)
    demo_df = load_demographics(paths.demographics)
    app_df = load_application_metadata(paths.application_metadata)
    loan_df = load_loan_details(paths.loan_details)
    credit_df = load_credit_history(paths.credit_history)
    
    print(f"Geographic data: {geo_df.shape}")
    print(f"Financial ratios: {financial_df.shape}")
    print(f"Demographics: {demo_df.shape}")
    print(f"Application metadata: {app_df.shape}")
    print(f"Loan details: {loan_df.shape}")
    print(f"Credit history: {credit_df.shape}")
    
    print("\n" + "="*60)
    print("STEP 2: Aggregating credit history features...")
    print("="*60)
    credit_agg = aggregate_credit_features(credit_df)
    print(f"Aggregated credit features: {credit_agg.shape}")
    
    print("\n" + "="*60)
    print("STEP 3: Merging datasets...")
    print("="*60)
    
    # Start with application metadata (has target)
    df = app_df.copy()
    
    # Merge with loan details
    df = df.merge(loan_df, on='customer_id', how='left')
    
    # Merge with demographics
    df = df.merge(demo_df, on='customer_id', how='left')
    
    # Merge with financial ratios
    df = df.merge(financial_df, on='customer_id', how='left')
    
    # Merge with aggregated credit history
    if not credit_agg.empty:
        df = df.merge(credit_agg, on='customer_id', how='left')
    
    # Merge with geographic data
    df = df.merge(geo_df, on='customer_id', how='left')
    
    print(f"Final merged dataset shape: {df.shape}")
    return df

