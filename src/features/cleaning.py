"""Data cleaning functions"""
import pandas as pd
import numpy as np


def clean_currency(value):
    """Clean currency strings to float"""
    if pd.isna(value) or value == '':
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    value_str = str(value).replace('$', '').replace(',', '').strip()
    try:
        return float(value_str)
    except:
        return np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Systematic data cleaning"""
    print("\n" + "="*60)
    print("STEP 4: Data Cleaning...")
    print("="*60)
    
    df_clean = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=['customer_id', 'application_id'], keep='first')
    print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Sanity checks
    if 'loan_amount' in df_clean.columns:
        df_clean = df_clean[df_clean['loan_amount'] > 0]
    if 'interest_rate' in df_clean.columns:
        df_clean = df_clean[df_clean['interest_rate'].between(0, 100)]
    if 'age' in df_clean.columns:
        df_clean = df_clean[df_clean['age'].between(18, 100)]
    
    print(f"After sanity checks: {df_clean.shape[0]} rows")
    
    return df_clean

