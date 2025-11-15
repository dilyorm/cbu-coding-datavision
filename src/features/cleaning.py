"""Data cleaning functions"""
import pandas as pd
import numpy as np
import re


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


def normalize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical values to merge similar variations
    
    This function handles common variations in categorical columns:
    - loan_type: cc, credit card, creditcard -> credit_card
    - employment_type: full-time, fulltime, full time -> full_time
    - marital_status: married, m -> married; single, s -> single
    - education: variations with different cases/spacing
    
    Args:
        df: DataFrame to normalize
        
    Returns:
        DataFrame with normalized categorical values
    """
    df_norm = df.copy()
    
    # Normalize loan_type
    if 'loan_type' in df_norm.columns:
        df_norm['loan_type'] = df_norm['loan_type'].astype(str).str.strip().str.lower()
        # Merge credit card variations: cc, credit card, creditcard -> credit_card
        df_norm['loan_type'] = df_norm['loan_type'].replace({
            'cc': 'credit_card',
            'credit card': 'credit_card',
            'creditcard': 'credit_card',
            'credit-card': 'credit_card'
        })
        # Merge personal loan variations
        df_norm['loan_type'] = df_norm['loan_type'].replace({
            'personal loan': 'personal',
            'personal_loan': 'personal',
            'personal-loan': 'personal'
        })
        # Merge auto loan variations
        df_norm['loan_type'] = df_norm['loan_type'].replace({
            'auto loan': 'auto',
            'auto_loan': 'auto',
            'auto-loan': 'auto',
            'car loan': 'auto',
            'car_loan': 'auto'
        })
        # Merge mortgage variations
        df_norm['loan_type'] = df_norm['loan_type'].replace({
            'mortgage': 'mortgage',
            'home loan': 'mortgage',
            'home_loan': 'mortgage',
            'home-loan': 'mortgage',
            'housing loan': 'mortgage'
        })
    
    # Normalize employment_type
    if 'employment_type' in df_norm.columns:
        df_norm['employment_type'] = df_norm['employment_type'].astype(str).str.strip().str.lower()
        # Merge full-time variations
        df_norm['employment_type'] = df_norm['employment_type'].replace({
            'full-time': 'full_time',
            'fulltime': 'full_time',
            'full time': 'full_time',
            'ft': 'full_time',
            'full': 'full_time'
        })
        # Merge part-time variations
        df_norm['employment_type'] = df_norm['employment_type'].replace({
            'part-time': 'part_time',
            'parttime': 'part_time',
            'part time': 'part_time',
            'pt': 'part_time',
            'part': 'part_time'
        })
        # Merge self-employed variations
        df_norm['employment_type'] = df_norm['employment_type'].replace({
            'self-employed': 'self_employed',
            'selfemployed': 'self_employed',
            'self employed': 'self_employed',
            'self emp': 'self_employed',
        })
        # Merge contractor variations (constructor, contract, etc.)
        df_norm['employment_type'] = df_norm['employment_type'].replace({
            'contractor': 'contractor',
            'constructor': 'contractor',
            'contract': 'contractor',
            'contructor': 'contractor',  # typo variation
            'contruct': 'contractor',  # typo variation
            'independent contractor': 'contractor',
            'ind contractor': 'contractor'
        })
        # Merge unemployed variations
        df_norm['employment_type'] = df_norm['employment_type'].replace({
            'unemployed': 'unemployed',
            'unemp': 'unemployed',
            'no job': 'unemployed',
            'not employed': 'unemployed'
        })
    
    # Normalize marital_status
    if 'marital_status' in df_norm.columns:
        df_norm['marital_status'] = df_norm['marital_status'].astype(str).str.strip().str.lower()
        df_norm['marital_status'] = df_norm['marital_status'].replace({
            'm': 'married',
            'married': 'married',
            'mar': 'married',
            's': 'single',
            'single': 'single',
            'sing': 'single',
            'divorced': 'divorced',
            'div': 'divorced',
            'd': 'divorced',
            'widowed': 'widowed',
            'wid': 'widowed',
            'w': 'widowed',
            'separated': 'separated',
            'sep': 'separated'
        })
    
    # Normalize education
    if 'education' in df_norm.columns:
        df_norm['education'] = df_norm['education'].astype(str).str.strip()
        # Normalize to title case and merge variations
        df_norm['education'] = df_norm['education'].str.title()
        df_norm['education'] = df_norm['education'].replace({
            'High School': 'High_School',
            'Highschool': 'High_School',
            'High School Diploma': 'High_School',
            'Hs': 'High_School',
            'Bachelor': 'Bachelor',
            'Bachelors': 'Bachelor',
            'Bachelor Degree': 'Bachelor',
            'Bs': 'Bachelor',
            'Ba': 'Bachelor',
            'Master': 'Master',
            'Masters': 'Master',
            'Master Degree': 'Master',
            'Ms': 'Master',
            'Ma': 'Master',
            'Doctorate': 'Doctorate',
            'Phd': 'Doctorate',
            'Ph.D': 'Doctorate',
            'Ph.D.': 'Doctorate',
            'Associate': 'Associate',
            'Associates': 'Associate',
            'Associate Degree': 'Associate',
            'Some College': 'Some_College',
            'Some College': 'Some_College',
            'College': 'Some_College'
        })
    
    return df_norm


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Systematic data cleaning with categorical normalization"""
    print("\n" + "="*60)
    print("STEP 4: Data Cleaning...")
    print("="*60)
    
    df_clean = df.copy()
    
    # Normalize categorical values (merge similar variations)
    print("Normalizing categorical values...")
    df_clean = normalize_categorical_values(df_clean)
    
    # Remove duplicates
    initial_rows = len(df_clean)
    if 'customer_id' in df_clean.columns and 'application_id' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['customer_id', 'application_id'], keep='first')
    elif 'customer_id' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['customer_id'], keep='first')
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

