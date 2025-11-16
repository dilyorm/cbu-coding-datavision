import pandas as pd
import numpy as np
import re


def parse_currency(value):
    """Parse currency strings to float, handling $, commas, quotes, and various formats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    # Convert to string and clean
    value_str = str(value).strip()
    # Remove $, commas, quotes, spaces, parentheses (for negative numbers)
    value_str = re.sub(r'[\$,\'"\s\(\)]', '', value_str)
    # Handle negative sign
    is_negative = value_str.startswith('-')
    if is_negative:
        value_str = value_str[1:]
    try:
        result = float(value_str)
        return -result if is_negative else result
    except (ValueError, TypeError):
        return np.nan


def standardize_employment_type(value):
    """Standardize employment type variants to standard categories.
    
    Based on data insights, standardizes all 16 variants found:
    - Full-time: 'Full-time', 'Full Time', 'FULL_TIME', 'Fulltime', 'FT'
    - Self-employed: 'SELF_EMPLOYED', 'Self Emp', 'Self Employed', 'Self-employed'
    - Part-time: 'PART_TIME', 'Part-time', 'PT', 'Part Time'
    - Contractor: 'Contractor', 'CONTRACT', 'Contract'
    """
    if pd.isna(value):
        return value
    value_str = str(value).strip()
    original_value = value_str
    value_lower = value_str.lower().replace('_', ' ').replace('-', ' ')
    
    # Full-time variants (5 found in data: Full-time, Full Time, FULL_TIME, Fulltime, FT)
    if any(variant in value_lower for variant in ['fulltime', 'full time', 'ft']):
        return 'Full-time'
    
    # Self-employed variants (4 found: SELF_EMPLOYED, Self Emp, Self Employed, Self-employed)
    elif any(variant in value_lower for variant in ['self employed', 'self emp', 'self-employed']):
        return 'Self-employed'
    
    # Part-time variants (4 found: PART_TIME, Part-time, PT, Part Time)
    elif any(variant in value_lower for variant in ['parttime', 'part time', 'pt']):
        return 'Part-time'
    
    # Contractor variants (3 found: Contractor, CONTRACT, Contract)
    elif any(variant in value_lower for variant in ['contractor', 'contract']):
        return 'Contractor'
    
    else:
        # Keep original if not recognized (XGBoost can handle it)
        return original_value


def standardize_loan_type(value):
    """Standardize loan type casing and variants."""
    if pd.isna(value):
        return value
    value_str = str(value).strip()
    # Map common variants
    loan_type_map = {
        'personal': 'Personal',
        'personnel': 'Personal',
        'home loan': 'Home Loan',
        'homeloan': 'Home Loan',
        'mortgage': 'Home Loan',
        'credit card': 'Credit Card',
        'creditcard': 'Credit Card',
        'credit': 'Credit Card'
    }
    lower_val = value_str.lower()
    return loan_type_map.get(lower_val, value_str.title())


def clean_currency_columns(df, columns):
    """Clean currency columns by parsing to float."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_currency)
    return df


def clean_categorical_columns(df):
    """Standardize categorical columns."""
    df = df.copy()
    
    # Standardize employment_type
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].apply(standardize_employment_type)
    
    # Standardize loan_type
    if 'loan_type' in df.columns:
        df['loan_type'] = df['loan_type'].apply(standardize_loan_type)
    
    return df


def convert_geographic_numeric_columns(df):
    """Convert geographic numeric columns from strings to numeric types."""
    df = df.copy()
    
    geographic_numeric_cols = [
        'regional_unemployment_rate',
        'regional_median_income',
        'regional_median_rent',
        'housing_price_index',
        'cost_of_living_index'
    ]
    
    for col in geographic_numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN (XGBoost can handle NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Also convert id column if it's a string of numbers
    if 'id' in df.columns and df['id'].dtype == 'object':
        # Check if it's numeric-like
        sample = df['id'].dropna().head(100)
        if sample.astype(str).str.match(r'^\d+$').all():
            df['id'] = pd.to_numeric(df['id'], errors='coerce')
    
    return df


def handle_missing_values(df):
    """Handle missing values - XGBoost can handle missing values, but we can impute for better performance."""
    df = df.copy()
    
    # Impute employment_length with median by employment_type (if employment_type is available)
    # This helps XGBoost but missing values are also acceptable
    if 'employment_length' in df.columns and 'employment_type' in df.columns:
        # Only impute if we have enough data
        if df['employment_length'].notna().sum() > 100:
            median_by_type = df.groupby('employment_type')['employment_length'].median()
            for emp_type in median_by_type.index:
                if pd.notna(median_by_type[emp_type]):
                    mask = (df['employment_type'] == emp_type) & df['employment_length'].isna()
                    if mask.any():
                        df.loc[mask, 'employment_length'] = median_by_type[emp_type]
            # If still missing, use overall median
            if df['employment_length'].isna().any():
                overall_median = df['employment_length'].median()
                if pd.notna(overall_median):
                    df['employment_length'] = df['employment_length'].fillna(overall_median)
    
    # Note: oldest_credit_line_age=0.0 might be valid (new accounts), so we keep it
    # XGBoost can handle zeros and missing values
    
    return df


# Removed winsorize_outliers - XGBoost can handle outliers naturally
# No need to cap/modify data values


def clean_data(df):
    """Main cleaning function that applies all cleaning steps.
    
    Note: This function does NOT drop any rows or features.
    XGBoost can handle missing values, outliers, and high cardinality features.
    We only clean/standardize data formats for consistency.
    """
    df = df.copy()
    
    # Drop random_noise_1 column if present (this is explicitly noise, not a real feature)
    if 'random_noise_1' in df.columns:
        df = df.drop(columns=['random_noise_1'])
    
    # Drop referral_code column (low importance, high cardinality, not useful for prediction)
    if 'referral_code' in df.columns:
        df = df.drop(columns=['referral_code'])
    
    # Clean currency columns - convert string formats to numeric
    currency_columns = [
        'annual_income', 'monthly_income', 'existing_monthly_debt', 
        'monthly_payment', 'revolving_balance', 'credit_usage_amount',
        'available_credit', 'total_monthly_debt_payment', 'annual_debt_payment',
        'total_debt_amount', 'monthly_free_cash_flow', 'loan_amount',
        'total_credit_limit'
    ]
    df = clean_currency_columns(df, currency_columns)
    
    # Convert geographic numeric columns from strings to numeric
    df = convert_geographic_numeric_columns(df)
    
    # Clean categorical columns - standardize variants
    df = clean_categorical_columns(df)
    
    # Handle missing values - impute where it makes sense, but XGBoost can handle NaN
    df = handle_missing_values(df)
    
    # Note: We do NOT winsorize outliers - XGBoost handles them naturally
    # Note: We do NOT drop any rows - all rows are preserved
    # Note: We drop random_noise_1 (explicitly noise) and referral_code (low importance, high cardinality)
    
    return df

