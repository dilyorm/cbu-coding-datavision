"""Data loading functions for various data sources"""
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from src.features.cleaning import clean_currency


def load_geographic_data(filepath: str) -> pd.DataFrame:
    """Load and parse XML geographic data"""
    tree = ET.parse(filepath)
    root = tree.getroot()
    data = []
    for customer in root.findall('customer'):
        row = {}
        for child in customer:
            row[child.tag] = child.text
        data.append(row)
    df = pd.DataFrame(data)
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
    numeric_cols = ['regional_unemployment_rate', 'regional_median_income', 
                    'regional_median_rent', 'housing_price_index', 'cost_of_living_index']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.rename(columns={'id': 'customer_id'})


def load_financial_ratios(filepath: str) -> pd.DataFrame:
    """Load JSONL financial ratios data"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    
    currency_cols = ['monthly_income', 'existing_monthly_debt', 'monthly_payment',
                     'revolving_balance', 'credit_usage_amount', 'available_credit',
                     'total_monthly_debt_payment', 'total_debt_amount', 'monthly_free_cash_flow']
    
    for col in currency_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency)
    
    numeric_cols = ['debt_to_income_ratio', 'debt_service_ratio', 'payment_to_income_ratio',
                    'credit_utilization', 'annual_debt_payment', 'loan_to_annual_income']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.rename(columns={'cust_num': 'customer_id'})


def load_demographics(filepath: str) -> pd.DataFrame:
    """Load and clean demographics CSV"""
    df = pd.read_csv(filepath)
    
    if 'cust_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['cust_id'], errors='coerce')
    
    if 'annual_income' in df.columns:
        df['annual_income'] = df['annual_income'].apply(clean_currency)
    
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].astype(str).str.strip().str.lower()
        # Merge full-time variations
        df['employment_type'] = df['employment_type'].replace({
            'full-time': 'full_time',
            'fulltime': 'full_time',
            'full time': 'full_time',
            'ft': 'full_time',
            'full': 'full_time'
        })
        # Merge part-time variations
        df['employment_type'] = df['employment_type'].replace({
            'part-time': 'part_time',
            'parttime': 'part_time',
            'part time': 'part_time',
            'pt': 'part_time',
            'part': 'part_time'
        })
        # Merge self-employed variations
        df['employment_type'] = df['employment_type'].replace({
            'self-employed': 'self_employed',
            'selfemployed': 'self_employed',
            'self employed': 'self_employed',
            'self emp': 'self_employed',
            'freelance': 'self_employed'
        })
        # Merge contractor variations (constructor, contract, etc.)
        df['employment_type'] = df['employment_type'].replace({
            'contractor': 'contractor',
            'constructor': 'contractor',
            'contract': 'contractor',
            'contructor': 'contractor',  # typo variation
            'contruct': 'contractor',  # typo variation
            'independent contractor': 'contractor',
            'ind contractor': 'contractor'
        })
        # Merge unemployed variations
        df['employment_type'] = df['employment_type'].replace({
            'unemployed': 'unemployed',
            'unemp': 'unemployed',
            'no job': 'unemployed',
            'not employed': 'unemployed'
        })
    
    if 'education' in df.columns:
        df['education'] = df['education'].astype(str).str.strip().str.title()
        # Normalize education variations
        df['education'] = df['education'].replace({
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
            'College': 'Some_College'
        })
    
    numeric_cols = ['age', 'employment_length', 'num_dependents']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_application_metadata(filepath: str) -> pd.DataFrame:
    """Load application metadata with target variable"""
    df = pd.read_csv(filepath)
    if 'customer_ref' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_ref'], errors='coerce')
    return df


def load_loan_details(filepath: str) -> pd.DataFrame:
    """Load loan details Excel file"""
    df = pd.read_excel(filepath)
    
    if 'loan_amount' in df.columns:
        df['loan_amount'] = df['loan_amount'].apply(clean_currency)
    
    if 'loan_type' in df.columns:
        df['loan_type'] = df['loan_type'].astype(str).str.strip().str.lower()
        # Merge credit card variations: cc, credit card, creditcard -> credit_card
        df['loan_type'] = df['loan_type'].replace({
            'cc': 'credit_card',
            'credit card': 'credit_card',
            'creditcard': 'credit_card',
            'credit-card': 'credit_card'
        })
        # Merge personal loan variations
        df['loan_type'] = df['loan_type'].replace({
            'personal loan': 'personal',
            'personal_loan': 'personal',
            'personal-loan': 'personal'
        })
        # Merge auto loan variations
        df['loan_type'] = df['loan_type'].replace({
            'auto loan': 'auto',
            'auto_loan': 'auto',
            'auto-loan': 'auto',
            'car loan': 'auto',
            'car_loan': 'auto'
        })
        # Merge mortgage variations
        df['loan_type'] = df['loan_type'].replace({
            'mortgage': 'mortgage',
            'home loan': 'mortgage',
            'home_loan': 'mortgage',
            'home-loan': 'mortgage',
            'housing loan': 'mortgage'
        })
    
    numeric_cols = ['loan_term', 'interest_rate', 'loan_to_value_ratio']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_credit_history(filepath: str) -> pd.DataFrame:
    """Load credit history parquet file"""
    df = pd.read_parquet(filepath)
    if 'customer_number' in df.columns:
        df = df.rename(columns={'customer_number': 'customer_id'})
    return df

