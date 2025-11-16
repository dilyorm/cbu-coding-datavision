import pandas as pd
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def load_application_metadata(file_path='data/application_metadata.csv'):
    """Load application metadata CSV and normalize ID column."""
    df = pd.read_csv(file_path)
    # Handle unnamed index column if present
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])
    # Normalize ID column
    if 'customer_ref' in df.columns:
        df = df.rename(columns={'customer_ref': 'customer_id'})
    elif 'application_id' in df.columns:
        # Use application_id as customer_id if customer_ref not present
        df = df.rename(columns={'application_id': 'customer_id'})
    # Convert customer_id to int64
    if 'customer_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce').astype('Int64')
    return df


def load_credit_history(file_path='data/credit_history.parquet'):
    """Load credit history Parquet file and normalize ID column."""
    df = pd.read_parquet(file_path)
    # Normalize ID column
    if 'customer_number' in df.columns:
        df = df.rename(columns={'customer_number': 'customer_id'})
    # Convert customer_id to int64
    if 'customer_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce').astype('Int64')
    return df


def load_demographics(file_path='data/demographics.csv'):
    """Load demographics CSV and normalize ID column."""
    df = pd.read_csv(file_path)
    # Handle unnamed index column if present
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])
    # Normalize ID column
    if 'cust_id' in df.columns:
        df = df.rename(columns={'cust_id': 'customer_id'})
    # Convert customer_id to int64
    if 'customer_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce').astype('Int64')
    return df


def load_financial_ratios(file_path='data/financial_ratios.jsonl'):
    """Load financial ratios JSONL file and normalize ID column."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Normalize ID column
    if 'cust_num' in df.columns:
        df = df.rename(columns={'cust_num': 'customer_id'})
    # Convert customer_id to int64
    if 'customer_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce').astype('Int64')
    return df


def load_geographic_data(file_path='data/geographic_data.xml'):
    """Load geographic data XML file and normalize ID column."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    records = []
    for customer in root.findall('customer'):
        record = {}
        for child in customer:
            record[child.tag] = child.text
        records.append(record)
    
    df = pd.DataFrame(records)
    # Normalize ID column
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'customer_id'})
    # Convert customer_id to int64
    if 'customer_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce').astype('Int64')
    # Convert numeric columns
    numeric_cols = ['regional_unemployment_rate', 'regional_median_income', 
                    'regional_median_rent', 'housing_price_index', 
                    'cost_of_living_index']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop previous_zip_code - high cardinality with low predictive value
    if 'previous_zip_code' in df.columns:
        df = df.drop(columns=['previous_zip_code'])
    
    return df


def load_loan_details(file_path='data/loan_details.xlsx'):
    """Load loan details Excel file and normalize ID column."""
    df = pd.read_excel(file_path)
    # Convert customer_id to int64
    if 'customer_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce').astype('Int64')
    return df


def load_all_data(data_dir='data'):
    """Load all data sources and return as dictionary."""
    data = {
        'application_metadata': load_application_metadata(f'{data_dir}/application_metadata.csv'),
        'credit_history': load_credit_history(f'{data_dir}/credit_history.parquet'),
        'demographics': load_demographics(f'{data_dir}/demographics.csv'),
        'financial_ratios': load_financial_ratios(f'{data_dir}/financial_ratios.jsonl'),
        'geographic_data': load_geographic_data(f'{data_dir}/geographic_data.xml'),
        'loan_details': load_loan_details(f'{data_dir}/loan_details.xlsx')
    }
    return data

