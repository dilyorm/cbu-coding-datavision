import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def load_all_data_sources():
    """Load all data sources without any cleaning."""
    data = {}
    
    # Load application_metadata
    print("Loading application_metadata.csv...")
    df_app = pd.read_csv('data/application_metadata.csv')
    if df_app.columns[0].startswith('Unnamed'):
        df_app = df_app.drop(columns=[df_app.columns[0]])
    data['application_metadata'] = df_app
    
    # Load credit_history
    print("Loading credit_history.parquet...")
    data['credit_history'] = pd.read_parquet('data/credit_history.parquet')
    
    # Load demographics
    print("Loading demographics.csv...")
    df_demo = pd.read_csv('data/demographics.csv')
    if df_demo.columns[0].startswith('Unnamed'):
        df_demo = df_demo.drop(columns=[df_demo.columns[0]])
    data['demographics'] = df_demo
    
    # Load financial_ratios
    print("Loading financial_ratios.jsonl...")
    records = []
    with open('data/financial_ratios.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    data['financial_ratios'] = pd.DataFrame(records)
    
    # Load geographic_data
    print("Loading geographic_data.xml...")
    tree = ET.parse('data/geographic_data.xml')
    root = tree.getroot()
    records = []
    for customer in root.findall('customer'):
        record = {}
        for child in customer:
            record[child.tag] = child.text
        records.append(record)
    data['geographic_data'] = pd.DataFrame(records)
    
    # Load loan_details
    print("Loading loan_details.xlsx...")
    data['loan_details'] = pd.read_excel('data/loan_details.xlsx')
    
    return data


def analyze_column(df, col_name, dataset_name):
    """Analyze a single column and return insights."""
    col = df[col_name]
    insights = {
        'column': col_name,
        'dataset': dataset_name,
        'dtype': str(col.dtype),
        'total_values': len(col),
        'non_null_count': col.notna().sum(),
        'null_count': col.isna().sum(),
        'null_percentage': (col.isna().sum() / len(col)) * 100,
        'unique_count': col.nunique(),
    }
    
    # Numeric analysis
    if pd.api.types.is_numeric_dtype(col):
        insights['type'] = 'numeric'
        insights['min'] = float(col.min()) if col.notna().any() else None
        insights['max'] = float(col.max()) if col.notna().any() else None
        insights['mean'] = float(col.mean()) if col.notna().any() else None
        insights['median'] = float(col.median()) if col.notna().any() else None
        insights['std'] = float(col.std()) if col.notna().any() else None
        insights['zeros'] = int((col == 0).sum())
        insights['negative'] = int((col < 0).sum()) if col.notna().any() else 0
        
        # Check for potential outliers (beyond 3 std)
        if col.notna().any() and col.std() > 0:
            mean_val = col.mean()
            std_val = col.std()
            outliers = col[(col < mean_val - 3*std_val) | (col > mean_val + 3*std_val)]
            insights['outliers_3std'] = len(outliers)
        else:
            insights['outliers_3std'] = 0
            
        # Percentiles
        if col.notna().any():
            insights['percentiles'] = {
                'p1': float(col.quantile(0.01)),
                'p5': float(col.quantile(0.05)),
                'p25': float(col.quantile(0.25)),
                'p50': float(col.quantile(0.50)),
                'p75': float(col.quantile(0.75)),
                'p95': float(col.quantile(0.95)),
                'p99': float(col.quantile(0.99))
            }
    else:
        insights['type'] = 'categorical'
        # Get unique values and their counts
        value_counts = col.value_counts()
        insights['unique_values'] = {}
        # Show all unique values if reasonable, otherwise top 50
        if len(value_counts) <= 100:
            insights['unique_values'] = {str(k): int(v) for k, v in value_counts.items()}
        else:
            insights['unique_values'] = {
                'top_50': {str(k): int(v) for k, v in value_counts.head(50).items()},
                'total_unique': len(value_counts)
            }
        
        # Check for potential issues
        insights['whitespace_values'] = int(col.astype(str).str.strip().ne(col.astype(str)).sum())
        insights['empty_strings'] = int((col.astype(str).str.strip() == '').sum())
        
        # Check for mixed types (numeric strings)
        if col.notna().any():
            sample_values = col.dropna().head(100).astype(str)
            numeric_like = sample_values.str.match(r'^[\d\s\$,\.\-]+$').sum()
            insights['numeric_like_strings'] = int(numeric_like)
    
    return insights


def generate_insights_report(data_dict):
    """Generate comprehensive insights report for all datasets."""
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("COMPREHENSIVE DATA INSIGHTS REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    all_insights = {}
    
    for dataset_name, df in data_dict.items():
        report_lines.append("=" * 100)
        report_lines.append(f"DATASET: {dataset_name.upper()}")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        # Basic info
        report_lines.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        report_lines.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report_lines.append("")
        
        # Column info
        report_lines.append("COLUMN INFORMATION:")
        report_lines.append("-" * 100)
        report_lines.append(f"{'Column':<30} {'Type':<15} {'Non-Null':<12} {'Null':<12} {'Unique':<10} {'% Null':<10}")
        report_lines.append("-" * 100)
        
        dataset_insights = {}
        
        for col in df.columns:
            insights = analyze_column(df, col, dataset_name)
            dataset_insights[col] = insights
            
            null_pct = insights['null_percentage']
            report_lines.append(
                f"{col:<30} {insights['dtype']:<15} "
                f"{insights['non_null_count']:<12,} {insights['null_count']:<12,} "
                f"{insights['unique_count']:<10} {null_pct:<10.2f}%"
            )
        
        report_lines.append("")
        all_insights[dataset_name] = dataset_insights
        
        # Detailed analysis for each column
        report_lines.append("DETAILED COLUMN ANALYSIS:")
        report_lines.append("-" * 100)
        
        for col in df.columns:
            insights = dataset_insights[col]
            report_lines.append("")
            report_lines.append(f"  Column: {col}")
            report_lines.append(f"    Type: {insights['type']} ({insights['dtype']})")
            report_lines.append(f"    Non-null: {insights['non_null_count']:,} ({100-insights['null_percentage']:.2f}%)")
            report_lines.append(f"    Null: {insights['null_count']:,} ({insights['null_percentage']:.2f}%)")
            report_lines.append(f"    Unique values: {insights['unique_count']}")
            
            if insights['type'] == 'numeric':
                report_lines.append(f"    Min: {insights.get('min', 'N/A')}")
                report_lines.append(f"    Max: {insights.get('max', 'N/A')}")
                report_lines.append(f"    Mean: {insights.get('mean', 'N/A'):.2f}" if insights.get('mean') is not None else "    Mean: N/A")
                report_lines.append(f"    Median: {insights.get('median', 'N/A'):.2f}" if insights.get('median') is not None else "    Median: N/A")
                report_lines.append(f"    Std: {insights.get('std', 'N/A'):.2f}" if insights.get('std') is not None else "    Std: N/A")
                report_lines.append(f"    Zeros: {insights.get('zeros', 0):,}")
                report_lines.append(f"    Negative values: {insights.get('negative', 0):,}")
                report_lines.append(f"    Outliers (3 std): {insights.get('outliers_3std', 0):,}")
                
                if 'percentiles' in insights:
                    p = insights['percentiles']
                    report_lines.append(f"    Percentiles: P1={p['p1']:.2f}, P5={p['p5']:.2f}, P25={p['p25']:.2f}, "
                                      f"P50={p['p50']:.2f}, P75={p['p75']:.2f}, P95={p['p95']:.2f}, P99={p['p99']:.2f}")
            
            elif insights['type'] == 'categorical':
                if 'unique_values' in insights:
                    unique_vals = insights['unique_values']
                    if 'top_50' in unique_vals:
                        report_lines.append(f"    Total unique values: {unique_vals['total_unique']}")
                        report_lines.append("    Top 50 unique values:")
                        for val, count in list(unique_vals['top_50'].items())[:20]:  # Show top 20
                            report_lines.append(f"      '{val}': {count:,}")
                        if len(unique_vals['top_50']) > 20:
                            report_lines.append(f"      ... and {len(unique_vals['top_50']) - 20} more")
                    else:
                        report_lines.append("    All unique values:")
                        for val, count in list(unique_vals.items())[:30]:  # Show first 30
                            report_lines.append(f"      '{val}': {count:,}")
                        if len(unique_vals) > 30:
                            report_lines.append(f"      ... and {len(unique_vals) - 30} more")
                
                report_lines.append(f"    Whitespace issues: {insights.get('whitespace_values', 0):,}")
                report_lines.append(f"    Empty strings: {insights.get('empty_strings', 0):,}")
                report_lines.append(f"    Numeric-like strings: {insights.get('numeric_like_strings', 0):,}")
        
        report_lines.append("")
        report_lines.append("")
    
    # Summary across all datasets
    report_lines.append("=" * 100)
    report_lines.append("CROSS-DATASET SUMMARY")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    # ID columns analysis
    report_lines.append("ID COLUMNS ANALYSIS:")
    report_lines.append("-" * 100)
    id_columns = {}
    for dataset_name, df in data_dict.items():
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'ref', 'number', 'num', 'customer']):
                if col not in id_columns:
                    id_columns[col] = []
                id_columns[col].append({
                    'dataset': dataset_name,
                    'dtype': str(df[col].dtype),
                    'unique_count': df[col].nunique(),
                    'null_count': df[col].isna().sum(),
                    'sample_values': df[col].dropna().head(5).tolist()
                })
    
    for col, info_list in id_columns.items():
        report_lines.append(f"  {col}:")
        for info in info_list:
            report_lines.append(f"    Dataset: {info['dataset']}")
            report_lines.append(f"      Type: {info['dtype']}, Unique: {info['unique_count']:,}, "
                              f"Null: {info['null_count']:,}")
            report_lines.append(f"      Sample values: {info['sample_values']}")
        report_lines.append("")
    
    # Data quality issues
    report_lines.append("DATA QUALITY ISSUES:")
    report_lines.append("-" * 100)
    
    issues = []
    for dataset_name, insights_dict in all_insights.items():
        for col, insights in insights_dict.items():
            # High null percentage
            if insights['null_percentage'] > 50:
                issues.append(f"{dataset_name}.{col}: {insights['null_percentage']:.2f}% null values")
            
            # Single value (no variance)
            if insights['unique_count'] == 1:
                issues.append(f"{dataset_name}.{col}: Only 1 unique value (no variance)")
            
            # Very high cardinality categorical
            if insights['type'] == 'categorical' and insights['unique_count'] > 1000:
                issues.append(f"{dataset_name}.{col}: Very high cardinality ({insights['unique_count']:,} unique values)")
            
            # Potential data type issues
            if insights['type'] == 'categorical' and insights.get('numeric_like_strings', 0) > 10:
                issues.append(f"{dataset_name}.{col}: Contains numeric-like strings (may need parsing)")
    
    if issues:
        for issue in issues:
            report_lines.append(f"  - {issue}")
    else:
        report_lines.append("  No major issues detected")
    
    report_lines.append("")
    report_lines.append("=" * 100)
    
    return "\n".join(report_lines), all_insights


def save_detailed_json(insights_dict, filepath='data/data_insights_detailed.json'):
    """Save detailed insights as JSON for programmatic access."""
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    json_data = convert_to_json_serializable(insights_dict)
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def main():
    """Main function to generate insights."""
    print("Loading all data sources...")
    data_dict = load_all_data_sources()
    
    print("\nGenerating comprehensive insights...")
    report, insights_dict = generate_insights_report(data_dict)
    
    # Save text report
    output_file = 'data/data_insights_report.txt'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nText report saved to: {output_file}")
    
    # Save detailed JSON
    json_file = 'data/data_insights_detailed.json'
    save_detailed_json(insights_dict, json_file)
    print(f"Detailed JSON saved to: {json_file}")
    
    # Print summary to console
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for dataset_name, df in data_dict.items():
        print(f"\n{dataset_name}:")
        print(f"  Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
        print(f"  Missing values: {df.isnull().sum().sum():,}")
        print(f"  Columns with missing values: {(df.isnull().sum() > 0).sum()}")
    
    print("\n" + "=" * 100)
    print("Full report available in: data/data_insights_report.txt")
    print("Detailed JSON available in: data/data_insights_detailed.json")
    print("=" * 100)


if __name__ == '__main__':
    main()

