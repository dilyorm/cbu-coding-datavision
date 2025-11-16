import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
from pathlib import Path

from src.data.load_data import load_all_data
from src.data.clean_data import clean_data
from src.features.feature_engineering import engineer_features
from src.features.feature_engineering_v2 import engineer_features_v2
from src.features.feature_engineering_v3 import engineer_features_v3
from src.data.preprocessing import (
    get_numeric_categorical_features,
    fit_preprocessing_pipeline,
    save_preprocessing_pipeline,
    get_feature_names
)


def merge_data(data_dict):
    """Merge all data sources on customer_id."""
    # Start with application_metadata as base (contains target)
    merged = data_dict['application_metadata'].copy()
    # Drop null customer_ids and convert to int64
    merged = merged.dropna(subset=['customer_id'])
    merged['customer_id'] = merged['customer_id'].astype('int64')
    
    # Merge other datasets
    datasets_to_merge = [
        'credit_history',
        'demographics',
        'financial_ratios',
        'geographic_data',
        'loan_details'
    ]
    
    for dataset_name in datasets_to_merge:
        if dataset_name in data_dict:
            df = data_dict[dataset_name].copy()
            # Drop null customer_ids and convert to int64
            df = df.dropna(subset=['customer_id'])
            df['customer_id'] = df['customer_id'].astype('int64')
            # Inner join to keep only complete records
            merged = merged.merge(
                df,
                on='customer_id',
                how='inner',
                suffixes=('', f'_{dataset_name}')
            )
    
    return merged


def create_data_report(df, X_train, X_test, y_train, y_test, feature_names):
    """Create a data report with statistics."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DATA PREPARATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Original data shape
    report_lines.append(f"Original merged dataset shape: {df.shape}")
    report_lines.append(f"  - Rows: {df.shape[0]:,}")
    report_lines.append(f"  - Columns: {df.shape[1]}")
    report_lines.append("")
    
    # Train/Test splits
    report_lines.append("Train/Test Splits:")
    report_lines.append(f"  - Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    report_lines.append(f"  - Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    report_lines.append("")
    
    # Feature counts
    report_lines.append(f"Total features after preprocessing: {len(feature_names)}")
    report_lines.append("")
    
    # Class balance
    report_lines.append("Target Variable (default) Distribution:")
    report_lines.append(f"  - Overall: {df['default'].value_counts().to_dict()}")
    report_lines.append(f"  - Train: {y_train.value_counts().to_dict()}")
    report_lines.append(f"  - Test: {y_test.value_counts().to_dict()}")
    report_lines.append("")
    
    # Missing values summary
    report_lines.append("Missing Values Summary (original data):")
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if len(missing_counts) > 0:
        for col, count in missing_counts.head(10).items():
            pct = count / len(df) * 100
            report_lines.append(f"  - {col}: {count:,} ({pct:.2f}%)")
    else:
        report_lines.append("  - No missing values after cleaning")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def load_selected_features(selected_features_file):
    """Load selected features from file."""
    if not Path(selected_features_file).exists():
        raise FileNotFoundError(f"Selected features file not found: {selected_features_file}")
    
    with open(selected_features_file, 'r') as f:
        selected_features = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(selected_features)} selected features from {selected_features_file}")
    return selected_features


def process_version(version, train_df_base, test_df_base, y_train, y_test, 
                   train_stats, target_col, selected_features_list=None):
    """Process a single feature engineering version.
    
    Args:
        version: Version name ('v1', 'v2', or 'v3')
        train_df_base: Base training DataFrame (after basic features)
        test_df_base: Base test DataFrame (after basic features)
        y_train: Training target
        y_test: Test target
        train_stats: Training statistics dict
        target_col: Target column name
        selected_features_list: Optional list of selected features to filter
    
    Returns:
        Tuple of (X_train_df, X_test_df, feature_names, output_dir)
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {version.upper()}")
    print(f"{'='*80}")
    
    # Apply version-specific feature engineering
    train_df = train_df_base.copy()
    test_df = test_df_base.copy()
    
    if version == 'v1':
        print("Applying v1 feature engineering...")
        train_df = engineer_features(train_df, train_stats=train_stats, train_value_counts=None)
        test_df = engineer_features(test_df, train_stats=train_stats, train_value_counts=None)
    elif version == 'v2':
        print("Applying v2 feature engineering...")
        train_df = engineer_features_v2(train_df, train_stats=train_stats, train_value_counts=None)
        test_df = engineer_features_v2(test_df, train_stats=train_stats, train_value_counts=None)
    elif version == 'v3':
        print("Applying v3 feature engineering...")
        train_df = engineer_features_v3(train_df, train_stats=train_stats, train_value_counts=None)
        test_df = engineer_features_v3(test_df, train_stats=train_stats, train_value_counts=None)
    else:
        raise ValueError(f"Unknown version: {version}")
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col, 'customer_id'], errors='ignore')
    X_test = test_df.drop(columns=[target_col, 'customer_id'], errors='ignore')
    
    print(f"After {version} features - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create combined df for reporting
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Filter to selected features if specified
    if selected_features_list is not None:
        print(f"\nFiltering to {len(selected_features_list)} selected features...")
        available_selected = [f for f in selected_features_list if f in X_train.columns]
        missing_features = [f for f in selected_features_list if f not in X_train.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} selected features not found in data")
            if len(missing_features) > 10:
                print(f"  Missing: {missing_features[:10]}...")
            else:
                print(f"  Missing: {missing_features}")
        
        X_train = X_train[available_selected]
        X_test = X_test[available_selected]
        print(f"Using {len(available_selected)} available selected features")
    
    # Identify numeric and categorical features
    numeric_features, categorical_features = get_numeric_categorical_features(df, target_col)
    
    # Filter to only features present in current data
    numeric_features = [f for f in numeric_features if f in X_train.columns]
    categorical_features = [f for f in categorical_features if f in X_train.columns]
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Fit preprocessing pipeline on training data only
    print("Fitting preprocessing pipeline...")
    preprocessor = fit_preprocessing_pipeline(X_train, numeric_features, categorical_features, y_train=y_train)
    
    # Transform all splits - ensure categorical columns are strings
    print("Transforming data...")
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    for col in categorical_features:
        if col in X_train_clean.columns:
            X_train_clean[col] = X_train_clean[col].astype(str)
        if col in X_test_clean.columns:
            X_test_clean[col] = X_test_clean[col].astype(str)
    
    X_train_transformed = preprocessor.transform(X_train_clean)
    X_test_transformed = preprocessor.transform(X_test_clean)
    
    # Convert to DataFrames with feature names
    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features, X_train)
    
    # Verify feature count matches actual transformation
    actual_feature_count = X_train_transformed.shape[1]
    if len(feature_names) != actual_feature_count:
        print(f"Warning: Feature name count ({len(feature_names)}) doesn't match transformation output ({actual_feature_count})")
        print("Generating feature names from actual output shape...")
        feature_names = [f'feature_{i}' for i in range(actual_feature_count)]
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)
    
    # Create output directory
    output_dir = Path(f'data/processed/{version}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preprocessed data
    print(f"Saving preprocessed data to {output_dir}...")
    print(f"  Saving X_train.csv ({X_train_df.shape[0]:,} rows, {X_train_df.shape[1]} cols)...")
    X_train_df.to_csv(output_dir / 'X_train.csv', index=False)
    print("  Saving y_train.csv...")
    y_train.to_frame().to_csv(output_dir / 'y_train.csv', index=False)
    print("  Saving X_test.csv...")
    X_test_df.to_csv(output_dir / 'X_test.csv', index=False)
    print("  Saving y_test.csv...")
    y_test.to_frame().to_csv(output_dir / 'y_test.csv', index=False)
    
    # Save preprocessing pipeline
    save_preprocessing_pipeline(preprocessor, output_dir / 'preprocessing_pipeline.pkl')
    
    # Save feature names
    with open(output_dir / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    # Create and save data report
    report = create_data_report(
        df, X_train_df, X_test_df,
        y_train, y_test, feature_names
    )
    with open(output_dir / 'data_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n{version.upper()} complete. Outputs saved to {output_dir}/")
    
    return X_train_df, X_test_df, feature_names, output_dir


def main(use_selected_features=False, selected_features_file=None):
    """Main pipeline execution - processes all 3 versions (v1, v2, v3).
    
    Args:
        use_selected_features: Whether to use pre-selected features
        selected_features_file: Path to file containing selected feature names (one per line)
    """
    # Load selected features if specified
    selected_features_list = None
    if use_selected_features and selected_features_file:
        selected_features_list = load_selected_features(selected_features_file)
    
    print("="*80)
    print("DATA PREPARATION - ALL VERSIONS")
    print("="*80)
    
    print("\nLoading data...")
    data_dict = load_all_data()
    
    print("Merging datasets...")
    df = merge_data(data_dict)
    print(f"Merged dataset shape: {df.shape}")
    
    print("Cleaning data...")
    df = clean_data(df)
    
    # Separate features and target BEFORE enhanced feature engineering
    target_col = 'default'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Split data BEFORE enhanced feature engineering to prevent leakage
    print("\nSplitting data (before enhanced feature engineering)...")
    X_basic = df.drop(columns=[target_col, 'customer_id'], errors='ignore')
    y = df[target_col]
    
    # Stratified split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_basic, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Basic split - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Reconstruct full DataFrames with customer_id and target for feature engineering
    train_df_base = X_train.copy()
    train_df_base['customer_id'] = df.loc[X_train.index, 'customer_id'].values
    train_df_base[target_col] = y_train.values
    
    test_df_base = X_test.copy()
    test_df_base['customer_id'] = df.loc[X_test.index, 'customer_id'].values
    test_df_base[target_col] = y_test.values
    
    # Apply basic feature engineering (no leakage risk) - common to all versions
    print("\nApplying basic feature engineering (common to all versions)...")
    from src.features.feature_engineering import (
        create_temporal_features, create_interaction_features,
        create_aggregation_features, create_geographic_features, create_loan_features
    )
    
    train_df_base = create_temporal_features(train_df_base)
    train_df_base = create_interaction_features(train_df_base)
    train_df_base = create_aggregation_features(train_df_base)
    train_df_base = create_geographic_features(train_df_base)
    train_df_base = create_loan_features(train_df_base)
    
    test_df_base = create_temporal_features(test_df_base)
    test_df_base = create_interaction_features(test_df_base)
    test_df_base = create_aggregation_features(test_df_base)
    test_df_base = create_geographic_features(test_df_base)
    test_df_base = create_loan_features(test_df_base)
    
    # Compute statistics from training set only (prevent leakage)
    print("\nComputing training set statistics for enhanced features...")
    train_stats = {}
    zscore_columns = [
        'credit_score', 'annual_income', 'monthly_income',
        'debt_to_income_ratio', 'credit_utilization', 'account_age',
        'loan_amount', 'total_debt_amount'
    ]
    for col in zscore_columns:
        if col in train_df_base.columns:
            train_stats[col] = {
                'mean': train_df_base[col].mean(),
                'std': train_df_base[col].std()
            }
    
    # Process all 3 versions
    versions = ['v1', 'v2', 'v3']
    for version in versions:
        process_version(
            version, train_df_base, test_df_base, y_train, y_test,
            train_stats, target_col, selected_features_list
        )
    
    print("\n" + "="*80)
    print("ALL VERSIONS PROCESSED SUCCESSFULLY")
    print("="*80)
    print("Outputs saved to:")
    for version in versions:
        print(f"  - data/processed/{version}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for ML model training')
    parser.add_argument('--use-selected-features', action='store_true',
                       help='Use pre-selected features from file')
    parser.add_argument('--features-file', type=str, default='data/feature_selection/selected_features.txt',
                       help='Path to file containing selected feature names (default: data/feature_selection/selected_features.txt)')
    
    args = parser.parse_args()
    
    main(use_selected_features=args.use_selected_features,
         selected_features_file=args.features_file if args.use_selected_features else None)

