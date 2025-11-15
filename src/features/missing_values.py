"""Missing value handling functions"""
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer, KNNImputer
# Enable IterativeImputer (experimental feature)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def handle_missing_values(df: pd.DataFrame, imputation_method: str = 'knn') -> pd.DataFrame:
    """Handle missing values systematically"""
    print("\n" + "="*60)
    print("STEP 6: Handling Missing Values...")
    print("="*60)
    
    df_clean = df.copy()
    dropped_cols = []
    
    # Identify columns to drop (IDs, leakage, noise)
    drop_cols = ['customer_id', 'customer_ref', 'application_id', 'random_noise_1', 
                 'referral_code', 'cust_id']
    drop_cols = [col for col in drop_cols if col in df_clean.columns]
    if drop_cols:
        print(f"  Dropping ID/leakage columns: {drop_cols}")
        dropped_cols.extend(drop_cols)
    df_clean = df_clean.drop(columns=drop_cols, errors='ignore')
    
    # Separate numeric and categorical
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'default' in numeric_cols:
        numeric_cols.remove('default')
    
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle categorical missing values
    for col in categorical_cols:
        missing_pct = df_clean[col].isna().sum() / len(df_clean)
        if missing_pct > 0.5:
            print(f"  Dropping {col} ({missing_pct:.1%} missing)")
            dropped_cols.append(col)
            df_clean = df_clean.drop(columns=[col])
        else:
            # Convert categorical to string first if it's a Categorical dtype
            if pd.api.types.is_categorical_dtype(df_clean[col]):
                # Add 'Unknown' to categories if it doesn't exist
                if 'Unknown' not in df_clean[col].cat.categories:
                    df_clean[col] = df_clean[col].cat.add_categories(['Unknown'])
                df_clean[col] = df_clean[col].fillna('Unknown')
            else:
                # For object/string columns, just fillna
                df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Handle numeric missing values with advanced imputation
    cols_to_impute = [col for col in numeric_cols 
                     if col in df_clean.columns and df_clean[col].isna().sum() > 0]
    
    if cols_to_impute:
        # Create missing indicators before imputation
        for col in cols_to_impute:
            missing_pct = df_clean[col].isna().sum() / len(df_clean)
            if missing_pct > 0.5:
                print(f"  Dropping {col} ({missing_pct:.1%} missing)")
                dropped_cols.append(col)
                df_clean = df_clean.drop(columns=[col])
            else:
                df_clean[f"{col}_was_missing"] = df_clean[col].isna().astype(int)
        
        # Re-filter after dropping
        cols_to_impute = [col for col in cols_to_impute if col in df_clean.columns]
        
        if cols_to_impute:
            # Separate employment_length for KNN imputation (always use KNN for this column)
            employment_length_col = 'employment_length' if 'employment_length' in cols_to_impute else None
            other_cols = [col for col in cols_to_impute if col != 'employment_length']
            
            # Impute employment_length with KNN if it has missing values
            if employment_length_col and df_clean[employment_length_col].isna().sum() > 0:
                print(f"  Using KNN imputation for {employment_length_col}...")
                # Use other numeric columns as features for KNN imputation of employment_length
                feature_cols = [col for col in numeric_cols 
                               if col in df_clean.columns and col != employment_length_col and col != 'default']
                if len(feature_cols) > 0:
                    try:
                        # Prepare data for KNN imputation
                        knn_data = df_clean[feature_cols + [employment_length_col]].copy()
                        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
                        knn_imputed = knn_imputer.fit_transform(knn_data)
                        df_clean[employment_length_col] = knn_imputed[:, -1]  # Last column is employment_length
                    except Exception as e:
                        print(f"  Warning: KNN imputation for {employment_length_col} failed, using median: {e}")
                        median_val = df_clean[employment_length_col].median()
                        df_clean[employment_length_col] = df_clean[employment_length_col].fillna(median_val)
            
            # Impute other columns with specified method
            if other_cols:
                print(f"  Using {imputation_method.upper()} imputation for {len(other_cols)} numeric columns...")
                impute_data = df_clean[other_cols].copy()
                
                try:
                    if imputation_method == 'knn':
                        imputer = KNNImputer(n_neighbors=5, weights='uniform')
                        imputed_values = imputer.fit_transform(impute_data)
                        df_clean[other_cols] = imputed_values
                    elif imputation_method == 'iterative':
                        imputer = IterativeImputer(max_iter=10, random_state=42, n_nearest_features=10)
                        imputed_values = imputer.fit_transform(impute_data)
                        df_clean[other_cols] = imputed_values
                    else:  # median
                        imputer = SimpleImputer(strategy='median')
                        imputed_values = imputer.fit_transform(impute_data)
                        df_clean[other_cols] = imputed_values
                except Exception as e:
                    print(f"  Warning: {imputation_method} imputation failed, falling back to median: {e}")
                    imputer = SimpleImputer(strategy='median')
                    imputed_values = imputer.fit_transform(impute_data)
                    df_clean[other_cols] = imputed_values
    
    # Log dropped columns
    if dropped_cols:
        os.makedirs('datas', exist_ok=True)
        dropped_df = pd.DataFrame({
            'dropped_column': dropped_cols, 
            'reason': 'high_missing_or_id'
        })
        dropped_df.to_csv('datas/dropped_columns_log.csv', index=False)
        print(f"  Logged {len(dropped_cols)} dropped columns to 'datas/dropped_columns_log.csv'")
    
    print(f"Final shape: {df_clean.shape}")
    print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
    
    return df_clean

