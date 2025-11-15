import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, roc_curve
)
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import RandomizedSearchCV
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')


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


def load_geographic_data(filepath):
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


def load_financial_ratios(filepath):
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


def load_demographics(filepath):
    """Load and clean demographics CSV"""
    df = pd.read_csv(filepath)
    
    if 'cust_id' in df.columns:
        df['customer_id'] = pd.to_numeric(df['cust_id'], errors='coerce')
    
    if 'annual_income' in df.columns:
        df['annual_income'] = df['annual_income'].apply(clean_currency)
    
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].str.strip().str.lower()
        df['employment_type'] = df['employment_type'].replace({
            'full-time': 'full_time', 'fulltime': 'full_time', 'full time': 'full_time'
        })
    
    if 'education' in df.columns:
        df['education'] = df['education'].str.strip()
    
    numeric_cols = ['age', 'employment_length', 'num_dependents']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_application_metadata(filepath):
    """Load application metadata with target variable"""
    df = pd.read_csv(filepath)
    if 'customer_ref' in df.columns:
        df['customer_id'] = pd.to_numeric(df['customer_ref'], errors='coerce')
    return df


def load_loan_details(filepath):
    """Load loan details Excel file"""
    df = pd.read_excel(filepath)
    
    if 'loan_amount' in df.columns:
        df['loan_amount'] = df['loan_amount'].apply(clean_currency)
    
    if 'loan_type' in df.columns:
        df['loan_type'] = df['loan_type'].str.strip().str.lower()
        df['loan_type'] = df['loan_type'].replace({
            'personal loan': 'personal', 'personal': 'personal'
        })
    
    numeric_cols = ['loan_term', 'interest_rate', 'loan_to_value_ratio']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_credit_history(filepath):
    """Load credit history parquet file"""
    df = pd.read_parquet(filepath)
    if 'customer_number' in df.columns:
        df = df.rename(columns={'customer_number': 'customer_id'})
    return df


def aggregate_credit_features(credit_df):
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


def merge_all_data():
    """Load and merge all datasets with proper aggregations"""
    print("="*60)
    print("STEP 1: Loading all data files...")
    print("="*60)
    
    geo_df = load_geographic_data('geographic_data.xml')
    financial_df = load_financial_ratios('financial_ratios.jsonl')
    demo_df = load_demographics('demographics.csv')
    app_df = load_application_metadata('application_metadata.csv')
    loan_df = load_loan_details('loan_details.xlsx')
    credit_df = load_credit_history('credit_history.parquet')
    
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


def clean_data(df):
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


def engineer_features(df):
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
        # Convert categorical to string to avoid issues later
        df_feat['age_band'] = df_feat['age_band'].astype(str)
        df_feat['is_young'] = (df_feat['age'] < 30).astype(int)
        df_feat['is_senior'] = (df_feat['age'] >= 55).astype(int)
    
    # 4. Credit score features
    if 'credit_score' in df_feat.columns:
        df_feat['credit_score_category'] = pd.cut(df_feat['credit_score'],
                                                  bins=[0, 580, 670, 740, 850],
                                                  labels=['Poor', 'Fair', 'Good', 'Excellent'])
        # Convert categorical to string to avoid issues later
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
        # Convert categorical to string to avoid issues later
        df_feat['application_hour_bin'] = df_feat['application_hour_bin'].astype(str)
    
    if 'num_customer_service_calls' in df_feat.columns:
        df_feat['frequent_service_calls'] = (df_feat['num_customer_service_calls'] > 3).astype(int)
    
    print(f"Created {len(df_feat.columns)} features (from {len(df.columns)})")
    return df_feat


def handle_missing_values(df):
    """Handle missing values systematically"""
    print("\n" + "="*60)
    print("STEP 6: Handling Missing Values...")
    print("="*60)
    
    df_clean = df.copy()
    
    # Identify columns to drop (IDs, leakage, noise)
    drop_cols = ['customer_id', 'customer_ref', 'application_id', 'random_noise_1', 
                 'referral_code', 'cust_id']
    drop_cols = [col for col in drop_cols if col in df_clean.columns]
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
    
    # Handle numeric missing values
    for col in numeric_cols:
        if col not in df_clean.columns:
            continue
        missing_pct = df_clean[col].isna().sum() / len(df_clean)
        if missing_pct > 0.5:
            print(f"  Dropping {col} ({missing_pct:.1%} missing)")
            df_clean = df_clean.drop(columns=[col])
        else:
            # Impute with median and create missing flag
            median_val = df_clean[col].median()
            if not pd.isna(median_val):
                df_clean[f"{col}_was_missing"] = df_clean[col].isna().astype(int)
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                # If median is NaN, fill with 0
                df_clean[f"{col}_was_missing"] = df_clean[col].isna().astype(int)
                df_clean[col] = df_clean[col].fillna(0)
    
    print(f"Final shape: {df_clean.shape}")
    print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def prepare_model_data(df):
    """Prepare data for modeling"""
    print("\n" + "="*60)
    print("STEP 7: Preparing Model Data...")
    print("="*60)
    
    # Ensure target exists
    if 'default' not in df.columns:
        raise ValueError("Target column 'default' not found!")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != 'default']
    X = df[feature_cols].copy()
    y = df['default'].copy()
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target distribution %:\n{y.value_counts(normalize=True)}")
    
    return X, y, feature_cols


def create_preprocessing_pipeline(X_train):
    """Create preprocessing pipeline"""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Convert categorical columns to string for encoding
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_cols, categorical_cols


def tune_lightgbm(X_train, y_train, X_valid, y_valid, n_trials=20):
    """Hyperparameter tuning for LightGBM using Optuna"""
    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'objective': 'binary',
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': 42,
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        pred = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, pred)
    
    study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_params.update({
        'n_estimators': 1000,
        'objective': 'binary',
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    })
    
    return best_params, study.best_value


def train_models(X_train, y_train, X_valid, y_valid, preprocessor, tune_hyperparams=False):
    """Train multiple models with optional hyperparameter tuning"""
    print("\n" + "="*60)
    print("STEP 9: Training Models...")
    print("="*60)
    
    models = {}
    results = {}
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)
    
    # Calculate class weight
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    # 1. Logistic Regression Baseline
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1, random_state=42)
    lr.fit(X_train_processed, y_train)
    lr_pred = lr.predict_proba(X_valid_processed)[:, 1]
    lr_auc = roc_auc_score(y_valid, lr_pred)
    models['LogisticRegression'] = lr
    results['LogisticRegression'] = {'auc': lr_auc, 'predictions': lr_pred}
    print(f"   AUC: {lr_auc:.4f}")
    
    # 2. LightGBM (with optional tuning)
    print("\n2. Training LightGBM...")
    if tune_hyperparams:
        print("   Tuning hyperparameters (this may take a while)...")
        best_params, best_score = tune_lightgbm(X_train_processed, y_train, X_valid_processed, y_valid, n_trials=20)
        print(f"   Best validation AUC during tuning: {best_score:.4f}")
        lgbm = lgb.LGBMClassifier(**best_params)
    else:
        lgbm = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )
    
    lgbm.fit(
        X_train_processed, y_train,
        eval_set=[(X_valid_processed, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    lgbm_pred = lgbm.predict_proba(X_valid_processed)[:, 1]
    lgbm_auc = roc_auc_score(y_valid, lgbm_pred)
    models['LightGBM'] = lgbm
    results['LightGBM'] = {'auc': lgbm_auc, 'predictions': lgbm_pred}
    print(f"   Final AUC: {lgbm_auc:.4f}")
    
    # 3. XGBoost
    print("\n3. Training XGBoost...")
    # In XGBoost 3.x, early_stopping_rounds goes in constructor, not fit()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric='auc',
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=42
    )
    xgb_model.fit(
        X_train_processed, y_train,
        eval_set=[(X_valid_processed, y_valid)],
        verbose=False
    )
    xgb_pred = xgb_model.predict_proba(X_valid_processed)[:, 1]
    xgb_auc = roc_auc_score(y_valid, xgb_pred)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {'auc': xgb_auc, 'predictions': xgb_pred}
    print(f"   AUC: {xgb_auc:.4f}")
    
    # 4. CatBoost
    print("\n4. Training CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        class_weights=[1, pos_weight],
        random_seed=42,
        verbose=False
    )
    cat_model.fit(
        X_train_processed, y_train,
        eval_set=(X_valid_processed, y_valid),
        early_stopping_rounds=50
    )
    cat_pred = cat_model.predict_proba(X_valid_processed)[:, 1]
    cat_auc = roc_auc_score(y_valid, cat_pred)
    models['CatBoost'] = cat_model
    results['CatBoost'] = {'auc': cat_auc, 'predictions': cat_pred}
    print(f"   AUC: {cat_auc:.4f}")
    
    return models, results


def create_ensemble(models, results, X_valid, y_valid, preprocessor):
    """Create ensemble model"""
    print("\n" + "="*60)
    print("STEP 10: Creating Ensemble...")
    print("="*60)
    
    # Get predictions from all models
    X_valid_processed = preprocessor.transform(X_valid)
    
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict_proba(X_valid_processed)[:, 1]
    
    # Try different ensemble weights
    best_auc = 0
    best_weights = None
    
    # Simple grid search for weights
    weight_options = [
        (0.25, 0.25, 0.25, 0.25),  # Equal weights
        (0.4, 0.3, 0.2, 0.1),      # Favor LightGBM
        (0.3, 0.3, 0.3, 0.1),      # Favor top 3
        (0.5, 0.3, 0.15, 0.05),    # Strong LightGBM
    ]
    
    model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'LogisticRegression']
    
    for weights in weight_options:
        ensemble_pred = (
            weights[0] * predictions['LightGBM'] +
            weights[1] * predictions['XGBoost'] +
            weights[2] * predictions['CatBoost'] +
            weights[3] * predictions['LogisticRegression']
        )
        ensemble_auc = roc_auc_score(y_valid, ensemble_pred)
        if ensemble_auc > best_auc:
            best_auc = ensemble_auc
            best_weights = weights
    
    print(f"Best ensemble AUC: {best_auc:.4f}")
    print(f"Best weights: {dict(zip(model_names, best_weights))}")
    
    return best_weights, best_auc


def evaluate_model(model, X_test, y_test, preprocessor, model_name):
    """Evaluate model on test set"""
    X_test_processed = preprocessor.transform(X_test)
    
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = model.predict(X_test_processed)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Test Results:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {'auc': auc, 'pr_auc': pr_auc, 'predictions': y_pred_proba}


def main():
    """Main pipeline"""
    print("="*60)
    print("ADVANCED DEFAULT PREDICTION MODEL PIPELINE")
    print("="*60)
    
    # Step 1-3: Load and merge data
    df = merge_all_data()
    
    # Step 4: Clean data
    df_clean = clean_data(df)
    
    # Step 5: Engineer features
    df_feat = engineer_features(df_clean)
    
    # Step 6: Handle missing values
    df_final = handle_missing_values(df_feat)
    
    # Step 7: Prepare model data
    X, y, feature_cols = prepare_model_data(df_final)
    
    # Step 8: Train/Validation/Test Split
    print("\n" + "="*60)
    print("STEP 8: Train/Validation/Test Split...")
    print("="*60)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"Train: {X_train.shape[0]}, Valid: {X_valid.shape[0]}, Test: {X_test.shape[0]}")
    
    # Create preprocessing pipeline
    preprocessor, numeric_cols, categorical_cols = create_preprocessing_pipeline(X_train)
    
    # Step 9: Train models (set tune_hyperparams=True for hyperparameter tuning)
    models, results = train_models(X_train, y_train, X_valid, y_valid, preprocessor, tune_hyperparams=False)
    
    # Step 10: Create ensemble
    best_weights, ensemble_auc = create_ensemble(models, results, X_valid, y_valid, preprocessor)
    
    # Step 11: Evaluate on test set
    print("\n" + "="*60)
    print("STEP 11: Test Set Evaluation...")
    print("="*60)
    
    test_results = {}
    for name, model in models.items():
        test_results[name] = evaluate_model(model, X_test, y_test, preprocessor, name)
    
    # Evaluate ensemble
    X_test_processed = preprocessor.transform(X_test)
    ensemble_pred = (
        best_weights[0] * models['LightGBM'].predict_proba(X_test_processed)[:, 1] +
        best_weights[1] * models['XGBoost'].predict_proba(X_test_processed)[:, 1] +
        best_weights[2] * models['CatBoost'].predict_proba(X_test_processed)[:, 1] +
        best_weights[3] * models['LogisticRegression'].predict_proba(X_test_processed)[:, 1]
    )
    ensemble_test_auc = roc_auc_score(y_test, ensemble_pred)
    ensemble_test_pr_auc = average_precision_score(y_test, ensemble_pred)
    print(f"\nEnsemble Test Results:")
    print(f"  ROC-AUC: {ensemble_test_auc:.4f}")
    print(f"  PR-AUC: {ensemble_test_pr_auc:.4f}")
    
    # Step 12: Save models
    print("\n" + "="*60)
    print("STEP 12: Saving Models...")
    print("="*60)
    
    # Create directories if they don't exist
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('datas', exist_ok=True)
    
    # Save best single model (highest validation AUC)
    best_single_model_name = max(results, key=lambda x: results[x]['auc'])
    best_single_model = models[best_single_model_name]
    
    # Save models in models/ folder
    joblib.dump(best_single_model, 'models/default_prediction_model_advanced.pkl')
    joblib.dump(preprocessor, 'models/preprocessor_advanced.pkl')
    joblib.dump(feature_cols, 'models/feature_columns_advanced.pkl')
    joblib.dump(best_weights, 'models/ensemble_weights.pkl')
    joblib.dump(models, 'models/all_models_advanced.pkl')
    print("Models saved in 'models/' folder")
    
    # Save feature importance from best model
    if hasattr(best_single_model, 'feature_importances_'):
        # Get feature names after preprocessing (OneHotEncoder creates more features)
        try:
            # Get transformed feature names from preprocessor
            feature_names = preprocessor.get_feature_names_out()
            # Ensure lengths match
            if len(feature_names) == len(best_single_model.feature_importances_):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': best_single_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                # Fallback: use generic names if lengths don't match
                feature_importance = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(best_single_model.feature_importances_))],
                    'importance': best_single_model.feature_importances_
                }).sort_values('importance', ascending=False)
            feature_importance.to_csv('datas/feature_importance_advanced.csv', index=False)
            print("Feature importance saved in 'datas/' folder")
        except Exception as e:
            print(f"Warning: Could not get feature names from preprocessor: {e}")
            # Fallback: save with generic feature names
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(best_single_model.feature_importances_))],
                'importance': best_single_model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance.to_csv('datas/feature_importance_advanced.csv', index=False)
            print("Feature importance saved with generic names in 'datas/' folder")
    
    print(f"Best single model: {best_single_model_name} (AUC: {results[best_single_model_name]['auc']:.4f})")
    print(f"Ensemble AUC: {ensemble_test_auc:.4f}")
    print("\nAll models saved successfully!")
    
    # Save cleaned dataset in datas/ folder
    df_final.to_csv('datas/cleaned_data_advanced.csv', index=False)
    print("Cleaned dataset saved as 'datas/cleaned_data_advanced.csv'")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

