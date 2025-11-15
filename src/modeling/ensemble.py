"""Ensemble model creation"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def create_stacking_ensemble(models, results, X_valid, y_valid, preprocessor):
    """Create stacking ensemble using meta-model (Logistic Regression)"""
    print("\n" + "="*60)
    print("STEP 10: Creating Stacking Ensemble...")
    print("="*60)
    
    # Get predictions from all base models on validation set
    X_valid_processed = preprocessor.transform(X_valid)
    
    # Collect base model predictions
    meta_X = []
    model_names = []
    
    # LightGBM
    meta_X.append(results['LightGBM']['predictions'])
    model_names.append('LightGBM')
    
    # XGBoost
    meta_X.append(results['XGBoost']['predictions'])
    model_names.append('XGBoost')
    
    # CatBoost
    if 'CatBoost_X_valid' in models:
        X_valid_cat = models['CatBoost_X_valid']
        cat_pred = models['CatBoost'].predict_proba(X_valid_cat)[:, 1]
    else:
        cat_pred = results['CatBoost']['predictions']
    meta_X.append(cat_pred)
    model_names.append('CatBoost')
    
    # Logistic Regression
    meta_X.append(results['LogisticRegression']['predictions'])
    model_names.append('LogisticRegression')
    
    # Stack predictions as features for meta-model
    meta_X = np.column_stack(meta_X)
    meta_y = y_valid.values
    
    # Train meta-model (Logistic Regression)
    meta_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    meta_clf.fit(meta_X, meta_y)
    
    # Evaluate stacking ensemble
    ensemble_pred = meta_clf.predict_proba(meta_X)[:, 1]
    ensemble_auc = roc_auc_score(y_valid, ensemble_pred)
    
    print(f"Stacking ensemble AUC: {ensemble_auc:.4f}")
    print(f"Meta-model coefficients: {dict(zip(model_names, meta_clf.coef_[0]))}")
    
    return meta_clf, ensemble_auc

