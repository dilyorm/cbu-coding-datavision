"""Model evaluation functions"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix
)


def find_best_threshold(y_true, y_scores, beta=1.0):
    """Find optimal classification threshold using F-beta score"""
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f_beta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-9)
    best_idx = np.argmax(f_beta)
    return thr[best_idx], prec[best_idx], rec[best_idx], f_beta[best_idx]


def evaluate_model(model, X_test, y_test, preprocessor, model_name, 
                   best_threshold=0.5, use_raw_catboost=False, X_test_raw=None):
    """Evaluate model on test set with optimal threshold"""
    if use_raw_catboost and X_test_raw is not None:
        # CatBoost uses raw data
        X_test_processed = X_test_raw
    elif preprocessor is not None:
        X_test_processed = preprocessor.transform(X_test)
    else:
        # No preprocessing needed (CatBoost with raw data)
        X_test_processed = X_test
    
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Test Results (threshold={best_threshold:.4f}):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {'auc': auc, 'pr_auc': pr_auc, 'predictions': y_pred_proba, 'threshold': best_threshold}

