"""Hyperparameter tuning functions"""
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
import catboost as cb


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


def tune_catboost(X_train, y_train, X_valid, y_valid, categorical_cols, n_trials=20):
    """Hyperparameter tuning for CatBoost using Optuna"""
    def objective(trial):
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 0.5, 10.0),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "class_weights": [1, pos_weight],
            "random_seed": 42,
            "verbose": False
        }
        
        # Prepare categorical features
        cat_feature_indices = [X_train.columns.get_loc(c) for c in categorical_cols if c in X_train.columns]
        
        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_feature_indices,
            early_stopping_rounds=50
        )
        pred = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, pred)
    
    study = optuna.create_study(direction="maximize", study_name='catboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    best_params.update({
        "iterations": 1000,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "class_weights": [1, pos_weight],
        "random_seed": 42,
        "verbose": False
    })
    
    return best_params, study.best_value

