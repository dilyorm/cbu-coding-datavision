"""IO utilities for saving and loading models and data"""
import os
import joblib
import pandas as pd


def save_artifacts(models, results, meta_clf, thresholds, preprocessor, feature_cols, 
                   categorical_cols, df_final, paths):
    """Save all model artifacts and data - CatBoost only"""
    print("\n" + "="*60)
    print("STEP 12: Saving Models...")
    print("="*60)
    
    # Create directories if they don't exist
    os.makedirs(paths.models_dir, exist_ok=True)
    os.makedirs(paths.datas_dir, exist_ok=True)
    
    # Get CatBoost model
    catboost_model = models['CatBoost']
    
    # Save CatBoost model and related data
    joblib.dump(catboost_model, f'{paths.models_dir}/default_prediction_model_advanced.pkl')
    joblib.dump(feature_cols, f'{paths.models_dir}/feature_columns_advanced.pkl')
    if categorical_cols:
        joblib.dump(categorical_cols, f'{paths.models_dir}/categorical_columns_advanced.pkl')
    if thresholds:
        joblib.dump(thresholds, f'{paths.models_dir}/optimal_thresholds_advanced.pkl')
    
    # Save model dict (without helper data)
    models_to_save = {'CatBoost': catboost_model}
    joblib.dump(models_to_save, f'{paths.models_dir}/all_models_advanced.pkl')
    print("Models saved in 'models/' folder")
    
    # Save feature importance from CatBoost model
    if hasattr(catboost_model, 'feature_importances_'):
        try:
            # Get feature names (CatBoost uses original feature names)
            feature_names = models['CatBoost_X_train'].columns.tolist()
            if len(feature_names) == len(catboost_model.feature_importances_):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': catboost_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                # Fallback: use generic names if lengths don't match
                feature_importance = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(catboost_model.feature_importances_))],
                    'importance': catboost_model.feature_importances_
                }).sort_values('importance', ascending=False)
            feature_importance.to_csv(f'{paths.datas_dir}/feature_importance_advanced.csv', index=False)
            print("Feature importance saved in 'datas/' folder")
        except Exception as e:
            print(f"Warning: Could not get feature names: {e}")
            # Fallback: save with generic feature names
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(catboost_model.feature_importances_))],
                'importance': catboost_model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance.to_csv(f'{paths.datas_dir}/feature_importance_advanced.csv', index=False)
            print("Feature importance saved with generic names in 'datas/' folder")
    
    print(f"CatBoost Validation AUC: {results['CatBoost']['auc']:.4f}")
    print("\nAll models saved successfully!")
    
    # Save cleaned dataset in datas/ folder
    df_final.to_csv(f'{paths.datas_dir}/cleaned_data_advanced.csv', index=False)
    print("Cleaned dataset saved as 'datas/cleaned_data_advanced.csv'")
