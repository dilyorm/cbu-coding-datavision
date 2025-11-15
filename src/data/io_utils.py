"""IO utilities for saving and loading models and data"""
import os
import joblib
import pandas as pd
import json


def save_artifacts(final_model, feature_cols, cat_feature_indices, feature_names_list,
                   results, df_final, paths, test_results=None):
    """Save all model artifacts and data for final CatBoost model
    
    Args:
        final_model: Trained CatBoostClassifier
        feature_cols: List of feature column names
        cat_feature_indices: List of categorical feature indices
        feature_names_list: List of feature names (for feature importance)
        results: Results dictionary with CV scores
        df_final: Final processed DataFrame
        paths: Paths configuration object
        test_results: Optional test results dictionary
    """
    print("\n" + "="*60)
    print("STEP 12: Saving Models and Artifacts...")
    print("="*60)
    
    # Create directories if they don't exist
    os.makedirs(paths.models_dir, exist_ok=True)
    os.makedirs(paths.datas_dir, exist_ok=True)
    
    # Save CatBoost model using CatBoost's native save method (more reliable)
    model_path = f'{paths.models_dir}/default_prediction_model_advanced.cbm'
    final_model.save_model(model_path)
    print(f"CatBoost model saved: {model_path}")
    
    # Also save with joblib for compatibility
    joblib.dump(final_model, f'{paths.models_dir}/default_prediction_model_advanced.pkl')
    print(f"CatBoost model saved (joblib): {paths.models_dir}/default_prediction_model_advanced.pkl")
    
    # Save feature columns
    joblib.dump(feature_cols, f'{paths.models_dir}/feature_columns_advanced.pkl')
    print(f"Feature columns saved: {len(feature_cols)} features")
    
    # Save categorical feature indices
    joblib.dump(cat_feature_indices, f'{paths.models_dir}/categorical_feature_indices_advanced.pkl')
    print(f"Categorical feature indices saved: {len(cat_feature_indices)} categorical features")
    
    # Save configuration as JSON for easy loading
    config = {
        'feature_cols': feature_cols,
        'cat_feature_indices': cat_feature_indices,
        'n_features': len(feature_cols),
        'n_categorical': len(cat_feature_indices)
    }
    with open(f'{paths.models_dir}/model_config_advanced.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Model configuration saved: model_config_advanced.json")
    
    # Save feature importance from CatBoost model
    if hasattr(final_model, 'feature_importances_'):
        try:
            if len(feature_names_list) == len(final_model.feature_importances_):
                feature_importance = pd.DataFrame({
                    'feature': feature_names_list,
                    'importance': final_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                # Fallback: use generic names if lengths don't match
                feature_importance = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(final_model.feature_importances_))],
                    'importance': final_model.feature_importances_
                }).sort_values('importance', ascending=False)
            feature_importance.to_csv(f'{paths.datas_dir}/feature_importance_advanced.csv', index=False)
            print("Feature importance saved in 'datas/' folder")
        except Exception as e:
            print(f"Warning: Could not get feature names: {e}")
            # Fallback: save with generic feature names
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(final_model.feature_importances_))],
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance.to_csv(f'{paths.datas_dir}/feature_importance_advanced.csv', index=False)
            print("Feature importance saved with generic names in 'datas/' folder")
    
    # Print results summary
    if 'cv_auc_mean' in results.get('CatBoost', {}):
        print(f"\nCatBoost CV AUC: {results['CatBoost']['cv_auc_mean']:.4f} (+/- {results['CatBoost']['cv_auc_std']:.4f})")
    if test_results:
        print(f"CatBoost Test AUC: {test_results.get('auc', 'N/A'):.4f}")
        print(f"CatBoost Test PR-AUC: {test_results.get('pr_auc', 'N/A'):.4f}")
    
    print("\nAll models and artifacts saved successfully!")
    
    # Save cleaned dataset in datas/ folder
    df_final.to_csv(f'{paths.datas_dir}/cleaned_data_advanced.csv', index=False)
    print("Cleaned dataset saved as 'datas/cleaned_data_advanced.csv'")
