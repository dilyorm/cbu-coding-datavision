"""
Script to clean up redundant files and organize project structure
"""
import os
import shutil
from pathlib import Path

def cleanup_project():
    """Remove redundant files and organize project"""
    
    print("="*60)
    print("CLEANING UP PROJECT FOR GITHUB")
    print("="*60)
    
    # Files to delete (redundant)
    files_to_delete = [
        'default_prediction_model.pkl',
        'scaler.pkl',
        'label_encoders.pkl',
        'median_values.pkl',
        'feature_columns.pkl',
        'feature_importance.csv',
        'model_pipeline.py',
        'predict.py',
        'cleaned_data.csv',
        'example_predictions_output.csv',
        'predictions.csv',
        'test_predictions.py',
        'usage.py',
        'example_customers.csv',
        'example_minimal.csv'
    ]
    
    # Directories to delete
    dirs_to_delete = [
        '__pycache__',
        'catboost_info'
    ]
    
    # Files to move to models/
    files_to_models = [
        'all_models_advanced.pkl',
        'default_prediction_model_advanced.pkl',
        'ensemble_weights.pkl',
        'feature_columns_advanced.pkl',
        'preprocessor_advanced.pkl'
    ]
    
    # Files to move to datas/
    files_to_datas = [
        'cleaned_data_advanced.csv',
        'feature_importance_advanced.csv'
    ]
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('datas', exist_ok=True)
    
    # Delete redundant files
    print("\n1. Deleting redundant files...")
    deleted_count = 0
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"   Error deleting {file}: {e}")
    
    # Delete redundant directories
    print("\n2. Deleting redundant directories...")
    for dir_name in dirs_to_delete:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"   Deleted: {dir_name}/")
            except Exception as e:
                print(f"   Error deleting {dir_name}: {e}")
    
    # Move files to models/
    print("\n3. Moving model files to models/...")
    moved_models = 0
    for file in files_to_models:
        if os.path.exists(file):
            try:
                dest = os.path.join('models', file)
                if os.path.exists(dest):
                    os.remove(dest)  # Remove if exists
                shutil.move(file, dest)
                print(f"   Moved: {file} -> models/{file}")
                moved_models += 1
            except Exception as e:
                print(f"   Error moving {file}: {e}")
    
    # Move files to datas/
    print("\n4. Moving data files to datas/...")
    moved_datas = 0
    for file in files_to_datas:
        if os.path.exists(file):
            try:
                dest = os.path.join('datas', file)
                if os.path.exists(dest):
                    os.remove(dest)  # Remove if exists
                shutil.move(file, dest)
                print(f"   Moved: {file} -> datas/{file}")
                moved_datas += 1
            except Exception as e:
                print(f"   Error moving {file}: {e}")
    
    print("\n" + "="*60)
    print("CLEANUP COMPLETE!")
    print("="*60)
    print(f"Deleted {deleted_count} redundant files")
    print(f"Moved {moved_models} files to models/")
    print(f"Moved {moved_datas} files to datas/")
    print("\nProject is now organized and ready for GitHub!")

if __name__ == "__main__":
    cleanup_project()

