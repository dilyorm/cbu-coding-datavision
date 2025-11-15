"""Entry point script for training the default prediction model"""
import sys
import os

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change to project root directory
os.chdir(project_root)

from src.pipelines.default_prediction import run_default_pipeline

if __name__ == "__main__":
    run_default_pipeline()

