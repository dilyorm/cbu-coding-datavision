"""Example script showing how to predict for judges' evaluation dataset"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling.prediction import predict_from_file

# Example usage:
# python scripts/example_predict_judges.py judges_data.csv

if __name__ == "__main__":
    # Example: Predict from a CSV file
    input_file = "judges_evaluation_data.csv"  # Change this to your file path
    output_file = "judges_predictions.csv"  # Output file
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("="*60)
    print("JUDGES PREDICTION EXAMPLE")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("="*60)
    
    # Run prediction
    results = predict_from_file(
        input_file=input_file,
        models_dir="models",
        output_file=output_file,
        imputation_method='knn',
        id_column='customer_id',
        threshold=0.5
    )
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)
    print(f"\nOutput file contains 3 columns:")
    print(f"  1. customer_id - Customer identifier")
    print(f"  2. probability_pct - Probability of default (0-100%)")
    print(f"  3. default - Default prediction (0 = no default, 1 = default)")
    print(f"\nFirst 10 predictions:")
    print(results.head(10).to_string(index=False))

