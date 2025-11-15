"""Script to predict defaults for judges' evaluation dataset"""
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modeling.prediction import predict_from_file
from config.default_config import paths


def main():
    """Main function to run predictions for judges"""
    parser = argparse.ArgumentParser(description='Predict defaults for judges evaluation dataset')
    parser.add_argument('input_file', type=str, help='Path to input CSV/Excel file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path (default: input_file_predictions.csv)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory containing saved model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    parser.add_argument('--id_column', type=str, default='customer_id', help='Name of ID column (default: customer_id)')
    parser.add_argument('--imputation', type=str, default='knn', choices=['knn', 'iterative', 'median'],
                       help='Imputation method (default: knn)')
    
    args = parser.parse_args()
    
    # Run predictions
    results_df = predict_from_file(
        input_file=args.input_file,
        models_dir=args.models_dir,
        output_file=args.output,
        imputation_method=args.imputation,
        id_column=args.id_column,
        threshold=args.threshold
    )
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)
    print(f"\nOutput columns:")
    print(f"  1. customer_id: Customer identifier")
    print(f"  2. probability_pct: Probability of default (percentage)")
    print(f"  3. default: Default prediction (0 = no default, 1 = default)")
    
    return results_df


if __name__ == "__main__":
    main()

