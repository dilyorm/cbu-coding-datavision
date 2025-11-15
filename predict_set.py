"""
Prediction script for default prediction model
Supports both single predictions and batch predictions
"""
import pandas as pd
import numpy as np
from predict_advanced import predict_default_advanced, predict_batch_advanced
import os


def predict_single_example():
    """Example: Single customer prediction"""
    print("="*60)
    print("SINGLE CUSTOMER PREDICTION")
    print("="*60)
    
    # Example customer data
    customer_data = {
        'customer_id': 10000,
        'age': 41,
        'annual_income': 61800,
        'credit_score': 696,
        'debt_to_income_ratio': 0.258,
        'loan_amount': 50000,
        'interest_rate': 5.5,
        'employment_type': 'full_time',
        'education': 'Graduate',
        'marital_status': 'Married'
    }
    
    result = predict_default_advanced(customer_data, use_ensemble=True)
    
    print(f"\nCustomer ID: {customer_data['customer_id']}")
    print(f"Default Probability: {result['default_probability']:.4f}")
    print(f"Predicted Default: {result['predicted_default']}")
    print(f"Risk Level: {result['risk_level']}")
    
    return result


def predict_batch_example():
    """Example: Batch prediction from CSV"""
    print("\n" + "="*60)
    print("BATCH PREDICTION FROM CSV")
    print("="*60)
    
    # Check if example file exists
    input_file = 'example_customers.csv'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please create it first.")
        return None
    
    output_file = 'datas/predictions_batch.csv'
    results = predict_batch_advanced(
        input_file, 
        use_ensemble=True, 
        output_file=output_file
    )
    
    print(f"\nProcessed {len(results)} customers")
    print(f"Results saved to: {output_file}")
    print("\nSummary Statistics:")
    print(results[['default_probability', 'predicted_default', 'risk_level']].describe())
    print("\nRisk Level Distribution:")
    print(results['risk_level'].value_counts())
    
    return results


def predict_from_dataframe(df):
    """Predict from a pandas DataFrame"""
    print("="*60)
    print("PREDICTING FROM DATAFRAME")
    print("="*60)
    
    results = predict_batch_advanced(
        df, 
        use_ensemble=True, 
        output_file='datas/predictions_dataframe.csv'
    )
    
    print(f"\nProcessed {len(results)} records")
    return results


def create_example_input():
    """Create example input CSV file"""
    example_data = pd.DataFrame({
        'customer_id': [100001, 100002, 100003, 100004, 100005],
        'age': [35, 28, 52, 22, 45],
        'annual_income': [75000, 45000, 95000, 30000, 120000],
        'credit_score': [720, 650, 780, 580, 750],
        'debt_to_income_ratio': [0.25, 0.45, 0.15, 0.65, 0.20],
        'loan_amount': [50000, 35000, 60000, 40000, 80000],
        'interest_rate': [5.5, 7.2, 4.8, 9.5, 5.0],
        'employment_type': ['full_time', 'full_time', 'full_time', 'full_time', 'full_time'],
        'education': ['Bachelor', 'High School', 'Graduate', 'Some College', 'Graduate'],
        'marital_status': ['Married', 'Single', 'Married', 'Single', 'Married'],
        'employment_length': [5.0, 2.5, 15.0, 1.0, 20.0],
        'num_dependents': [1, 0, 2, 0, 3]
    })
    
    os.makedirs('datas', exist_ok=True)
    example_data.to_csv('datas/example_input.csv', index=False)
    print("Created example input file: datas/example_input.csv")
    return example_data


if __name__ == "__main__":
    import sys
    
    # Create example input if it doesn't exist
    if not os.path.exists('datas/example_input.csv'):
        print("Creating example input file...")
        create_example_input()
    
    if len(sys.argv) > 1:
        # Command line usage
        if sys.argv[1] == 'single':
            predict_single_example()
        elif sys.argv[1] == 'batch':
            input_file = sys.argv[2] if len(sys.argv) > 2 else 'datas/example_input.csv'
            output_file = sys.argv[3] if len(sys.argv) > 3 else 'datas/predictions_batch.csv'
            results = predict_batch_advanced(input_file, use_ensemble=True, output_file=output_file)
            print(f"\nPredictions saved to: {output_file}")
        else:
            print("Usage:")
            print("  python predict_set.py single          - Single prediction example")
            print("  python predict_set.py batch [input] [output]  - Batch prediction")
    else:
        # Run examples
        print("Running prediction examples...\n")
        
        # Single prediction
        predict_single_example()
        
        # Batch prediction
        if os.path.exists('example_customers.csv'):
            predict_batch_example()
        else:
            print("\nNote: example_customers.csv not found. Using example_input.csv...")
            if os.path.exists('datas/example_input.csv'):
                results = predict_batch_advanced(
                    'datas/example_input.csv',
                    use_ensemble=True,
                    output_file='datas/predictions_batch.csv'
                )
                print(f"\nProcessed {len(results)} customers")
                print("Results saved to: datas/predictions_batch.csv")

