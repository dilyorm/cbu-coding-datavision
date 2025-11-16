"""Main script to execute the complete ML pipeline workflow."""
import argparse
import sys
from pathlib import Path
import time
from datetime import datetime

# Import main functions from each module
from src.data.prepare_data import main as prepare_data_main
from src.features.run_feature_selection import main as feature_selection_main
from src.training.train_all_feature_sets import main as train_models_main
from src.ensemble.create_meta_features import main as create_meta_features_main
from src.ensemble.train_stacking_models import main as train_stacking_main
from src.ensemble.ensemble import main as ensemble_main
from src.ensemble.generate_final_predictions import main as generate_final_main


def print_section(title, step_num, total_steps):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("=" * 80)


def print_step_complete(step_name, elapsed_time):
    """Print step completion message."""
    print(f"\n✓ {step_name} completed in {elapsed_time:.2f} seconds")
    print("-" * 80)


def check_prerequisites(step_name, required_paths):
    """Check if required files/directories exist."""
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        print(f"\n✗ Error: Missing prerequisites for {step_name}:")
        for path in missing:
            print(f"  - {path}")
        return False
    return True


def run_workflow(
    skip_data_prep=False,
    skip_feature_selection=False,
    skip_training=False,
    skip_meta_features=False,
    skip_stacking=False,
    skip_ensemble=False,
    skip_final=False
):
    """Execute the complete ML pipeline workflow.
    
    Args:
        skip_data_prep: Skip data preparation step
        skip_feature_selection: Skip feature selection step
        skip_training: Skip model training step
        skip_meta_features: Skip meta features creation step
        skip_stacking: Skip stacking models training step
        skip_ensemble: Skip ensemble creation step
        skip_final: Skip final predictions generation step
    """
    start_time = time.time()
    total_steps = 7
    current_step = 0
    
    print("\n" + "=" * 80)
    print("DEFAULT PREDICTION ML PIPELINE - COMPLETE WORKFLOW")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Data Preparation
    current_step += 1
    if not skip_data_prep:
        print_section("DATA PREPARATION", current_step, total_steps)
        step_start = time.time()
        try:
            prepare_data_main()
            elapsed = time.time() - step_start
            print_step_complete("Data preparation", elapsed)
        except Exception as e:
            print(f"\n✗ Error in data preparation: {e}")
            print("Workflow aborted.")
            return False
    else:
        print_section("DATA PREPARATION (SKIPPED)", current_step, total_steps)
        if not check_prerequisites("Feature Selection", ["data/processed/v2"]):
            print("Warning: Data preparation was skipped but required files are missing.")
    
    # Step 2: Feature Selection (Optional)
    current_step += 1
    if not skip_feature_selection:
        print_section("FEATURE SELECTION", current_step, total_steps)
        step_start = time.time()
        try:
            # Check prerequisites
            if not check_prerequisites("Feature Selection", ["data/processed/v2"]):
                print("Skipping feature selection due to missing prerequisites.")
            else:
                feature_selection_main()
                elapsed = time.time() - step_start
                print_step_complete("Feature selection", elapsed)
        except Exception as e:
            print(f"\n✗ Error in feature selection: {e}")
            print("Continuing workflow (feature selection is optional)...")
    else:
        print_section("FEATURE SELECTION (SKIPPED)", current_step, total_steps)
    
    # Step 3: Train All Models
    current_step += 1
    if not skip_training:
        print_section("MODEL TRAINING", current_step, total_steps)
        step_start = time.time()
        try:
            # Check prerequisites
            if not check_prerequisites("Model Training", ["data/processed/v2"]):
                print("Error: Cannot proceed with training. Missing required data files.")
                return False
            
            train_models_main()
            elapsed = time.time() - step_start
            print_step_complete("Model training", elapsed)
        except Exception as e:
            print(f"\n✗ Error in model training: {e}")
            print("Workflow aborted.")
            return False
    else:
        print_section("MODEL TRAINING (SKIPPED)", current_step, total_steps)
        if not check_prerequisites("Meta Features", ["data/models"]):
            print("Warning: Model training was skipped but required models are missing.")
    
    # Step 4: Create Meta Features
    current_step += 1
    if not skip_meta_features:
        print_section("META FEATURES CREATION", current_step, total_steps)
        step_start = time.time()
        try:
            # Check prerequisites
            if not check_prerequisites("Meta Features", ["data/models"]):
                print("Error: Cannot create meta features. Missing model predictions.")
                return False
            
            create_meta_features_main()
            elapsed = time.time() - step_start
            print_step_complete("Meta features creation", elapsed)
        except Exception as e:
            print(f"\n✗ Error in meta features creation: {e}")
            print("Workflow aborted.")
            return False
    else:
        print_section("META FEATURES CREATION (SKIPPED)", current_step, total_steps)
        if not check_prerequisites("Stacking", ["data/stacking"]):
            print("Warning: Meta features creation was skipped but required files are missing.")
    
    # Step 5: Train Stacking Models
    current_step += 1
    if not skip_stacking:
        print_section("STACKING MODELS TRAINING", current_step, total_steps)
        step_start = time.time()
        try:
            # Check prerequisites
            if not check_prerequisites("Stacking", ["data/stacking"]):
                print("Error: Cannot train stacking models. Missing stacking dataset.")
                return False
            
            train_stacking_main()
            elapsed = time.time() - step_start
            print_step_complete("Stacking models training", elapsed)
        except Exception as e:
            print(f"\n✗ Error in stacking models training: {e}")
            print("Workflow aborted.")
            return False
    else:
        print_section("STACKING MODELS TRAINING (SKIPPED)", current_step, total_steps)
    
    # Step 6: Create Ensemble
    current_step += 1
    if not skip_ensemble:
        print_section("ENSEMBLE CREATION", current_step, total_steps)
        step_start = time.time()
        try:
            # Check prerequisites
            if not check_prerequisites("Ensemble", ["data/models"]):
                print("Error: Cannot create ensemble. Missing model predictions.")
                return False
            
            ensemble_main()
            elapsed = time.time() - step_start
            print_step_complete("Ensemble creation", elapsed)
        except Exception as e:
            print(f"\n✗ Error in ensemble creation: {e}")
            print("Workflow aborted.")
            return False
    else:
        print_section("ENSEMBLE CREATION (SKIPPED)", current_step, total_steps)
    
    # Step 7: Generate Final Predictions
    current_step += 1
    if not skip_final:
        print_section("FINAL PREDICTIONS GENERATION", current_step, total_steps)
        step_start = time.time()
        try:
            # Check prerequisites
            if not check_prerequisites("Final Predictions", ["data/final_submission"]):
                print("Warning: Ensemble results may be missing, but continuing...")
            
            generate_final_main()
            elapsed = time.time() - step_start
            print_step_complete("Final predictions generation", elapsed)
        except Exception as e:
            print(f"\n✗ Error in final predictions generation: {e}")
            print("Workflow completed with errors.")
            return False
    else:
        print_section("FINAL PREDICTIONS GENERATION (SKIPPED)", current_step, total_steps)
    
    # Summary
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Total execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nFinal submission file: data/final_submission/submission.csv")
    print("=" * 80)
    
    return True


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Execute the complete ML pipeline workflow for default prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow
  python main.py
  
  # Skip feature selection (optional step)
  python main.py --skip-feature-selection
  
  # Resume from meta features (skip data prep, feature selection, and training)
  python main.py --skip-data-prep --skip-feature-selection --skip-training
  
  # Only run ensemble and final predictions
  python main.py --skip-data-prep --skip-feature-selection --skip-training \\
                 --skip-meta-features --skip-stacking
        """
    )
    
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Skip data preparation step'
    )
    parser.add_argument(
        '--skip-feature-selection',
        action='store_true',
        help='Skip feature selection step (optional)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )
    parser.add_argument(
        '--skip-meta-features',
        action='store_true',
        help='Skip meta features creation step'
    )
    parser.add_argument(
        '--skip-stacking',
        action='store_true',
        help='Skip stacking models training step'
    )
    parser.add_argument(
        '--skip-ensemble',
        action='store_true',
        help='Skip ensemble creation step'
    )
    parser.add_argument(
        '--skip-final',
        action='store_true',
        help='Skip final predictions generation step'
    )
    
    args = parser.parse_args()
    
    # Run workflow
    success = run_workflow(
        skip_data_prep=args.skip_data_prep,
        skip_feature_selection=args.skip_feature_selection,
        skip_training=args.skip_training,
        skip_meta_features=args.skip_meta_features,
        skip_stacking=args.skip_stacking,
        skip_ensemble=args.skip_ensemble,
        skip_final=args.skip_final
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

