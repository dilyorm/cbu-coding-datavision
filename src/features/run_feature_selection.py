"""Main script to run feature selection pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.features.feature_selection import (
    zero_importance_filter,
    permutation_importance_filter,
    forward_feature_selection,
    # stepped_permutation_selection,  # Commented out
    train_baseline_lgbm
)
from sklearn.metrics import roc_auc_score, log_loss


def load_preprocessed_data(version='v1'):
    """Load preprocessed train/val/test splits for a specific version.
    
    Args:
        version: Version name ('v1', 'v2', or 'v3')
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names)
    """
    data_dir = Path(f'data/processed/{version}')
    
    print(f"Loading preprocessed data for {version}...")
    X_train = pd.read_csv(data_dir / 'X_train.csv')
    y_train = pd.read_csv(data_dir / 'y_train.csv')['default']
    X_test = pd.read_csv(data_dir / 'X_test.csv')
    y_test = pd.read_csv(data_dir / 'y_test.csv')['default']
    
    # Load feature names
    with open(data_dir / 'feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {version} data:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    
    return X_train, y_train, X_test, y_test, feature_names


def define_feature_groups(feature_names):
    """Define feature groups for forward selection.
    
    Groups features by their prefix/type to enable group-wise selection.
    """
    groups = {
        'original': [],
        'temporal': [],
        'rank': [],
        'advanced_ratios': [],
        'population_stats': [],
        'binned_categoricals': []
    }
    
    for feat in feature_names:
        if any(x in feat for x in ['rank_pct', 'rank_pct_by']):
            groups['rank'].append(feat)
        elif any(x in feat for x in ['hour_sin', 'hour_cos', 'weekend', 'account_age']):
            groups['temporal'].append(feat)
        elif any(x in feat for x in ['adjusted', 'burden', 'depth', 'available_credit_to', 'payment_to_available']):
            groups['advanced_ratios'].append(feat)
        elif any(x in feat for x in ['zscore', 'quartile', 'dist_from']):
            groups['population_stats'].append(feat)
        elif 'binned' in feat:
            groups['binned_categoricals'].append(feat)
        elif 'oneHot' in feat:
            # One-hot encoded features go to original
            groups['original'].append(feat)
        else:
            groups['original'].append(feat)
    
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    
    print(f"\nFeature groups:")
    for group_name, features in groups.items():
        print(f"  {group_name}: {len(features)} features")
    
    return groups


def generate_selection_report(stage_results, output_dir):
    """Generate comprehensive selection report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FEATURE SELECTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Stage 1: Zero Importance
    if 'stage1' in stage_results:
        s1 = stage_results['stage1']
        report_lines.append("STAGE 1: Zero Importance Filtering")
        report_lines.append("-" * 80)
        report_lines.append(f"Initial features: {s1['initial_count']}")
        report_lines.append(f"Removed: {s1['removed_count']}")
        report_lines.append(f"Remaining: {s1['final_count']} ({s1['final_count']/s1['initial_count']*100:.1f}%)")
        report_lines.append(f"Baseline AUC: {s1['metrics']['auc']:.6f}")
        report_lines.append(f"Baseline Log Loss: {s1['metrics']['log_loss']:.6f}")
        report_lines.append("")
    
    # Stage 2: Permutation Importance
    if 'stage2' in stage_results:
        s2 = stage_results['stage2']
        report_lines.append("STAGE 2: Permutation Importance Filtering")
        report_lines.append("-" * 80)
        report_lines.append(f"Initial features: {s2['initial_count']}")
        report_lines.append(f"Removed: {s2['removed_count']}")
        report_lines.append(f"Remaining: {s2['final_count']} ({s2['final_count']/s2['initial_count']*100:.1f}%)")
        report_lines.append(f"Final AUC: {s2['metrics']['auc']:.6f}")
        report_lines.append(f"Final Log Loss: {s2['metrics']['log_loss']:.6f}")
        report_lines.append("")
    
    # Stage 2B: Stepped Permutation (if used)
    if 'stage2b' in stage_results:
        s2b = stage_results['stage2b']
        report_lines.append("STAGE 2B: Stepped Permutation Selection")
        report_lines.append("-" * 80)
        report_lines.append(f"Initial features: {s2b['initial_count']}")
        report_lines.append(f"Final features: {s2b['final_count']}")
        report_lines.append(f"Iterations: {len(s2b['history'])}")
        if s2b['history']:
            report_lines.append(f"Final AUC: {s2b['history'][-1]['auc']:.6f}")
        report_lines.append("")
    
    # Stage 3: Forward Selection
    if 'stage3' in stage_results:
        s3 = stage_results['stage3']
        report_lines.append("STAGE 3: Forward Feature Selection")
        report_lines.append("-" * 80)
        report_lines.append(f"Final features: {s3['final_count']}")
        report_lines.append(f"Groups tested: {len(s3['history'])}")
        for step in s3['history']:
            status = step.get('status', 'accepted')
            improvement = step.get('improvement', 0)
            report_lines.append(f"  {step['group']}: {step['features_added']} features, "
                              f"AUC={step['auc']:.6f}, improvement={improvement:+.6f} ({status})")
        report_lines.append("")
    
    # Final Summary
    if 'final' in stage_results:
        final = stage_results['final']
        report_lines.append("FINAL SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total features selected: {final['count']}")
        report_lines.append(f"Reduction: {final['reduction_pct']:.1f}%")
        report_lines.append(f"Final validation AUC: {final['val_auc']:.6f}")
        report_lines.append(f"Final validation Log Loss: {final['val_log_loss']:.6f}")
        report_lines.append(f"Test AUC: {final['test_auc']:.6f}")
        report_lines.append(f"Test Log Loss: {final['test_log_loss']:.6f}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'selection_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text


def run_feature_selection_for_version(version='v1'):
    """Run feature selection pipeline for a specific version.
    
    Args:
        version: Version name ('v1', 'v2', or 'v3')
    
    Returns:
        List of selected feature names
    """
    print("\n" + "=" * 80)
    print(f"FEATURE SELECTION PIPELINE FOR {version.upper()}")
    print("=" * 80)
    
    # Load data for this version
    X_train, y_train, X_test, y_test, feature_names = load_preprocessed_data(version)
    
    # Create version-specific output directory
    output_dir = Path(f'data/feature_selection/{version}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage_results = {}
    selected_features = feature_names.copy()
    
    # Stage 1: Zero Importance Filtering
    print("\n" + "=" * 80)
    print("RUNNING STAGE 1: Zero Importance Filtering")
    print("=" * 80)
    
    selected_features, importances, metrics = zero_importance_filter(
        X_train, y_train, selected_features, n_estimators=500
    )
    
    stage_results['stage1'] = {
        'initial_count': len(feature_names),
        'final_count': len(selected_features),
        'removed_count': len(feature_names) - len(selected_features),
        'metrics': metrics,
        'importances': importances
    }
    
    # Update dataframes with selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # Save intermediate results
    with open(output_dir / 'stage1_selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    # Stage 2: Permutation Importance (optional - can be slow)
    print("\n" + "=" * 80)
    print("RUNNING STAGE 2: Permutation Importance Filtering")
    print("=" * 80)
    print("Note: This stage can take a long time. Consider using Stage 2B instead.")
    
    use_stage2 = True  # Set to False to skip
    if use_stage2:
        initial_count = len(selected_features)
        selected_features, perm_importances, metrics = permutation_importance_filter(
            X_train, y_train, selected_features,
            n_repeats=3,  # Reduced for speed
            threshold=0.0,
            n_estimators=300,
            cv_folds=3,  # Reduced for speed
            random_seeds=[42, 206]
        )
        
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        stage_results['stage2'] = {
            'initial_count': initial_count,
            'final_count': len(selected_features),
            'removed_count': initial_count - len(selected_features),
            'metrics': metrics,
            'importances': perm_importances
        }
        
        with open(output_dir / 'stage2_selected_features.txt', 'w') as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
    else:
        print("Skipping Stage 2 (use_stage2=False)")
    
    # Stage 2B: Stepped Permutation (alternative to Stage 2)
    # Commented out - stepped_permutation_selection
    # print("\n" + "=" * 80)
    # print("RUNNING STAGE 2B: Stepped Permutation Selection")
    # print("=" * 80)
    # 
    # initial_count = len(selected_features)
    # selected_features_step2b, step2b_history = stepped_permutation_selection(
    #     X_train, y_train, selected_features,
    #     drop_pct=0.2,
    #     max_iterations=5,  # Limit iterations for speed
    #     n_estimators=300
    # )
    # 
    # if len(selected_features_step2b) < len(selected_features):
    #     selected_features = selected_features_step2b
    #     X_train = X_train[selected_features]
    #     X_test = X_test[selected_features]
    # 
    # stage_results['stage2b'] = {
    #     'initial_count': initial_count,
    #     'final_count': len(selected_features),
    #     'history': step2b_history
    # }
    
    # Skip Stage 2B
    stage_results['stage2b'] = {
        'initial_count': len(selected_features),
        'final_count': len(selected_features),
        'history': []
    }
    
    # Stage 3: Forward Feature Selection (optional)
    print("\n" + "=" * 80)
    print("RUNNING STAGE 3: Forward Feature Selection")
    print("=" * 80)
    
    use_stage3 = False  # Set to True to enable (can be slow)
    if use_stage3:
        feature_groups = define_feature_groups(selected_features)
        selected_features, forward_history = forward_feature_selection(
            X_train, y_train, feature_groups,
            threshold=0.0003,
            n_estimators=300
        )
        
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        stage_results['stage3'] = {
            'final_count': len(selected_features),
            'history': forward_history
        }
    else:
        print("Skipping Stage 3 (use_stage3=False)")
    
    # Final validation on test set
    print("\n" + "=" * 80)
    print("FINAL VALIDATION")
    print("=" * 80)
    
    print(f"Final feature count: {len(selected_features)}")
    print(f"Reduction: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")
    
    # Train final model
    print("\nTraining final model with selected features...")
    final_model, final_metrics = train_baseline_lgbm(
        X_train, y_train, n_estimators=500
    )
    
    # Evaluate on test set
    y_test_pred = final_model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_logloss = log_loss(y_test, y_test_pred)
    
    print(f"\nCV AUC: {final_metrics['auc']:.6f}")
    print(f"CV Log Loss: {final_metrics['log_loss']:.6f}")
    print(f"Test AUC: {test_auc:.6f}")
    print(f"Test Log Loss: {test_logloss:.6f}")
    
    stage_results['final'] = {
        'count': len(selected_features),
        'reduction_pct': (1 - len(selected_features)/len(feature_names)) * 100,
        'val_auc': final_metrics['auc'],
        'val_log_loss': final_metrics['log_loss'],
        'test_auc': test_auc,
        'test_log_loss': test_logloss
    }
    
    # Save selected features
    print(f"\nSaving selected features to {output_dir}/selected_features.txt")
    with open(output_dir / 'selected_features.txt', 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    
    # Save feature importances
    feature_importances_df = pd.DataFrame({
        'feature': selected_features,
        'importance': [final_model.feature_importance()[i] for i in range(len(selected_features))]
    }).sort_values('importance', ascending=False)
    
    feature_importances_df.to_csv(output_dir / 'feature_importances.csv', index=False)
    
    # Save selection history
    with open(output_dir / 'selection_history.json', 'w') as f:
        json.dump(stage_results, f, indent=2, default=str)
    
    # Generate and print report
    report = generate_selection_report(stage_results, output_dir)
    print("\n" + report)
    
    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION COMPLETE FOR {version.upper()}")
    print(f"{'='*80}")
    print(f"Selected features saved to: {output_dir}/selected_features.txt")
    print(f"Full report saved to: {output_dir}/selection_report.txt")
    
    return selected_features


def main():
    """Main feature selection pipeline - runs on all versions."""
    print("=" * 80)
    print("FEATURE SELECTION PIPELINE FOR ALL VERSIONS")
    print("=" * 80)
    
    versions = ['v1', 'v2', 'v3']
    all_selected_features = {}
    
    for version in versions:
        try:
            selected_features = run_feature_selection_for_version(version)
            all_selected_features[version] = selected_features
        except Exception as e:
            print(f"\nError processing {version}: {e}")
            print(f"Skipping {version} and continuing...")
            continue
    
    # Save summary
    summary_dir = Path('data/feature_selection')
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    with open(summary_dir / 'selection_summary.txt', 'w') as f:
        f.write("FEATURE SELECTION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for version, features in all_selected_features.items():
            f.write(f"{version.upper()}:\n")
            f.write(f"  Selected features: {len(features)}\n")
            f.write(f"  Features saved to: data/feature_selection/{version}/selected_features.txt\n\n")
    
    print("\n" + "=" * 80)
    print("ALL VERSIONS FEATURE SELECTION COMPLETE")
    print("=" * 80)
    print(f"Summary saved to: {summary_dir}/selection_summary.txt")


if __name__ == '__main__':
    main()

