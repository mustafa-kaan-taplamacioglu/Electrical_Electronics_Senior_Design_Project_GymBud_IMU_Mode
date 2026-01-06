#!/usr/bin/env python3
"""
Comprehensive ML Model Training for All Exercises and All Modes
================================================================
Trains ML models for:
- 4 Exercises: bicep_curls, lateral_shoulder_raises, squats, tricep_extensions
- 3 Modes: Camera Only, IMU Only, Sensory Fusion (Camera + IMU)
Total: 12 ML models (4 x 3)

Also trains One-Class Classifiers for movement validation (reject option).

Each model includes:
1. ML-based form scoring (Random Forest with multi-output or single-output)
2. One-class classification for movement validation
3. Rule-based ensemble integration
4. LLM + rule-based + scientific feedback integration
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import traceback

# Import training functions
from train_ml_models import (
    train_camera_model,
    train_imu_model,
    train_fusion_model
)

# Exercises to train
EXERCISES = [
    'bicep_curls',
    'lateral_shoulder_raises',
    'squats',
    'tricep_extensions'
]

# Training modes
MODES = {
    'camera': train_camera_model,
    'imu': train_imu_model,
    'fusion': train_fusion_model
}

# One-class classifier training
def train_one_class_classifier(exercise: str, mode: str):
    """
    Train one-class classifier for movement validation (reject option).
    
    Args:
        exercise: Exercise name
        mode: 'camera', 'imu', or 'fusion'
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n🎯 Training One-Class Classifier for {exercise} ({mode} mode)...")
    
    if mode == 'camera':
        # Camera-based one-class classifier uses MediaPipe landmarks
        # This would need a separate implementation for camera mode
        print(f"   ⚠️  Camera-based one-class classifier not yet implemented")
        print(f"   📝 TODO: Implement MediaPipe landmark-based one-class classifier")
        return False
    
    elif mode in ['imu', 'fusion']:
        # IMU-based one-class classifier
        # Import helper functions from bicep curl one-class trainer
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Define helper functions inline (based on train_bicep_curl_one_class_classifier.py)
        def csv_to_imu_sequence(df_rep):
            """Convert CSV DataFrame to IMU sequence format."""
            import pandas as pd
            imu_sequence = []
            for timestamp in df_rep['timestamp'].unique():
                ts_data = df_rep[df_rep['timestamp'] == timestamp].copy()
                sample = {'timestamp': timestamp}
                for _, row in ts_data.iterrows():
                    node_name = row['node_name']
                    sample[node_name] = {
                        'ax': float(row['ax']), 'ay': float(row['ay']), 'az': float(row['az']),
                        'gx': float(row['gx']), 'gy': float(row['gy']), 'gz': float(row['gz']),
                        'qw': float(row['qw']), 'qx': float(row['qx']), 'qy': float(row['qy']), 'qz': float(row['qz']),
                        'roll': float(row['roll']), 'pitch': float(row['pitch']), 'yaw': float(row['yaw']),
                    }
                imu_sequence.append(sample)
            imu_sequence.sort(key=lambda x: x['timestamp'])
            return imu_sequence
        
        def extract_rep_sequences_from_csv(csv_file):
            """Extract rep sequences from CSV file."""
            import pandas as pd
            df = pd.read_csv(csv_file)
            valid_rep_numbers = df[df['rep_number'] > 0]['rep_number'].unique()
            reps = []
            for rep_num in sorted(valid_rep_numbers):
                rep_data = df[df['rep_number'] == rep_num].copy()
                if len(rep_data) >= 5:
                    rep_data = rep_data.sort_values('timestamp')
                    reps.append(rep_data)
            return reps
        
        import pandas as pd
        import numpy as np
        import joblib
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
        from imu_feature_extractor import extract_imu_features
        
        BASE_DIR = Path(__file__).parent
        MLTRAIN_DIR = BASE_DIR / "MLTRAINIMU" / exercise
        
        if not MLTRAIN_DIR.exists():
            print(f"   ❌ MLTRAINIMU/{exercise}/ directory not found")
            return False
        
        # Find training folders
        training_folders = sorted([d for d in MLTRAIN_DIR.iterdir() if d.is_dir()])
        
        if len(training_folders) < 2:
            print(f"   ⚠️  Need at least 2 training sessions, found {len(training_folders)}")
            print(f"   📁 Available folders: {[f.name for f in training_folders]}")
            return False
        
        print(f"   📁 Found {len(training_folders)} training sessions")
        
        # Collect all rep sequences
        all_features = []
        
        for folder in training_folders:
            csv_file = folder / "imu_samples.csv"
            if not csv_file.exists():
                print(f"   ⚠️  Skipping {folder.name}: imu_samples.csv not found")
                continue
            
            print(f"   📂 Processing {folder.name}...")
            rep_sequences = extract_rep_sequences_from_csv(csv_file)
            
            for df_rep in rep_sequences:
                # Convert to IMU sequence format
                imu_sequence = csv_to_imu_sequence(df_rep)
                
                # Extract features
                features = extract_imu_features(imu_sequence)
                
                if features:
                    all_features.append(features)
        
        if len(all_features) < 10:
            print(f"   ❌ Need at least 10 rep sequences, found {len(all_features)}")
            return False
        
        print(f"   ✅ Collected {len(all_features)} rep sequences with features")
        
        # Convert to numpy array
        feature_names = sorted(list(all_features[0].keys()))
        X = np.array([[f.get(k, 0.0) for k in feature_names] for f in all_features])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train One-Class SVM
        # Adjust NU parameter based on exercise (more lenient for variations)
        nu_map = {
            'bicep_curls': 0.35,
            'lateral_shoulder_raises': 0.40,
            'squats': 0.35,
            'tricep_extensions': 0.40
        }
        nu = nu_map.get(exercise, 0.35)
        
        print(f"   🎯 Training OneClassSVM with nu={nu}...")
        classifier = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        classifier.fit(X_scaled)
        
        # Save model
        model_dir = BASE_DIR / "models" / exercise / f"one_class_{mode}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = model_dir / "one_class_svm.joblib"
        scaler_file = model_dir / "one_class_scaler.joblib"
        features_file = model_dir / "one_class_features.json"
        
        joblib.dump(classifier, model_file)
        joblib.dump(scaler, scaler_file)
        
        with open(features_file, 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        print(f"   ✅ One-class classifier saved to {model_dir}")
        print(f"      - Model: {model_file}")
        print(f"      - Scaler: {scaler_file}")
        print(f"      - Features: {features_file}")
        
        # Test on training data
        predictions = classifier.predict(X_scaled)
        inliers = np.sum(predictions == 1)
        outliers = np.sum(predictions == -1)
        print(f"   📊 Training data validation:")
        print(f"      - Inliers (accepted): {inliers} ({100*inliers/len(predictions):.1f}%)")
        print(f"      - Outliers (rejected): {outliers} ({100*outliers/len(predictions):.1f}%)")
        
        return True
    
    else:
        print(f"   ❌ Unknown mode: {mode}")
        return False


def train_exercise_all_modes(exercise: str, multi_output: bool = True, 
                             train_one_class: bool = True, skip_existing: bool = False):
    """
    Train all models for one exercise (camera, IMU, fusion).
    
    Args:
        exercise: Exercise name
        multi_output: If True, train multi-output model (regional scores)
        train_one_class: If True, also train one-class classifiers
        skip_existing: If True, skip if model already exists
    
    Returns:
        Dict with results for each mode
    """
    print(f"\n{'='*80}")
    print(f"🏋️  TRAINING ALL MODELS FOR: {exercise.upper()}")
    print(f"{'='*80}")
    
    results = {
        'exercise': exercise,
        'camera': False,
        'imu': False,
        'fusion': False,
        'one_class_camera': False,
        'one_class_imu': False,
        'one_class_fusion': False
    }
    
    # Check if data exists
    BASE_DIR = Path(__file__).parent
    has_camera_data = (BASE_DIR / "MLTRAINCAMERA" / exercise).exists()
    has_imu_data = (BASE_DIR / "MLTRAINIMU" / exercise).exists()
    
    print(f"\n📊 Data Availability:")
    print(f"   - Camera data: {'✅' if has_camera_data else '❌'}")
    print(f"   - IMU data: {'✅' if has_imu_data else '❌'}")
    
    # Train Camera model
    if has_camera_data:
        model_dir = BASE_DIR / "models" / exercise / f"form_score_camera_random_forest_{'multi_output' if multi_output else 'single_output'}"
        if skip_existing and model_dir.exists():
            print(f"\n⏭️  Skipping camera model (already exists)")
            results['camera'] = True
        else:
            try:
                results['camera'] = train_camera_model(exercise, multi_output=multi_output)
            except Exception as e:
                print(f"   ❌ Camera model training failed: {e}")
                traceback.print_exc()
        
        # Train one-class classifier for camera
        if train_one_class and results['camera']:
            try:
                results['one_class_camera'] = train_one_class_classifier(exercise, 'camera')
            except Exception as e:
                print(f"   ⚠️  One-class camera classifier training failed: {e}")
    else:
        print(f"\n⏭️  Skipping camera model (no data)")
    
    # Train IMU model
    if has_imu_data:
        model_dir = BASE_DIR / "models" / exercise / f"form_score_imu_random_forest_{'multi_output' if multi_output else 'single_output'}"
        if skip_existing and model_dir.exists():
            print(f"\n⏭️  Skipping IMU model (already exists)")
            results['imu'] = True
        else:
            try:
                results['imu'] = train_imu_model(exercise, multi_output=multi_output)
            except Exception as e:
                print(f"   ❌ IMU model training failed: {e}")
                traceback.print_exc()
        
        # Train one-class classifier for IMU
        if train_one_class and results['imu']:
            try:
                results['one_class_imu'] = train_one_class_classifier(exercise, 'imu')
            except Exception as e:
                print(f"   ⚠️  One-class IMU classifier training failed: {e}")
    else:
        print(f"\n⏭️  Skipping IMU model (no data)")
    
    # Train Fusion model (requires both camera and IMU data)
    if has_camera_data and has_imu_data:
        model_dir = BASE_DIR / "models" / exercise / f"form_score_fusion_random_forest_{'multi_output' if multi_output else 'single_output'}"
        if skip_existing and model_dir.exists():
            print(f"\n⏭️  Skipping fusion model (already exists)")
            results['fusion'] = True
        else:
            try:
                results['fusion'] = train_fusion_model(exercise, multi_output=multi_output)
            except Exception as e:
                print(f"   ❌ Fusion model training failed: {e}")
                traceback.print_exc()
        
        # Train one-class classifier for fusion
        if train_one_class and results['fusion']:
            try:
                results['one_class_fusion'] = train_one_class_classifier(exercise, 'fusion')
            except Exception as e:
                print(f"   ⚠️  One-class fusion classifier training failed: {e}")
    else:
        print(f"\n⏭️  Skipping fusion model (requires both camera and IMU data)")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY FOR {exercise.upper()}:")
    print(f"{'='*80}")
    print(f"   Camera Model:        {'✅' if results['camera'] else '❌'}")
    print(f"   IMU Model:           {'✅' if results['imu'] else '❌'}")
    print(f"   Fusion Model:        {'✅' if results['fusion'] else '❌'}")
    if train_one_class:
        print(f"   One-Class (Camera):  {'✅' if results['one_class_camera'] else '❌'}")
        print(f"   One-Class (IMU):     {'✅' if results['one_class_imu'] else '❌'}")
        print(f"   One-Class (Fusion):  {'✅' if results['one_class_fusion'] else '❌'}")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for all exercises and all modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models for all exercises
  python train_all_exercises_all_modes.py --all
  
  # Train only bicep_curls
  python train_all_exercises_all_modes.py --exercise bicep_curls
  
  # Train only camera models
  python train_all_exercises_all_modes.py --all --mode camera
  
  # Single-output models (overall score only)
  python train_all_exercises_all_modes.py --all --single-output
  
  # Skip existing models
  python train_all_exercises_all_modes.py --all --skip-existing
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Train all exercises')
    parser.add_argument('--exercise', type=str,
                       choices=EXERCISES,
                       help='Train specific exercise only')
    parser.add_argument('--mode', type=str,
                       choices=['camera', 'imu', 'fusion'],
                       help='Train specific mode only (default: all modes)')
    parser.add_argument('--single-output', action='store_true',
                       help='Train single-output models (overall score) instead of multi-output (regional scores)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip training if model already exists')
    parser.add_argument('--no-one-class', action='store_true',
                       help='Skip one-class classifier training')
    parser.add_argument('--list-exercises', action='store_true',
                       help='List available exercises and exit')
    
    args = parser.parse_args()
    
    if args.list_exercises:
        print("Available exercises:")
        for ex in EXERCISES:
            print(f"  - {ex}")
        return 0
    
    if not args.all and not args.exercise:
        parser.print_help()
        print("\n❌ Error: Must specify --all or --exercise")
        return 1
    
    # Determine exercises to train
    exercises_to_train = [args.exercise] if args.exercise else EXERCISES
    
    print(f"\n{'='*80}")
    print(f"🤖 COMPREHENSIVE ML MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Exercises: {', '.join(exercises_to_train)}")
    print(f"Mode: {'All modes' if not args.mode else args.mode}")
    print(f"Output type: {'Single-output' if args.single_output else 'Multi-output'}")
    print(f"One-class classifiers: {'❌ Disabled' if args.no_one_class else '✅ Enabled'}")
    print(f"Skip existing: {'✅ Yes' if args.skip_existing else '❌ No'}")
    print(f"{'='*80}\n")
    
    # Train each exercise
    all_results = []
    for exercise in exercises_to_train:
        try:
            if args.mode:
                # Train specific mode only
                results = train_exercise_all_modes(
                    exercise,
                    multi_output=not args.single_output,
                    train_one_class=not args.no_one_class,
                    skip_existing=args.skip_existing
                )
                # Filter to only requested mode
                if args.mode == 'camera' and not results['camera']:
                    continue
                elif args.mode == 'imu' and not results['imu']:
                    continue
                elif args.mode == 'fusion' and not results['fusion']:
                    continue
            else:
                results = train_exercise_all_modes(
                    exercise,
                    multi_output=not args.single_output,
                    train_one_class=not args.no_one_class,
                    skip_existing=args.skip_existing
                )
            
            all_results.append(results)
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Training interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error training {exercise}: {e}")
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 FINAL TRAINING SUMMARY")
    print(f"{'='*80}")
    
    total_models = 0
    successful_models = 0
    
    for result in all_results:
        ex = result['exercise']
        camera_ok = result['camera']
        imu_ok = result['imu']
        fusion_ok = result['fusion']
        
        print(f"\n{ex.upper()}:")
        print(f"   Camera:  {'✅' if camera_ok else '❌'}")
        print(f"   IMU:     {'✅' if imu_ok else '❌'}")
        print(f"   Fusion:  {'✅' if fusion_ok else '❌'}")
        
        for mode_ok in [camera_ok, imu_ok, fusion_ok]:
            total_models += 1
            if mode_ok:
                successful_models += 1
    
    print(f"\n{'='*80}")
    print(f"Total models trained: {successful_models}/{total_models}")
    print(f"{'='*80}\n")
    
    return 0 if successful_models > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

