"""
ML Model Training Script
=========================
Manually train ML models using collected datasets from MLTRAINCAMERA and MLTRAINIMU.
This script can be run independently to train models or update existing models.
"""

import argparse
from pathlib import Path
from dataset_collector import DatasetCollector
from imu_dataset_collector import IMUDatasetCollector
from ml_trainer import FormScorePredictor, BaselineCalculator
from dataset_tracker import DatasetTracker

def train_camera_model(exercise: str, use_unused_only: bool = False):
    """Train camera-based ML model (exercise-specific)."""
    print(f"\n📹 Training Camera Model for {exercise}...")
    
    # Load dataset from exercise-specific folder
    collector = DatasetCollector("MLTRAINCAMERA")
    samples = collector.load_dataset(exercise=exercise)  # Load only this exercise's data
    
    if len(samples) == 0:
        print(f"❌ No samples found for exercise: {exercise}")
        return False
    
    # Filter by unused sessions if requested
    if use_unused_only:
        tracker = DatasetTracker()
        unused_sessions = set(tracker.get_unused_camera_sessions("MLTRAINCAMERA"))
        # Filter samples (simplified - assumes session_id is in sample metadata)
        print(f"   Filtering to unused sessions: {len(unused_sessions)} sessions")
        # TODO: Implement proper session filtering based on sample metadata
    
    print(f"   Found {len(samples)} samples for {exercise}")
    
    # Auto-label if not labeled
    labeled_samples = [s for s in samples if s.expert_score is not None or s.is_perfect_form is not None]
    if len(labeled_samples) == 0:
        print("   Auto-labeling samples based on regional scores...")
        for sample in samples:
            if sample.regional_scores:
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                sample.expert_score = avg_score
                sample.is_perfect_form = (avg_score >= 90)
        labeled_samples = samples
    
    if len(labeled_samples) < 10:
        print(f"❌ Not enough labeled samples (need >=10, got {len(labeled_samples)})")
        return False
    
    # Extract features if not already extracted
    for sample in labeled_samples:
        if sample.features is None:
            collector.extract_features(sample)
    
    # Train model
    print(f"   Training model with {len(labeled_samples)} samples...")
    predictor = FormScorePredictor(model_type="random_forest")
    results = predictor.train(labeled_samples, verbose=True, use_imu_features=False)
    
    # Save model (exercise-specific path) with extended metadata
    model_dir = Path("models") / exercise / f"form_score_camera_random_forest"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(
        str(model_dir),
        exercise=exercise,
        training_samples=len(labeled_samples),
        performance_metrics=results
    )
    print(f"✅ Model saved to {model_dir}")
    
    # Calculate and save baselines
    perfect_samples = [s for s in labeled_samples if s.is_perfect_form == True]
    if perfect_samples:
        baselines = BaselineCalculator.calculate_baselines(perfect_samples)
        baseline_file = model_dir / "baselines.json"
        import json
        with open(baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2, default=str)
        print(f"✅ Baselines saved to {baseline_file}")
    
    # Mark sessions as used if tracker is enabled
    if use_unused_only and tracker:
        # TODO: Mark sessions as used after successful training
        pass
    
    print(f"✅ Camera model training completed!")
    print(f"   - Samples used: {len(labeled_samples)}")
    print(f"   - Perfect samples: {len(perfect_samples)}")
    print(f"   - Model performance: {results}")
    
    return True

def train_imu_model(exercise: str, use_unused_only: bool = False):
    """Train IMU-based ML model (exercise-specific)."""
    print(f"\n🎚️ Training IMU Model for {exercise}...")
    
    # Load dataset from exercise-specific folder
    collector = IMUDatasetCollector("MLTRAINIMU")
    samples_data = collector.load_dataset(exercise=exercise)  # Load only this exercise's data
    
    if len(samples_data) == 0:
        print(f"❌ No IMU samples found in MLTRAINIMU/{exercise}/")
        return False
    
    print(f"   Found {len(samples_data)} IMU rep samples for {exercise}")
    
    # TODO: Convert IMU samples_data to RepSample format and extract features
    # For now, IMU model training is not fully implemented
    print("⚠️  IMU model training not fully implemented yet")
    print("   IMU feature extraction and model training requires integration with camera samples for labels")
    
    return False
    
    if len(labeled_samples) < 10:
        print(f"❌ Not enough labeled samples (need >=10, got {len(labeled_samples)})")
        return False
    
    # Extract IMU features if not already extracted
    from imu_feature_extractor import extract_imu_features
    for sample in labeled_samples:
        if sample.imu_features is None:
            sample.imu_features = extract_imu_features(sample.imu_sequence)
    
    # Train model (using IMU features)
    print(f"   Training model with {len(labeled_samples)} samples...")
    predictor = FormScorePredictor(model_type="random_forest")
    # Note: We need to adapt the training to use IMU features
    # For now, this is a placeholder
    print("⚠️  IMU model training not fully implemented yet")
    
    return False

def update_existing_model(exercise: str):
    """Update existing model using only unused datasets."""
    print(f"\n🔄 Updating existing model for {exercise}...")
    
    # Check if model exists
    model_dir = Path("models") / f"form_score_{exercise}_random_forest"
    if not model_dir.exists():
        print(f"❌ No existing model found at {model_dir}")
        print("   Use --train to create a new model")
        return False
    
    # Train using only unused datasets
    return train_camera_model(exercise, use_unused_only=True)

def main():
    parser = argparse.ArgumentParser(description="Train ML models for exercise form analysis")
    parser.add_argument('--exercise', type=str, required=True,
                       choices=['bicep_curls', 'squats', 'lunges', 'pushups', 
                               'lateral_shoulder_raises', 'tricep_extensions',
                               'dumbbell_rows', 'dumbbell_shoulder_press'],
                       help='Exercise type to train')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'update'],
                       help='Mode: train (new model) or update (existing model)')
    parser.add_argument('--camera-only', action='store_true',
                       help='Train only camera model')
    parser.add_argument('--imu-only', action='store_true',
                       help='Train only IMU model')
    
    args = parser.parse_args()
    
    print(f"\n🤖 ML Model Training Script")
    print(f"   Exercise: {args.exercise}")
    print(f"   Mode: {args.mode}")
    
    success = False
    
    if args.mode == 'update':
        success = update_existing_model(args.exercise)
    else:
        if args.imu_only:
            success = train_imu_model(args.exercise)
        elif args.camera_only:
            success = train_camera_model(args.exercise)
        else:
            # Train both
            camera_success = train_camera_model(args.exercise)
            imu_success = train_imu_model(args.exercise)
            success = camera_success or imu_success
    
    if success:
        print(f"\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed or incomplete")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

