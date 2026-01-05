"""
ML Model Training Script
=========================
Manually train ML models using collected datasets from MLTRAINCAMERA and MLTRAINIMU.
This script can be run independently to train models or update existing models.
"""

import warnings
# Suppress sklearn parallel warnings (harmless but annoying)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.parallel')

import argparse
from pathlib import Path
from typing import List
from dataset_collector import DatasetCollector
from imu_dataset_collector import IMUDatasetCollector
from ml_trainer import FormScorePredictor, BaselineCalculator
from dataset_tracker import DatasetTracker

def train_camera_model(exercise: str, use_unused_only: bool = False, multi_output: bool = True):
    """Train camera-based ML model (exercise-specific).
    
    Args:
        exercise: Exercise name
        use_unused_only: If True, use only unused sessions
        multi_output: If True, train multi-output model (regional scores), else single-output (overall score)
    """
    output_type = "multi-output (regional scores)" if multi_output else "single-output (overall score)"
    print(f"\n📹 Training Camera Model ({output_type}) for {exercise}...")
    
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
    
    # Extract features first (required for Z-score perfect form selection)
    print("   Extracting features for all samples...")
    for sample in samples:
        if sample.features is None:
            collector.extract_features(sample)
    
    # Select perfect form samples using Z-score analysis
    from zscore_perfect_form_selector import ZScorePerfectFormSelector
    print("\n   Using Z-score statistical analysis to select perfect form samples...")
    selector = ZScorePerfectFormSelector(z_threshold=1.0, min_features_acceptable=0.9)
    perfect_samples, non_perfect_samples, selection_stats = selector.select_perfect_form_samples(
        samples, 
        use_imu_features=False, 
        verbose=True
    )
    
    # Auto-label if not labeled (fallback for expert_score)
    labeled_samples = [s for s in samples if s.expert_score is not None or s.is_perfect_form is not None]
    if len(labeled_samples) == 0:
        print("   Auto-labeling samples based on regional scores (for expert_score only)...")
        for sample in samples:
            if sample.regional_scores:
                # For single-output: use average of regional scores as overall score
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                sample.expert_score = avg_score
                # is_perfect_form already set by Z-score selector
        labeled_samples = samples
    
    if len(labeled_samples) < 10:
        print(f"❌ Not enough labeled samples (need >=10, got {len(labeled_samples)})")
        return False
    
    # Add temporal features to samples (periodic movement analysis)
    print("   Extracting temporal features (periodic movement patterns)...")
    from temporal_feature_extractor import extract_temporal_features
    temporal_feature_count = 0
    for sample in labeled_samples:
        if sample.landmarks_sequence and len(sample.landmarks_sequence) > 0:
            temporal_features = extract_temporal_features(sample.landmarks_sequence, fps=20.0)
            if temporal_features:
                # Merge temporal features into main features
                if sample.features is None:
                    sample.features = {}
                sample.features.update({f'temporal_{k}': v for k, v in temporal_features.items()})
                temporal_feature_count += len(temporal_features)
    
    if temporal_feature_count > 0:
        print(f"   ✅ Added {temporal_feature_count} temporal features per sample")
    
    # Train model (features already extracted above + temporal features)
    print(f"   Training {output_type} model with {len(labeled_samples)} samples...")
    predictor = FormScorePredictor(model_type="random_forest", multi_output=multi_output)
    
    # HYPERPARAMETER TUNING (with improved parameters to reduce overfitting)
    print(f"\n🔧 Hyperparameter Tuning (to improve generalization and reduce overfitting)...")
    try:
        best_params = predictor.tune_hyperparameters(
            labeled_samples,
            cv=5,
            method="random",
            n_iter=30,  # Try 30 different parameter combinations
            verbose=False,  # Less verbose
            use_imu_features=False
        )
        if best_params:
            print(f"   ✅ Best hyperparameters found:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")
            # Note: tune_hyperparameters already fits the model with best params
            # So we need to evaluate it, not train again
            print(f"   📊 Evaluating tuned model...")
            # Prepare data for evaluation
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            X, y = predictor.prepare_features(labeled_samples, use_imu_features=False)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = predictor.scaler.fit_transform(X_train)
            X_test_scaled = predictor.scaler.transform(X_test)
            
            # Model is already fitted by tune_hyperparameters, just evaluate
            y_train_pred = predictor.model.predict(X_train_scaled)
            y_test_pred = predictor.model.predict(X_test_scaled)
            
            if multi_output:
                train_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
                test_r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
                train_mae = mean_absolute_error(y_train, y_train_pred, multioutput='uniform_average')
                test_mae = mean_absolute_error(y_test, y_test_pred, multioutput='uniform_average')
                train_mse = mean_squared_error(y_train, y_train_pred, multioutput='uniform_average')
                test_mse = mean_squared_error(y_test, y_test_pred, multioutput='uniform_average')
            else:
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
            
            results = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            predictor.is_trained = True
            
            print(f"\n📈 Tuned Model Results:")
            print(f"   Train MSE: {train_mse:.2f}")
            print(f"   Test MSE:  {test_mse:.2f}")
            print(f"   Train MAE: {train_mae:.2f}")
            print(f"   Test MAE:  {test_mae:.2f}")
            print(f"   Train R²:  {train_r2:.3f}")
            print(f"   Test R²:   {test_r2:.3f}")
            gap = train_r2 - test_r2
            if gap < 0.1:
                print(f"   Gap: {gap:.3f} ✅ Excellent (no overfitting)")
            elif gap < 0.2:
                print(f"   Gap: {gap:.3f} ✅ Good (minimal overfitting)")
            elif gap < 0.5:
                print(f"   Gap: {gap:.3f} ⚠️  Moderate overfitting")
            else:
                print(f"   Gap: {gap:.3f} ❌ High overfitting")
        else:
            print(f"   ⚠️  Tuning skipped. Training with default parameters...")
            results = predictor.train(labeled_samples, verbose=True, use_imu_features=False)
    except Exception as e:
        print(f"   ⚠️  Tuning failed: {e}. Using default parameters...")
        import traceback
        traceback.print_exc()
    results = predictor.train(labeled_samples, verbose=True, use_imu_features=False)
    
    # Save model (exercise-specific path) with extended metadata
    model_name_suffix = "multi_output" if multi_output else "single_output"
    model_dir = Path("models") / exercise / f"form_score_camera_random_forest_{model_name_suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(
        str(model_dir),
        exercise=exercise,
        training_samples=len(labeled_samples),
        performance_metrics=results
    )
    print(f"✅ Model saved to {model_dir}")
    
    # Calculate and save baselines (perfect_samples already selected via Z-score)
    if perfect_samples:
        baselines = BaselineCalculator.calculate_baselines(perfect_samples, regional=multi_output)
        baseline_file = model_dir / "baselines.json"
        import json
        with open(baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2, default=str)
        baseline_type = "regional baselines" if multi_output else "overall baselines"
        print(f"✅ Baselines saved to {baseline_file} (including {baseline_type})")
    
    # Mark sessions as used if tracker is enabled
    if use_unused_only and tracker:
        # TODO: Mark sessions as used after successful training
        pass
    
    print(f"✅ Camera model training completed!")
    print(f"   - Output type: {output_type}")
    print(f"   - Samples used: {len(labeled_samples)}")
    print(f"   - Perfect samples: {len(perfect_samples)}")
    print(f"   - Model performance: {results}")
    
    return True

def train_imu_model(exercise: str, use_unused_only: bool = False, multi_output: bool = True):
    """Train IMU-based ML model (exercise-specific).
    
    Args:
        exercise: Exercise name
        use_unused_only: If True, use only unused sessions
        multi_output: If True, train multi-output model (regional scores), else single-output (overall score)
    """
    output_type = "multi-output (regional scores)" if multi_output else "single-output (overall score)"
    print(f"\n🎚️ Training IMU Model ({output_type}) for {exercise}...")
    
    # Load IMU dataset from exercise-specific folder
    imu_collector = IMUDatasetCollector("MLTRAINIMU")
    imu_samples_data = imu_collector.load_dataset(exercise=exercise)  # List[Dict] with rep_number, samples, etc.
    
    if len(imu_samples_data) == 0:
        print(f"❌ No IMU samples found in MLTRAINIMU/{exercise}/")
        return False
    
    print(f"   Found {len(imu_samples_data)} IMU rep sequences for {exercise}")
    
    # Load camera dataset to get labels (regional_scores, expert_score, etc.)
    camera_collector = DatasetCollector("MLTRAINCAMERA")
    camera_samples = camera_collector.load_dataset(exercise=exercise)  # List[RepSample]
    
    if len(camera_samples) == 0:
        print(f"⚠️  No camera samples found for labels. Using IMU-only labels...")
        camera_samples = []
    
    # Create a mapping: rep_number + timestamp -> camera sample (for label matching)
    camera_label_map = {}
    for cam_sample in camera_samples:
        key = (cam_sample.rep_number, round(cam_sample.timestamp, 1))  # Round to 0.1s for matching
        camera_label_map[key] = cam_sample
    
    # Convert IMU samples_data to RepSample format
    from dataset_collector import RepSample
    from imu_feature_extractor import extract_imu_features
    
    rep_samples = []
    for imu_rep_data in imu_samples_data:
        rep_number = imu_rep_data.get('rep_number', 0)
        rep_start_time = imu_rep_data.get('rep_start_time', 0)
        # Handle both old format (samples) and new format (imu_sequence)
        imu_sequence = imu_rep_data.get('samples', imu_rep_data.get('imu_sequence', []))
        
        if len(imu_sequence) == 0:
            continue
        
        # Try to find matching camera sample for labels
        matching_camera_sample = None
        key = (rep_number, round(rep_start_time, 1))
        if key in camera_label_map:
            matching_camera_sample = camera_label_map[key]
        else:
            # Try to find by rep_number only (if timestamp doesn't match exactly)
            for cam_sample in camera_samples:
                if cam_sample.rep_number == rep_number:
                    matching_camera_sample = cam_sample
                    break
        
        # Create RepSample from IMU data
        rep_sample = RepSample(
            timestamp=rep_start_time,
            exercise=exercise,
            rep_number=rep_number,
            landmarks_sequence=[],  # Empty for IMU-only model
            imu_sequence=imu_sequence
        )
        
        # Copy labels from camera sample if available
        if matching_camera_sample:
            rep_sample.expert_score = matching_camera_sample.expert_score
            rep_sample.regional_scores = matching_camera_sample.regional_scores
            rep_sample.regional_issues = matching_camera_sample.regional_issues
            rep_sample.is_perfect_form = matching_camera_sample.is_perfect_form
            rep_sample.user_feedback = matching_camera_sample.user_feedback
            rep_sample.min_angle = matching_camera_sample.min_angle
            rep_sample.max_angle = matching_camera_sample.max_angle
            rep_sample.range_of_motion = matching_camera_sample.range_of_motion
        
        # Extract IMU features
        rep_sample.imu_features = extract_imu_features(imu_sequence)
        
        if rep_sample.imu_features:
            rep_samples.append(rep_sample)
    
    if len(rep_samples) == 0:
        print(f"❌ No valid IMU samples with features extracted")
        return False
    
    print(f"   Converted {len(rep_samples)} IMU samples to RepSample format")
    
    # Extract features first (required for Z-score perfect form selection)
    print("   Extracting IMU features for all samples...")
    for sample in rep_samples:
        if sample.imu_features is None:
            sample.imu_features = extract_imu_features(sample.imu_sequence)
    
    # Select perfect form samples using Z-score analysis
    from zscore_perfect_form_selector import ZScorePerfectFormSelector
    print("\n   Using Z-score statistical analysis to select perfect form samples...")
    selector = ZScorePerfectFormSelector(z_threshold=1.0, min_features_acceptable=0.9)
    perfect_samples, non_perfect_samples, selection_stats = selector.select_perfect_form_samples(
        rep_samples, 
        use_imu_features=True,  # Use IMU features for selection
        verbose=True
    )
    
    # Auto-label if not labeled (fallback for expert_score)
    labeled_samples = [s for s in rep_samples if s.expert_score is not None or s.is_perfect_form is not None]
    if len(labeled_samples) == 0:
        print("   Auto-labeling samples based on regional scores (for expert_score only)...")
        for sample in rep_samples:
            if sample.regional_scores:
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                sample.expert_score = avg_score
                # is_perfect_form already set by Z-score selector
        labeled_samples = rep_samples
    
    if len(labeled_samples) < 10:
        print(f"❌ Not enough labeled samples (need >=10, got {len(labeled_samples)})")
        return False
    
    # Train model (features already extracted above)
    print(f"   Training {output_type} IMU model with {len(labeled_samples)} samples...")
    predictor = FormScorePredictor(model_type="random_forest", multi_output=multi_output)
    
    # HYPERPARAMETER TUNING (to improve generalization and reduce overfitting)
    print(f"\n🔧 Hyperparameter Tuning (to improve generalization and reduce overfitting)...")
    try:
        best_params = predictor.tune_hyperparameters(
            labeled_samples,
            cv=5,
            method="random",
            n_iter=30,
            verbose=False,
            use_imu_features=True
        )
        if best_params:
            print(f"   ✅ Best hyperparameters found:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")
            # Evaluate tuned model (already fitted by tune_hyperparameters)
            print(f"   📊 Evaluating tuned model...")
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            X, y = predictor.prepare_features(labeled_samples, use_imu_features=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = predictor.scaler.fit_transform(X_train)
            X_test_scaled = predictor.scaler.transform(X_test)
            
            y_train_pred = predictor.model.predict(X_train_scaled)
            y_test_pred = predictor.model.predict(X_test_scaled)
            
            if multi_output:
                train_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
                test_r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
                train_mae = mean_absolute_error(y_train, y_train_pred, multioutput='uniform_average')
                test_mae = mean_absolute_error(y_test, y_test_pred, multioutput='uniform_average')
                train_mse = mean_squared_error(y_train, y_train_pred, multioutput='uniform_average')
                test_mse = mean_squared_error(y_test, y_test_pred, multioutput='uniform_average')
            else:
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
            
            results = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            predictor.is_trained = True
            
            print(f"\n📈 Tuned Model Results:")
            print(f"   Train MSE: {train_mse:.2f}")
            print(f"   Test MSE:  {test_mse:.2f}")
            print(f"   Train MAE: {train_mae:.2f}")
            print(f"   Test MAE:  {test_mae:.2f}")
            print(f"   Train R²:  {train_r2:.3f}")
            print(f"   Test R²:   {test_r2:.3f}")
            gap = train_r2 - test_r2
            if gap < 0.1:
                print(f"   Gap: {gap:.3f} ✅ Excellent (no overfitting)")
            elif gap < 0.2:
                print(f"   Gap: {gap:.3f} ✅ Good (minimal overfitting)")
            elif gap < 0.5:
                print(f"   Gap: {gap:.3f} ⚠️  Moderate overfitting")
            else:
                print(f"   Gap: {gap:.3f} ❌ High overfitting")
        else:
            print(f"   ⚠️  Tuning skipped. Training with default parameters...")
            results = predictor.train(labeled_samples, verbose=True, use_imu_features=True)
    except Exception as e:
        print(f"   ⚠️  Tuning failed: {e}. Using default parameters...")
        import traceback
        traceback.print_exc()
        results = predictor.train(labeled_samples, verbose=True, use_imu_features=True)
    
    # Save model (exercise-specific path) with extended metadata
    model_name_suffix = "multi_output" if multi_output else "single_output"
    model_dir = Path("models") / exercise / f"form_score_imu_random_forest_{model_name_suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(
        str(model_dir),
        exercise=exercise,
        training_samples=len(labeled_samples),
        performance_metrics=results
    )
    print(f"✅ Model saved to {model_dir}")
    
    # Calculate and save baselines (perfect_samples already selected via Z-score)
    if perfect_samples:
        baselines = BaselineCalculator.calculate_baselines(perfect_samples, regional=multi_output)
        baseline_file = model_dir / "baselines.json"
        import json
        with open(baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2, default=str)
        baseline_type = "regional baselines" if multi_output else "overall baselines"
        print(f"✅ Baselines saved to {baseline_file} (including {baseline_type})")
    
    print(f"✅ IMU model training completed!")
    print(f"   - Output type: {output_type}")
    print(f"   - Samples used: {len(labeled_samples)}")
    print(f"   - Perfect samples: {len(perfect_samples)}")
    print(f"   - Model performance: {results}")
    
    return True

def train_fusion_model(exercise: str, use_unused_only: bool = False, multi_output: bool = True):
    """Train Sensor Fusion ML model (Camera + IMU combined).
    
    Args:
        exercise: Exercise name
        use_unused_only: If True, use only unused sessions
        multi_output: If True, train multi-output model (regional scores), else single-output (overall score)
    """
    output_type = "multi-output (regional scores)" if multi_output else "single-output (overall score)"
    print(f"\n🔀 Training Sensor Fusion Model ({output_type}) for {exercise}...")
    
    # Load camera dataset
    camera_collector = DatasetCollector("MLTRAINCAMERA")
    camera_samples = camera_collector.load_dataset(exercise=exercise)
    
    if len(camera_samples) == 0:
        print(f"❌ No camera samples found for exercise: {exercise}")
        return False
    
    print(f"   Found {len(camera_samples)} camera samples")
    
    # Load IMU dataset
    imu_collector = IMUDatasetCollector("MLTRAINIMU")
    imu_samples_data = imu_collector.load_dataset(exercise=exercise)
    
    if len(imu_samples_data) == 0:
        print(f"⚠️  No IMU samples found. Fusion model requires both camera and IMU data.")
        return False
    
    print(f"   Found {len(imu_samples_data)} IMU rep sequences")
    
    # Create mapping: rep_number + timestamp -> IMU data
    imu_data_map = {}
    from imu_feature_extractor import extract_imu_features
    for imu_rep_data in imu_samples_data:
        rep_number = imu_rep_data.get('rep_number', 0)
        # Try multiple timestamp fields
        rep_start_time = imu_rep_data.get('rep_start_time', 
                                          imu_rep_data.get('camera_rep_timestamp', 
                                                          imu_rep_data.get('timestamp', 0)))
        # Handle both old format (samples) and new format (imu_sequence)
        imu_sequence = imu_rep_data.get('samples', imu_rep_data.get('imu_sequence', []))
        
        if len(imu_sequence) > 0:
            key = (rep_number, round(rep_start_time, 1))
            imu_data_map[key] = {
                'sequence': imu_sequence,
                'features': extract_imu_features(imu_sequence)
            }
    
    print(f"   Created IMU data map with {len(imu_data_map)} entries")
    
    # Combine camera and IMU data
    from dataset_collector import RepSample
    fusion_samples = []
    
    print(f"   Matching camera samples with IMU data...")
    matched_count = 0
    unmatched_count = 0
    
    for cam_sample in camera_samples:
        # Find matching IMU data
        key = (cam_sample.rep_number, round(cam_sample.timestamp, 1))
        matching_imu = None
        
        if key in imu_data_map:
            matching_imu = imu_data_map[key]
            matched_count += 1
        else:
            # Try to find by rep_number only (more flexible matching)
            for imu_key, imu_data in imu_data_map.items():
                if imu_key[0] == cam_sample.rep_number:  # Same rep_number
                    matching_imu = imu_data
                    matched_count += 1
                    break
            
            if matching_imu is None:
                unmatched_count += 1
                # Skip samples without matching IMU data
                continue
        
        # Create fusion sample (copy camera sample and add IMU features)
        fusion_sample = RepSample(
            timestamp=cam_sample.timestamp,
            exercise=cam_sample.exercise,
            rep_number=cam_sample.rep_number,
            landmarks_sequence=cam_sample.landmarks_sequence,
            imu_sequence=matching_imu['sequence'] if matching_imu else None,
            user_id=cam_sample.user_id,
            features=cam_sample.features,  # Camera features
            imu_features=matching_imu['features'] if matching_imu else None,  # IMU features
            expert_score=cam_sample.expert_score,
            user_feedback=cam_sample.user_feedback,
            is_perfect_form=cam_sample.is_perfect_form,
            regional_scores=cam_sample.regional_scores,
            regional_issues=cam_sample.regional_issues,
            min_angle=cam_sample.min_angle,
            max_angle=cam_sample.max_angle,
            range_of_motion=cam_sample.range_of_motion
        )
        
        # Extract camera features if not already extracted
        if fusion_sample.features is None:
            fusion_sample.features = camera_collector.extract_camera_features(fusion_sample)
        
        # Extract IMU features if not already extracted
        if fusion_sample.imu_features is None and fusion_sample.imu_sequence:
            fusion_sample.imu_features = extract_imu_features(fusion_sample.imu_sequence)
        
        # Only include samples that have both camera and IMU features
        if fusion_sample.features and fusion_sample.imu_features:
            fusion_samples.append(fusion_sample)
    
    print(f"   Matched: {matched_count}, Unmatched: {unmatched_count}")
    
    if len(fusion_samples) == 0:
        print(f"❌ No samples with both camera and IMU features found")
        print(f"   Camera samples: {len(camera_samples)}")
        print(f"   IMU samples: {len(imu_data_map)}")
        print(f"   Matched: {matched_count}, Unmatched: {unmatched_count}")
        # Debug: Check feature extraction
        if len(camera_samples) > 0:
            cam_sample = camera_samples[0]
            print(f"   Debug - Camera sample features: {cam_sample.features is not None}, IMU features: {cam_sample.imu_features is not None}")
        return False
    
    print(f"   Created {len(fusion_samples)} fusion samples (camera + IMU)")
    
    # Extract features first (required for Z-score perfect form selection)
    print("   Extracting features for all samples...")
    for sample in fusion_samples:
        if sample.features is None:
            sample.features = camera_collector.extract_camera_features(sample)
        if sample.imu_features is None and sample.imu_sequence:
            sample.imu_features = extract_imu_features(sample.imu_sequence)
    
    # Select perfect form samples using Z-score analysis (using combined features)
    from zscore_perfect_form_selector import ZScorePerfectFormSelector
    print("\n   Using Z-score statistical analysis to select perfect form samples...")
    selector = ZScorePerfectFormSelector(z_threshold=1.0, min_features_acceptable=0.9)
    
    # For fusion, we need to combine camera and IMU features for selection
    # Create temporary combined features for Z-score selection
    for sample in fusion_samples:
        if sample.features and sample.imu_features:
            # Combine features for Z-score selection
            combined_features = {**sample.features, **{f'imu_{k}': v for k, v in sample.imu_features.items()}}
            # Temporarily store in features dict for selector
            original_features = sample.features
            sample.features = combined_features
    
    perfect_samples, non_perfect_samples, selection_stats = selector.select_perfect_form_samples(
        fusion_samples, 
        use_imu_features=False,  # Using combined features in features dict
        verbose=True
    )
    
    # Restore original features structure
    for sample in fusion_samples:
        if isinstance(sample.features, dict) and any(k.startswith('imu_') for k in sample.features.keys()):
            # Split back to camera and IMU features
            sample.features = {k: v for k, v in sample.features.items() if not k.startswith('imu_')}
            if not sample.imu_features:
                sample.imu_features = {k[4:]: v for k, v in sample.features.items() if k.startswith('imu_')}
    
    # Auto-label if not labeled
    labeled_samples = [s for s in fusion_samples if s.expert_score is not None or s.is_perfect_form is not None]
    if len(labeled_samples) == 0:
        print("   Auto-labeling samples based on regional scores...")
        for sample in fusion_samples:
            if sample.regional_scores:
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                sample.expert_score = avg_score
        labeled_samples = fusion_samples
    
    if len(labeled_samples) < 10:
        print(f"❌ Not enough labeled samples (need >=10, got {len(labeled_samples)})")
    return False
    
    # Add temporal features to fusion samples
    print("   Extracting temporal features (periodic movement patterns)...")
    from temporal_feature_extractor import extract_temporal_features, extract_imu_temporal_features
    temporal_feature_count = 0
    for sample in labeled_samples:
        # Camera temporal features
        if sample.landmarks_sequence and len(sample.landmarks_sequence) > 0:
            temporal_features = extract_temporal_features(sample.landmarks_sequence, fps=20.0)
            if temporal_features:
                if sample.features is None:
                    sample.features = {}
                sample.features.update({f'temporal_{k}': v for k, v in temporal_features.items()})
                temporal_feature_count += len(temporal_features)
        
        # IMU temporal features
        if sample.imu_sequence and len(sample.imu_sequence) > 0:
            imu_temporal_features = extract_imu_temporal_features(sample.imu_sequence, fps=20.0)
            if imu_temporal_features:
                if sample.imu_features is None:
                    sample.imu_features = {}
                sample.imu_features.update({f'temporal_{k}': v for k, v in imu_temporal_features.items()})
                temporal_feature_count += len(imu_temporal_features)
    
    if temporal_feature_count > 0:
        print(f"   ✅ Added {temporal_feature_count} temporal features per sample")
    
    # Train model with combined features (both camera and IMU)
    print(f"   Training {output_type} fusion model with {len(labeled_samples)} samples...")
    print("   🔧 Performing hyperparameter tuning...")
    
    # Create a custom FormScorePredictor that combines camera and IMU features
    # We'll modify prepare_features to combine both feature sets
    class FusionPredictor(FormScorePredictor):
        def prepare_features(self, samples: List, use_imu_features: bool = False):
            """Combine camera and IMU features for fusion model."""
            feature_vectors = []
            labels = []
            
            for sample in samples:
                # Combine camera and IMU features
                combined_features = {}
                if sample.features:
                    combined_features.update(sample.features)
                if sample.imu_features:
                    # Prefix IMU features to avoid conflicts
                    combined_features.update({f'imu_{k}': v for k, v in sample.imu_features.items()})
                
                if not combined_features:
                    continue
                
                # Prepare label FIRST - if no label, skip this sample
                label = None
                if self.multi_output:
                    if sample.regional_scores:
                        label = [
                            sample.regional_scores.get(region, 0.0)
                            for region in self.REGIONAL_OUTPUTS
                        ]
                    elif sample.expert_score is not None:
                        label = [sample.expert_score] * 4
                    elif sample.is_perfect_form is not None:
                        score = 100.0 if sample.is_perfect_form else 50.0
                        label = [score] * 4
                else:
                    if sample.expert_score is not None:
                        label = sample.expert_score
                    elif sample.regional_scores:
                        label = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                
                # Skip if no label
                if label is None:
                    continue
                
                # Use combined features as feature vector (only if we have a label)
                if self.feature_names is None:
                    self.feature_names = sorted(combined_features.keys())
                
                feature_vec = [combined_features.get(name, 0.0) for name in self.feature_names]
                feature_vectors.append(feature_vec)
                labels.append(label)
            
            import numpy as np
            X = np.array(feature_vectors)
            y = np.array(labels)
            return X, y
    
    predictor = FusionPredictor(model_type="random_forest", multi_output=multi_output)
    
    # Prepare features first to get initial feature count
    print(f"\n📊 Preparing features for fusion model...")
    X_init, y_init = predictor.prepare_features(labeled_samples, use_imu_features=False)
    initial_feature_count = X_init.shape[1]
    print(f"   Initial feature count: {initial_feature_count}")
    
    # FEATURE SELECTION (to reduce curse of dimensionality for fusion model)
    # Use RandomForest feature importance instead of univariate selection for better performance
    print(f"\n🔍 Feature Selection (reducing from {initial_feature_count} features)...")
    from sklearn.ensemble import RandomForestRegressor as RF
    import numpy as np
    
    # Train/test split for feature selection
    from sklearn.model_selection import train_test_split
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
        X_init, y_init, test_size=0.2, random_state=42
    )
    
    # Scale features
    X_train_fs_scaled = predictor.scaler.fit_transform(X_train_fs)
    X_test_fs_scaled = predictor.scaler.transform(X_test_fs)
    
    # Method: Use RandomForest feature importance (better than univariate)
    # More aggressive selection: keep only top 40-50% of features (vs 65% before)
    # This reduces overfitting and improves generalization
    if multi_output:
        # For multi-output, use MultiOutputRegressor
        from sklearn.multioutput import MultiOutputRegressor
        rf_selector = RF(n_estimators=100, max_depth=10, min_samples_split=10, 
                        min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf_selector = MultiOutputRegressor(rf_selector, n_jobs=-1)
        y_for_selection = y_train_fs
    else:
        rf_selector = RF(n_estimators=100, max_depth=10, min_samples_split=10, 
                        min_samples_leaf=5, random_state=42, n_jobs=-1)
        y_for_selection = y_train_fs
    
    # Train a quick RF to get feature importances
    rf_selector.fit(X_train_fs_scaled, y_for_selection)
    
    # Get feature importances (handle multi-output)
    if multi_output:
        # Average importance across all outputs
        importances = np.mean([est.feature_importances_ for est in rf_selector.estimators_], axis=0)
    else:
        importances = rf_selector.feature_importances_
    
    # Select top features based on importance threshold
    # Optimized selection: Balance between reducing overfitting and maintaining performance
    # Use 45-50% to reduce overfitting while keeping enough features for good performance
    k_select = max(int(initial_feature_count * 0.48), 140)  # 48% or min 140 features
    k_select = min(k_select, 200)  # Max 200 features (reduce overfitting)
    
    # Get top k features by importance
    top_k_indices = np.argsort(importances)[-k_select:][::-1]
    selected_feature_indices = sorted(top_k_indices)
    
    print(f"   Selected {len(selected_feature_indices)} features using RandomForest importance (top {k_select})")
    print(f"   Feature importance range: {importances[selected_feature_indices].min():.4f} - {importances[selected_feature_indices].max():.4f}")
    
    # Apply feature selection to feature names
    if predictor.feature_names:
        predictor.selected_feature_names = [predictor.feature_names[i] for i in selected_feature_indices]
        predictor.feature_names = predictor.selected_feature_names
        print(f"   Reduced feature count: {initial_feature_count} -> {len(predictor.feature_names)} ({len(predictor.feature_names)/initial_feature_count*100:.1f}%)")
    
    # Apply feature selection to the data (slice X_init to only selected features)
    X_selected = X_init[:, selected_feature_indices]
    y_selected = y_init
    
    # HYPERPARAMETER TUNING (with improved parameters for fusion model - more regularization)
    print(f"\n🔧 Hyperparameter Tuning (with improved regularization for fusion model)...")
    try:
        # Use custom hyperparameter tuning with more aggressive regularization
        # For fusion model with many features, we want to prevent overfitting
        
        # Apply feature selection to the data
        X_train_hp, X_test_hp, y_train_hp, y_test_hp = train_test_split(
            X_selected, y_selected, test_size=0.2, random_state=42
        )
        X_train_hp_scaled = predictor.scaler.fit_transform(X_train_hp)
        X_test_hp_scaled = predictor.scaler.transform(X_test_hp)
        
        # Custom hyperparameter tuning with more regularization
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        # Optimized regularization for fusion model - "normally fitted" (balanced train/test)
        # Goal: Reduce overfitting (small gap) while maintaining good test performance
        # For MultiOutputRegressor, we need to use 'estimator__' prefix
        if multi_output:
            param_dist = {
                'estimator__n_estimators': randint(150, 300),  # Moderate trees for generalization
                'estimator__max_depth': [10, 12, 15, 18, 20, None],  # Limit depth to reduce overfitting
                'estimator__min_samples_split': randint(10, 30),  # Higher = more regularization (reduce overfitting)
                'estimator__min_samples_leaf': randint(3, 15),  # Higher = more regularization (reduce overfitting)
                'estimator__max_features': ['sqrt', 'log2', 0.5]  # Limit features per split (reduce overfitting)
            }
            base_model = RF(random_state=42, n_jobs=-1)
            from sklearn.multioutput import MultiOutputRegressor
            search_model = MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            param_dist = {
                'n_estimators': randint(150, 300),  # Moderate trees for generalization
                'max_depth': [10, 12, 15, 18, 20, None],  # Limit depth to reduce overfitting
                'min_samples_split': randint(10, 30),  # Higher = more regularization (reduce overfitting)
                'min_samples_leaf': randint(3, 15),  # Higher = more regularization (reduce overfitting)
                'max_features': ['sqrt', 'log2', 0.5]  # Limit features per split (reduce overfitting)
            }
            search_model = RF(random_state=42, n_jobs=-1)
        
        search = RandomizedSearchCV(
            search_model,
            param_dist,
            n_iter=100,  # More iterations for better tuning (increased from 50)
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        print(f"   Running RandomizedSearchCV with 100 iterations...")
        search.fit(X_train_hp_scaled, y_train_hp)
        
        best_params = search.best_params_
        predictor.model = search.best_estimator_
        
        print(f"   ✅ Best hyperparameters found:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")
        print(f"      Best CV MAE: {-search.best_score_:.2f}")
        # Evaluate tuned model with feature selection
        print(f"   📊 Evaluating tuned model with feature selection...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_train_pred = predictor.model.predict(X_train_hp_scaled)
        y_test_pred = predictor.model.predict(X_test_hp_scaled)
            
        if multi_output:
            train_r2 = r2_score(y_train_hp, y_train_pred, multioutput='uniform_average')
            test_r2 = r2_score(y_test_hp, y_test_pred, multioutput='uniform_average')
            train_mae = mean_absolute_error(y_train_hp, y_train_pred, multioutput='uniform_average')
            test_mae = mean_absolute_error(y_test_hp, y_test_pred, multioutput='uniform_average')
            train_mse = mean_squared_error(y_train_hp, y_train_pred, multioutput='uniform_average')
            test_mse = mean_squared_error(y_test_hp, y_test_pred, multioutput='uniform_average')
        else:
            train_r2 = r2_score(y_train_hp, y_train_pred)
            test_r2 = r2_score(y_test_hp, y_test_pred)
            train_mae = mean_absolute_error(y_train_hp, y_train_pred)
            test_mae = mean_absolute_error(y_test_hp, y_test_pred)
            train_mse = mean_squared_error(y_train_hp, y_train_pred)
            test_mse = mean_squared_error(y_test_hp, y_test_pred)
        
        results = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        predictor.is_trained = True
        
        print(f"\n📈 Tuned Model Results (with feature selection):")
        print(f"   Features: {initial_feature_count} -> {len(predictor.feature_names)} (selected)")
        print(f"   Train MSE: {train_mse:.2f}")
        print(f"   Test MSE:  {test_mse:.2f}")
        print(f"   Train MAE: {train_mae:.2f}")
        print(f"   Test MAE:  {test_mae:.2f}")
        print(f"   Train R²:  {train_r2:.3f}")
        print(f"   Test R²:   {test_r2:.3f}")
        gap = train_r2 - test_r2
        if gap < 0:
            print(f"   Gap: {gap:.3f} ✅ Excellent (Test > Train = Great generalization!)")
        elif gap < 0.1:
            print(f"   Gap: {gap:.3f} ✅ Excellent (no overfitting)")
        elif gap < 0.2:
            print(f"   Gap: {gap:.3f} ✅ Good (minimal overfitting)")
        elif gap < 0.5:
            print(f"   Gap: {gap:.3f} ⚠️  Moderate overfitting")
        else:
            print(f"   Gap: {gap:.3f} ❌ High overfitting")
    except Exception as e:
        print(f"   ⚠️  Tuning failed: {e}. Using default parameters with feature selection...")
        import traceback
        traceback.print_exc()
        # Apply feature selection even if tuning fails
        try:
            X_fs, y_fs = predictor.prepare_features(labeled_samples, use_imu_features=False)
            X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(
                X_fs, y_fs, test_size=0.2, random_state=42
            )
            X_train_fs_scaled = predictor.scaler.fit_transform(X_train_fs)
            X_test_fs_scaled = predictor.scaler.transform(X_test_fs)
            
            # Use default model with optimized regularization (normally fitted - balanced)
            from sklearn.ensemble import RandomForestRegressor as RF
            base_model = RF(
                n_estimators=200,  # Moderate trees for generalization
                max_depth=15,  # Moderate depth (reduce overfitting)
                min_samples_split=15,  # Higher = more regularization
                min_samples_leaf=5,  # Higher = more regularization
                max_features='sqrt',  # Standard feature selection
                random_state=42,
                n_jobs=-1
            )
            if multi_output:
                from sklearn.multioutput import MultiOutputRegressor
                predictor.model = MultiOutputRegressor(base_model, n_jobs=-1)
            else:
                predictor.model = base_model
            
            predictor.model.fit(X_train_fs_scaled, y_train_fs)
            predictor.is_trained = True
            
            y_train_pred = predictor.model.predict(X_train_fs_scaled)
            y_test_pred = predictor.model.predict(X_test_fs_scaled)
            
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            if multi_output:
                train_r2 = r2_score(y_train_fs, y_train_pred, multioutput='uniform_average')
                test_r2 = r2_score(y_test_fs, y_test_pred, multioutput='uniform_average')
                train_mae = mean_absolute_error(y_train_fs, y_train_pred, multioutput='uniform_average')
                test_mae = mean_absolute_error(y_test_fs, y_test_pred, multioutput='uniform_average')
                train_mse = mean_squared_error(y_train_fs, y_train_pred, multioutput='uniform_average')
                test_mse = mean_squared_error(y_test_fs, y_test_pred, multioutput='uniform_average')
            else:
                train_r2 = r2_score(y_train_fs, y_train_pred)
                test_r2 = r2_score(y_test_fs, y_test_pred)
                train_mae = mean_absolute_error(y_train_fs, y_train_pred)
                test_mae = mean_absolute_error(y_test_fs, y_test_pred)
                train_mse = mean_squared_error(y_train_fs, y_train_pred)
                test_mse = mean_squared_error(y_test_fs, y_test_pred)
            
            results = {
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            print(f"   ✅ Trained with default parameters (regularized) and feature selection")
        except Exception as e2:
            print(f"   ⚠️  Feature selection also failed: {e2}. Using original training method...")
            results = predictor.train(labeled_samples, verbose=True, use_imu_features=False)
    
    # Save model (exercise-specific path)
    model_name_suffix = "multi_output" if multi_output else "single_output"
    model_dir = Path("models") / exercise / f"form_score_fusion_random_forest_{model_name_suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(
        str(model_dir),
        exercise=exercise,
        training_samples=len(labeled_samples),
        performance_metrics=results
    )
    print(f"✅ Model saved to {model_dir}")
    
    # Calculate and save baselines (perfect_samples already selected via Z-score)
    if perfect_samples:
        baselines = BaselineCalculator.calculate_baselines(perfect_samples, regional=multi_output)
        baseline_file = model_dir / "baselines.json"
        import json
        with open(baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2, default=str)
        baseline_type = "regional baselines" if multi_output else "overall baselines"
        print(f"✅ Baselines saved to {baseline_file} (including {baseline_type})")
    
    print(f"✅ Fusion model training completed!")
    print(f"   - Output type: {output_type}")
    print(f"   - Samples used: {len(labeled_samples)}")
    print(f"   - Perfect samples: {len(perfect_samples)}")
    print(f"   - Model performance: {results}")
    
    return True

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
                       choices=['bicep_curls', 'squats', 
                               'lateral_shoulder_raises', 'triceps_pushdown',
                               'dumbbell_rows', 'dumbbell_shoulder_press'],
                       help='Exercise type to train')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'update'],
                       help='Mode: train (new model) or update (existing model)')
    parser.add_argument('--camera-only', action='store_true',
                       help='Train only camera model')
    parser.add_argument('--imu-only', action='store_true',
                       help='Train only IMU model')
    parser.add_argument('--fusion', action='store_true',
                       help='Train fusion model (Camera + IMU combined)')
    parser.add_argument('--single-output', action='store_true',
                       help='Train single-output model (overall score) instead of multi-output (regional scores)')
    
    args = parser.parse_args()
    
    print(f"\n🤖 ML Model Training Script")
    print(f"   Exercise: {args.exercise}")
    print(f"   Mode: {args.mode}")
    
    success = False
    
    if args.mode == 'update':
        success = update_existing_model(args.exercise)
    else:
        multi_output = not args.single_output  # If --single-output, use False, else True
        
        if args.fusion:
            success = train_fusion_model(args.exercise, multi_output=multi_output)
        elif args.imu_only:
            success = train_imu_model(args.exercise, multi_output=multi_output)
        elif args.camera_only:
            success = train_camera_model(args.exercise, multi_output=multi_output)
        else:
            # Train all three: camera, IMU, and fusion (same output type)
            output_type = "single-output" if args.single_output else "multi-output"
            print(f"\n📋 Training all {output_type} models (Camera, IMU, Fusion)...")
            camera_success = train_camera_model(args.exercise, multi_output=multi_output)
            imu_success = train_imu_model(args.exercise, multi_output=multi_output)
            fusion_success = train_fusion_model(args.exercise, multi_output=multi_output)
            success = camera_success or imu_success or fusion_success
    
    if success:
        print(f"\n✅ Training completed successfully!")
    else:
        print(f"\n❌ Training failed or incomplete")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

