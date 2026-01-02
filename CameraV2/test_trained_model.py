#!/usr/bin/env python3
"""
Test Trained ML Model
=====================
Test the trained model with actual samples from the dataset.
"""

import sys
from pathlib import Path
from dataset_collector import DatasetCollector
from ml_trainer import FormScorePredictor, BaselineCalculator
import json

def test_model_inference(exercise: str = "bicep_curls"):
    """Test trained model with actual samples."""
    print(f"\n🧪 Testing Trained Model for {exercise}")
    print("=" * 70)
    
    # Load model
    model_dir = Path("models") / exercise / "form_score_camera_random_forest"
    if not model_dir.exists():
        print(f"❌ Model not found at {model_dir}")
        print(f"   Please train the model first: python train_ml_models.py --exercise {exercise} --camera-only")
        return
    
    print(f"\n📂 Loading model from {model_dir}...")
    try:
        predictor = FormScorePredictor.load(str(model_dir))
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {predictor.model_type}")
        print(f"   Features: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load baselines
    baseline_file = model_dir / "baselines.json"
    baselines = {}
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baselines = json.load(f)
        print(f"✅ Baselines loaded ({len(baselines)} features)")
    else:
        print(f"⚠️  No baselines file found")
    
    # Load dataset
    print(f"\n📊 Loading test dataset...")
    collector = DatasetCollector("MLTRAINCAMERA")
    samples = collector.load_dataset(exercise=exercise)
    
    if len(samples) == 0:
        print(f"❌ No samples found for {exercise}")
        return
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Extract features for all samples
    print(f"\n🔧 Extracting features for all samples...")
    samples_with_features = []
    for i, sample in enumerate(samples):
        if sample.features is None:
            collector.extract_features(sample)
        if sample.features:
            samples_with_features.append(sample)
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(samples)} samples...")
    
    print(f"✅ {len(samples_with_features)} samples with features")
    
    if len(samples_with_features) == 0:
        print(f"❌ No samples with features found")
        return
    
    # Test predictions
    print(f"\n" + "=" * 70)
    print("🔮 PREDICTION TEST")
    print("=" * 70)
    
    # Test with a few samples
    test_samples = samples_with_features[:10]  # Test first 10 samples
    
    predictions = []
    errors = []
    
    for i, sample in enumerate(test_samples):
        try:
            # Predict score
            predicted_score = predictor.predict(sample.features)
            
            # Get actual score (if available)
            actual_score = sample.expert_score
            if actual_score is None and sample.regional_scores:
                actual_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
            
            error = abs(predicted_score - actual_score) if actual_score is not None else None
            
            predictions.append({
                'sample_idx': i,
                'rep_number': sample.rep_number,
                'predicted': predicted_score,
                'actual': actual_score,
                'error': error,
                'is_perfect': sample.is_perfect_form
            })
            
            if error is not None:
                errors.append(error)
            
            print(f"\n📊 Sample {i+1} (Rep #{sample.rep_number}):")
            print(f"   Predicted Score: {predicted_score:.2f}")
            if actual_score is not None:
                print(f"   Actual Score:    {actual_score:.2f}")
                print(f"   Error:           {error:.2f}")
            print(f"   Perfect Form:     {sample.is_perfect_form}")
        
        except Exception as e:
            print(f"❌ Error predicting sample {i+1}: {e}")
            continue
    
    # Statistics
    print(f"\n" + "=" * 70)
    print("📈 PREDICTION STATISTICS")
    print("=" * 70)
    
    if errors:
        import numpy as np
        print(f"   Tested samples: {len(predictions)}")
        print(f"   Mean Error (MAE): {np.mean(errors):.2f}")
        print(f"   Std Error:        {np.std(errors):.2f}")
        print(f"   Min Error:        {np.min(errors):.2f}")
        print(f"   Max Error:        {np.max(errors):.2f}")
    
    # Test baseline similarity (if baselines available)
    if baselines:
        print(f"\n" + "=" * 70)
        print("📏 BASELINE SIMILARITY TEST (Sample)")
        print("=" * 70)
        
        # Test with one perfect form sample
        perfect_samples = [s for s in samples_with_features if s.is_perfect_form == True]
        if perfect_samples:
            test_sample = perfect_samples[0]
            print(f"\n🔍 Testing baseline similarity for a perfect form sample:")
            
            # Calculate similarity for a few key features
            key_features = ['left_elbow_range', 'left_elbow_mean', 'left_elbow_std', 
                          'left_elbow_vel_mean', 'right_elbow_range']
            
            similarities = []
            for feat_name in key_features:
                if feat_name in test_sample.features and feat_name in baselines:
                    current_val = test_sample.features[feat_name]
                    baseline_mean = baselines[feat_name]['mean']
                    baseline_std = baselines[feat_name]['std']
                    
                    if baseline_std > 0:
                        z_score = abs(current_val - baseline_mean) / baseline_std
                        similarity = max(0, 100 - (z_score * 16))  # Scale to 0-100
                        similarities.append(similarity)
                        print(f"   {feat_name:25s}: Z={z_score:.2f}, Similarity={similarity:.1f}%")
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                print(f"\n   Average Baseline Similarity: {avg_similarity:.1f}%")
    
    # Feature importance
    print(f"\n" + "=" * 70)
    print("🎯 TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 70)
    
    try:
        importances = predictor.get_feature_importance(top_n=10)
        for i, (feature, importance) in enumerate(importances.items(), 1):
            print(f"   {i:2d}. {feature:30s}: {importance:.4f}")
    except Exception as e:
        print(f"   ⚠️  Could not get feature importance: {e}")
    
    print(f"\n✅ Test completed!")
    print(f"\n💡 Next steps:")
    print(f"   1. Model is ready for real-time inference")
    print(f"   2. Use model_inference.py for production inference")
    print(f"   3. Baselines can be used for similarity-based correction scoring")

if __name__ == "__main__":
    exercise = sys.argv[1] if len(sys.argv) > 1 else "bicep_curls"
    test_model_inference(exercise)

