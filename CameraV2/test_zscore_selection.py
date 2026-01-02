#!/usr/bin/env python3
"""
Test Z-Score Perfect Form Selection
===================================
Quick test script to see how Z-score selection works on existing data.
"""

import sys
from pathlib import Path
from dataset_collector import DatasetCollector
from zscore_perfect_form_selector import ZScorePerfectFormSelector

def test_zscore_selection(exercise: str = "bicep_curls"):
    """Test Z-score perfect form selection."""
    print(f"\n🧪 Testing Z-Score Perfect Form Selection for {exercise}")
    print("=" * 70)
    
    # Load dataset
    print(f"\n📂 Loading dataset from MLTRAINCAMERA/{exercise}/...")
    collector = DatasetCollector("MLTRAINCAMERA")
    samples = collector.load_dataset(exercise=exercise)
    
    if len(samples) == 0:
        print(f"❌ No samples found for {exercise}")
        print(f"   Please collect some training data first!")
        return
    
    print(f"✅ Loaded {len(samples)} samples")
    
    # Check if features are extracted
    samples_with_features = [s for s in samples if s.features is not None]
    samples_without_features = len(samples) - len(samples_with_features)
    
    if samples_without_features > 0:
        print(f"\n📊 Extracting features for {samples_without_features} samples...")
        for i, sample in enumerate(samples):
            if sample.features is None:
                collector.extract_features(sample)
                if (i + 1) % 10 == 0:
                    print(f"   Extracted features for {i + 1}/{len(samples)} samples...")
        print(f"✅ Feature extraction completed")
    else:
        print(f"✅ All samples already have features")
    
    # Test Z-score selection with different thresholds
    print(f"\n" + "=" * 70)
    print("🔍 Testing Z-Score Selection with Different Thresholds")
    print("=" * 70)
    
    thresholds = [1.0, 1.5, 2.0]
    min_ratios = [0.7, 0.8, 0.9]
    
    results = []
    
    for z_threshold in thresholds:
        for min_ratio in min_ratios:
            print(f"\n📊 Testing: Z-threshold={z_threshold}, Min-ratio={min_ratio*100:.0f}%")
            print("-" * 70)
            
            selector = ZScorePerfectFormSelector(
                z_threshold=z_threshold,
                min_features_acceptable=min_ratio
            )
            
            perfect_samples, non_perfect, stats = selector.select_perfect_form_samples(
                samples,
                use_imu_features=False,
                verbose=False  # Don't print detailed stats for each test
            )
            
            results.append({
                'z_threshold': z_threshold,
                'min_ratio': min_ratio,
                'perfect_count': len(perfect_samples),
                'perfect_percentage': stats['perfect_ratio'] * 100,
                'avg_ratio_within_threshold': stats['avg_ratio_within_threshold'] * 100,
                'avg_max_abs_z': stats['avg_max_abs_z']
            })
            
            print(f"   ✅ Perfect samples: {len(perfect_samples)} ({stats['perfect_ratio']*100:.1f}%)")
            print(f"   📈 Avg ratio within threshold: {stats['avg_ratio_within_threshold']*100:.1f}%")
            print(f"   📊 Avg max |Z-score|: {stats['avg_max_abs_z']:.2f}")
    
    # Summary table
    print(f"\n" + "=" * 70)
    print("📋 SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Z-Threshold':<12} {'Min-Ratio':<12} {'Perfect':<10} {'Perfect %':<12} {'Avg Ratio %':<15} {'Avg Max |Z|':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['z_threshold']:<12} {r['min_ratio']*100:<12.0f} {r['perfect_count']:<10} "
              f"{r['perfect_percentage']:<12.1f} {r['avg_ratio_within_threshold']:<15.1f} {r['avg_max_abs_z']:<12.2f}")
    
    # Recommended settings
    print(f"\n" + "=" * 70)
    print("💡 RECOMMENDED SETTINGS")
    print("=" * 70)
    
    # Find settings that give 20-40% perfect samples (good balance)
    good_results = [r for r in results if 20 <= r['perfect_percentage'] <= 40]
    
    if good_results:
        # Prefer z_threshold=1.5 and min_ratio=0.8 if available
        recommended = next((r for r in good_results if r['z_threshold'] == 1.5 and r['min_ratio'] == 0.8), good_results[0])
        print(f"✅ Recommended: Z-threshold={recommended['z_threshold']}, Min-ratio={recommended['min_ratio']*100:.0f}%")
        print(f"   → This gives {recommended['perfect_percentage']:.1f}% perfect samples")
        print(f"   → Average ratio within threshold: {recommended['avg_ratio_within_threshold']:.1f}%")
    else:
        # Use default if no good match
        recommended = next((r for r in results if r['z_threshold'] == 1.5 and r['min_ratio'] == 0.8), results[-1])
        print(f"⚠️  No ideal range found. Using default: Z-threshold={recommended['z_threshold']}, Min-ratio={recommended['min_ratio']*100:.0f}%")
        print(f"   → This gives {recommended['perfect_percentage']:.1f}% perfect samples")
    
    print(f"\n✅ Test completed!")
    print(f"\nTo use these settings in training:")
    print(f"   The default settings (Z=1.5, Min-ratio=80%) are already used in train_ml_models.py")
    print(f"   To customize, edit train_ml_models.py and change:")
    print(f"      selector = ZScorePerfectFormSelector(z_threshold={recommended['z_threshold']}, min_features_acceptable={recommended['min_ratio']})")

if __name__ == "__main__":
    exercise = sys.argv[1] if len(sys.argv) > 1 else "bicep_curls"
    test_zscore_selection(exercise)

