"""
Z-Score Based Perfect Form Selection
====================================
Uses statistical Z-score analysis to identify perfect form samples based on
feature distributions (IMU patterns, camera landmark patterns: velocity, position, 
acceleration, and relative positions).

Approach:
1. Calculate mean and std for all features across all samples
2. For each sample, calculate Z-scores for all features
3. Select samples with |Z-score| < threshold (e.g., |Z| < 1.5) for most features
4. These samples represent "normal" patterns (statistically consistent)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataset_collector import RepSample


class ZScorePerfectFormSelector:
    """Select perfect form samples using Z-score statistical analysis."""
    
    def __init__(self, z_threshold: float = 1.5, min_features_acceptable: float = 0.8):
        """
        Initialize selector.
        
        Args:
            z_threshold: Maximum absolute Z-score for a feature to be considered "normal" (default: 1.5)
                        |Z| < 1.5 means within ~87% of normal distribution
            min_features_acceptable: Minimum ratio of features that must be within threshold (default: 0.8 = 80%)
        """
        self.z_threshold = z_threshold
        self.min_features_acceptable = min_features_acceptable
    
    def calculate_feature_statistics(self, samples: List[RepSample], use_imu_features: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate mean and std for all features across all samples.
        
        Args:
            samples: List of RepSample objects
            use_imu_features: If True, use imu_features; otherwise use features
        
        Returns:
            Dict: {feature_name: {'mean': float, 'std': float}}
        """
        # Collect all feature values
        all_features_dict = {}
        
        for sample in samples:
            # Choose which features to use
            if use_imu_features:
                features = sample.imu_features
            else:
                features = sample.features
            
            if features is None:
                continue
            
            # Collect values for each feature
            for feature_name, value in features.items():
                if feature_name not in all_features_dict:
                    all_features_dict[feature_name] = []
                all_features_dict[feature_name].append(float(value))
        
        # Calculate statistics
        statistics = {}
        for feature_name, values in all_features_dict.items():
            values_array = np.array(values)
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            
            # Avoid division by zero
            if std_val == 0:
                std_val = 1e-6
            
            statistics[feature_name] = {
                'mean': float(mean_val),
                'std': float(std_val)
            }
        
        return statistics
    
    def calculate_sample_z_scores(self, sample: RepSample, statistics: Dict[str, Dict[str, float]], 
                                  use_imu_features: bool = False) -> Dict[str, float]:
        """
        Calculate Z-scores for a single sample.
        
        Args:
            sample: RepSample object
            statistics: Feature statistics (from calculate_feature_statistics)
            use_imu_features: If True, use imu_features; otherwise use features
        
        Returns:
            Dict: {feature_name: z_score}
        """
        # Choose which features to use
        if use_imu_features:
            features = sample.imu_features
        else:
            features = sample.features
        
        if features is None:
            return {}
        
        z_scores = {}
        for feature_name, value in features.items():
            if feature_name in statistics:
                mean_val = statistics[feature_name]['mean']
                std_val = statistics[feature_name]['std']
                
                if std_val > 0:
                    z_score = (float(value) - mean_val) / std_val
                else:
                    z_score = 0.0  # Perfect match if std = 0
                
                z_scores[feature_name] = float(z_score)
        
        return z_scores
    
    def is_perfect_form(self, z_scores: Dict[str, float]) -> Tuple[bool, float]:
        """
        Check if a sample is perfect form based on Z-scores.
        
        Args:
            z_scores: Dict of feature_name -> z_score
        
        Returns:
            Tuple: (is_perfect, ratio_within_threshold)
                - is_perfect: True if sample passes criteria
                - ratio_within_threshold: Ratio of features within Z-threshold
        """
        if len(z_scores) == 0:
            return (False, 0.0)
        
        # Count how many features are within threshold
        within_threshold = sum(1 for z in z_scores.values() if abs(z) < self.z_threshold)
        total_features = len(z_scores)
        ratio = within_threshold / total_features if total_features > 0 else 0.0
        
        # Perfect if ratio >= min_features_acceptable
        is_perfect = ratio >= self.min_features_acceptable
        
        return (is_perfect, ratio)
    
    def select_perfect_form_samples(self, samples: List[RepSample], use_imu_features: bool = False, 
                                   verbose: bool = True) -> Tuple[List[RepSample], List[RepSample], Dict]:
        """
        Select perfect form samples using Z-score analysis.
        
        Args:
            samples: List of all RepSample objects
            use_imu_features: If True, use IMU features; otherwise use camera features
            verbose: Print selection statistics
        
        Returns:
            Tuple: (perfect_samples, non_perfect_samples, selection_stats)
                - perfect_samples: List of samples marked as perfect form
                - non_perfect_samples: List of samples not perfect form
                - selection_stats: Dict with selection statistics
        """
        if len(samples) == 0:
            return ([], [], {})
        
        # Filter samples with features
        samples_with_features = [
            s for s in samples 
            if (use_imu_features and s.imu_features is not None) or 
               (not use_imu_features and s.features is not None)
        ]
        
        if len(samples_with_features) == 0:
            print("⚠️  No samples with features found")
            return ([], samples, {})
        
        # Calculate feature statistics
        if verbose:
            feature_type = "IMU" if use_imu_features else "Camera"
            print(f"📊 Calculating {feature_type} feature statistics for {len(samples_with_features)} samples...")
        
        statistics = self.calculate_feature_statistics(samples_with_features, use_imu_features=use_imu_features)
        
        if verbose:
            print(f"   Found {len(statistics)} features")
        
        # Calculate Z-scores for each sample and select perfect ones
        perfect_samples = []
        non_perfect_samples = []
        z_score_details = []
        
        for sample in samples_with_features:
            z_scores = self.calculate_sample_z_scores(sample, statistics, use_imu_features=use_imu_features)
            is_perfect, ratio = self.is_perfect_form(z_scores)
            
            # Store max absolute Z-score for analysis
            max_abs_z = max([abs(z) for z in z_scores.values()]) if z_scores else 0.0
            
            z_score_details.append({
                'sample_idx': len(z_score_details),
                'is_perfect': is_perfect,
                'ratio_within_threshold': ratio,
                'max_abs_z': max_abs_z,
                'total_features': len(z_scores)
            })
            
            if is_perfect:
                perfect_samples.append(sample)
                sample.is_perfect_form = True  # Mark sample as perfect
            else:
                non_perfect_samples.append(sample)
                sample.is_perfect_form = False  # Mark sample as not perfect
        
        # Calculate selection statistics
        selection_stats = {
            'total_samples': len(samples),
            'samples_with_features': len(samples_with_features),
            'perfect_samples': len(perfect_samples),
            'non_perfect_samples': len(non_perfect_samples),
            'perfect_ratio': len(perfect_samples) / len(samples_with_features) if samples_with_features else 0.0,
            'z_threshold': self.z_threshold,
            'min_features_acceptable': self.min_features_acceptable,
            'feature_type': 'IMU' if use_imu_features else 'Camera',
            'total_features': len(statistics),
            'avg_ratio_within_threshold': np.mean([d['ratio_within_threshold'] for d in z_score_details]) if z_score_details else 0.0,
            'avg_max_abs_z': np.mean([d['max_abs_z'] for d in z_score_details]) if z_score_details else 0.0
        }
        
        if verbose:
            print(f"\n✅ Z-Score Perfect Form Selection Results:")
            print(f"   Total samples: {selection_stats['total_samples']}")
            print(f"   Samples with features: {selection_stats['samples_with_features']}")
            print(f"   Perfect form samples: {selection_stats['perfect_samples']} ({selection_stats['perfect_ratio']*100:.1f}%)")
            print(f"   Non-perfect samples: {selection_stats['non_perfect_samples']}")
            print(f"   Z-threshold: ±{self.z_threshold}")
            print(f"   Min features acceptable: {self.min_features_acceptable*100:.0f}%")
            print(f"   Avg ratio within threshold: {selection_stats['avg_ratio_within_threshold']*100:.1f}%")
            print(f"   Avg max |Z-score|: {selection_stats['avg_max_abs_z']:.2f}")
        
        return (perfect_samples, non_perfect_samples, selection_stats)
    
    def select_perfect_form_hybrid(self, samples: List[RepSample], camera_weight: float = 0.6, 
                                   imu_weight: float = 0.4, verbose: bool = True) -> Tuple[List[RepSample], List[RepSample], Dict]:
        """
        Select perfect form samples using both camera and IMU features (hybrid approach).
        
        Args:
            samples: List of all RepSample objects
            camera_weight: Weight for camera features (default: 0.6)
            imu_weight: Weight for IMU features (default: 0.4)
            verbose: Print selection statistics
        
        Returns:
            Tuple: (perfect_samples, non_perfect_samples, selection_stats)
        """
        if len(samples) == 0:
            return ([], [], {})
        
        # Filter samples with both camera and IMU features
        samples_with_both = [
            s for s in samples 
            if s.features is not None and s.imu_features is not None
        ]
        
        if len(samples_with_both) == 0:
            print("⚠️  No samples with both camera and IMU features found. Falling back to camera-only selection.")
            return self.select_perfect_form_samples(samples, use_imu_features=False, verbose=verbose)
        
        if verbose:
            print(f"📊 Hybrid Z-Score Perfect Form Selection ({len(samples_with_both)} samples with both features)...")
        
        # Calculate statistics for both feature types
        camera_stats = self.calculate_feature_statistics(samples_with_both, use_imu_features=False)
        imu_stats = self.calculate_feature_statistics(samples_with_both, use_imu_features=True)
        
        if verbose:
            print(f"   Camera features: {len(camera_stats)}")
            print(f"   IMU features: {len(imu_stats)}")
        
        # Calculate combined Z-scores and select perfect samples
        perfect_samples = []
        non_perfect_samples = []
        z_score_details = []
        
        for sample in samples_with_both:
            # Calculate Z-scores for both feature types
            camera_z_scores = self.calculate_sample_z_scores(sample, camera_stats, use_imu_features=False)
            imu_z_scores = self.calculate_sample_z_scores(sample, imu_stats, use_imu_features=True)
            
            # Calculate ratios for each feature type
            camera_within = sum(1 for z in camera_z_scores.values() if abs(z) < self.z_threshold)
            camera_ratio = camera_within / len(camera_z_scores) if camera_z_scores else 0.0
            
            imu_within = sum(1 for z in imu_z_scores.values() if abs(z) < self.z_threshold)
            imu_ratio = imu_within / len(imu_z_scores) if imu_z_scores else 0.0
            
            # Weighted combined ratio
            combined_ratio = (camera_ratio * camera_weight) + (imu_ratio * imu_weight)
            
            # Perfect if combined ratio >= min_features_acceptable
            is_perfect = combined_ratio >= self.min_features_acceptable
            
            # Store max absolute Z-score (across both types)
            all_z_scores = list(camera_z_scores.values()) + list(imu_z_scores.values())
            max_abs_z = max([abs(z) for z in all_z_scores]) if all_z_scores else 0.0
            
            z_score_details.append({
                'sample_idx': len(z_score_details),
                'is_perfect': is_perfect,
                'camera_ratio': camera_ratio,
                'imu_ratio': imu_ratio,
                'combined_ratio': combined_ratio,
                'max_abs_z': max_abs_z
            })
            
            if is_perfect:
                perfect_samples.append(sample)
                sample.is_perfect_form = True
            else:
                non_perfect_samples.append(sample)
                sample.is_perfect_form = False
        
        # Calculate selection statistics
        selection_stats = {
            'total_samples': len(samples),
            'samples_with_both_features': len(samples_with_both),
            'perfect_samples': len(perfect_samples),
            'non_perfect_samples': len(non_perfect_samples),
            'perfect_ratio': len(perfect_samples) / len(samples_with_both) if samples_with_both else 0.0,
            'z_threshold': self.z_threshold,
            'min_features_acceptable': self.min_features_acceptable,
            'camera_weight': camera_weight,
            'imu_weight': imu_weight,
            'camera_features': len(camera_stats),
            'imu_features': len(imu_stats),
            'avg_combined_ratio': np.mean([d['combined_ratio'] for d in z_score_details]) if z_score_details else 0.0,
            'avg_max_abs_z': np.mean([d['max_abs_z'] for d in z_score_details]) if z_score_details else 0.0
        }
        
        if verbose:
            print(f"\n✅ Hybrid Z-Score Perfect Form Selection Results:")
            print(f"   Total samples: {selection_stats['total_samples']}")
            print(f"   Samples with both features: {selection_stats['samples_with_both_features']}")
            print(f"   Perfect form samples: {selection_stats['perfect_samples']} ({selection_stats['perfect_ratio']*100:.1f}%)")
            print(f"   Non-perfect samples: {selection_stats['non_perfect_samples']}")
            print(f"   Camera weight: {camera_weight}, IMU weight: {imu_weight}")
            print(f"   Avg combined ratio: {selection_stats['avg_combined_ratio']*100:.1f}%")
            print(f"   Avg max |Z-score|: {selection_stats['avg_max_abs_z']:.2f}")
        
        return (perfect_samples, non_perfect_samples, selection_stats)


# Example usage
if __name__ == "__main__":
    from dataset_collector import DatasetCollector
    
    # Load dataset
    collector = DatasetCollector("MLTRAINCAMERA")
    samples = collector.load_dataset(exercise="bicep_curls")
    
    # Extract features if not already extracted
    for sample in samples:
        if sample.features is None:
            collector.extract_features(sample)
    
    # Select perfect form samples using Z-score
    selector = ZScorePerfectFormSelector(z_threshold=1.5, min_features_acceptable=0.8)
    perfect_samples, non_perfect, stats = selector.select_perfect_form_samples(
        samples, 
        use_imu_features=False, 
        verbose=True
    )
    
    print(f"\n📊 Selected {len(perfect_samples)} perfect form samples out of {len(samples)} total")

