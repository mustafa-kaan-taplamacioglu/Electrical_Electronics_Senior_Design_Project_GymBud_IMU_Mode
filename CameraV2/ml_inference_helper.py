"""
ML Inference Helper Functions
==============================
Helper functions for ML model inference and baseline similarity calculation.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import json


def calculate_baseline_similarity(
    current_features: Dict[str, float] = None, 
    baselines: Dict[str, Dict[str, float]] = None,
    current_regional_scores: Dict[str, float] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate how similar current features/regional scores are to perfect form baselines.
    
    Args:
        current_features: Dict of feature_name -> value for current rep (optional, for feature-based similarity)
        baselines: Dict of feature_name -> {mean, std, min, max, median} from perfect form samples (optional)
        current_regional_scores: Dict of regional scores {"arms": float, "legs": float, ...} (optional, for regional similarity)
    
    Returns:
        Tuple: (similarity_score, feature_similarities)
            - similarity_score: 0-100 (100 = perfect match), or Dict[str, float] for regional similarity
            - feature_similarities: Dict of feature_name -> similarity_score, or Dict[str, Dict[str, float]] for regional
    """
    # Regional similarity (preferred if regional scores provided)
    if current_regional_scores and baselines:
        regional_similarities = {}
        regional_avg_similarities = {}
        
        for region in ['arms', 'legs', 'core', 'head']:
            baseline_key = f'regional_score_{region}'
            if baseline_key in baselines and region in current_regional_scores:
                baseline = baselines[baseline_key]
                current_score = current_regional_scores[region]
                
                baseline_mean = baseline.get('mean', 100.0)
                baseline_std = baseline.get('std', 1.0)
                
                # Calculate Z-score
                if baseline_std > 0:
                    z_score = abs(current_score - baseline_mean) / baseline_std
                    similarity = max(0, 100 - (z_score * 16))
                else:
                    if abs(current_score - baseline_mean) < 0.001:
                        similarity = 100.0
                    else:
                        similarity = 50.0
                
                regional_avg_similarities[region] = float(similarity)
                regional_similarities[region] = {
                    'similarity': float(similarity),
                    'current': float(current_score),
                    'baseline_mean': float(baseline_mean)
                }
        
        # Return regional similarity dict
        if regional_avg_similarities:
            return (regional_avg_similarities, regional_similarities)
    
    # Feature-based similarity (fallback or if no regional scores)
    if not baselines or not current_features:
        return ({}, {})
    
    feature_similarities = {}
    
    for feature_name, baseline in baselines.items():
        # Skip regional_score entries (handled above)
        if feature_name.startswith('regional_score_'):
            continue
        
        # Skip non-feature entries
        if not isinstance(baseline, dict) or 'mean' not in baseline:
            continue
        
        current_value = current_features.get(feature_name)
        if current_value is None:
            continue
        
        baseline_mean = baseline.get('mean', 0)
        baseline_std = baseline.get('std', 1.0)
        
        # Calculate Z-score
        if baseline_std > 0:
            z_score = abs(current_value - baseline_mean) / baseline_std
            similarity = max(0, 100 - (z_score * 16))
        else:
            if abs(current_value - baseline_mean) < 0.001:
                similarity = 100.0
            else:
                similarity = 50.0
        
        feature_similarities[feature_name] = float(similarity)
    
    # Calculate average similarity
    if feature_similarities:
        avg_similarity = sum(feature_similarities.values()) / len(feature_similarities)
    else:
        avg_similarity = 0.0
    
    return (float(avg_similarity), feature_similarities)


def calculate_hybrid_correction_score(
    ml_prediction: Dict[str, float],
    baseline_similarity: Dict[str, float],
    ml_weight: float = 0.6,
    baseline_weight: float = 0.4
) -> Dict[str, float]:
    """
    Calculate hybrid correction score combining ML prediction and baseline similarity.
    
    Args:
        ml_prediction: ML model prediction dict (either {"score": float} or {"arms": float, "legs": float, ...})
        baseline_similarity: Baseline similarity dict (either float or {"arms": float, "legs": float, ...})
        ml_weight: Weight for ML prediction (default: 0.6)
        baseline_weight: Weight for baseline similarity (default: 0.4)
    
    Returns:
        Hybrid correction score dict (same structure as ml_prediction)
    """
    # Handle regional scores
    if isinstance(baseline_similarity, dict) and 'arms' in baseline_similarity:
        # Regional scores
        hybrid_scores = {}
        for region in ['arms', 'legs', 'core', 'head']:
            ml_score = ml_prediction.get(region, 0.0) if isinstance(ml_prediction, dict) else 0.0
            baseline_score = baseline_similarity.get(region, 0.0)
            hybrid_score = (ml_score * ml_weight) + (baseline_score * baseline_weight)
            hybrid_scores[region] = float(np.clip(hybrid_score, 0, 100))
        return hybrid_scores
    else:
        # Single overall score
        ml_score = ml_prediction.get('score', 0.0) if isinstance(ml_prediction, dict) else float(ml_prediction)
        baseline_score = float(baseline_similarity) if not isinstance(baseline_similarity, dict) else baseline_similarity.get('score', 0.0)
        hybrid_score = (ml_score * ml_weight) + (baseline_score * baseline_weight)
        return {"score": float(np.clip(hybrid_score, 0, 100))}


def load_baselines(exercise: str) -> Dict:
    """
    Load baseline values for an exercise.
    
    Args:
        exercise: Exercise name (e.g., "bicep_curls")
    
    Returns:
        Dict of baselines or empty dict if not found
    """
    baseline_file = Path("models") / exercise / "form_score_camera_random_forest" / "baselines.json"
    
    if not baseline_file.exists():
        return {}
    
    try:
        with open(baseline_file, 'r') as f:
            baselines = json.load(f)
        return baselines
    except Exception as e:
        print(f"⚠️  Failed to load baselines: {e}")
        return {}


if __name__ == "__main__":
    # Test
    test_features = {
        'left_elbow_range': 120.5,
        'left_elbow_mean': 90.2,
        'left_elbow_std': 35.5
    }
    
    test_baselines = {
        'left_elbow_range': {
            'mean': 120.0,
            'std': 8.3,
            'min': 105.2,
            'max': 135.8,
            'median': 120.0
        },
        'left_elbow_mean': {
            'mean': 90.0,
            'std': 5.1,
            'min': 85.0,
            'max': 95.0,
            'median': 90.0
        }
    }
    
    similarity, feature_sims = calculate_baseline_similarity(test_features, test_baselines)
    print(f"Baseline Similarity: {similarity:.1f}%")
    print(f"Feature similarities: {feature_sims}")

