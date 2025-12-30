"""
Feature Extraction for Exercise Motion Analysis
Extracts kinematic features: angles, ROM, velocity, smoothness
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import skew, kurtosis


# Joint angle definitions using common joint indices (after mapping)
# Format: (parent, joint, child) - angle is computed at 'joint'
# Common joint order: nose(0), l_shoulder(1), r_shoulder(2), l_elbow(3), r_elbow(4),
#                     l_wrist(5), r_wrist(6), l_hip(7), r_hip(8), l_knee(9), r_knee(10),
#                     l_ankle(11), r_ankle(12)

ANGLE_DEFINITIONS = {
    "left_elbow": (1, 3, 5),      # shoulder-elbow-wrist
    "right_elbow": (2, 4, 6),
    "left_shoulder": (3, 1, 7),   # elbow-shoulder-hip
    "right_shoulder": (4, 2, 8),
    "left_hip": (1, 7, 9),        # shoulder-hip-knee
    "right_hip": (2, 8, 10),
    "left_knee": (7, 9, 11),      # hip-knee-ankle
    "right_knee": (8, 10, 12),
}


def compute_angle(
    p1: np.ndarray, 
    p2: np.ndarray, 
    p3: np.ndarray
) -> np.ndarray:
    """
    Compute angle at p2 formed by vectors p2->p1 and p2->p3.
    
    Args:
        p1, p2, p3: Points with shape (..., 2) or (..., 3)
    
    Returns:
        Angles in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Compute dot product and magnitudes
    dot = np.sum(v1 * v2, axis=-1)
    mag1 = np.linalg.norm(v1, axis=-1)
    mag2 = np.linalg.norm(v2, axis=-1)
    
    # Avoid division by zero
    denom = mag1 * mag2
    denom = np.maximum(denom, 1e-8)
    
    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    return np.degrees(angle_rad)


def extract_joint_angles(
    pose_data: np.ndarray,
    angle_definitions: Dict = None
) -> Dict[str, np.ndarray]:
    """
    Extract joint angles over time from pose data.
    
    Args:
        pose_data: Pose data, shape (coords, frames, joints)
        angle_definitions: Dict mapping angle name to (p1_idx, p2_idx, p3_idx)
    
    Returns:
        Dict mapping angle name to angle time series (frames,)
    """
    if angle_definitions is None:
        angle_definitions = ANGLE_DEFINITIONS
    
    coords = pose_data.shape[0]
    frames = pose_data.shape[1]
    
    # Transpose to (frames, joints, coords) for easier indexing
    pose = pose_data.transpose(1, 2, 0)  # (frames, joints, coords)
    
    angles = {}
    for angle_name, (p1_idx, p2_idx, p3_idx) in angle_definitions.items():
        try:
            p1 = pose[:, p1_idx, :2]  # Use only x, y for 2D angles
            p2 = pose[:, p2_idx, :2]
            p3 = pose[:, p3_idx, :2]
            
            angles[angle_name] = compute_angle(p1, p2, p3)
        except IndexError:
            # Joint not available
            angles[angle_name] = np.zeros(frames)
    
    return angles


def compute_velocity(
    values: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Compute velocity (first derivative) of a time series.
    
    Args:
        values: Time series, shape (frames,)
        fps: Frames per second
    
    Returns:
        Velocity time series
    """
    if len(values) < 2:
        return np.zeros_like(values)
    
    velocity = np.gradient(values) * fps
    return velocity


def compute_acceleration(
    values: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Compute acceleration (second derivative) of a time series.
    
    Args:
        values: Time series, shape (frames,)
        fps: Frames per second
    
    Returns:
        Acceleration time series
    """
    velocity = compute_velocity(values, fps)
    acceleration = compute_velocity(velocity, fps)
    return acceleration


def compute_smoothness(
    values: np.ndarray,
    fps: float = 30.0
) -> float:
    """
    Compute smoothness metric using spectral arc length (SPARC).
    Lower values indicate smoother movement.
    
    Args:
        values: Time series, shape (frames,)
        fps: Frames per second
    
    Returns:
        Smoothness score (negative, higher = smoother)
    """
    if len(values) < 4:
        return 0.0
    
    # Compute velocity
    velocity = compute_velocity(values, fps)
    
    # Compute power spectrum
    n = len(velocity)
    freq = np.fft.rfftfreq(n, 1/fps)
    spectrum = np.abs(np.fft.rfft(velocity))
    
    # Normalize spectrum
    if spectrum.max() > 0:
        spectrum = spectrum / spectrum.max()
    
    # Compute spectral arc length
    # Only consider frequencies up to a threshold
    freq_threshold = 10.0  # Hz
    mask = freq <= freq_threshold
    
    if mask.sum() < 2:
        return 0.0
    
    freq_masked = freq[mask]
    spectrum_masked = spectrum[mask]
    
    # Arc length
    dfreq = np.diff(freq_masked)
    dspec = np.diff(spectrum_masked)
    arc_length = -np.sum(np.sqrt(dfreq**2 + dspec**2))
    
    return arc_length


def extract_rom_features(angles: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Extract Range of Motion (ROM) features from angle time series.
    
    Args:
        angles: Dict mapping angle name to angle time series
    
    Returns:
        Dict with ROM features for each angle
    """
    features = {}
    
    for angle_name, angle_series in angles.items():
        if len(angle_series) == 0:
            continue
            
        features[f"{angle_name}_min"] = float(np.min(angle_series))
        features[f"{angle_name}_max"] = float(np.max(angle_series))
        features[f"{angle_name}_range"] = float(np.max(angle_series) - np.min(angle_series))
        features[f"{angle_name}_mean"] = float(np.mean(angle_series))
        features[f"{angle_name}_std"] = float(np.std(angle_series))
    
    return features


def extract_dynamics_features(
    angles: Dict[str, np.ndarray],
    fps: float = 30.0
) -> Dict[str, float]:
    """
    Extract dynamic features (velocity, acceleration, smoothness).
    
    Args:
        angles: Dict mapping angle name to angle time series
        fps: Frames per second
    
    Returns:
        Dict with dynamics features
    """
    features = {}
    
    for angle_name, angle_series in angles.items():
        if len(angle_series) < 2:
            continue
        
        velocity = compute_velocity(angle_series, fps)
        acceleration = compute_acceleration(angle_series, fps)
        smoothness = compute_smoothness(angle_series, fps)
        
        features[f"{angle_name}_vel_mean"] = float(np.mean(np.abs(velocity)))
        features[f"{angle_name}_vel_max"] = float(np.max(np.abs(velocity)))
        features[f"{angle_name}_acc_mean"] = float(np.mean(np.abs(acceleration)))
        features[f"{angle_name}_smoothness"] = float(smoothness)
    
    return features


def extract_temporal_features(
    angles: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Extract temporal pattern features.
    
    Args:
        angles: Dict mapping angle name to angle time series
    
    Returns:
        Dict with temporal features
    """
    features = {}
    
    for angle_name, angle_series in angles.items():
        if len(angle_series) < 4:
            continue
        
        # Skewness and kurtosis
        features[f"{angle_name}_skew"] = float(skew(angle_series))
        features[f"{angle_name}_kurtosis"] = float(kurtosis(angle_series))
        
        # Zero crossings (normalized by length)
        zero_mean = angle_series - np.mean(angle_series)
        zero_crossings = np.sum(np.diff(np.sign(zero_mean)) != 0)
        features[f"{angle_name}_zero_crossings"] = float(zero_crossings / len(angle_series))
        
        # Peak count (normalized)
        peaks, _ = signal.find_peaks(angle_series)
        features[f"{angle_name}_peak_count"] = float(len(peaks) / len(angle_series))
    
    return features


def extract_all_features(
    pose_data: np.ndarray,
    fps: float = 30.0
) -> Dict[str, float]:
    """
    Extract all kinematic features from pose data.
    
    Args:
        pose_data: Pose data, shape (coords, frames, joints)
        fps: Frames per second
    
    Returns:
        Dict with all features
    """
    # Extract joint angles
    angles = extract_joint_angles(pose_data)
    
    # Extract feature groups
    rom_features = extract_rom_features(angles)
    dynamics_features = extract_dynamics_features(angles, fps)
    temporal_features = extract_temporal_features(angles)
    
    # Combine all features
    all_features = {}
    all_features.update(rom_features)
    all_features.update(dynamics_features)
    all_features.update(temporal_features)
    
    return all_features


def features_to_vector(
    features: Dict[str, float],
    feature_order: List[str] = None
) -> np.ndarray:
    """
    Convert feature dict to fixed-length vector.
    
    Args:
        features: Dict of features
        feature_order: List of feature names defining order.
                      If None, uses sorted keys.
    
    Returns:
        Feature vector as numpy array
    """
    if feature_order is None:
        feature_order = sorted(features.keys())
    
    vector = np.array([
        features.get(name, 0.0) for name in feature_order
    ])
    
    return vector

