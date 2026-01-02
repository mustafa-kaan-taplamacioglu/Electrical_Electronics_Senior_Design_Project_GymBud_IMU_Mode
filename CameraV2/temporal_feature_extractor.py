"""
Temporal Feature Extraction for Exercise Repetitions
Analyzes periodic movement patterns in rep sequences
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import signal
from scipy.fft import fft, fftfreq


def extract_temporal_features(landmarks_sequence: List[List[Dict]], fps: float = 20.0) -> Dict[str, float]:
    """
    Extract temporal/periodic features from a rep sequence.
    
    Args:
        landmarks_sequence: List of frames, each frame has 33 landmarks
        fps: Frames per second (default: 20.0)
    
    Returns:
        Dictionary of temporal features
    """
    if not landmarks_sequence or len(landmarks_sequence) < 10:
        return {}
    
    features = {}
    
    # Extract key joint positions over time
    # For bicep curls: left/right wrist Y positions (vertical movement)
    # For squats: hip Y positions
    # We'll use a generic approach: track multiple key points
    
    # Key points to track (indices from MediaPipe)
    key_points = {
        'left_wrist': 15,
        'right_wrist': 16,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
    }
    
    # Extract time series for each key point (Y coordinate - vertical movement)
    time_series = {}
    for point_name, point_idx in key_points.items():
        if point_idx < len(landmarks_sequence[0]) if landmarks_sequence else False:
            y_values = []
            for frame in landmarks_sequence:
                if point_idx < len(frame) and frame[point_idx].get('y') is not None:
                    y_values.append(frame[point_idx]['y'])
                else:
                    y_values.append(np.nan)
            
            if len(y_values) > 0 and not all(np.isnan(y_values)):
                # Interpolate NaN values
                y_array = np.array(y_values)
                nan_mask = np.isnan(y_array)
                if np.any(nan_mask) and not np.all(nan_mask):
                    y_array[nan_mask] = np.interp(
                        np.where(nan_mask)[0],
                        np.where(~nan_mask)[0],
                        y_array[~nan_mask]
                    )
                time_series[point_name] = y_array
    
    if not time_series:
        return {}
    
    # 1. Movement Frequency Analysis (FFT)
    for point_name, values in time_series.items():
        if len(values) < 10:
            continue
        
        # Detrend (remove linear trend)
        values_detrended = signal.detrend(values)
        
        # FFT to find dominant frequency
        fft_vals = np.abs(fft(values_detrended))
        freqs = fftfreq(len(values_detrended), 1.0 / fps)
        
        # Find dominant frequency (excluding DC component)
        positive_freqs = freqs[1:len(freqs)//2]
        positive_fft = fft_vals[1:len(fft_vals)//2]
        
        if len(positive_freqs) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            dominant_freq = positive_freqs[dominant_freq_idx]
            dominant_magnitude = positive_fft[dominant_freq_idx]
            
            features[f'{point_name}_dominant_freq'] = float(dominant_freq)
            features[f'{point_name}_freq_magnitude'] = float(dominant_magnitude)
            
            # Period (time for one cycle)
            if dominant_freq > 0:
                period = 1.0 / dominant_freq
                features[f'{point_name}_period'] = float(period)
    
    # 2. Movement Smoothness (velocity and acceleration variance)
    for point_name, values in time_series.items():
        if len(values) < 3:
            continue
        
        # Calculate velocity (first derivative)
        velocity = np.diff(values) * fps  # Convert to units per second
        
        if len(velocity) > 1:
            velocity_std = np.std(velocity)
            velocity_mean = np.abs(np.mean(velocity))
            features[f'{point_name}_velocity_std'] = float(velocity_std)
            features[f'{point_name}_velocity_mean'] = float(velocity_mean)
            
            # Smoothness: lower std = smoother movement
            if velocity_mean > 0:
                smoothness = 1.0 / (1.0 + velocity_std / velocity_mean)
                features[f'{point_name}_smoothness'] = float(smoothness)
        
        # Calculate acceleration (second derivative)
        if len(velocity) > 1:
            acceleration = np.diff(velocity) * fps
            if len(acceleration) > 0:
                acceleration_std = np.std(acceleration)
                features[f'{point_name}_acceleration_std'] = float(acceleration_std)
    
    # 3. Movement Range (amplitude)
    for point_name, values in time_series.items():
        if len(values) > 0:
            range_val = np.max(values) - np.min(values)
            features[f'{point_name}_range'] = float(range_val)
            features[f'{point_name}_mean'] = float(np.mean(values))
            features[f'{point_name}_std'] = float(np.std(values))
    
    # 4. Cross-correlation between left and right (symmetry)
    if 'left_wrist' in time_series and 'right_wrist' in time_series:
        left_vals = time_series['left_wrist']
        right_vals = time_series['right_wrist']
        
        if len(left_vals) == len(right_vals) and len(left_vals) > 10:
            # Cross-correlation
            correlation = np.corrcoef(left_vals, right_vals)[0, 1]
            features['wrist_symmetry_correlation'] = float(correlation)
            
            # Phase difference (if periodic)
            if 'left_wrist_dominant_freq' in features and 'right_wrist_dominant_freq' in features:
                # Simple phase difference using cross-correlation lag
                cross_corr = np.correlate(
                    signal.detrend(left_vals),
                    signal.detrend(right_vals),
                    mode='full'
                )
                max_lag = np.argmax(cross_corr) - len(left_vals) + 1
                phase_diff = (max_lag / len(left_vals)) * 2 * np.pi
                features['wrist_phase_difference'] = float(phase_diff)
    
    # 5. Rep duration and consistency
    if len(landmarks_sequence) > 0:
        duration = len(landmarks_sequence) / fps
        features['rep_duration'] = float(duration)
        features['num_frames'] = float(len(landmarks_sequence))
    
    return features


def extract_imu_temporal_features(imu_sequence: List[Dict], fps: float = 20.0) -> Dict[str, float]:
    """
    Extract temporal features from IMU sequence.
    
    Args:
        imu_sequence: List of IMU samples
        fps: Samples per second (default: 20.0)
    
    Returns:
        Dictionary of temporal features
    """
    if not imu_sequence or len(imu_sequence) < 10:
        return {}
    
    features = {}
    
    # Extract time series for key IMU measurements
    # Left wrist roll (rotation around X axis - typical for bicep curls)
    left_wrist_roll = []
    right_wrist_roll = []
    left_wrist_ax = []  # Accelerometer X
    right_wrist_ax = []
    
    for sample in imu_sequence:
        if sample.get('left_wrist'):
            lw = sample['left_wrist']
            left_wrist_roll.append(lw.get('roll', 0))
            left_wrist_ax.append(lw.get('ax', 0))
        
        if sample.get('right_wrist'):
            rw = sample['right_wrist']
            right_wrist_roll.append(rw.get('roll', 0))
            right_wrist_ax.append(rw.get('ax', 0))
    
    # FFT analysis for periodic movement
    for name, values in [
        ('lw_roll', left_wrist_roll),
        ('rw_roll', right_wrist_roll),
        ('lw_ax', left_wrist_ax),
        ('rw_ax', right_wrist_ax)
    ]:
        if len(values) >= 10:
            values_array = np.array(values)
            values_detrended = signal.detrend(values_array)
            
            # FFT
            fft_vals = np.abs(fft(values_detrended))
            freqs = fftfreq(len(values_detrended), 1.0 / fps)
            
            positive_freqs = freqs[1:len(freqs)//2]
            positive_fft = fft_vals[1:len(fft_vals)//2]
            
            if len(positive_freqs) > 0:
                dominant_freq_idx = np.argmax(positive_fft)
                dominant_freq = positive_freqs[dominant_freq_idx]
                features[f'{name}_dominant_freq'] = float(dominant_freq)
                
                if dominant_freq > 0:
                    period = 1.0 / dominant_freq
                    features[f'{name}_period'] = float(period)
            
            # Velocity and smoothness
            velocity = np.diff(values_array) * fps
            if len(velocity) > 0:
                features[f'{name}_velocity_std'] = float(np.std(velocity))
                features[f'{name}_velocity_mean'] = float(np.abs(np.mean(velocity)))
    
    # Symmetry between left and right
    if len(left_wrist_roll) == len(right_wrist_roll) and len(left_wrist_roll) > 10:
        correlation = np.corrcoef(left_wrist_roll, right_wrist_roll)[0, 1]
        features['imu_wrist_symmetry_correlation'] = float(correlation)
    
    # Rep duration
    if len(imu_sequence) > 0:
        duration = len(imu_sequence) / fps
        features['imu_rep_duration'] = float(duration)
    
    return features

