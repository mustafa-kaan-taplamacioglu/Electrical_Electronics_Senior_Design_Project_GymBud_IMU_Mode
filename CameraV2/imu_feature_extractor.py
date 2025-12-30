"""
IMU Feature Extraction for ML Training
Extracts features from IMU sensor data sequences
"""

import numpy as np
from typing import Dict, List, Optional


def extract_imu_features(imu_sequence: List[Dict]) -> Dict[str, float]:
    """
    Extract features from IMU data sequence.
    
    Args:
        imu_sequence: List of IMU samples, each containing:
            - left_wrist: {quaternion: {w,x,y,z}, euler: {roll,pitch,yaw}, accel: {x,y,z}, gyro: {x,y,z}}
            - right_wrist: same structure
            - chest: same structure (optional)
            - timestamp: float
    
    Returns:
        Dictionary of extracted features
    """
    if not imu_sequence or len(imu_sequence) == 0:
        return {}
    
    features = {}
    
    # Extract data arrays for each sensor
    left_wrist_roll = []
    left_wrist_pitch = []
    left_wrist_yaw = []
    left_wrist_quat_w = []
    left_wrist_quat_x = []
    left_wrist_quat_y = []
    left_wrist_quat_z = []
    
    right_wrist_roll = []
    right_wrist_pitch = []
    right_wrist_yaw = []
    right_wrist_quat_w = []
    right_wrist_quat_x = []
    right_wrist_quat_y = []
    right_wrist_quat_z = []
    
    chest_roll = []
    chest_pitch = []
    chest_yaw = []
    chest_quat_w = []
    chest_quat_x = []
    chest_quat_y = []
    chest_quat_z = []
    
    # Extract IMU data from sequence
    for sample in imu_sequence:
        if sample.get('left_wrist'):
            lw = sample['left_wrist']
            if 'euler' in lw:
                left_wrist_roll.append(lw['euler'].get('roll', 0))
                left_wrist_pitch.append(lw['euler'].get('pitch', 0))
                left_wrist_yaw.append(lw['euler'].get('yaw', 0))
            if 'quaternion' in lw:
                q = lw['quaternion']
                left_wrist_quat_w.append(q.get('w', 1))
                left_wrist_quat_x.append(q.get('x', 0))
                left_wrist_quat_y.append(q.get('y', 0))
                left_wrist_quat_z.append(q.get('z', 0))
        
        if sample.get('right_wrist'):
            rw = sample['right_wrist']
            if 'euler' in rw:
                right_wrist_roll.append(rw['euler'].get('roll', 0))
                right_wrist_pitch.append(rw['euler'].get('pitch', 0))
                right_wrist_yaw.append(rw['euler'].get('yaw', 0))
            if 'quaternion' in rw:
                q = rw['quaternion']
                right_wrist_quat_w.append(q.get('w', 1))
                right_wrist_quat_x.append(q.get('x', 0))
                right_wrist_quat_y.append(q.get('y', 0))
                right_wrist_quat_z.append(q.get('z', 0))
        
        if sample.get('chest'):
            ch = sample['chest']
            if 'euler' in ch:
                chest_roll.append(ch['euler'].get('roll', 0))
                chest_pitch.append(ch['euler'].get('pitch', 0))
                chest_yaw.append(ch['euler'].get('yaw', 0))
            if 'quaternion' in ch:
                q = ch['quaternion']
                chest_quat_w.append(q.get('w', 1))
                chest_quat_x.append(q.get('x', 0))
                chest_quat_y.append(q.get('y', 0))
                chest_quat_z.append(q.get('z', 0))
    
    # Helper function to compute statistics
    def compute_stats(arr, prefix):
        if len(arr) == 0:
            return {}
        arr_np = np.array(arr)
        return {
            f'{prefix}_mean': float(np.mean(arr_np)),
            f'{prefix}_std': float(np.std(arr_np)),
            f'{prefix}_min': float(np.min(arr_np)),
            f'{prefix}_max': float(np.max(arr_np)),
            f'{prefix}_range': float(np.max(arr_np) - np.min(arr_np)),
            f'{prefix}_median': float(np.median(arr_np)),
        }
    
    # Extract features for each sensor
    # Left Wrist
    if left_wrist_roll:
        features.update(compute_stats(left_wrist_roll, 'lw_roll'))
        features.update(compute_stats(left_wrist_pitch, 'lw_pitch'))
        features.update(compute_stats(left_wrist_yaw, 'lw_yaw'))
        features.update(compute_stats(left_wrist_quat_w, 'lw_qw'))
        features.update(compute_stats(left_wrist_quat_x, 'lw_qx'))
        features.update(compute_stats(left_wrist_quat_y, 'lw_qy'))
        features.update(compute_stats(left_wrist_quat_z, 'lw_qz'))
    
    # Right Wrist
    if right_wrist_roll:
        features.update(compute_stats(right_wrist_roll, 'rw_roll'))
        features.update(compute_stats(right_wrist_pitch, 'rw_pitch'))
        features.update(compute_stats(right_wrist_yaw, 'rw_yaw'))
        features.update(compute_stats(right_wrist_quat_w, 'rw_qw'))
        features.update(compute_stats(right_wrist_quat_x, 'rw_qx'))
        features.update(compute_stats(right_wrist_quat_y, 'rw_qy'))
        features.update(compute_stats(right_wrist_quat_z, 'rw_qz'))
    
    # Cross-sensor features (symmetry, coordination)
    if left_wrist_roll and right_wrist_roll:
        lw_arr = np.array(left_wrist_roll)
        rw_arr = np.array(right_wrist_roll)
        # Symmetry: difference between left and right
        if len(lw_arr) == len(rw_arr):
            symmetry_diff = np.abs(lw_arr - rw_arr)
            features['wrist_symmetry_mean'] = float(np.mean(symmetry_diff))
            features['wrist_symmetry_max'] = float(np.max(symmetry_diff))
    
    # Chest (optional)
    if chest_roll:
        features.update(compute_stats(chest_roll, 'ch_roll'))
        features.update(compute_stats(chest_pitch, 'ch_pitch'))
        features.update(compute_stats(chest_yaw, 'ch_yaw'))
        features.update(compute_stats(chest_quat_w, 'ch_qw'))
        features.update(compute_stats(chest_quat_x, 'ch_qx'))
        features.update(compute_stats(chest_quat_y, 'ch_qy'))
        features.update(compute_stats(chest_quat_z, 'ch_qz'))
        features['has_chest'] = 1.0
    else:
        features['has_chest'] = 0.0
    
    # Sequence length
    features['sequence_length'] = float(len(imu_sequence))
    features['has_left_wrist'] = float(len(left_wrist_roll) > 0)
    features['has_right_wrist'] = float(len(right_wrist_roll) > 0)
    
    return features

