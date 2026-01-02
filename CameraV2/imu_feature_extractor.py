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
    # Euler angles and quaternions
    left_wrist_roll = []
    left_wrist_pitch = []
    left_wrist_yaw = []
    left_wrist_quat_w = []
    left_wrist_quat_x = []
    left_wrist_quat_y = []
    left_wrist_quat_z = []
    
    # Accelerometer and gyroscope (NEW - for new format)
    left_wrist_ax = []
    left_wrist_ay = []
    left_wrist_az = []
    left_wrist_gx = []
    left_wrist_gy = []
    left_wrist_gz = []
    
    right_wrist_roll = []
    right_wrist_pitch = []
    right_wrist_yaw = []
    right_wrist_quat_w = []
    right_wrist_quat_x = []
    right_wrist_quat_y = []
    right_wrist_quat_z = []
    
    right_wrist_ax = []
    right_wrist_ay = []
    right_wrist_az = []
    right_wrist_gx = []
    right_wrist_gy = []
    right_wrist_gz = []
    
    chest_roll = []
    chest_pitch = []
    chest_yaw = []
    chest_quat_w = []
    chest_quat_x = []
    chest_quat_y = []
    chest_quat_z = []
    
    chest_ax = []
    chest_ay = []
    chest_az = []
    chest_gx = []
    chest_gy = []
    chest_gz = []
    
    # Extract IMU data from sequence
    # Support both formats: new format (flat) and old format (nested)
    for sample in imu_sequence:
        if sample.get('left_wrist'):
            lw = sample['left_wrist']
            # New format: direct keys (ax, ay, az, gx, gy, gz, qw, qx, qy, qz, roll, pitch, yaw)
            if 'roll' in lw or 'ax' in lw:
                left_wrist_roll.append(lw.get('roll', 0))
                left_wrist_pitch.append(lw.get('pitch', 0))
                left_wrist_yaw.append(lw.get('yaw', 0))
                left_wrist_quat_w.append(lw.get('qw', 1))
                left_wrist_quat_x.append(lw.get('qx', 0))
                left_wrist_quat_y.append(lw.get('qy', 0))
                left_wrist_quat_z.append(lw.get('qz', 0))
                # Accelerometer and gyroscope
                left_wrist_ax.append(lw.get('ax', 0))
                left_wrist_ay.append(lw.get('ay', 0))
                left_wrist_az.append(lw.get('az', 0))
                left_wrist_gx.append(lw.get('gx', 0))
                left_wrist_gy.append(lw.get('gy', 0))
                left_wrist_gz.append(lw.get('gz', 0))
            # Old format: nested (euler, quaternion, accel, gyro)
            else:
                if 'euler' in lw:
                    left_wrist_roll.append(lw['euler'].get('roll', 0))
                    left_wrist_pitch.append(lw['euler'].get('pitch', 0))
                    left_wrist_yaw.append(lw['euler'].get('yaw', 0))
                if 'quaternion' in lw:
                    q = lw['quaternion']
                    left_wrist_quat_w.append(q.get('w', q.get('qw', 1)))
                    left_wrist_quat_x.append(q.get('x', q.get('qx', 0)))
                    left_wrist_quat_y.append(q.get('y', q.get('qy', 0)))
                    left_wrist_quat_z.append(q.get('z', q.get('qz', 0)))
                if 'accel' in lw:
                    a = lw['accel']
                    left_wrist_ax.append(a.get('x', 0))
                    left_wrist_ay.append(a.get('y', 0))
                    left_wrist_az.append(a.get('z', 0))
                if 'gyro' in lw:
                    g = lw['gyro']
                    left_wrist_gx.append(g.get('x', 0))
                    left_wrist_gy.append(g.get('y', 0))
                    left_wrist_gz.append(g.get('z', 0))
        
        if sample.get('right_wrist'):
            rw = sample['right_wrist']
            # New format: direct keys
            if 'roll' in rw or 'ax' in rw:
                right_wrist_roll.append(rw.get('roll', 0))
                right_wrist_pitch.append(rw.get('pitch', 0))
                right_wrist_yaw.append(rw.get('yaw', 0))
                right_wrist_quat_w.append(rw.get('qw', 1))
                right_wrist_quat_x.append(rw.get('qx', 0))
                right_wrist_quat_y.append(rw.get('qy', 0))
                right_wrist_quat_z.append(rw.get('qz', 0))
                # Accelerometer and gyroscope
                right_wrist_ax.append(rw.get('ax', 0))
                right_wrist_ay.append(rw.get('ay', 0))
                right_wrist_az.append(rw.get('az', 0))
                right_wrist_gx.append(rw.get('gx', 0))
                right_wrist_gy.append(rw.get('gy', 0))
                right_wrist_gz.append(rw.get('gz', 0))
            # Old format: nested
            else:
                if 'euler' in rw:
                    right_wrist_roll.append(rw['euler'].get('roll', 0))
                    right_wrist_pitch.append(rw['euler'].get('pitch', 0))
                    right_wrist_yaw.append(rw['euler'].get('yaw', 0))
                if 'quaternion' in rw:
                    q = rw['quaternion']
                    right_wrist_quat_w.append(q.get('w', q.get('qw', 1)))
                    right_wrist_quat_x.append(q.get('x', q.get('qx', 0)))
                    right_wrist_quat_y.append(q.get('y', q.get('qy', 0)))
                    right_wrist_quat_z.append(q.get('z', q.get('qz', 0)))
                if 'accel' in rw:
                    a = rw['accel']
                    right_wrist_ax.append(a.get('x', 0))
                    right_wrist_ay.append(a.get('y', 0))
                    right_wrist_az.append(a.get('z', 0))
                if 'gyro' in rw:
                    g = rw['gyro']
                    right_wrist_gx.append(g.get('x', 0))
                    right_wrist_gy.append(g.get('y', 0))
                    right_wrist_gz.append(g.get('z', 0))
        
        if sample.get('chest'):
            ch = sample['chest']
            # New format: direct keys
            if 'roll' in ch or 'ax' in ch:
                chest_roll.append(ch.get('roll', 0))
                chest_pitch.append(ch.get('pitch', 0))
                chest_yaw.append(ch.get('yaw', 0))
                chest_quat_w.append(ch.get('qw', 1))
                chest_quat_x.append(ch.get('qx', 0))
                chest_quat_y.append(ch.get('qy', 0))
                chest_quat_z.append(ch.get('qz', 0))
                # Accelerometer and gyroscope
                chest_ax.append(ch.get('ax', 0))
                chest_ay.append(ch.get('ay', 0))
                chest_az.append(ch.get('az', 0))
                chest_gx.append(ch.get('gx', 0))
                chest_gy.append(ch.get('gy', 0))
                chest_gz.append(ch.get('gz', 0))
            # Old format: nested
            else:
                if 'euler' in ch:
                    chest_roll.append(ch['euler'].get('roll', 0))
                    chest_pitch.append(ch['euler'].get('pitch', 0))
                    chest_yaw.append(ch['euler'].get('yaw', 0))
                if 'quaternion' in ch:
                    q = ch['quaternion']
                    chest_quat_w.append(q.get('w', q.get('qw', 1)))
                    chest_quat_x.append(q.get('x', q.get('qx', 0)))
                    chest_quat_y.append(q.get('y', q.get('qy', 0)))
                    chest_quat_z.append(q.get('z', q.get('qz', 0)))
                if 'accel' in ch:
                    a = ch['accel']
                    chest_ax.append(a.get('x', 0))
                    chest_ay.append(a.get('y', 0))
                    chest_az.append(a.get('z', 0))
                if 'gyro' in ch:
                    g = ch['gyro']
                    chest_gx.append(g.get('x', 0))
                    chest_gy.append(g.get('y', 0))
                    chest_gz.append(g.get('z', 0))
    
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
    if left_wrist_ax:
        features.update(compute_stats(left_wrist_ax, 'lw_ax'))
        features.update(compute_stats(left_wrist_ay, 'lw_ay'))
        features.update(compute_stats(left_wrist_az, 'lw_az'))
        features.update(compute_stats(left_wrist_gx, 'lw_gx'))
        features.update(compute_stats(left_wrist_gy, 'lw_gy'))
        features.update(compute_stats(left_wrist_gz, 'lw_gz'))
    
    # Right Wrist
    if right_wrist_roll:
        features.update(compute_stats(right_wrist_roll, 'rw_roll'))
        features.update(compute_stats(right_wrist_pitch, 'rw_pitch'))
        features.update(compute_stats(right_wrist_yaw, 'rw_yaw'))
        features.update(compute_stats(right_wrist_quat_w, 'rw_qw'))
        features.update(compute_stats(right_wrist_quat_x, 'rw_qx'))
        features.update(compute_stats(right_wrist_quat_y, 'rw_qy'))
        features.update(compute_stats(right_wrist_quat_z, 'rw_qz'))
    if right_wrist_ax:
        features.update(compute_stats(right_wrist_ax, 'rw_ax'))
        features.update(compute_stats(right_wrist_ay, 'rw_ay'))
        features.update(compute_stats(right_wrist_az, 'rw_az'))
        features.update(compute_stats(right_wrist_gx, 'rw_gx'))
        features.update(compute_stats(right_wrist_gy, 'rw_gy'))
        features.update(compute_stats(right_wrist_gz, 'rw_gz'))
    
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
    if chest_ax:
        features.update(compute_stats(chest_ax, 'ch_ax'))
        features.update(compute_stats(chest_ay, 'ch_ay'))
        features.update(compute_stats(chest_az, 'ch_az'))
        features.update(compute_stats(chest_gx, 'ch_gx'))
        features.update(compute_stats(chest_gy, 'ch_gy'))
        features.update(compute_stats(chest_gz, 'ch_gz'))
    
    # Sequence length
    features['sequence_length'] = float(len(imu_sequence))
    features['has_left_wrist'] = float(len(left_wrist_roll) > 0)
    features['has_right_wrist'] = float(len(right_wrist_roll) > 0)
    
    return features

