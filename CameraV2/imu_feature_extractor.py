"""
IMU Feature Extraction for ML Training
Extracts features from IMU sensor data sequences
"""

import numpy as np
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add utils directory to path for quaternion_vectors import
sys.path.append(str(Path(__file__).parent.parent))
try:
    from utils.quaternion_vectors import extract_orientation_vectors, calculate_orientation_features
    QUATERNION_VECTORS_AVAILABLE = True
except ImportError:
    QUATERNION_VECTORS_AVAILABLE = False
    print("⚠️  quaternion_vectors module not available - unit vectors features will be skipped")


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
    
    # ORIENTATION UNIT VECTORS FEATURES (Normal, Tangent, Binormal)
    # These represent the sensor's orientation in 3D space more robustly than Euler angles
    if QUATERNION_VECTORS_AVAILABLE:
        # Extract unit vectors for each sample and compute statistics
        left_wrist_normal_x = []
        left_wrist_normal_y = []
        left_wrist_normal_z = []
        left_wrist_tangent_x = []
        left_wrist_tangent_y = []
        left_wrist_tangent_z = []
        left_wrist_binormal_x = []
        left_wrist_binormal_y = []
        left_wrist_binormal_z = []
        
        right_wrist_normal_x = []
        right_wrist_normal_y = []
        right_wrist_normal_z = []
        right_wrist_tangent_x = []
        right_wrist_tangent_y = []
        right_wrist_tangent_z = []
        right_wrist_binormal_x = []
        right_wrist_binormal_y = []
        right_wrist_binormal_z = []
        
        for sample in imu_sequence:
            # Left wrist unit vectors
            lw_vectors = extract_orientation_vectors(sample, 'left_wrist')
            if lw_vectors:
                left_wrist_normal_x.append(lw_vectors['normal'][0])
                left_wrist_normal_y.append(lw_vectors['normal'][1])
                left_wrist_normal_z.append(lw_vectors['normal'][2])
                left_wrist_tangent_x.append(lw_vectors['tangent'][0])
                left_wrist_tangent_y.append(lw_vectors['tangent'][1])
                left_wrist_tangent_z.append(lw_vectors['tangent'][2])
                left_wrist_binormal_x.append(lw_vectors['binormal'][0])
                left_wrist_binormal_y.append(lw_vectors['binormal'][1])
                left_wrist_binormal_z.append(lw_vectors['binormal'][2])
            
            # Right wrist unit vectors
            rw_vectors = extract_orientation_vectors(sample, 'right_wrist')
            if rw_vectors:
                right_wrist_normal_x.append(rw_vectors['normal'][0])
                right_wrist_normal_y.append(rw_vectors['normal'][1])
                right_wrist_normal_z.append(rw_vectors['normal'][2])
                right_wrist_tangent_x.append(rw_vectors['tangent'][0])
                right_wrist_tangent_y.append(rw_vectors['tangent'][1])
                right_wrist_tangent_z.append(rw_vectors['tangent'][2])
                right_wrist_binormal_x.append(rw_vectors['binormal'][0])
                right_wrist_binormal_y.append(rw_vectors['binormal'][1])
                right_wrist_binormal_z.append(rw_vectors['binormal'][2])
        
        # Compute statistics for left wrist unit vectors
        if left_wrist_normal_x:
            features.update(compute_stats(left_wrist_normal_x, 'lw_normal_x'))
            features.update(compute_stats(left_wrist_normal_y, 'lw_normal_y'))
            features.update(compute_stats(left_wrist_normal_z, 'lw_normal_z'))
            features.update(compute_stats(left_wrist_tangent_x, 'lw_tangent_x'))
            features.update(compute_stats(left_wrist_tangent_y, 'lw_tangent_y'))
            features.update(compute_stats(left_wrist_tangent_z, 'lw_tangent_z'))
            features.update(compute_stats(left_wrist_binormal_x, 'lw_binormal_x'))
            features.update(compute_stats(left_wrist_binormal_y, 'lw_binormal_y'))
            features.update(compute_stats(left_wrist_binormal_z, 'lw_binormal_z'))
        
        # Compute statistics for right wrist unit vectors
        if right_wrist_normal_x:
            features.update(compute_stats(right_wrist_normal_x, 'rw_normal_x'))
            features.update(compute_stats(right_wrist_normal_y, 'rw_normal_y'))
            features.update(compute_stats(right_wrist_normal_z, 'rw_normal_z'))
            features.update(compute_stats(right_wrist_tangent_x, 'rw_tangent_x'))
            features.update(compute_stats(right_wrist_tangent_y, 'rw_tangent_y'))
            features.update(compute_stats(right_wrist_tangent_z, 'rw_tangent_z'))
            features.update(compute_stats(right_wrist_binormal_x, 'rw_binormal_x'))
            features.update(compute_stats(right_wrist_binormal_y, 'rw_binormal_y'))
            features.update(compute_stats(right_wrist_binormal_z, 'rw_binormal_z'))
    
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
    
    # Rep duration (temporal feature - important for speed detection)
    timestamps = [s.get('timestamp', 0) for s in imu_sequence if 'timestamp' in s]
    if len(timestamps) >= 2:
        rep_duration = timestamps[-1] - timestamps[0]  # Seconds
        features['rep_duration'] = float(rep_duration)
        # Sample rate (Hz) - useful for normalization
        if rep_duration > 0:
            features['avg_sample_rate'] = float(len(timestamps) / rep_duration)
    else:
        features['rep_duration'] = 0.0
        features['avg_sample_rate'] = 0.0
    
    # Gyroscope magnitude features (for speed detection)
    # Calculate gyro magnitude for each sample and compute statistics
    gyro_magnitudes = []
    for sample in imu_sequence:
        mags = []
        for node_name in ['left_wrist', 'right_wrist', 'chest']:
            node_data = sample.get(node_name, {})
            if node_data and isinstance(node_data, dict):
                gx = node_data.get('gx', 0) or 0
                gy = node_data.get('gy', 0) or 0
                gz = node_data.get('gz', 0) or 0
                mag = np.sqrt(gx**2 + gy**2 + gz**2)
                mags.append(mag)
        if len(mags) > 0:
            gyro_magnitudes.append(np.mean(mags))
    
    if len(gyro_magnitudes) > 0:
        gyro_mag_arr = np.array(gyro_magnitudes)
        features['gyro_magnitude_mean'] = float(np.mean(gyro_mag_arr))
        features['gyro_magnitude_std'] = float(np.std(gyro_mag_arr))
        features['gyro_magnitude_max'] = float(np.max(gyro_mag_arr))
        features['gyro_magnitude_min'] = float(np.min(gyro_mag_arr))
        features['gyro_magnitude_range'] = float(np.max(gyro_mag_arr) - np.min(gyro_mag_arr))
        # Speed indicator: higher gyro magnitude mean = faster movement
        features['movement_intensity'] = float(np.mean(gyro_mag_arr))
    else:
        features['gyro_magnitude_mean'] = 0.0
        features['gyro_magnitude_std'] = 0.0
        features['gyro_magnitude_max'] = 0.0
        features['gyro_magnitude_min'] = 0.0
        features['gyro_magnitude_range'] = 0.0
        features['movement_intensity'] = 0.0
    
    return features

