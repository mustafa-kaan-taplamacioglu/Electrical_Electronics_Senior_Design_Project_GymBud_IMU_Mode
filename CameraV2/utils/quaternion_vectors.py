"""
Quaternion to Unit Vectors Conversion
Converts quaternion orientation to unit normal, tangent, and binormal vectors.

For IMU sensors attached to body parts, these vectors represent:
- Normal: Sensor's "up" direction (perpendicular to sensor surface)
- Tangent: Forward direction (movement direction)
- Binormal: Lateral direction (cross product of normal and tangent)
"""

import numpy as np
from typing import Dict, Tuple, Optional


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        qw, qx, qy, qz: Quaternion components (w, x, y, z)
    
    Returns:
        3x3 rotation matrix
    """
    # Normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm < 1e-10:
        return np.eye(3)  # Identity matrix for zero quaternion
    
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Build rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    return R


def quaternion_rotate_vector(qw: float, qx: float, qy: float, qz: float, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by quaternion.
    
    Args:
        qw, qx, qy, qz: Quaternion components (w, x, y, z)
        v: 3D vector to rotate
    
    Returns:
        Rotated 3D vector
    """
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    return R @ v


def quaternion_to_unit_vectors(qw: float, qx: float, qy: float, qz: float) -> Dict[str, np.ndarray]:
    """
    Convert quaternion to unit normal, tangent, and binormal vectors.
    
    For IMU sensors:
    - Normal vector (up): Z-axis direction after rotation (sensor surface normal)
    - Tangent vector (forward): Y-axis direction after rotation (movement direction)
    - Binormal vector (right): X-axis direction after rotation (lateral direction)
    
    These form an orthonormal basis (Frenet frame) representing the sensor's orientation.
    
    Args:
        qw, qx, qy, qz: Quaternion components (w, x, y, z)
    
    Returns:
        Dictionary with 'normal', 'tangent', 'binormal' vectors (each is 3D numpy array)
    """
    # Normalize quaternion
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm < 1e-10:
        # Return default identity orientation
        return {
            'normal': np.array([0.0, 0.0, 1.0]),  # Z-axis (up)
            'tangent': np.array([0.0, 1.0, 0.0]),  # Y-axis (forward)
            'binormal': np.array([1.0, 0.0, 0.0])  # X-axis (right)
        }
    
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Get rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    
    # Standard basis vectors (before rotation)
    # For IMU sensors, we use:
    # - Z-axis as "up" (normal to sensor surface)
    # - Y-axis as "forward" (movement direction)
    # - X-axis as "right" (lateral direction)
    
    z_axis = np.array([0.0, 0.0, 1.0])  # Normal (up)
    y_axis = np.array([0.0, 1.0, 0.0])  # Tangent (forward)
    x_axis = np.array([1.0, 0.0, 0.0])  # Binormal (right)
    
    # Rotate basis vectors
    normal = R @ z_axis      # Sensor's "up" direction
    tangent = R @ y_axis     # Sensor's "forward" direction
    binormal = R @ x_axis    # Sensor's "right" direction
    
    # Normalize (should already be unit vectors, but ensure)
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
    binormal = binormal / (np.linalg.norm(binormal) + 1e-10)
    
    return {
        'normal': normal,
        'tangent': tangent,
        'binormal': binormal
    }


def extract_orientation_vectors(imu_sample: Dict, node_name: str = 'left_wrist') -> Optional[Dict[str, np.ndarray]]:
    """
    Extract unit vectors from IMU sample for a specific node.
    
    Args:
        imu_sample: IMU data dictionary containing node data
        node_name: Name of the node ('left_wrist', 'right_wrist', 'chest')
    
    Returns:
        Dictionary with 'normal', 'tangent', 'binormal' vectors or None if not available
    """
    node_data = imu_sample.get(node_name)
    if not node_data or not isinstance(node_data, dict):
        return None
    
    # Try to get quaternion components
    qw = node_data.get('qw') or node_data.get('quaternion', {}).get('w')
    qx = node_data.get('qx') or node_data.get('quaternion', {}).get('x')
    qy = node_data.get('qy') or node_data.get('quaternion', {}).get('y')
    qz = node_data.get('qz') or node_data.get('quaternion', {}).get('z')
    
    if qw is None or qx is None or qy is None or qz is None:
        return None
    
    try:
        return quaternion_to_unit_vectors(float(qw), float(qx), float(qy), float(qz))
    except (ValueError, TypeError):
        return None


def calculate_orientation_features(unit_vectors: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculate features from unit vectors for ML training.
    
    Args:
        unit_vectors: Dictionary with 'normal', 'tangent', 'binormal' vectors
    
    Returns:
        Dictionary of features (24 features: 3 vectors × 8 features per vector)
    """
    features = {}
    
    for vec_name, vec in unit_vectors.items():
        # Vector components
        features[f'{vec_name}_x'] = float(vec[0])
        features[f'{vec_name}_y'] = float(vec[1])
        features[f'{vec_name}_z'] = float(vec[2])
        
        # Vector magnitude (should be ~1.0 for unit vectors)
        features[f'{vec_name}_magnitude'] = float(np.linalg.norm(vec))
        
        # Angles with respect to coordinate axes
        features[f'{vec_name}_angle_x'] = float(np.arccos(np.clip(np.abs(vec[0]), -1, 1)) * 180 / np.pi)
        features[f'{vec_name}_angle_y'] = float(np.arccos(np.clip(np.abs(vec[1]), -1, 1)) * 180 / np.pi)
        features[f'{vec_name}_angle_z'] = float(np.arccos(np.clip(np.abs(vec[2]), -1, 1)) * 180 / np.pi)
    
    return features

