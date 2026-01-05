"""Helper functions for pose analysis."""

import numpy as np

def check_required_landmarks(landmarks: list, required: list, threshold: float = 0.5, min_visibility_ratio: float = 0.8) -> tuple:
    """Check if required landmarks are visible.
    Returns (all_visible, visible_count, missing_landmarks)
    
    Args:
        landmarks: List of landmark dicts
        required: List of required landmark indices
        threshold: Minimum visibility score (0.0-1.0)
        min_visibility_ratio: Minimum ratio of landmarks that must be visible (default 0.8 = 80%)
    """
    visible_count = 0
    missing = []
    
    LANDMARK_NAMES = {
        0: "nose", 2: "right eye", 5: "right ear",
        11: "left shoulder", 12: "right shoulder",
        13: "left elbow", 14: "right elbow",
        15: "left wrist", 16: "right wrist",
        23: "left hip", 24: "right hip",
        25: "left knee", 26: "right knee",
        27: "left ankle", 28: "right ankle",
    }
    
    for idx in required:
        if idx < len(landmarks):
            visibility = landmarks[idx].get('visibility', 0)
            if visibility >= threshold:
                visible_count += 1
            else:
                missing.append(LANDMARK_NAMES.get(idx, f"nokta {idx}"))
    
    # Require at least min_visibility_ratio (default 80%) of landmarks to be visible
    min_required = int(len(required) * min_visibility_ratio)
    all_visible = visible_count >= min_required
    return (all_visible, visible_count, missing)


def calculate_angle(a, b, c):
    """Calculate angle between three points (at point b)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle


# ============ BONE-BASED ANALYSIS ============

# Key bones for exercise analysis
BONES = {
    # Upper body
    'left_upper_arm': (11, 13),   # Omuz -> Dirsek
    'left_forearm': (13, 15),     # Dirsek -> Bilek
    'right_upper_arm': (12, 14),
    'right_forearm': (14, 16),
    'shoulders': (11, 12),        # Omuz çizgisi
    'left_torso': (11, 23),       # Sol gövde
    'right_torso': (12, 24),      # Sağ gövde
    'hips': (23, 24),             # Kalça çizgisi
    
    # Lower body
    'left_thigh': (23, 25),       # Kalça -> Diz
    'left_shin': (25, 27),        # Diz -> Ayak
    'right_thigh': (24, 26),
    'right_shin': (26, 28),
}


def get_bone_vector(landmarks, bone_name):
    """Get the vector representing a bone."""
    if bone_name not in BONES:
        return None
    start_idx, end_idx = BONES[bone_name]
    start = landmarks[start_idx]
    end = landmarks[end_idx]
    return {
        'start': (start['x'], start['y']),
        'end': (end['x'], end['y']),
        'dx': end['x'] - start['x'],
        'dy': end['y'] - start['y'],
    }


def get_bone_length(landmarks, bone_name):
    """Get the length of a bone."""
    vec = get_bone_vector(landmarks, bone_name)
    if vec is None:
        return 0
    return np.sqrt(vec['dx']**2 + vec['dy']**2)


def get_bone_angle_from_vertical(landmarks, bone_name):
    """Get angle of bone from vertical (0° = pointing down, 90° = horizontal)."""
    vec = get_bone_vector(landmarks, bone_name)
    if vec is None:
        return 0
    # Angle from vertical (positive y is down in screen coords)
    angle = np.degrees(np.arctan2(vec['dx'], vec['dy']))
    return abs(angle)


def get_bone_angle_from_horizontal(landmarks, bone_name):
    """Get angle of bone from horizontal (0° = horizontal, 90° = vertical)."""
    vec = get_bone_vector(landmarks, bone_name)
    if vec is None:
        return 0
    angle = np.degrees(np.arctan2(vec['dy'], vec['dx']))
    return abs(angle)


def get_angle_between_bones(landmarks, bone1_name, bone2_name):
    """Get angle between two connected bones."""
    vec1 = get_bone_vector(landmarks, bone1_name)
    vec2 = get_bone_vector(landmarks, bone2_name)
    if vec1 is None or vec2 is None:
        return 0
    
    # Use dot product to find angle
    dot = vec1['dx'] * vec2['dx'] + vec1['dy'] * vec2['dy']
    mag1 = np.sqrt(vec1['dx']**2 + vec1['dy']**2)
    mag2 = np.sqrt(vec2['dx']**2 + vec2['dy']**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    cos_angle = np.clip(dot / (mag1 * mag2), -1, 1)
    return np.degrees(np.arccos(cos_angle))

