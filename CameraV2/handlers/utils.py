"""
Utility functions for API server
"""

import numpy as np
from typing import Tuple, List


def check_required_landmarks(
    landmarks: list,
    required: list,
    threshold: float = 0.5,
    min_visibility_ratio: float = 0.8
) -> Tuple[bool, int, List[str]]:
    """Check if required landmarks are visible.
    Returns (all_visible, visible_count, missing_landmarks)
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
            visibility = landmarks[idx].get('visibility', 0.0)
            if visibility >= threshold:
                visible_count += 1
            else:
                missing.append(LANDMARK_NAMES.get(idx, f"nokta {idx}"))
    
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
        angle = 360.0 - angle
    
    return angle

