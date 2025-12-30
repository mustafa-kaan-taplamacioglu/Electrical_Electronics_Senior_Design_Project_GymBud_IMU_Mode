"""
Joint Mapping between MM-Fit and MediaPipe
Provides utilities for converting between joint representations.
"""

import numpy as np
from typing import Dict, List, Tuple

# MM-Fit joint indices (common COCO-style format)
MMFIT_JOINTS_2D = {
    0: "nose",
    1: "left_eye",
    2: "right_eye", 
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
    17: "neck",      # derived: midpoint of shoulders
    18: "pelvis"     # derived: midpoint of hips
}

# MediaPipe joint indices
MEDIAPIPE_JOINTS = {
    0: "nose",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle"
}

# Mapping from MM-Fit index to MediaPipe index
MMFIT_TO_MEDIAPIPE = {
    0: 0,    # nose
    1: 2,    # left_eye (use MediaPipe left_eye)
    2: 5,    # right_eye (use MediaPipe right_eye)
    3: 7,    # left_ear
    4: 8,    # right_ear
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}

# Key joints used for exercise analysis (common between both)
COMMON_KEY_JOINTS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# MediaPipe indices for common key joints
MEDIAPIPE_KEY_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def convert_mmfit_to_common(
    pose_data: np.ndarray,
    is_3d: bool = False
) -> np.ndarray:
    """
    Convert MM-Fit pose data to common joint representation.
    
    Args:
        pose_data: MM-Fit pose data, shape (coords, frames, joints)
        is_3d: Whether this is 3D data (3 coords) or 2D (2 coords)
    
    Returns:
        Pose data with only common joints, shape (coords, frames, 13)
    """
    # MM-Fit joint indices for common joints
    mmfit_indices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    # Handle different number of joints in MM-Fit
    max_joint = pose_data.shape[2]
    valid_indices = [i for i in mmfit_indices if i < max_joint]
    
    return pose_data[:, :, valid_indices]


def convert_mediapipe_to_common(pose_data: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe pose data to common joint representation.
    
    Args:
        pose_data: MediaPipe pose data, shape (coords, frames, 33)
    
    Returns:
        Pose data with only common joints, shape (coords, frames, 13)
    """
    return pose_data[:, :, MEDIAPIPE_KEY_INDICES]


def normalize_pose_to_relative(
    pose_data: np.ndarray,
    reference_joint: int = 0  # nose
) -> np.ndarray:
    """
    Normalize pose to relative coordinates (remove absolute position).
    Centers pose on reference joint.
    
    Args:
        pose_data: Pose data, shape (coords, frames, joints)
        reference_joint: Joint index to use as origin
    
    Returns:
        Normalized pose data
    """
    # Get reference joint position for each frame
    reference = pose_data[:, :, reference_joint:reference_joint+1]  # (coords, frames, 1)
    
    # Subtract reference from all joints
    normalized = pose_data - reference
    
    return normalized


def normalize_pose_scale(
    pose_data: np.ndarray,
    left_shoulder_idx: int = 1,
    right_shoulder_idx: int = 2
) -> np.ndarray:
    """
    Normalize pose scale using shoulder width.
    
    Args:
        pose_data: Pose data, shape (coords, frames, joints)
        left_shoulder_idx: Index of left shoulder in common joints
        right_shoulder_idx: Index of right shoulder in common joints
    
    Returns:
        Scale-normalized pose data
    """
    # Calculate shoulder width for each frame
    left_shoulder = pose_data[:2, :, left_shoulder_idx]  # (2, frames)
    right_shoulder = pose_data[:2, :, right_shoulder_idx]  # (2, frames)
    
    shoulder_width = np.linalg.norm(
        left_shoulder - right_shoulder, axis=0
    )  # (frames,)
    
    # Avoid division by zero
    shoulder_width = np.maximum(shoulder_width, 1e-6)
    
    # Normalize all coordinates
    # Reshape shoulder_width to broadcast: (frames,) -> (1, frames, 1)
    normalized = pose_data / shoulder_width[np.newaxis, :, np.newaxis]
    
    return normalized

