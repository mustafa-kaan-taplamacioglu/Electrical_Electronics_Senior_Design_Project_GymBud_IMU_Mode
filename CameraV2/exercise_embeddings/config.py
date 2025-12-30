"""
Configuration for Exercise Embeddings Pipeline
"""

from pathlib import Path

# Dataset paths
MMFIT_PATH = Path("mm-fit")

# Included exercises (excluding Cardio: jumping_jacks, Core: situps)
INCLUDED_EXERCISES = [
    "squats",
    "lunges", 
    "pushups",
    "dumbbell_shoulder_press",
    "tricep_extensions",
    "dumbbell_rows",
    "bicep_curls",
    "lateral_shoulder_raises"
]

# MM-Fit uses a subset of joints. We map to MediaPipe indices.
# MM-Fit 2D has 19 joints, 3D has 18 joints
# MediaPipe has 33 landmarks

# MM-Fit joint order (based on common pose estimation formats):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle,
# 17: neck, 18: pelvis (for 2D only)

# MediaPipe Pose landmarks:
MEDIAPIPE_LANDMARKS = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

# Key joints for exercise analysis (MediaPipe indices)
KEY_JOINTS = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28
}

# Joint angles to compute for each exercise type
JOINT_ANGLES = {
    # Format: (joint1, joint2, joint3) -> angle at joint2
    "left_elbow": (11, 13, 15),   # shoulder-elbow-wrist
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),  # elbow-shoulder-hip
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),      # shoulder-hip-knee
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),     # hip-knee-ankle
    "right_knee": (24, 26, 28),
}

# Embedding dimension
EMBEDDING_DIM = 64

