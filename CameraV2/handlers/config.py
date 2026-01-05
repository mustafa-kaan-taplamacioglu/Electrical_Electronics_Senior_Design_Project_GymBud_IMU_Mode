"""
Configuration and constants for API server
"""

# Exercise configs with required landmarks for calibration
EXERCISE_CONFIG = {
    "bicep_curls": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 60, "down": 140},
        "required_landmarks": list(range(0, 23)),
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "dumbbell_shoulder_press": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 160, "down": 90},
        "required_landmarks": list(range(0, 23)),
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "lateral_shoulder_raises": {
        "joints": {"left": (23, 11, 13), "right": (24, 12, 14)},
        "rep_threshold": {"up": 80, "down": 20},
        "required_landmarks": list(range(0, 23)),
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "triceps_pushdown": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 160, "down": 60},
        "required_landmarks": list(range(0, 23)),
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "dumbbell_rows": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 60, "down": 150},
        "required_landmarks": list(range(0, 33)),
        "calibration_message": "Full body must be visible",
    },
    "squats": {
        "joints": {"left": (23, 25, 27), "right": (24, 26, 28)},
        "rep_threshold": {"up": 160, "down": 90},
        "required_landmarks": list(range(0, 33)),
        "calibration_message": "Full body must be visible",
    },
    "dev_mode": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 0, "down": 0},
        "required_landmarks": [0, 2, 5, 15, 16],
        "calibration_message": "Face and hands must be visible",
    },
}

# Feedback templates
FEEDBACK_TEMPLATES = [
    "Great job! {detail}",
    "Looking good! {detail}",
    "Nice work! {detail}",
    "Keep it up! {detail}",
    "Excellent! {detail}",
    "{detail} Keep going!",
    "{detail} You're doing great!",
]

