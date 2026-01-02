"""
FastAPI Backend for Fitness AI Coach
=====================================
Connects React frontend to Python pose analysis.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
from typing import Optional, Dict
import asyncio
import time
from datetime import datetime
from openai import OpenAI
import subprocess
import sys
import copy
import websockets
WEBSOCKETS_AVAILABLE = True

# Dataset collection (optional)
try:
    from dataset_collector import DatasetCollector
    DATASET_COLLECTION_ENABLED = True
except ImportError:
    DATASET_COLLECTION_ENABLED = False

# IMU Dataset collection (for ML training)
try:
    from imu_dataset_collector import IMUDatasetCollector
    IMU_DATASET_COLLECTION_ENABLED = True
except ImportError:
    IMU_DATASET_COLLECTION_ENABLED = False
    print("⚠️  IMU dataset collection disabled (imu_dataset_collector not found)")

# ML Training (optional)
try:
    from ml_trainer import FormScorePredictor, BaselineCalculator
    from dataset_collector import DatasetCollector as DC
    ML_TRAINING_ENABLED = True
except ImportError:
    ML_TRAINING_ENABLED = False
    print("⚠️  ML training disabled (ml_trainer not found)")

# ML Inference (optional)
try:
    from model_inference import ModelInference
    from ml_inference_helper import calculate_baseline_similarity, calculate_hybrid_correction_score, load_baselines
    ML_INFERENCE_ENABLED = True
except ImportError as e:
    ML_INFERENCE_ENABLED = False
    print(f"⚠️  ML inference disabled: {e}")

# Dataset Tracker (optional)
try:
    from dataset_tracker import DatasetTracker
    DATASET_TRACKER_ENABLED = True
except ImportError:
    DATASET_TRACKER_ENABLED = False
    print("⚠️  Dataset tracker disabled (dataset_tracker not found)")

# Import from existing backend
import sys
sys.path.insert(0, '.')

app = FastAPI(title="Fitness AI Coach API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
openai_client: Optional[OpenAI] = None

# Store connected clients
connected_clients = set()


def init_openai(api_key: str):
    global openai_client
    openai_client = OpenAI(api_key=api_key)


# Exercise configs with required landmarks for calibration
# Only 6 exercises: bicep_curls, dumbbell_shoulder_press, lateral_shoulder_raises, triceps_pushdown, dumbbell_rows, squats
EXERCISE_CONFIG = {
    "bicep_curls": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 60, "down": 140},
        # Upper body: Face (0-10), Upper Body (11-16), Hands (17-22) = 23 landmarks
        "required_landmarks": list(range(0, 23)),  # 0-22: Face + Upper Body + Hands
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "dumbbell_shoulder_press": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 160, "down": 90},
        # Upper body: Face (0-10), Upper Body (11-16), Hands (17-22) = 23 landmarks
        "required_landmarks": list(range(0, 23)),  # 0-22: Face + Upper Body + Hands
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "lateral_shoulder_raises": {
        "joints": {"left": (23, 11, 13), "right": (24, 12, 14)},
        "rep_threshold": {"up": 80, "down": 20},
        # Upper body: Face (0-10), Upper Body (11-16), Hands (17-22) = 23 landmarks
        "required_landmarks": list(range(0, 23)),  # 0-22: Face + Upper Body + Hands
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "triceps_pushdown": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 160, "down": 60},
        # Upper body: Face (0-10), Upper Body (11-16), Hands (17-22) = 23 landmarks
        "required_landmarks": list(range(0, 23)),  # 0-22: Face + Upper Body + Hands
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "dumbbell_rows": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 60, "down": 150},
        # Full body: All 33 landmarks (0-32)
        "required_landmarks": list(range(0, 33)),  # 0-32: All landmarks
        "calibration_message": "Full body must be visible",
    },
    "squats": {
        "joints": {"left": (23, 25, 27), "right": (24, 26, 28)},
        "rep_threshold": {"up": 160, "down": 90},
        # Full body: All 33 landmarks (0-32)
        "required_landmarks": list(range(0, 33)),  # 0-32: All landmarks
        "calibration_message": "Full body must be visible",
    },
    "dev_mode": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 0, "down": 0},
        # Only face and hands: nose, eyes, wrists
        "required_landmarks": [0, 2, 5, 15, 16],
        "calibration_message": "Face and hands must be visible",
    },
}


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


class FormAnalyzer:
    """Analyzes exercise form based on pose landmarks."""
    
    CALIBRATION_TIMEOUT = 8.0  # seconds (increased to allow more time)
    CALIBRATION_FRAMES = 20  # Reduced from 30 to 20 (faster calibration)
    CALIBRATION_VISIBILITY_THRESHOLD = 0.5  # Lowered from 0.6 to 0.5 (more lenient)
    CALIBRATION_MIN_VISIBILITY_RATIO = 0.75  # Require 75% of landmarks visible (instead of 100%)
    
    def __init__(self, exercise: str):
        self.exercise = exercise
        self.config = EXERCISE_CONFIG.get(exercise, {})
        self.calibrated = False
        self.calibration_frames = []
        self.calibration_start_time = None
        self.required_landmarks = self.config.get('required_landmarks', [])
        
        # Body proportions (calibrated)
        self.shoulder_width = None
        self.torso_height = None
        self.hip_width = None
        self.upper_arm_length = None
        self.forearm_length = None
        self.thigh_length = None
        self.shin_length = None
        
        # Initial positions for all key joints
        self.initial_positions = {}
    
    def reset_calibration(self):
        """Reset calibration state."""
        self.calibration_frames = []
        self.calibration_start_time = None
        self.calibrated = False
    
    def calibrate(self, landmarks: list) -> tuple:
        """Collect calibration data. Returns (completed, timed_out)
        Only collects frames where ALL required landmarks are above visibility threshold.
        """
        current_time = time.time()
        
        # Start timer on first frame
        if self.calibration_start_time is None:
            self.calibration_start_time = current_time
        
        # Check timeout
        elapsed = current_time - self.calibration_start_time
        if elapsed > self.CALIBRATION_TIMEOUT and len(self.calibration_frames) < self.CALIBRATION_FRAMES:
            self.reset_calibration()
            return (False, True)  # Not completed, timed out
        
        # Check if enough required landmarks are above threshold before adding frame
        all_visible, visible_count, missing = check_required_landmarks(
            landmarks, 
            self.required_landmarks, 
            threshold=self.CALIBRATION_VISIBILITY_THRESHOLD,
            min_visibility_ratio=self.CALIBRATION_MIN_VISIBILITY_RATIO
        )
        
        # Add frame if enough landmarks are visible (e.g., 75% instead of 100%)
        if all_visible:
            self.calibration_frames.append(landmarks)
        # If not enough visible, skip this frame (don't reset, just wait for next frame)
        
        if len(self.calibration_frames) >= self.CALIBRATION_FRAMES:
            # Calculate averages for all 33 landmarks
            # Only use frames where landmark visibility is above threshold
            frames = self.calibration_frames
            avg = []
            for i in range(33):
                # Filter frames where this landmark is visible (visibility >= threshold)
                visible_frames = [
                    f[i] for f in frames 
                    if i < len(f) and f[i].get('visibility', 0) >= self.CALIBRATION_VISIBILITY_THRESHOLD
                ]
                
                if len(visible_frames) > 0:
                    # Calculate average only from visible frames
                    avg_x = sum(f['x'] for f in visible_frames) / len(visible_frames)
                    avg_y = sum(f['y'] for f in visible_frames) / len(visible_frames)
                    avg.append({'x': avg_x, 'y': avg_y, 'calibrated': True, 'visible_frames': len(visible_frames)})
                else:
                    # Landmark was never visible during calibration - mark as not calibrated
                    # Use a default position (0,0) but mark it as not calibrated
                    avg.append({'x': 0.0, 'y': 0.0, 'calibrated': False, 'visible_frames': 0})
            
            # === Body proportions ===
            # Only calculate if required landmarks are calibrated
            if avg[11].get('calibrated', False) and avg[12].get('calibrated', False):
                self.shoulder_width = abs(avg[11]['x'] - avg[12]['x'])
            else:
                self.shoulder_width = None
                print(f"⚠️  Warning: Shoulders not calibrated (left: {avg[11].get('calibrated', False)}, right: {avg[12].get('calibrated', False)})")
            
            if avg[23].get('calibrated', False) and avg[24].get('calibrated', False):
                self.hip_width = abs(avg[23]['x'] - avg[24]['x'])
            else:
                self.hip_width = None
            
            if (avg[11].get('calibrated', False) and avg[12].get('calibrated', False) and 
                avg[23].get('calibrated', False) and avg[24].get('calibrated', False)):
                self.torso_height = abs(
                (avg[11]['y'] + avg[12]['y']) / 2 -
                (avg[23]['y'] + avg[24]['y']) / 2
                )
            else:
                self.torso_height = None
            
            # Arm lengths
            if avg[11].get('calibrated', False) and avg[13].get('calibrated', False):
                self.upper_arm_length = np.sqrt(
                (avg[11]['x'] - avg[13]['x'])**2 + (avg[11]['y'] - avg[13]['y'])**2
                )
            else:
                self.upper_arm_length = None
            
            if avg[13].get('calibrated', False) and avg[15].get('calibrated', False):
                self.forearm_length = np.sqrt(
                (avg[13]['x'] - avg[15]['x'])**2 + (avg[13]['y'] - avg[15]['y'])**2
                )
            else:
                self.forearm_length = None
            
            # Leg lengths
            if avg[23].get('calibrated', False) and avg[25].get('calibrated', False):
                self.thigh_length = np.sqrt(
                (avg[23]['x'] - avg[25]['x'])**2 + (avg[23]['y'] - avg[25]['y'])**2
                )
            else:
                self.thigh_length = None
            
            if avg[25].get('calibrated', False) and avg[27].get('calibrated', False):
                self.shin_length = np.sqrt(
                (avg[25]['x'] - avg[27]['x'])**2 + (avg[25]['y'] - avg[27]['y'])**2
                )
            else:
                self.shin_length = None
            
            # === Initial positions for ALL key joints ===
            # Only include landmarks that were calibrated
            self.initial_positions = {}
            
                # Shoulders
            if avg[11].get('calibrated', False):
                self.initial_positions['left_shoulder'] = {'x': avg[11]['x'], 'y': avg[11]['y']}
            if avg[12].get('calibrated', False):
                self.initial_positions['right_shoulder'] = {'x': avg[12]['x'], 'y': avg[12]['y']}
            
                # Elbows
            if avg[13].get('calibrated', False):
                self.initial_positions['left_elbow'] = {'x': avg[13]['x'], 'y': avg[13]['y']}
            if avg[14].get('calibrated', False):
                self.initial_positions['right_elbow'] = {'x': avg[14]['x'], 'y': avg[14]['y']}
            
                # Wrists
            if avg[15].get('calibrated', False):
                self.initial_positions['left_wrist'] = {'x': avg[15]['x'], 'y': avg[15]['y']}
            if avg[16].get('calibrated', False):
                self.initial_positions['right_wrist'] = {'x': avg[16]['x'], 'y': avg[16]['y']}
            
                # Hips
            if avg[23].get('calibrated', False):
                self.initial_positions['left_hip'] = {'x': avg[23]['x'], 'y': avg[23]['y']}
            if avg[24].get('calibrated', False):
                self.initial_positions['right_hip'] = {'x': avg[24]['x'], 'y': avg[24]['y']}
            
                # Knees
            if avg[25].get('calibrated', False):
                self.initial_positions['left_knee'] = {'x': avg[25]['x'], 'y': avg[25]['y']}
            if avg[26].get('calibrated', False):
                self.initial_positions['right_knee'] = {'x': avg[26]['x'], 'y': avg[26]['y']}
            
                # Ankles
            if avg[27].get('calibrated', False):
                self.initial_positions['left_ankle'] = {'x': avg[27]['x'], 'y': avg[27]['y']}
            if avg[28].get('calibrated', False):
                self.initial_positions['right_ankle'] = {'x': avg[28]['x'], 'y': avg[28]['y']}
            
            # Spine center (only if all 4 landmarks are calibrated)
            if (avg[11].get('calibrated', False) and avg[12].get('calibrated', False) and 
                avg[23].get('calibrated', False) and avg[24].get('calibrated', False)):
                self.initial_positions['spine_center'] = {
                    'x': (avg[11]['x'] + avg[12]['x'] + avg[23]['x'] + avg[24]['x']) / 4,
                    'y': (avg[11]['y'] + avg[12]['y'] + avg[23]['y'] + avg[24]['y']) / 4,
            }
            
            # Log calibration status
            calibrated_count = sum(1 for lm in avg if lm.get('calibrated', False))
            print(f"📊 Calibration complete: {calibrated_count}/33 landmarks calibrated")
            
            self.calibrated = True
            return (True, False)  # Completed, not timed out
        
        return (False, False)  # Not completed, not timed out
    
    def _get_distance(self, p1, p2):
        """Calculate distance between two points."""
        return np.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
    
    def _check_drift(self, current, initial, tolerance, label):
        """Check if a joint has drifted from its initial position.
        Returns None if initial position is not available (landmark not calibrated).
        """
        if initial is None:
            return None  # Landmark not calibrated, skip drift check
        drift = self._get_distance(current, initial)
        if drift > tolerance:
            severity = min(50, (drift / tolerance) * 30)
            return (label, max(40, 100 - severity))
        return None
    
    def check_form(self, landmarks: list) -> dict:
        """Check form and return score + issues with regional breakdown."""
        if not self.calibrated:
            return {
                'score': 100, 
                'issues': [],
                'regional_scores': {
                    'arms': 100,
                    'legs': 100,
                    'core': 100,
                    'head': 100
                }
            }
        
        issues = []
        scores = []
        
        # Regional scores and issues
        arms_issues = []
        arms_scores = []
        legs_issues = []
        legs_scores = []
        core_issues = []
        core_scores = []
        head_issues = []
        head_scores = []
        
        # Convert to easier format
        lm = {i: {'x': landmarks[i]['x'], 'y': landmarks[i]['y']} for i in range(33)}
        init = self.initial_positions
        
        # === BICEP CURLS ===
        if self.exercise == 'bicep_curls':
            drift_tolerance = self.shoulder_width * 0.15
            
            # --- ARMS REGION ---
            # 1. Upper arm angle from vertical (should stay close to vertical ~0-20°)
            left_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'left_upper_arm')
            right_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'right_upper_arm')
            
            if left_upper_arm_angle > 30:
                arms_issues.append(f'Sol üst kol çok açık ({left_upper_arm_angle:.0f}°)')
                arms_scores.append(max(50, 100 - left_upper_arm_angle))
            
            if right_upper_arm_angle > 30:
                arms_issues.append(f'Sağ üst kol çok açık ({right_upper_arm_angle:.0f}°)')
                arms_scores.append(max(50, 100 - right_upper_arm_angle))
            
            # 2. Elbow angle (between upper arm and forearm)
            left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
            right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
            
            # 3. Elbow drift from initial position
            result = self._check_drift(lm[13], init.get('left_elbow'), drift_tolerance, 'Sol dirsek oynuyor')
            if result: 
                arms_issues.append(result[0])
                arms_scores.append(result[1])
            
            result = self._check_drift(lm[14], init.get('right_elbow'), drift_tolerance, 'Sağ dirsek oynuyor')
            if result: 
                arms_issues.append(result[0])
                arms_scores.append(result[1])
            
            # 4. Shoulder rise (arms region - affects arm form)
            if self.torso_height is not None:
                rise_tolerance = self.torso_height * 0.08
                left_rise = 0.0  # Initialize to avoid UnboundLocalError
                right_rise = 0.0  # Initialize to avoid UnboundLocalError
                
                left_shoulder_init = init.get('left_shoulder')
                if left_shoulder_init:
                    left_rise = left_shoulder_init['y'] - lm[11]['y']
            if left_rise > rise_tolerance:
                arms_issues.append('Sol omuz kalkıyor')
                arms_scores.append(max(50, 100 - (left_rise / rise_tolerance) * 25))
            
                right_shoulder_init = init.get('right_shoulder')
                if right_shoulder_init:
                    right_rise = right_shoulder_init['y'] - lm[12]['y']
            if right_rise > rise_tolerance:
                arms_issues.append('Sağ omuz kalkıyor')
                arms_scores.append(max(50, 100 - (right_rise / rise_tolerance) * 25))
            
            # 5. Elbow above shoulder (critical - means arm is raised, not curling)
            if lm[13]['y'] < lm[11]['y'] - 0.03:
                arms_issues.append('Sol dirsek omuzun üstünde!')
                arms_scores.append(20)
            if lm[14]['y'] < lm[12]['y'] - 0.03:
                arms_issues.append('Sağ dirsek omuzun üstünde!')
                arms_scores.append(20)
            
            # --- CORE REGION ---
            # 6. Torso stability (shoulders line should stay level)
            shoulders_angle = get_bone_angle_from_horizontal(landmarks, 'shoulders')
            if shoulders_angle > 15:
                core_issues.append('Omuzlar eğik - düz dur')
                core_scores.append(max(60, 100 - shoulders_angle * 2))
            
            # Torso lean check
            left_torso_angle = get_bone_angle_from_vertical(landmarks, 'left_torso')
            right_torso_angle = get_bone_angle_from_vertical(landmarks, 'right_torso')
            avg_torso_angle = (left_torso_angle + right_torso_angle) / 2
            if avg_torso_angle > 20:
                core_issues.append(f'Gövde eğiliyor ({avg_torso_angle:.0f}°)')
                core_scores.append(max(55, 100 - avg_torso_angle * 2))
            
            # Hip stability
            spine_center = init.get('spine_center')
            hip_shift = 0.0  # Initialize to avoid UnboundLocalError
            if spine_center is not None and self.hip_width is not None:
                hip_shift = abs((lm[23]['x'] + lm[24]['x']) / 2 - spine_center['x'])
            if hip_shift > self.hip_width * 0.1:
                core_issues.append('Kalça kayıyor')
                core_scores.append(max(60, 100 - hip_shift / self.hip_width * 100))
            
            # --- HEAD REGION ---
            # Head position check (should be neutral)
            head_y = lm[0]['y']
            shoulder_y = (lm[11]['y'] + lm[12]['y']) / 2
            if head_y < shoulder_y - 0.15:
                head_issues.append('Kafan çok öne eğik')
                head_scores.append(70)
            elif head_y > shoulder_y + 0.1:
                head_issues.append('Kafan çok geride')
                head_scores.append(75)
            
            # --- LEGS REGION ---
            # For bicep curls, legs should be stable
            left_knee_angle = get_angle_between_bones(landmarks, 'left_thigh', 'left_shin')
            right_knee_angle = get_angle_between_bones(landmarks, 'right_thigh', 'right_shin')
            # Knees should be slightly bent but stable
            if left_knee_angle < 150 or right_knee_angle < 150:
                # Knees too bent - might be compensating
                if abs(left_knee_angle - right_knee_angle) > 15:
                    legs_issues.append('Bacaklar asimetrik')
                    legs_scores.append(75)
            
            # Combine all regional issues and scores
            issues.extend(arms_issues)
            issues.extend(legs_issues)
            issues.extend(core_issues)
            issues.extend(head_issues)
            scores.extend(arms_scores)
            scores.extend(legs_scores)
            scores.extend(core_scores)
            scores.extend(head_scores)
        
        # === SQUATS ===
        elif self.exercise == 'squats':
            # --- LEGS REGION ---
            # 1. Thigh angle from horizontal (0° = parallel to ground = good depth)
            left_thigh_angle = get_bone_angle_from_horizontal(landmarks, 'left_thigh')
            right_thigh_angle = get_bone_angle_from_horizontal(landmarks, 'right_thigh')
            avg_thigh_angle = (left_thigh_angle + right_thigh_angle) / 2
            
            # 2. Shin angle from vertical (should stay relatively vertical)
            left_shin_angle = get_bone_angle_from_vertical(landmarks, 'left_shin')
            right_shin_angle = get_bone_angle_from_vertical(landmarks, 'right_shin')
            
            if left_shin_angle > 35:
                legs_issues.append(f'Sol bacak çok eğik ({left_shin_angle:.0f}°)')
                legs_scores.append(max(50, 100 - left_shin_angle))
            
            if right_shin_angle > 35:
                legs_issues.append(f'Sağ bacak çok eğik ({right_shin_angle:.0f}°)')
                legs_scores.append(max(50, 100 - right_shin_angle))
            
            # 3. Knee tracking (knees should be over toes, not caving in)
            knee_width = abs(lm[25]['x'] - lm[26]['x'])
            ankle_width = abs(lm[27]['x'] - lm[28]['x'])
            
            if knee_width < ankle_width * 0.8:
                legs_issues.append('Dizler içe çöküyor!')
                legs_scores.append(40)
            
            # Knee angles
            left_knee_angle = get_angle_between_bones(landmarks, 'left_thigh', 'left_shin')
            right_knee_angle = get_angle_between_bones(landmarks, 'right_thigh', 'right_shin')
            if abs(left_knee_angle - right_knee_angle) > 15:
                legs_issues.append('Dizler asimetrik')
                legs_scores.append(65)
            
            # --- CORE REGION ---
            # 3. Torso angle from vertical (should stay upright-ish)
            left_torso_angle = get_bone_angle_from_vertical(landmarks, 'left_torso')
            right_torso_angle = get_bone_angle_from_vertical(landmarks, 'right_torso')
            avg_torso_angle = (left_torso_angle + right_torso_angle) / 2
            
            if avg_torso_angle > 45:
                core_issues.append(f'Gövde çok öne eğiliyor ({avg_torso_angle:.0f}°)')
                core_scores.append(max(40, 100 - avg_torso_angle))
            
            # 5. Hip symmetry
            hip_shift = abs((lm[23]['x'] + lm[24]['x']) / 2 - init['spine_center']['x'])
            if hip_shift > self.hip_width * 0.15:
                core_issues.append('Kalça kayıyor')
                core_scores.append(max(50, 100 - hip_shift / self.hip_width * 100))
            
            # 6. Shoulders level
            shoulders_angle = get_bone_angle_from_horizontal(landmarks, 'shoulders')
            if shoulders_angle > 10:
                core_issues.append('Omuzlar eğik')
                core_scores.append(max(70, 100 - shoulders_angle * 2))
            
            # --- HEAD REGION ---
            head_y = lm[0]['y']
            shoulder_y = (lm[11]['y'] + lm[12]['y']) / 2
            if head_y < shoulder_y - 0.2:
                head_issues.append('Kafan çok öne eğik')
                head_scores.append(60)
            
            # --- ARMS REGION ---
            # Arms should be relatively stable (less critical for squats)
            left_arm_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
            right_arm_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
            if abs(left_arm_angle - right_arm_angle) > 20:
                arms_issues.append('Kollar asimetrik')
                arms_scores.append(75)
            
            # Combine all regional issues and scores
            issues.extend(arms_issues)
            issues.extend(legs_issues)
            issues.extend(core_issues)
            issues.extend(head_issues)
            scores.extend(arms_scores)
            scores.extend(legs_scores)
            scores.extend(core_scores)
            scores.extend(head_scores)
        
        # === LATERAL SHOULDER RAISES ===
        elif self.exercise == 'lateral_shoulder_raises':
            # --- ARMS REGION ---
            # 1. Upper arm angle from vertical (should rise to ~90° at top)
            left_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'left_upper_arm')
            right_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'right_upper_arm')
            
            # 2. Arm symmetry
            arm_asymmetry = abs(left_upper_arm_angle - right_upper_arm_angle)
            if arm_asymmetry > 15:
                arms_issues.append(f'Kollar asimetrik ({arm_asymmetry:.0f}° fark)')
                arms_scores.append(max(55, 100 - arm_asymmetry * 2))
            
            # 3. Elbow bend (arms should be slightly bent, not locked)
            left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
            right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
            
            # Elbows should have slight bend (150-170°)
            if left_elbow_angle > 175:
                arms_issues.append('Sol dirsek kilitli - hafifçe bük')
                arms_scores.append(70)
            if left_elbow_angle < 140:
                arms_issues.append('Sol dirsek çok bükülü')
                arms_scores.append(65)
            
            if right_elbow_angle > 175:
                arms_issues.append('Sağ dirsek kilitli - hafifçe bük')
                arms_scores.append(70)
            if right_elbow_angle < 140:
                arms_issues.append('Sağ dirsek çok bükülü')
                arms_scores.append(65)
            
            # 6. Wrist position (wrists shouldn't be higher than elbows)
            if lm[15]['y'] < lm[13]['y'] - 0.05:  # Left wrist above elbow
                arms_issues.append('Sol bilek dirseğin üstünde')
                arms_scores.append(55)
            if lm[16]['y'] < lm[14]['y'] - 0.05:  # Right wrist above elbow
                arms_issues.append('Sağ bilek dirseğin üstünde')
                arms_scores.append(55)
            
            # --- CORE REGION ---
            # 4. Shoulder shrug (shoulders shouldn't rise toward ears)
            if self.torso_height is not None:
                left_rise = init['left_shoulder']['y'] - lm[11]['y']
                right_rise = init['right_shoulder']['y'] - lm[12]['y']
                rise_tolerance = self.torso_height * 0.08
            
            if left_rise > rise_tolerance or right_rise > rise_tolerance:
                core_issues.append("Omuzlar kalkıyor - omuzları aşağıda tut")
                core_scores.append(50)
            
            # 5. Torso stability (shouldn't lean to compensate)
            left_torso_angle = get_bone_angle_from_vertical(landmarks, 'left_torso')
            right_torso_angle = get_bone_angle_from_vertical(landmarks, 'right_torso')
            
            if abs(left_torso_angle - right_torso_angle) > 10:
                core_issues.append('Gövde yana kayıyor')
                core_scores.append(60)
            
            # Shoulders level
            shoulders_angle = get_bone_angle_from_horizontal(landmarks, 'shoulders')
            if shoulders_angle > 10:
                core_issues.append('Omuzlar eşit seviyede değil')
                core_scores.append(max(70, 100 - shoulders_angle * 2))
            
            # Hip stability
            hip_shift = abs((lm[23]['x'] + lm[24]['x']) / 2 - init['spine_center']['x'])
            if hip_shift > self.hip_width * 0.1:
                core_issues.append('Kalça kayıyor')
                core_scores.append(max(60, 100 - hip_shift / self.hip_width * 100))
            
            # --- HEAD REGION ---
            head_y = lm[0]['y']
            shoulder_y = (lm[11]['y'] + lm[12]['y']) / 2
            if head_y < shoulder_y - 0.15:
                head_issues.append('Kafan çok öne eğik')
                head_scores.append(70)
            
            # Head alignment
            head_x = lm[0]['x']
            shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
            if abs(head_x - shoulder_center_x) > 0.08:
                head_issues.append('Kafa merkezde değil')
                head_scores.append(75)
            
            # --- LEGS REGION ---
            # Legs should be stable (less critical for lateral raises)
            left_knee_angle = get_angle_between_bones(landmarks, 'left_thigh', 'left_shin')
            right_knee_angle = get_angle_between_bones(landmarks, 'right_thigh', 'right_shin')
            if abs(left_knee_angle - right_knee_angle) > 20:
                legs_issues.append('Bacaklar asimetrik')
                legs_scores.append(80)
            
            # Combine all regional issues and scores
            issues.extend(arms_issues)
            issues.extend(legs_issues)
            issues.extend(core_issues)
            issues.extend(head_issues)
            scores.extend(arms_scores)
            scores.extend(legs_scores)
            scores.extend(core_scores)
            scores.extend(head_scores)
        
        # === TRICEP EXTENSIONS ===
        elif self.exercise == 'triceps_pushdown':
            # --- ARMS REGION ---
            # 1. Upper arm angle (should be close to vertical, pointing up)
            left_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'left_upper_arm')
            right_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'right_upper_arm')
            
            # Upper arm should be near vertical (close to ears)
            if left_upper_arm_angle > 25:
                arms_issues.append(f'Sol üst kol dikey olmalı ({left_upper_arm_angle:.0f}°)')
                arms_scores.append(max(50, 100 - left_upper_arm_angle * 2))
            
            if right_upper_arm_angle > 25:
                arms_issues.append(f'Sağ üst kol dikey olmalı ({right_upper_arm_angle:.0f}°)')
                arms_scores.append(max(50, 100 - right_upper_arm_angle * 2))
            
            # Arm symmetry
            arm_asymmetry = abs(left_upper_arm_angle - right_upper_arm_angle)
            if arm_asymmetry > 15:
                arms_issues.append(f'Kollar asimetrik ({arm_asymmetry:.0f}° fark)')
                arms_scores.append(max(55, 100 - arm_asymmetry * 2))
            
            # 2. Elbow angle (the key movement)
            left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
            right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
            
            # Elbow symmetry
            if abs(left_elbow_angle - right_elbow_angle) > 15:
                arms_issues.append('Dirsekler asimetrik')
                arms_scores.append(65)
            
            # 3. Upper arm stability (shouldn't drift during movement)
            left_drift = self._get_distance(lm[13], init['left_elbow'])
            right_drift = self._get_distance(lm[14], init['right_elbow'])
            drift_tolerance = self.shoulder_width * 0.15
            
            if left_drift > drift_tolerance:
                arms_issues.append('Sol dirsek kayıyor - sabit tut')
                arms_scores.append(max(45, 100 - (left_drift / drift_tolerance) * 40))
            
            if right_drift > drift_tolerance:
                arms_issues.append('Sağ dirsek kayıyor - sabit tut')
                arms_scores.append(max(45, 100 - (right_drift / drift_tolerance) * 40))
            
            # 4. Elbow flare (elbows should stay close to head)
            elbow_width = abs(lm[13]['x'] - lm[14]['x'])
            head_width = self.shoulder_width * 0.3  # Approximate head width
            
            if elbow_width > head_width * 2.5:
                arms_issues.append('Dirsekler açılıyor - kafaya yakın tut')
                arms_scores.append(55)
            
            # --- CORE REGION ---
            # 5. Torso stability
            torso_angle = get_bone_angle_from_vertical(landmarks, 'left_torso')
            if torso_angle > 15:
                core_issues.append('Gövde sallanıyor - sabit dur')
                core_scores.append(max(60, 100 - torso_angle * 2))
            
            # Shoulders level
            shoulders_angle = get_bone_angle_from_horizontal(landmarks, 'shoulders')
            if shoulders_angle > 10:
                core_issues.append('Omuzlar eşit seviyede değil')
                core_scores.append(max(70, 100 - shoulders_angle * 2))
            
            # Hip stability
            hip_shift = abs((lm[23]['x'] + lm[24]['x']) / 2 - init['spine_center']['x'])
            if hip_shift > self.hip_width * 0.1:
                core_issues.append('Kalça kayıyor')
                core_scores.append(max(60, 100 - hip_shift / self.hip_width * 100))
            
            # --- HEAD REGION ---
            head_y = lm[0]['y']
            shoulder_y = (lm[11]['y'] + lm[12]['y']) / 2
            if head_y < shoulder_y - 0.2:
                head_issues.append('Kafan çok öne eğik')
                head_scores.append(60)
            
            # Head alignment
            head_x = lm[0]['x']
            shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
            if abs(head_x - shoulder_center_x) > 0.08:
                head_issues.append('Kafa merkezde değil')
                head_scores.append(75)
            
            # --- LEGS REGION ---
            # Legs should be stable (less critical for tricep extensions)
            left_knee_angle = get_angle_between_bones(landmarks, 'left_thigh', 'left_shin')
            right_knee_angle = get_angle_between_bones(landmarks, 'right_thigh', 'right_shin')
            if abs(left_knee_angle - right_knee_angle) > 20:
                legs_issues.append('Bacaklar asimetrik')
                legs_scores.append(80)
            
            # Combine all regional issues and scores
            issues.extend(arms_issues)
            issues.extend(legs_issues)
            issues.extend(core_issues)
            issues.extend(head_issues)
            scores.extend(arms_scores)
            scores.extend(legs_scores)
            scores.extend(core_scores)
            scores.extend(head_scores)
        
        # === DUMBBELL ROWS ===
        elif self.exercise == 'dumbbell_rows':
            # --- ARMS REGION ---
            # 3. Elbow path (should pull toward hip, not out to side)
            # At top of row, elbow should be behind the body line
            left_elbow_behind = lm[13]['y'] > lm[11]['y']  # Elbow behind shoulder
            right_elbow_behind = lm[14]['y'] > lm[12]['y']
            
            # 4. Upper arm angle at top of pull
            left_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'left_upper_arm')
            right_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'right_upper_arm')
            
            # Arm symmetry
            arm_asymmetry = abs(left_upper_arm_angle - right_upper_arm_angle)
            if arm_asymmetry > 20:
                arms_issues.append(f'Kollar asimetrik ({arm_asymmetry:.0f}° fark)')
                arms_scores.append(max(55, 100 - arm_asymmetry * 2))
            
            # 5. Arm path (elbow should stay close to body)
            left_elbow_x = lm[13]['x']
            left_hip_x = lm[23]['x']
            left_shoulder_x = lm[11]['x']
            right_elbow_x = lm[14]['x']
            right_hip_x = lm[24]['x']
            right_shoulder_x = lm[12]['x']
            
            # Elbow should be between hip and shoulder (not flared out)
            if left_elbow_x < min(left_hip_x, left_shoulder_x) - 0.1:
                arms_issues.append('Sol dirsek çok açık - vücuda yakın çek')
                arms_scores.append(55)
            if right_elbow_x < min(right_hip_x, right_shoulder_x) - 0.1:
                arms_issues.append('Sağ dirsek çok açık - vücuda yakın çek')
                arms_scores.append(55)
            
            # Elbow angles
            left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
            right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
            if abs(left_elbow_angle - right_elbow_angle) > 15:
                arms_issues.append('Dirsekler asimetrik')
                arms_scores.append(65)
            
            # --- CORE REGION ---
            # 1. Torso angle (should be bent forward ~45°)
            left_torso_angle = get_bone_angle_from_vertical(landmarks, 'left_torso')
            right_torso_angle = get_bone_angle_from_vertical(landmarks, 'right_torso')
            avg_torso_angle = (left_torso_angle + right_torso_angle) / 2
            
            if avg_torso_angle < 30:
                core_issues.append('Daha fazla öne eğil')
                core_scores.append(65)
            elif avg_torso_angle > 60:
                core_issues.append('Çok eğik - biraz kalk')
                core_scores.append(60)
            
            # 2. Back straight (shoulders and hips should form a line)
            shoulders_angle = get_bone_angle_from_horizontal(landmarks, 'shoulders')
            if shoulders_angle > 15:
                core_issues.append('Omuzlar düz değil - sırtı düz tut')
                core_scores.append(55)
            
            # 6. Shoulder rotation (shouldn't twist)
            if self.torso_height is not None:
                shoulder_y_diff = abs(lm[11]['y'] - lm[12]['y'])
            if shoulder_y_diff > self.torso_height * 0.12:
                core_issues.append('Omuzlar dönüyor - sabit tut')
                core_scores.append(max(50, 100 - shoulder_y_diff / self.torso_height * 150))
            
            # Hip stability
            hip_shift = abs((lm[23]['x'] + lm[24]['x']) / 2 - init['spine_center']['x'])
            if hip_shift > self.hip_width * 0.1:
                core_issues.append('Kalça kayıyor')
                core_scores.append(max(60, 100 - hip_shift / self.hip_width * 100))
            
            # --- HEAD REGION ---
            # 7. Head position (neutral spine)
            head_y = lm[0]['y']
            shoulder_y = (lm[11]['y'] + lm[12]['y']) / 2
            if head_y < shoulder_y - 0.15:
                head_issues.append("Kafanı düşürme")
                head_scores.append(65)
            elif head_y > shoulder_y + 0.1:
                head_issues.append('Kafan çok yukarıda')
                head_scores.append(70)
            
            # Head alignment
            head_x = lm[0]['x']
            shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
            if abs(head_x - shoulder_center_x) > 0.1:
                head_issues.append('Kafa merkezde değil')
                head_scores.append(75)
            
            # --- LEGS REGION ---
            # Legs should be stable (less critical for rows)
            left_knee_angle = get_angle_between_bones(landmarks, 'left_thigh', 'left_shin')
            right_knee_angle = get_angle_between_bones(landmarks, 'right_thigh', 'right_shin')
            if abs(left_knee_angle - right_knee_angle) > 20:
                legs_issues.append('Bacaklar asimetrik')
                legs_scores.append(80)
            
            # Combine all regional issues and scores
            issues.extend(arms_issues)
            issues.extend(legs_issues)
            issues.extend(core_issues)
            issues.extend(head_issues)
            scores.extend(arms_scores)
            scores.extend(legs_scores)
            scores.extend(core_scores)
            scores.extend(head_scores)
        
        # === SHOULDER PRESS ===
        elif self.exercise == 'dumbbell_shoulder_press':
            # --- ARMS REGION ---
            # 1. Upper arm angles (should be symmetric and moving toward vertical)
            left_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'left_upper_arm')
            right_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'right_upper_arm')
            
            # Symmetry check
            arm_asymmetry = abs(left_upper_arm_angle - right_upper_arm_angle)
            if arm_asymmetry > 15:
                arms_issues.append(f'Kollar asimetrik ({arm_asymmetry:.0f}° fark)')
                arms_scores.append(max(55, 100 - arm_asymmetry * 2))
            
            # 2. Elbow angles (should extend as pressing up)
            left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
            right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
            
            # Elbow symmetry
            if abs(left_elbow_angle - right_elbow_angle) > 15:
                arms_issues.append('Dirsekler asimetrik')
                arms_scores.append(65)
            
            # 3. Wrist over elbow (at bottom position, wrist should be above elbow)
            left_wrist_y = lm[15]['y']
            left_elbow_y = lm[13]['y']
            right_wrist_y = lm[16]['y']
            right_elbow_y = lm[14]['y']
            
            # At start position, wrists should be at or above shoulder level
            if left_wrist_y > lm[11]['y'] + 0.1:
                arms_issues.append('Sol el çok alçak - omuz seviyesine getir')
                arms_scores.append(60)
            if right_wrist_y > lm[12]['y'] + 0.1:
                arms_issues.append('Sağ el çok alçak - omuz seviyesine getir')
                arms_scores.append(60)
            
            # 6. Elbow position (shouldn't flare too wide)
            elbow_width = abs(lm[13]['x'] - lm[14]['x'])
            if elbow_width > self.shoulder_width * 2:
                arms_issues.append('Dirsekler çok açık')
                arms_scores.append(60)
            
            # 7. Lockout check (at top, elbows should be nearly straight but not locked)
            if left_elbow_angle > 175 and right_elbow_angle > 175:
                # Full lockout - okay but check for hyperextension
                if left_upper_arm_angle > 15:
                    arms_issues.append('Kolları tamamen yukarı uzat')
                    arms_scores.append(70)
            
            # Wrist stability
            if abs(left_wrist_y - right_wrist_y) > 0.1:
                arms_issues.append('Bilekler eşit seviyede değil')
                arms_scores.append(70)
            
            # --- CORE REGION ---
            # 4. Torso stability (shouldn't lean back)
            left_torso_angle = get_bone_angle_from_vertical(landmarks, 'left_torso')
            right_torso_angle = get_bone_angle_from_vertical(landmarks, 'right_torso')
            avg_torso_angle = (left_torso_angle + right_torso_angle) / 2
            
            if avg_torso_angle > 15:
                core_issues.append('Gövde geriye eğiliyor - dik dur')
                core_scores.append(max(45, 100 - avg_torso_angle * 3))
            
            # 5. Shoulders level
            shoulders_angle = get_bone_angle_from_horizontal(landmarks, 'shoulders')
            if shoulders_angle > 10:
                core_issues.append('Omuzlar eşit seviyede değil')
                core_scores.append(max(65, 100 - shoulders_angle * 3))
            
            # Hip stability
            hip_shift = abs((lm[23]['x'] + lm[24]['x']) / 2 - init['spine_center']['x'])
            if hip_shift > self.hip_width * 0.1:
                core_issues.append('Kalça kayıyor')
                core_scores.append(max(60, 100 - hip_shift / self.hip_width * 100))
            
            # --- HEAD REGION ---
            head_y = lm[0]['y']
            shoulder_y = (lm[11]['y'] + lm[12]['y']) / 2
            if head_y < shoulder_y - 0.15:
                head_issues.append('Kafan çok öne eğik')
                head_scores.append(70)
            elif head_y > shoulder_y + 0.1:
                head_issues.append('Kafan çok geride')
                head_scores.append(75)
            
            # Head alignment
            head_x = lm[0]['x']
            shoulder_center_x = (lm[11]['x'] + lm[12]['x']) / 2
            if abs(head_x - shoulder_center_x) > 0.08:
                head_issues.append('Kafa merkezde değil')
                head_scores.append(75)
            
            # --- LEGS REGION ---
            # Legs should be stable (less critical for shoulder press)
            left_knee_angle = get_angle_between_bones(landmarks, 'left_thigh', 'left_shin')
            right_knee_angle = get_angle_between_bones(landmarks, 'right_thigh', 'right_shin')
            if abs(left_knee_angle - right_knee_angle) > 20:
                legs_issues.append('Bacaklar asimetrik')
                legs_scores.append(80)
            
            # Knee stability (shouldn't lock or bend too much)
            if left_knee_angle > 175 or right_knee_angle > 175:
                legs_issues.append('Dizler kilitli - hafifçe bük')
                legs_scores.append(75)
            
            # Combine all regional issues and scores
            issues.extend(arms_issues)
            issues.extend(legs_issues)
            issues.extend(core_issues)
            issues.extend(head_issues)
            scores.extend(arms_scores)
            scores.extend(legs_scores)
            scores.extend(core_scores)
            scores.extend(head_scores)
        
        # Calculate regional scores
        arms_score = sum(arms_scores) / len(arms_scores) if arms_scores else 100
        legs_score = sum(legs_scores) / len(legs_scores) if legs_scores else 100
        core_score = sum(core_scores) / len(core_scores) if core_scores else 100
        head_score = sum(head_scores) / len(head_scores) if head_scores else 100
        
        # Calculate final score (weighted average based on exercise type)
        if self.exercise in ['bicep_curls', 'lateral_shoulder_raises', 'triceps_pushdown', 'dumbbell_shoulder_press']:
            # Upper body exercises: arms 50%, core 30%, head 10%, legs 10%
            final_score = (arms_score * 0.5 + core_score * 0.3 + head_score * 0.1 + legs_score * 0.1)
        elif self.exercise == 'squats':
            # Lower body exercises: legs 50%, core 40%, arms 5%, head 5%
            final_score = (legs_score * 0.5 + core_score * 0.4 + arms_score * 0.05 + head_score * 0.05)
        elif self.exercise == 'dumbbell_rows':
            # Back exercise: core 45%, arms 40%, head 10%, legs 5%
            final_score = (core_score * 0.45 + arms_score * 0.4 + head_score * 0.1 + legs_score * 0.05)
        else:
            # Default: equal weight
            final_score = (arms_score + legs_score + core_score + head_score) / 4
        
        # Penalty for critical issues
            if any(s <= 30 for s in scores):
                final_score = min(final_score, 40)
        
        # Ensure minimum score when no issues
        if not scores:
            final_score = 88
            arms_score = 88
            legs_score = 88
            core_score = 88
            head_score = 88
        
        return {
            'score': round(final_score, 1),
            'issues': issues,
            'regional_scores': {
                'arms': round(arms_score, 1),
                'legs': round(legs_score, 1),
                'core': round(core_score, 1),
                'head': round(head_score, 1)
            },
            'regional_issues': {
                'arms': arms_issues,
                'legs': legs_issues,
                'core': core_issues,
                'head': head_issues
            }
        }


class RepCounter:
    """Counts exercise repetitions with form validation."""
    
    # Ultra-Strict minimum form score to count as valid rep
    MIN_FORM_SCORE = 70
    
    # Angle requirements per exercise (min_angle, max_angle, required_range_percent)
    ANGLE_REQUIREMENTS = {
        'bicep_curls': {'min': 40, 'max': 160, 'range_pct': 0.7},
        'squats': {'min': 75, 'max': 165, 'range_pct': 0.6},
        'lateral_shoulder_raises': {'min': 20, 'max': 80, 'range_pct': 0.7},
        'triceps_pushdown': {'min': 50, 'max': 160, 'range_pct': 0.6},
        'dumbbell_rows': {'min': 50, 'max': 155, 'range_pct': 0.6},
        'dumbbell_shoulder_press': {'min': 80, 'max': 165, 'range_pct': 0.6},
    }
    
    def __init__(self, exercise: str):
        self.exercise = exercise
        self.config = EXERCISE_CONFIG.get(exercise, {})
        self.phase = 'down'
        self.count = 0           # Total reps (valid + invalid)
        self.valid_count = 0     # Only valid reps with good form
        self.form_scores = []
        self.min_angle_reached = 180
        self.max_angle_reached = 0
        self.rep_feedback = ""
    
    def validate_rep(self, avg_score: float) -> tuple:
        """Validate if rep meets quality standards. Returns (is_valid, feedback)."""
        requirements = self.ANGLE_REQUIREMENTS.get(self.exercise, {'min': 30, 'max': 160, 'range_pct': 0.5})
        
        range_of_motion = self.max_angle_reached - self.min_angle_reached
        required_range = (requirements['max'] - requirements['min']) * requirements['range_pct']
        
        # Check 1: Form score
        if avg_score < self.MIN_FORM_SCORE:
            return False, f"❌ Form düşük ({avg_score:.0f}%) - Rep sayılmadı!"
        
        # Check 2: Range of motion
        if range_of_motion < required_range:
            return False, f"❌ Yetersiz hareket ({range_of_motion:.0f}°) - Tam hareketi yap!"
        
        # Check 3: Min angle reached (contracted position)
        if self.min_angle_reached > requirements['min'] + 15:
            return False, f"❌ Tam bükülmedi ({self.min_angle_reached:.0f}° > {requirements['min']}°)"
        
        # Check 4: Max angle reached (extended position)
        if self.max_angle_reached < requirements['max'] - 15:
            return False, f"❌ Tam açılmadı ({self.max_angle_reached:.0f}° < {requirements['max']}°)"
        
        # Valid rep!
        if avg_score >= 90:
            return True, f"✅ Mükemmel! ({avg_score:.0f}%)"
        elif avg_score >= 75:
            return True, f"✅ İyi rep ({avg_score:.0f}%)"
        else:
            return True, f"⚠️ Geçerli ama formu düzelt ({avg_score:.0f}%)"
    
    def complete_rep(self) -> dict:
        """Complete a rep and return result with validation."""
        avg_score = sum(self.form_scores) / len(self.form_scores) if self.form_scores else 0
        
        is_valid, feedback = self.validate_rep(avg_score)
        
        self.count += 1
        if is_valid:
            self.valid_count += 1
        
        result = {
            'rep': self.count,
            'valid_rep': self.valid_count,
            'form_score': round(avg_score, 1),
            'is_valid': is_valid,
            'feedback': feedback,
            'min_angle': round(self.min_angle_reached, 1),
            'max_angle': round(self.max_angle_reached, 1),
        }
        
        # Reset for next rep
        self.form_scores = []
        self.min_angle_reached = 180
        self.max_angle_reached = 0
        self.rep_feedback = feedback
        
        return result
    
    def update(self, angle: float, form_score: float, landmarks: list = None) -> Optional[dict]:
        """Update with new angle, return rep data if completed."""
        self.form_scores.append(form_score)
        
        # Track min/max angles
        self.min_angle_reached = min(self.min_angle_reached, angle)
        self.max_angle_reached = max(self.max_angle_reached, angle)
        
        up_threshold = self.config.get('rep_threshold', {}).get('up', 60)
        down_threshold = self.config.get('rep_threshold', {}).get('down', 140)
        
        result = None
        
        if self.exercise == 'bicep_curls':
            # SIMPLE ANGLE-BASED DETECTION: Use standard angle thresholds
            # Phase 'down' = arm extended (large angle ~120-180°)
            # Phase 'up' = arm curled (small angle ~30-80°)
            # Rep is complete when going from curled back to extended
            # Note: More flexible thresholds for better detection
            curl_threshold = 80  # Angle below this = curled (was 60, now 80 for easier detection)
            extend_threshold = 120  # Angle above this = extended (was 140, now 120 for easier detection)
            
            if self.phase == 'down' and angle < curl_threshold:  # Going into curl
                if self.count == 0:  # Only log on first transition
                    print(f"🔄 Phase: down → up (angle: {angle:.1f}° < {curl_threshold}°)")
                self.phase = 'up'
            elif self.phase == 'up' and angle > extend_threshold:  # Going back to extended
                print(f"🔄 Phase: up → down (angle: {angle:.1f}° > {extend_threshold}°) - REP COMPLETE!")
                self.phase = 'down'
                result = self.complete_rep()
                if result:
                    print(f"✅ REP #{result.get('rep', 0)} COMPLETED! Valid: {result.get('is_valid', False)}, Score: {result.get('form_score', 0):.1f}%")
        
        elif self.exercise == 'squats':
            # BONE-BASED SQUAT REP COUNTING
            if landmarks:
                # Use thigh angle from horizontal
                left_thigh_angle = get_bone_angle_from_horizontal(landmarks, 'left_thigh')
                right_thigh_angle = get_bone_angle_from_horizontal(landmarks, 'right_thigh')
                thigh_angle = (left_thigh_angle + right_thigh_angle) / 2
                
                # Thigh parallel = ~0-20° from horizontal
                # Standing = ~70-90° from horizontal
                
                # Validate: hips must be visible
                hip_y = (landmarks[23]['y'] + landmarks[24]['y']) / 2
                if hip_y < 0.2:  # Hips not visible
                    return None
                
                if self.phase == 'down' and thigh_angle < 30:  # Deep squat (thigh near parallel)
                    self.phase = 'up'
                elif self.phase == 'up' and thigh_angle > 60:  # Standing up
                    self.phase = 'down'
                    result = self.complete_rep()
        
        elif self.exercise == 'lateral_shoulder_raises':
            # BONE-BASED LATERAL RAISE REP COUNTING
            if landmarks:
                # Use upper arm angle from vertical
                left_arm_angle = get_bone_angle_from_vertical(landmarks, 'left_upper_arm')
                right_arm_angle = get_bone_angle_from_vertical(landmarks, 'right_upper_arm')
                arm_angle = (left_arm_angle + right_arm_angle) / 2
                
                # Down: arms at sides ~0-15°
                # Up: arms horizontal ~80-100°
                
                if self.phase == 'down' and arm_angle > 70:  # Arms raised
                    self.phase = 'up'
                elif self.phase == 'up' and arm_angle < 25:  # Arms down
                    self.phase = 'down'
                    result = self.complete_rep()
        
        elif self.exercise == 'triceps_pushdown':
            # BONE-BASED TRICEP EXTENSION REP COUNTING
            if landmarks:
                # Use elbow angle
                left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
                right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
                elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                
                # Validate: upper arm should be raised (overhead)
                left_upper_arm_angle = get_bone_angle_from_vertical(landmarks, 'left_upper_arm')
                if left_upper_arm_angle > 50:  # Arm not raised
                    return None
                
                # Down: elbow bent ~60°
                # Up: elbow extended ~160°
                
                if self.phase == 'down' and elbow_angle > 150:  # Extended
                    self.phase = 'up'
                elif self.phase == 'up' and elbow_angle < 80:  # Bent
                    self.phase = 'down'
                    result = self.complete_rep()
        
        elif self.exercise == 'dumbbell_rows':
            # BONE-BASED ROW REP COUNTING
            if landmarks:
                # Use elbow angle on rowing arm (typically right)
                right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
                
                # Validate: torso should be bent forward
                torso_angle = get_bone_angle_from_vertical(landmarks, 'right_torso')
                if torso_angle < 20:  # Not bent over enough
                    return None
                
                # Down: arm extended ~160°
                # Up: arm pulled ~70°
                
                if self.phase == 'down' and right_elbow_angle < 90:  # Pulled up
                    self.phase = 'up'
                elif self.phase == 'up' and right_elbow_angle > 150:  # Extended
                    self.phase = 'down'
                    result = self.complete_rep()
        
        elif self.exercise == 'dumbbell_shoulder_press':
            # BONE-BASED SHOULDER PRESS REP COUNTING
            if landmarks:
                # Use elbow angle
                left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
                right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
                elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                
                # Also check wrist height
                left_wrist_y = landmarks[15]['y']
                left_shoulder_y = landmarks[11]['y']
                
                # Down: elbows ~90°, wrists near shoulders
                # Up: elbows extended ~170°, wrists above head
                
                if self.phase == 'down' and elbow_angle > 160:  # Pressed up
                    self.phase = 'up'
                elif self.phase == 'up' and elbow_angle < 100:  # Lowered
                    self.phase = 'down'
                    result = self.complete_rep()
        
        else:
            # Generic fallback using angle thresholds
            if self.phase == 'down' and angle > up_threshold:
                self.phase = 'up'
            elif self.phase == 'up' and angle < down_threshold:
                self.phase = 'down'
                result = self.complete_rep()
        
        return result


# AI Feedback with variety
FEEDBACK_TEMPLATES = [
    "Great job! {detail}",
    "Looking good! {detail}",
    "Nice work! {detail}",
    "Keep it up! {detail}",
    "Excellent! {detail}",
    "{detail} Keep going!",
    "Amazing energy! {detail}",
]

CORRECTION_TEMPLATES = [
    "{issue} - pay attention.",
    "Small fix needed: {issue}",
    "{issue} - stay controlled.",
    "Watch out: {issue}",
]

import random

# EXERCISE FEEDBACK LIBRARY - 72 feedback options (6 exercises x 12 categories)
EXERCISE_FEEDBACK_LIBRARY = {
    'bicep_curls': {
        1: "🎉 Mükemmel biceps curl! Form, hız ve kontrol harika. Devam et!",
        2: "💪 Çok iyi! Dirsekler sabit, hareket kontrollü. İyi gidiyorsun!",
        3: "👍 İyi form, dirseklerin biraz daha sabit kalmalı. Küçük bir iyileştirme yap.",
        4: "✅ İyi gidiyorsun, omuzların daha düşük kalmalı. Gövdeni sabitle.",
        5: "⚠️ Orta seviye, dirsekleri gövdene sabitle. Daha kontrollü hareket et.",
        6: "🔴 Kollarına odaklan: dirsekleri sabit tut, sallama. Gövdeni sabitle.",
        7: "🔴 Gövdeni sabitle, öne eğilme. Dikey dur ve dirsekleri sabit tut.",
        8: "🔴 Kafanı nötr tut, aşağı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: dirsekleri sabitle ve gövdeni düz tut. Yavaşla.",
        10: "🟡 Hareketi tamamla, kolları tam uzat. Tam hareket menzili kullan.",
        11: "🟡 Kontrolü artır, daha yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Dirseklerin omuzun üstüne çıkmasın, daha düşük tut. Yanlış açıda hareket ediyorsun."
    },
    'squats': {
        1: "🎉 Mükemmel squat! Derinlik ve form harika. Mükemmel çalışma!",
        2: "💪 Çok iyi! Dizler ayak parmaklarının üzerinde, gövde düz. İyi gidiyorsun!",
        3: "👍 İyi form, biraz daha derine inebilirsin. Derinliği artır.",
        4: "✅ İyi gidiyorsun, gövdeni daha dik tut. Omurganı düzleştir.",
        5: "⚠️ Orta seviye, dizlerin içe düşmesin. Dizlerini dışarı doğru it.",
        6: "🔴 Bacaklarına odaklan: dizleri dışarı doğru it. İçe çökmesin.",
        7: "🔴 Gövdeni düz tut, öne çok eğilme. Dikey dur, göğsünü kaldır.",
        8: "🔴 İleri bak, kafanı öne eğme. Gözlerin öne baksın.",
        9: "🔴 Birkaç sorun var: diz pozisyonu ve gövde düzgünlüğüne dikkat. Yavaşla.",
        10: "🟡 Daha derine in, kalçalar diz seviyesinin altına gelsin. Derinlik artır.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Dizlerin içe çökmesin! Ayak parmaklarınla hizalı tut, dışarı doğru it."
    },
    'lateral_shoulder_raises': {
        1: "🎉 Mükemmel lateral raise! Omuz kontrolü harika. Devam et!",
        2: "💪 Çok iyi! Kollar omuz hizasında, simetrik. İyi gidiyorsun!",
        3: "👍 İyi form, kolları biraz daha simetrik kaldır. Eşit yüksekliğe getir.",
        4: "✅ İyi gidiyorsun, omuzların yukarı kalkmasın. Omuzları düşük tut.",
        5: "⚠️ Orta seviye, kolları omuz hizasına kadar kaldır. Yeterince yükseğe çık.",
        6: "🔴 Kollarına odaklan: simetrik kaldır, eşit yüksekliğe getir. Asimetri var.",
        7: "🔴 Gövdeni sabitle, sallanma. Dikey dur, core'unu sık.",
        8: "🔴 Kafanı nötr tut, yukarı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: simetrik kaldır ve gövdeni sabitle. Yavaşla.",
        10: "🟡 Kolları omuz hizasına kadar kaldır, daha yukarı çıkar. Tam menzil kullan.",
        11: "🟡 Kontrolü artır, omuzları silkmeyi bırak. Yavaş ve kontrollü hareket et.",
        12: "🔴 Omuzlarını yukarı kaldırma! Sadece kolları kaldır, omuzlar düşük kalsın."
    },
    'triceps_pushdown': {
        1: "🎉 Mükemmel triceps pushdown! Üst kol sabit, form harika. Devam et!",
        2: "💪 Çok iyi! Üst kol sabit, sadece dirsek hareket ediyor. İyi gidiyorsun!",
        3: "👍 İyi form, üst kolunu biraz daha sabit tut. Sallanmayı azalt.",
        4: "✅ İyi gidiyorsun, dirseği tam aç. Tam hareket menzili kullan.",
        5: "⚠️ Orta seviye, üst kolunu sabit tut, sallama. Kontrolü artır.",
        6: "🔴 Kollarına odaklan: üst kol sabit, sadece dirsek hareket etsin. Sallama.",
        7: "🔴 Gövdeni sabitle, öne eğilme. Dikey dur, core'unu sık.",
        8: "🔴 Kafanı nötr tut, aşağı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: üst kol sabitliği ve gövde pozisyonuna dikkat. Yavaşla.",
        10: "🟡 Dirseği tam aç, kolları tam uzat. Tam hareket menzili kullan.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Üst kolunu sabit tut! Sadece ön kol hareket etmeli, üst kol sabit kalmalı."
    },
    'dumbbell_rows': {
        1: "🎉 Mükemmel row! Sırt kasların aktif, form harika. Devam et!",
        2: "💪 Çok iyi! Gövde sabit, kürek kemikleri sıkılıyor. İyi gidiyorsun!",
        3: "👍 İyi form, gövdeni biraz daha sabit tut. Sallanmayı azalt.",
        4: "✅ İyi gidiyorsun, dirseği vücuda daha yakın çek. Daha yakın tut.",
        5: "⚠️ Orta seviye, sırtını düz tut, eğilme. Gövdeni sabitle.",
        6: "🔴 Gövdeni sabitle, sırtını düz tut. Öne çok eğilme, düz kal.",
        7: "🔴 Kollarına odaklan: dirseği vücuda yakın çek. Daha yakın tut.",
        8: "🔴 Kafanı nötr tut, boynunu eğme. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: sırt düzgünlüğü ve dirsek pozisyonuna dikkat. Yavaşla.",
        10: "🟡 Daha geriye çek, kürek kemiklerini sıkıştır. Tam menzil kullan.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Sırtını düz tut, fazla kavisli olmasın! Omurganı nötr tut."
    },
    'dumbbell_shoulder_press': {
        1: "🎉 Mükemmel shoulder press! Core aktif, form harika. Devam et!",
        2: "💪 Çok iyi! Kollar tam yukarı, gövde sabit. İyi gidiyorsun!",
        3: "👍 İyi form, kolları biraz daha tam yukarı it. Tam aç.",
        4: "✅ İyi gidiyorsun, gövdeni daha sabit tut. Core'unu sık.",
        5: "⚠️ Orta seviye, core'unu sık, sırtına yaslanma. Dikey dur.",
        6: "🔴 Kollarına odaklan: tam yukarı it, tam aç. Yeterince yukarı çıkmıyor.",
        7: "🔴 Gövdeni sabitle, core'unu sık. Sallanmayı azalt.",
        8: "🔴 Kafanı nötr tut, yukarı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: core stabilitesi ve kol hareketi düzgünlüğüne dikkat. Yavaşla.",
        10: "🟡 Kolları tam yukarı it, tam aç. Tam hareket menzili kullan.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Arkaya yaslanma! Gövdeni dik tut, core'unu sık. Öne eğilme."
    }
}


def select_feedback_category(
    exercise: str,
    score: float,
    regional_scores: dict,
    regional_issues: dict,
    min_angle: float = None,
    max_angle: float = None,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary'  # 'camera_only', 'imu_only', 'camera_primary'
) -> int:
    """
    Select appropriate feedback category (1-12) based on ML predictions, scores, IMU data, and landmarks.
    Supports Camera-only, IMU-only, and Sensor Fusion modes.
    
    Args:
        exercise: Exercise name
        score: Overall form score (0-100)
        regional_scores: Dict with regional scores {'arms': float, 'legs': float, ...}
        regional_issues: Dict with regional issues {'arms': [str, ...], ...}
        min_angle: Minimum angle during rep
        max_angle: Maximum angle during rep
        ml_prediction: ML model prediction dict (regional scores) - from camera or fusion
        imu_data: IMU data dict (left_wrist, right_wrist, chest)
        landmarks: Raw landmark data (list of 33 landmarks)
        initial_positions: Calibration initial positions
        fusion_mode: 'camera_only', 'imu_only', or 'camera_primary' (sensor fusion)
    
    Returns:
        Feedback category ID (1-12)
    """
    # Use ML prediction if available (preferred - works for all modes)
    if ml_prediction and isinstance(ml_prediction, dict) and 'arms' in ml_prediction:
        score = sum(ml_prediction.values()) / len(ml_prediction)
        regional_scores = ml_prediction
    
    # Calculate range of motion
    rom = None
    if min_angle is not None and max_angle is not None:
        rom = max_angle - min_angle
    
    # Expected ROM ranges per exercise
    expected_rom = {
        'bicep_curls': (90, 120),
        'squats': (70, 100),
        'lateral_shoulder_raises': (50, 80),
        'triceps_pushdown': (80, 120),
        'dumbbell_rows': (80, 110),
        'dumbbell_shoulder_press': (60, 90)
    }
    
    # Category 1: Perfect Form (Score >=95, no issues)
    if score >= 95 and not any(regional_issues.values() if regional_issues else []):
        return 1
    
    # Category 2: Excellent Form (Score 90-94)
    if score >= 90:
        return 2
    
    # Category 3: Good - Minor Issues (Score 85-89)
    if score >= 85:
        return 3
    
    # Category 4: Good - Needs Improvement (Score 80-84)
    if score >= 80:
        return 4
    
    # Category 5: Moderate Form (Score 70-79)
    if score >= 70:
        return 5
    
    # Category 11: Range Too Limited (works for all modes - uses angle data)
    if rom is not None and expected_rom.get(exercise):
        exp_min, exp_max = expected_rom[exercise]
        if rom < exp_min * 0.8:
            return 11
    
    # Category 12: Range Too Wide or specific landmark-based issues
    if rom is not None and expected_rom.get(exercise):
        exp_min, exp_max = expected_rom[exercise]
        if rom > exp_max * 1.2:
            return 12
    
    # ✅ Landmark-based checks (Camera mode or Fusion mode)
    if (fusion_mode in ['camera_only', 'camera_primary']) and landmarks and initial_positions:
        lm = {i: {'x': landmarks[i]['x'], 'y': landmarks[i]['y']} for i in range(min(len(landmarks), 33))}
        
        # Bicep curls - elbow above shoulder check
        if exercise == 'bicep_curls' and 13 < len(landmarks) and 'left_elbow' in initial_positions:
            left_elbow_current = lm.get(13, {})
            left_elbow_init = initial_positions.get('left_elbow', {})
            if left_elbow_current.get('x') and left_elbow_init.get('x'):
                elbow_drift = abs(left_elbow_current['x'] - left_elbow_init['x'])
                if elbow_drift > 0.1:
                    return 6  # Arms issue
        
        # Squats - knee valgus check
        if exercise == 'squats' and len(landmarks) >= 28:
            left_knee_x = lm.get(25, {}).get('x', 0)
            left_ankle_x = lm.get(27, {}).get('x', 0)
            right_knee_x = lm.get(26, {}).get('x', 0)
            right_ankle_x = lm.get(28, {}).get('x', 0)
            
            knee_width = abs(right_knee_x - left_knee_x)
            ankle_width = abs(right_ankle_x - left_ankle_x)
            
            if ankle_width > 0 and knee_width < ankle_width * 0.8:
                return 12  # Knee valgus
    
    # ✅ IMU-based checks (IMU mode or Fusion mode)
    if (fusion_mode in ['imu_only', 'camera_primary']) and imu_data:
        # Check for excessive wrist movement (bicep curls, triceps pushdown)
        if exercise in ['bicep_curls', 'triceps_pushdown']:
            left_wrist = imu_data.get('left_wrist', {})
            right_wrist = imu_data.get('right_wrist', {})
            
            # Check gyroscope magnitude (indicates movement)
            if left_wrist and right_wrist:
                left_gyro = left_wrist.get('gyro', {})
                right_gyro = right_wrist.get('gyro', {})
                
                if left_gyro and right_gyro:
                    left_mag = (left_gyro.get('x', 0)**2 + left_gyro.get('y', 0)**2 + left_gyro.get('z', 0)**2)**0.5
                    right_mag = (right_gyro.get('x', 0)**2 + right_gyro.get('y', 0)**2 + right_gyro.get('z', 0)**2)**0.5
                    
                    # High gyro magnitude indicates excessive movement
                    if left_mag > 500 or right_mag > 500:  # Threshold in deg/s
                        return 6  # Arms issue - too much movement
    
    # Category 10: Multiple Issues (3+ issues across regions)
    total_issues = sum(len(issues) for issues in (regional_issues.values() if regional_issues else []))
    if total_issues >= 3:
        return 10
    
    # Category 6-9: Poor Form - Region-specific (Score <70, find lowest region)
    if score < 70 and regional_scores:
        min_region = min(regional_scores.items(), key=lambda x: x[1])
        region_name = min_region[0]
        
        region_to_category = {
            'arms': 6,
            'legs': 7,
            'core': 8,
            'head': 9
        }
        return region_to_category.get(region_name, 6)
    
    # Default: Category 5 (Moderate)
    return 5


def get_smart_feedback(
    exercise: str,
    score: float,
    regional_scores: dict,
    regional_issues: dict,
    min_angle: float = None,
    max_angle: float = None,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary',
    rep_num: int = 0
) -> str:
    """
    Get smart feedback using ML predictions, IMU data, and landmark analysis.
    Supports Camera-only, IMU-only, and Sensor Fusion modes.
    
    Args:
        exercise: Exercise name
        score: Overall form score
        regional_scores: Regional scores dict
        regional_issues: Regional issues dict
        min_angle: Min angle
        max_angle: Max angle
        ml_prediction: ML model prediction (regional scores)
        imu_data: IMU data (left_wrist, right_wrist, chest)
        landmarks: Raw landmark data
        initial_positions: Calibration initial positions
        fusion_mode: 'camera_only', 'imu_only', or 'camera_primary'
        rep_num: Rep number
    
    Returns:
        Feedback message string
    """
    # Select feedback category based on mode
    category = select_feedback_category(
        exercise, score, regional_scores, regional_issues,
        min_angle, max_angle, ml_prediction, imu_data,
        landmarks=landmarks,
        initial_positions=initial_positions,
        fusion_mode=fusion_mode
    )
    
    # Get feedback from library
    feedback_lib = EXERCISE_FEEDBACK_LIBRARY.get(exercise, {})
    feedback = feedback_lib.get(category, "Formunu iyileştirmeye devam et.")
    
    if rep_num > 0:
        return f"Rep #{rep_num}: {feedback}"
    
    return feedback


def get_rule_based_regional_feedback(
    exercise: str,
    region: str,
    region_score: float,
    region_issues: list,
    rep_num: int,
    min_angle: float = None,
    max_angle: float = None
) -> str:
    """Get rule-based feedback for a specific body region using MediaPipe data."""
    region_names = {
        'arms': 'Kollar',
        'legs': 'Bacaklar',
        'core': 'Gövde',
        'head': 'Kafa'
    }
    
    region_name_tr = region_names.get(region, region)
    
    # If score is high, give positive feedback
    if region_score >= 85:
        if region == 'arms':
            return f"💪 Kollar mükemmel! Form çok iyi."
        elif region == 'legs':
            return f"🦵 Bacaklar mükemmel! Form çok iyi."
        elif region == 'core':
            return f"✅ Gövde mükemmel! Duruş çok iyi."
        elif region == 'head':
            return f"👍 Kafa pozisyonu mükemmel!"
        else:
            return f"{region_name_tr} mükemmel! Skor: %{region_score:.0f}"
    
    # If there are specific issues, provide targeted feedback
    if region_issues:
        # Exercise-specific feedback based on issues
        issue_lower = region_issues[0].lower()
        
        # Arms feedback
        if region == 'arms':
            if 'dirsek' in issue_lower or 'elbow' in issue_lower or 'oynuyor' in issue_lower:
                if 'sol' in issue_lower or 'left' in issue_lower:
                    return "Sol dirseğini gövdene sabitle, daha az oynatmalısın."
                elif 'sağ' in issue_lower or 'right' in issue_lower:
                    return "Sağ dirseğini gövdene sabitle, daha az oynatmalısın."
                else:
                    return "Dirseklerini sabit tutmalısın, gövdene yakın tut."
            elif 'kol' in issue_lower and 'esit' in issue_lower:
                return "Kollarını eşit yüksekliğe getirmelisin, simetrik hareket et."
            elif 'uzat' in issue_lower or 'extend' in issue_lower:
                return "Kollarını daha fazla uzatmalısın, tam hareket menzili kullan."
            elif 'bük' in issue_lower or 'curl' in issue_lower:
                return "Kollarını daha fazla bük, hareket menzilini artır."
            else:
                return f"Kollar: {region_issues[0]}"
        
        # Legs feedback
        elif region == 'legs':
            if 'diz' in issue_lower or 'knee' in issue_lower:
                if 'içe' in issue_lower or 'valgus' in issue_lower:
                    return "Dizlerini ayak parmaklarınla hizalı tut, içe düşmesin."
                elif 'öne' in issue_lower or 'forward' in issue_lower:
                    return "Dizlerini ayak bileklerinin üzerinde tut, çok öne çıkmasın."
                else:
                    return "Diz pozisyonuna dikkat et, doğru açıda tut."
            elif 'duruş' in issue_lower or 'genişlik' in issue_lower:
                return "Bacaklarını omuz genişliğinde tut, daha dengeli dur."
            elif 'derinlik' in issue_lower or 'depth' in issue_lower:
                return "Daha derin inmelisin, tam hareket menzili kullan."
            else:
                return f"Bacaklar: {region_issues[0]}"
        
        # Core feedback
        elif region == 'core':
            if 'gövde' in issue_lower or 'sırt' in issue_lower or 'omurga' in issue_lower:
                if 'düz' in issue_lower or 'straight' in issue_lower:
                    return "Gövdeni düz tut, omurganı nötr pozisyonda tut."
                elif 'kavis' in issue_lower or 'arch' in issue_lower:
                    return "Sırtını düzleştir, fazla kavisli olmasın."
                elif 'eğil' in issue_lower or 'lean' in issue_lower:
                    return "Gövdeni dikey tut, öne veya arkaya eğilme."
                else:
                    return "Gövdeni stabilize et, düz ve dengeli tut."
            elif 'pelvis' in issue_lower or 'kalça' in issue_lower:
                return "Kalça pozisyonunu kontrol et, pelvis nötr olsun."
            else:
                return f"Gövde: {region_issues[0]}"
        
        # Head feedback
        elif region == 'head':
            if 'öne' in issue_lower or 'forward' in issue_lower:
                return "Başını öne eğme, ileri bak."
            elif 'yukarı' in issue_lower or 'up' in issue_lower:
                return "Başını çok yukarı kaldırma, nötr pozisyonda tut."
            elif 'aşağı' in issue_lower or 'down' in issue_lower:
                return "Başını aşağı bakma, öne doğru bak."
            else:
                return f"Kafa: {region_issues[0]}"
    
    # Default feedback based on score range
    if region_score >= 70:
        return f"{region_name_tr} iyi (Skor: %{region_score:.0f}), küçük iyileştirmeler yapabilirsin."
    elif region_score >= 50:
        return f"{region_name_tr} orta (Skor: %{region_score:.0f}), formunu iyileştirmeye odaklan."
    else:
        return f"{region_name_tr} düşük (Skor: %{region_score:.0f}), formunu düzeltmeye öncelik ver."


async def get_regional_ai_feedback(
    exercise: str,
    region: str,
    region_score: float,
    region_issues: list,
    rep_num: int,
    min_angle: float = None,
    max_angle: float = None
) -> str:
    """Get AI feedback for a specific body region. Falls back to rule-based if OpenAI unavailable."""
    # Always use rule-based feedback (faster and more reliable)
    return get_rule_based_regional_feedback(
        exercise, region, region_score, region_issues, rep_num, min_angle, max_angle
    )
    


def get_rule_based_overall_feedback(
    exercise: str,
    rep_num: int,
    score: float,
    issues: list,
    regional_scores: dict = None,
    regional_issues: dict = None,
    min_angle: float = None,
    max_angle: float = None,
    is_valid: bool = True,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary'
) -> str:
    """Get rule-based overall feedback using MediaPipe data, ML predictions, IMU data, and landmarks."""
    if not is_valid:
        if issues:
            return f"Rep #{rep_num}: Geçersiz rep. {issues[0] if issues else 'Form hatası'}."
        return f"Rep #{rep_num}: Geçersiz rep, formunu düzelt."
    
    # Use smart feedback system (includes ML, IMU, and landmark data)
    return get_smart_feedback(
        exercise=exercise,
        score=score,
        regional_scores=regional_scores or {},
        regional_issues=regional_issues or {},
        min_angle=min_angle,
        max_angle=max_angle,
        ml_prediction=ml_prediction,
        imu_data=imu_data,
        landmarks=landmarks,
        initial_positions=initial_positions,
        fusion_mode=fusion_mode,
        rep_num=rep_num
    )


async def get_ai_feedback(exercise: str, rep_data: dict, issues: list, regional_scores: dict = None, regional_issues: dict = None) -> dict:
    """Get technical and specific AI feedback based on rep quality data with regional breakdown.
    Uses OpenAI if available, otherwise falls back to rule-based feedback.
    """
    rep_num = rep_data.get('rep', 0)
    score = rep_data.get('form_score', 0)
    min_angle = rep_data.get('min_angle', 0)
    max_angle = rep_data.get('max_angle', 0)
    is_valid = rep_data.get('is_valid', True)
    
    # Try OpenAI first (if available)
    if openai_client:
        try:
            # Build comprehensive prompt with all available data
            exercise_names = {
                'bicep_curls': 'Biceps Curl',
                'squats': 'Squat',
                'lunges': 'Lunge',
                'pushups': 'Push-up',
                'lateral_shoulder_raises': 'Lateral Shoulder Raise',
                'triceps_pushdown': 'Triceps Extension',
                'dumbbell_rows': 'Dumbbell Row',
                'dumbbell_shoulder_press': 'Shoulder Press'
            }
            ex_name = exercise_names.get(exercise, exercise)
            
            issues_text = ', '.join(issues) if issues else 'None'
            regional_info = ""
            if regional_scores:
                regional_info = f"\nRegional Scores:\n"
                for region, reg_score in regional_scores.items():
                    region_name = {'arms': 'Arms', 'legs': 'Legs', 'core': 'Core/Torso', 'head': 'Head/Neck'}.get(region, region)
                    region_issues_str = ', '.join(regional_issues.get(region, [])) if regional_issues else 'None'
                    regional_info += f"- {region_name}: {reg_score:.1f}% (Issues: {region_issues_str})\n"
            
            angle_info = ""
            if min_angle and max_angle:
                angle_info = f"\nMovement Range: {min_angle:.1f}° to {max_angle:.1f}° (Range: {max_angle - min_angle:.1f}°)"
            
            prompt = f"""You are an expert fitness coach analyzing a {ex_name} exercise rep.

Rep #{rep_num} Analysis:
- Form Score: {score:.1f}%
- Valid Rep: {'Yes' if is_valid else 'No'}
- Detected Issues: {issues_text}
{regional_info}{angle_info}

Provide SHORT, MOTIVATING, and ACTIONABLE feedback in Turkish:
1. Start with a positive note (even if score is low)
2. Mention the most critical issue to fix (if any)
3. Give one specific correction tip
4. End with encouragement

Keep it under 2 sentences. Be friendly and supportive."""

            response = openai_client.chat.completions.create(
                model='gpt-4o-mini',  # Faster and cheaper than gpt-4
                messages=[
                    {'role': 'system', 'content': 'You are a professional fitness coach. Provide concise, actionable feedback in Turkish.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=150,
                temperature=0.7,
            )
            
            overall_feedback = response.choices[0].message.content.strip()
            
            # Get regional feedbacks using rule-based (faster for regions, OpenAI for overall)
            regional_feedbacks = {}
            if regional_scores and regional_issues:
                for region in ['arms', 'legs', 'core', 'head']:
                    region_score = regional_scores.get(region, 100)
                    region_issues_list = regional_issues.get(region, [])
                    regional_feedbacks[region] = get_rule_based_regional_feedback(
                        exercise, region, region_score, region_issues_list,
                        rep_num, min_angle, max_angle
                    )
            
            return {
                'overall': overall_feedback,
                'regional': regional_feedbacks
            }
        except Exception as e:
            print(f"⚠️  OpenAI feedback error: {e}, falling back to rule-based")
            # Fall through to rule-based feedback
    


async def countdown_task(websocket: WebSocket, session_id: int):
    """Handle countdown after calibration: 3, 2, 1, START!"""
    session = sessions.get(session_id)
    if not session:
        print("⚠️  Countdown task: Session not found!")
        return
    
    try:
        # State is already 'countdown' (set before this task was created)
        print("⏳ Countdown starting...")
        
        # Send calibration complete message
        await websocket.send_json({
            'type': 'state',
            'state': 'countdown',
            'message': 'Calibration complete! Get ready...'
        })
        await asyncio.sleep(0.5)  # Brief pause before countdown
        
        # Countdown: 3, 2, 1
        for count in [3, 2, 1]:
            if websocket.client_state.name != 'CONNECTED':
                print("⚠️  WebSocket disconnected during countdown")
                break
            print(f"⏳ Countdown: {count}")
            await websocket.send_json({
                'type': 'countdown',
                'number': count
            })
            await asyncio.sleep(1)
        
        # START!
        if websocket.client_state.name == 'CONNECTED':
            print("⏳ Countdown: START!")
            await websocket.send_json({
                'type': 'countdown',
                'number': 0,  # 0 = START
                'message': 'START!'
            })
            await asyncio.sleep(0.5)
            
            # Start tracking - THIS IS THE CRITICAL PART
            print("🏁 TRACKING STATE ACTIVATED!")
            session['state'] = 'tracking'
            session['tracking_frame_count'] = 0  # Reset frame count
            await websocket.send_json({
                'type': 'state',
                'state': 'tracking',
                'message': 'Start exercising!'
            })
            print(f"✅ Session state is now: {session['state']}")
    except Exception as e:
        print(f"⚠️  Countdown error: {e}")
        import traceback
        traceback.print_exc()


async def rest_countdown_task(websocket: WebSocket, session_id: int, rest_time: int, next_set: int):
    """Handle rest period countdown between sets."""
    session = sessions.get(session_id)
    if not session:
        return
    
    try:
        # Countdown from rest_time to 0
        for remaining in range(rest_time, 0, -1):
            if websocket.client_state.name != 'CONNECTED':
                break
            
            await websocket.send_json({
                'type': 'rest_countdown',
                'remaining': remaining,
                'total': rest_time
            })
            await asyncio.sleep(1)
        
        # Rest complete - start next set
        if websocket.client_state.name == 'CONNECTED' and session:
            session['current_set'] = next_set
            session['current_rep_in_set'] = 0
            session['state'] = 'tracking'
            
            await websocket.send_json({
                'type': 'state',
                'state': 'tracking',
                'message': f'Set {next_set} starting!'
            })
            print(f"✅ Rest complete, starting Set {next_set} - Data collection resumed for MLTRAINCAMERA and MLTRAINIMU")
    except Exception as e:
        print(f"⚠️  Rest countdown error: {e}")


async def send_ai_feedback_async(
    websocket: WebSocket,
    exercise: str,
    rep_result: dict,
    issues: list,
    regional_scores: dict = None,
    regional_issues: dict = None,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary'
):
    """Send AI feedback asynchronously without blocking rep detection.
    Supports Camera-only, IMU-only, and Sensor Fusion modes.
    """
    try:
        feedback_data = await get_ai_feedback(
            exercise,
            rep_result,
            issues,
            regional_scores,
            regional_issues,
            ml_prediction=ml_prediction,
            imu_data=imu_data,
            landmarks=landmarks,
            initial_positions=initial_positions,
            fusion_mode=fusion_mode
        )
        
        # Send feedback as separate message
        if websocket.client_state.name == 'CONNECTED':
            if isinstance(feedback_data, dict):
                await websocket.send_json({
                    'type': 'rep_feedback',
                    'rep': rep_result.get('rep', 0),
                    'feedback': feedback_data.get('overall', ''),
                    'regional_feedback': feedback_data.get('regional', {})
                })
            else:
                await websocket.send_json({
                    'type': 'rep_feedback',
                    'rep': rep_result.get('rep', 0),
                    'feedback': feedback_data,
                    'regional_feedback': {}
                })
    except Exception as e:
        print(f"⚠️  Error sending async AI feedback: {e}")
        # Silently fail - feedback is optional


async def train_ml_model_async(
    exercise: str,
    websocket: WebSocket,
    camera_session_id: str = None,
    imu_session_id: str = None
):
    """Train ML model using collected datasets (both camera and IMU)."""
    if not ML_TRAINING_ENABLED:
        error_msg = "ML training not available (ml_trainer not found)"
        print(f"⚠️  {error_msg}")
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_json({
                'type': 'training_status',
                'status': 'error',
                'message': error_msg
            })
        return
    
    try:
        print(f"🤖 Starting ML model training for {exercise}...")
        
        # Load datasets
        camera_collector = DC("MLTRAINCAMERA")
        all_camera_samples = camera_collector.load_dataset()
        
        # Filter by exercise and unused sessions (if tracker enabled)
        camera_samples = [s for s in all_camera_samples if s.exercise == exercise]
        
        if dataset_tracker:
            # Only use unused sessions
            unused_camera_sessions = set(dataset_tracker.get_unused_camera_sessions("MLTRAINCAMERA"))
            # Filter samples by session ID (need to check how session_id is stored in samples)
            # For now, use all samples if tracker is enabled (will be improved)
            pass  # TODO: Implement proper session filtering
        
        if len(camera_samples) < 10:
            raise ValueError(f"Not enough camera samples for training (need >=10, got {len(camera_samples)})")
        
        # Auto-label if not labeled
        labeled_camera_samples = [s for s in camera_samples if s.expert_score is not None or s.is_perfect_form is not None]
        if len(labeled_camera_samples) == 0:
            print("   Auto-labeling camera samples based on regional scores...")
            for sample in camera_samples:
                if sample.regional_scores:
                    avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                    sample.expert_score = avg_score
                    sample.is_perfect_form = (avg_score >= 90)
            labeled_camera_samples = camera_samples
        
        # Train model (run in thread to avoid blocking)
        def train_model():
            predictor = FormScorePredictor(model_type="random_forest")
            
            # Extract camera features if not already extracted
            for sample in labeled_camera_samples:
                if sample.features is None:
                    camera_collector.extract_features(sample)
            
            # Train camera model
            results = predictor.train(labeled_camera_samples, verbose=False, use_imu_features=False)
            
            # Save model with extended metadata
            from pathlib import Path
            model_dir = Path("models") / f"form_score_{exercise}_random_forest"
            model_dir.mkdir(parents=True, exist_ok=True)
            predictor.save(
                str(model_dir),
                exercise=exercise,
                training_samples=len(labeled_camera_samples),
                performance_metrics=results
            )
            
            # Calculate baselines if perfect samples exist
            perfect_samples = [s for s in labeled_camera_samples if s.is_perfect_form == True]
            if perfect_samples:
                baselines = BaselineCalculator.calculate_baselines(perfect_samples)
                
                # Save baselines
                baseline_file = model_dir / "baselines.json"
                import json
                with open(baseline_file, 'w') as f:
                    json.dump(baselines, f, indent=2, default=str)
            
            return (results, len(labeled_camera_samples))
        
        # Run training in thread pool
        training_result = await asyncio.to_thread(train_model)
        results, sample_count = training_result
        
        # Format performance metrics for display
        metrics_text = ""
        if results:
            metrics_text = f"\n📊 Model Performance:\n"
            metrics_text += f"   Test R²: {results.get('test_r2', 0):.3f}\n"
            metrics_text += f"   Test MAE: {results.get('test_mae', 0):.2f}\n"
            metrics_text += f"   Train R²: {results.get('train_r2', 0):.3f}"
        
        # Send success notification with performance metrics
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_json({
                'type': 'training_status',
                'status': 'completed',
                'message': f'✅ ML model training completed! Model saved to models/form_score_{exercise}_random_forest/ ({sample_count} samples){metrics_text}',
                'performance_metrics': results,
                'sample_count': sample_count,
                'model_path': f'models/form_score_{exercise}_random_forest/'
            })
        
        print(f"✅ ML model training completed for {exercise}")
        print(f"   - Model: models/form_score_{exercise}_random_forest/")
        print(f"   - Samples used: {sample_count}")
        
    except Exception as e:
        error_msg = f"⚠️  ML training failed: {str(e)}"
        print(error_msg)
        
        # Send error notification
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_json({
                'type': 'training_status',
                'status': 'error',
                'message': error_msg
            })


async def get_session_feedback(exercise: str, reps_data: list, all_issues: list) -> str:
    """Get comprehensive feedback at session end. Uses OpenAI if available, otherwise rule-based."""
    
    if not reps_data:
        return "Henüz rep tamamlanmadı. Devam et, daha uzun süre yapmaya çalış!"
    
    total_reps = len(reps_data)
    avg_score = sum(r['form_score'] for r in reps_data) / total_reps
    best_score = max(r['form_score'] for r in reps_data)
    worst_score = min(r['form_score'] for r in reps_data)
    
    # Find most common issues
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    top_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:3]
    
    # Exercise names
    exercise_names = {
        'bicep_curls': 'Biceps Curl',
        'squats': 'Squat',
        'lateral_shoulder_raises': 'Lateral Raise',
        'triceps_pushdown': 'Triceps Extension',
        'dumbbell_rows': 'Dumbbell Row',
        'dumbbell_shoulder_press': 'Shoulder Press'
    }
    ex_name = exercise_names.get(exercise, exercise)
    
    # Try OpenAI first (if available)
    if openai_client:
        try:
            top_issues_text = ', '.join([f"{issue} ({count}x)" for issue, count in top_issues]) if top_issues else 'None'
            
            prompt = f"""You are an expert fitness coach providing workout session feedback.

📊 WORKOUT SUMMARY ({ex_name}):
- Total Reps Completed: {total_reps}
- Average Form Score: {avg_score:.1f}%
- Best Rep Score: {best_score:.1f}%
- Worst Rep Score: {worst_score:.1f}%
- Most Common Issues: {top_issues_text}

Provide comprehensive feedback in Turkish:
1. Congratulate them for completing the workout
2. Overall performance assessment (be encouraging but honest)
3. 2-3 specific improvement areas based on the most common issues
4. Motivating closing message

Keep it friendly, professional, and under 4-5 sentences. Focus on actionable advice."""

            response = openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': 'You are a professional fitness coach. Provide detailed, encouraging workout feedback in Turkish.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=300,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  OpenAI session feedback error: {e}, falling back to rule-based")
            # Fall through to rule-based feedback
    
    # Fallback: Build feedback based on performance (rule-based)
    feedback_parts = []
    
    # Opening
    if avg_score >= 85:
        feedback_parts.append(f"🎉 Harika iş! {total_reps} rep {ex_name} tamamladın!")
    elif avg_score >= 70:
        feedback_parts.append(f"👍 İyi gidiyorsun! {total_reps} rep {ex_name} tamamladın!")
    else:
        feedback_parts.append(f"💪 Tebrikler! {total_reps} rep {ex_name} tamamladın!")
    
    # Performance summary
    feedback_parts.append(f"Ortalama form skoru: %{avg_score:.0f}")
    if best_score >= 85:
        feedback_parts.append(f"En iyi rep: %{best_score:.0f} (Mükemmel!)")
    
    # Improvement areas
    if top_issues:
        if len(top_issues) == 1:
            feedback_parts.append(f"İyileştirme alanı: {top_issues[0][0]} ({top_issues[0][1]} kez tespit edildi).")
        else:
            issues_str = ", ".join([f"{issue} ({count}x)" for issue, count in top_issues[:2]])
            feedback_parts.append(f"İyileştirme alanları: {issues_str}.")
    elif avg_score >= 80:
        feedback_parts.append("Formun çok iyi, devam et!")
    else:
        feedback_parts.append("Formunu iyileştirmeye devam et, yavaş ve kontrollü hareket et.")
    
    # Closing motivation
    if avg_score >= 85:
        feedback_parts.append("Harika çalışma, bu şekilde devam et! 💪")
    elif avg_score >= 70:
        feedback_parts.append("İyi performans, bir sonraki antrenmanda daha da iyileşeceksin!")
    else:
        feedback_parts.append("İlk adımlar zor, ama devam ettiğin sürece ilerleyeceksin!")
    
    return " ".join(feedback_parts)


# Session storage
sessions = {}

# Dataset collectors (optional)
dataset_collector = None
if DATASET_COLLECTION_ENABLED:
    dataset_collector = DatasetCollector("dataset")

# Training mode collectors (for ML training)
# We'll create collectors per exercise in the WebSocket handler
# Store collectors in a dict: {exercise: collector}
camera_training_collectors: Dict[str, DatasetCollector] = {}
imu_training_collectors: Dict[str, IMUDatasetCollector] = {}

# IMU bridge WebSocket client tasks: {session_id: task}
imu_bridge_tasks: Dict[str, asyncio.Task] = {}


# Dataset tracker (tracks which datasets are used for training)
if DATASET_TRACKER_ENABLED:
    dataset_tracker = DatasetTracker()
else:
    dataset_tracker = None


@app.websocket("/ws/{exercise}")
async def websocket_endpoint(websocket: WebSocket, exercise: str):
    """WebSocket endpoint for real-time pose analysis."""
    await websocket.accept()
    
    # Create session
    session_id = id(websocket)
    
    # Initialize ML inference if enabled
    ml_inference_instance = None
    baselines_dict = {}
    if ML_INFERENCE_ENABLED:
        try:
            ml_inference_instance = ModelInference(exercise)
            baselines_dict = load_baselines(exercise)
            if ml_inference_instance.has_camera_model():
                print(f"🤖 ML Inference enabled for {exercise}")
        except Exception as e:
            print(f"⚠️  Failed to initialize ML inference: {e}")
            ml_inference_instance = None
    
    sessions[session_id] = {
        'form_analyzer': FormAnalyzer(exercise),
        'rep_counter': RepCounter(exercise),
        'state': 'detecting',
        'reps_data': [],       # Store all rep data
        'all_issues': [],      # Store all issues for session summary
        'landmarks_history': [],  # Store landmarks sequence for dataset collection
        'current_rep_landmarks': [],  # Landmarks for current rep
        'current_rep_imu': [],  # IMU data for current rep (list of IMU samples) - for usage mode
        'current_rep_imu_samples': [],  # IMU samples for training mode (separate from usage mode)
        'dataset_collection_enabled': False,  # Track if collection is active
        'collected_reps_count': 0,  # Count collected reps
        'last_camera_sample_time': None,  # For 20Hz throttling (50ms interval)
        'last_imu_sample_time': {},  # For 20Hz throttling per node (50ms interval per node)
        'ml_mode': 'usage',  # 'usage' (basic mode + data recording) or 'train' (ML training only)
        'training_session_started': False,  # Track if training collectors are started
        # ML Inference (for real-time predictions)
        'ml_inference': ml_inference_instance,
        'baselines': baselines_dict,
        # Workout configuration
        'workout_config': {
            'numberOfSets': 3,
            'repsPerSet': 10,
            'restTimeSeconds': 60
        },
        # Set tracking
        'current_set': 1,
        'current_rep_in_set': 0,
        'total_reps_in_session': 0,
    }
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'init':
                # Initialize OpenAI (optional - if provided, will be used for enhanced feedback)
                if data.get('api_key'):
                    init_openai(data['api_key'])
                    print("✅ OpenAI API initialized (will be used for enhanced feedback)")
                else:
                    print("ℹ️  No API key provided - using rule-based feedback only")
                
                # Store ML mode (default: usage - basic mode with data recording)
                ml_mode = data.get('ml_mode', 'usage')
                session = sessions.get(session_id)
                if session:
                    session['ml_mode'] = ml_mode
                    # Usage mode: data recording enabled automatically
                    # Train mode: dataset collection enabled automatically
                    if ml_mode == 'usage':
                        session['dataset_collection_enabled'] = True  # Usage mode records data
                        # Usage mode: automatically start dataset collector
                        if dataset_collector and DATASET_COLLECTION_ENABLED:
                            try:
                                user_id = data.get('user_id', 'default')
                                dataset_collector.start_session(exercise, user_id=user_id)
                                session['collected_reps_count'] = 0
                                print(f"✅ Usage mode: Dataset collector started for {exercise}")
                            except Exception as e:
                                print(f"⚠️  Failed to start dataset collector in usage mode: {e}")
                        print(f"📊 Mode: {ml_mode} (Usage mode - data recording enabled)")
                    elif ml_mode == 'train':
                        # Train mode: start separate collectors for camera and IMU (exercise-specific)
                        session['dataset_collection_enabled'] = True  # Train mode collects data
                        
                        # Create a shared session_id to ensure synchronization
                        # Both camera and IMU collectors will use the same session_id
                        shared_session_id = f"{exercise}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        session['training_session_id'] = shared_session_id
                        
                        # Get or create exercise-specific collectors
                        if DATASET_COLLECTION_ENABLED:
                            if exercise not in camera_training_collectors:
                                camera_training_collectors[exercise] = DatasetCollector("MLTRAINCAMERA")
                            # Start with shared session_id
                            camera_training_collectors[exercise].start_session(exercise)
                            # Override session_id to ensure sync (in case they were created in different seconds)
                            camera_training_collectors[exercise].current_session_id = shared_session_id
                            print(f"📹 Camera training collector started for {exercise} (session: {shared_session_id})")
                        
                        if IMU_DATASET_COLLECTION_ENABLED:
                            if exercise not in imu_training_collectors:
                                imu_training_collectors[exercise] = IMUDatasetCollector("MLTRAINIMU")
                            # Start with shared session_id
                            imu_training_collectors[exercise].start_session(exercise)
                            # Override session_id to ensure sync (in case they were created in different seconds)
                            imu_training_collectors[exercise].current_session_id = shared_session_id
                            print(f"📡 IMU training collector started for {exercise} (session: {shared_session_id})")
                            
                            # Connect to gymbud_imu_bridge WebSocket to receive IMU data directly (bypass frontend throttling)
                            if WEBSOCKETS_AVAILABLE:
                                async def imu_bridge_client_task():
                                    """Connect to gymbud_imu_bridge WebSocket and receive IMU data directly."""
                                    IMU_BRIDGE_WS_URL = "ws://localhost:8765"
                                    
                                    while True:
                                        try:
                                            async with websockets.connect(IMU_BRIDGE_WS_URL) as ws:
                                                print(f"✅ Connected to gymbud_imu_bridge WebSocket for {exercise} (session: {shared_session_id})")
                                                
                                                async for message in ws:
                                                    try:
                                                        data = json.loads(message)
                                                        if data.get('type') == 'imu_update' and 'nodes' in data:
                                                            # Get current session state
                                                            session = sessions.get(session_id)
                                                            if not session or not session.get('training_session_started'):
                                                                continue
                                                            
                                                            # Get current rep_number from session (check rep_counter if available)
                                                            rep_number = 0
                                                            if 'rep_counter' in session and session['rep_counter']:
                                                                rep_counter = session['rep_counter']
                                                                if rep_counter.phase == 'up' and rep_counter.count >= 0:
                                                                    rep_number = rep_counter.count + 1
                                                            
                                                            # Create IMU sample data from gymbud_imu_bridge format
                                                            timestamp = data.get('timestamp', time.time())
                                                            imu_sample_data = {
                                                                'timestamp': timestamp,
                                                                'rep_number': rep_number
                                                            }
                                                            
                                                            # Add each node's data (convert from nested to flat format)
                                                            nodes_data = data.get('nodes', {})
                                                            for node_name in ['left_wrist', 'right_wrist', 'chest']:
                                                                if node_name in nodes_data:
                                                                    node_data = nodes_data[node_name]
                                                                    accel = node_data.get('accel', {})
                                                                    gyro = node_data.get('gyro', {})
                                                                    quaternion = node_data.get('quaternion', {})
                                                                    euler = node_data.get('euler', {})
                                                                    imu_sample_data[node_name] = {
                                                                        'ax': accel.get('x'),
                                                                        'ay': accel.get('y'),
                                                                        'az': accel.get('z'),
                                                                        'gx': gyro.get('x'),
                                                                        'gy': gyro.get('y'),
                                                                        'gz': gyro.get('z'),
                                                                        'qw': quaternion.get('w'),
                                                                        'qx': quaternion.get('x'),
                                                                        'qy': quaternion.get('y'),
                                                                        'qz': quaternion.get('z'),
                                                                        'roll': euler.get('roll'),
                                                                        'pitch': euler.get('pitch'),
                                                                        'yaw': euler.get('yaw')
                                                                    }
                                                            
                                                            # Add to session-level buffer (continuous collection)
                                                            if 'session_imu_samples' not in session:
                                                                session['session_imu_samples'] = []
                                                            session['session_imu_samples'].append(copy.deepcopy(imu_sample_data))
                                                            
                                                            # Add to rep-level buffer if rep is active
                                                            if rep_number > 0:
                                                                if 'current_rep_imu_samples' not in session:
                                                                    session['current_rep_imu_samples'] = []
                                                                session['current_rep_imu_samples'].append(copy.deepcopy(imu_sample_data))
                                                    except json.JSONDecodeError:
                                                        continue
                                                    except Exception as e:
                                                        print(f"⚠️  Error processing IMU data from gymbud_imu_bridge: {e}")
                                        except websockets.exceptions.ConnectionClosedOK:
                                            print(f"🔌 Disconnected from gymbud_imu_bridge WebSocket for {exercise}. Reconnecting...")
                                            await asyncio.sleep(1)
                                        except Exception as e:
                                            print(f"❌ Failed to connect to gymbud_imu_bridge WebSocket for {exercise}: {e}. Retrying in 1 second...")
                                            await asyncio.sleep(1)
                                
                                # Start the WebSocket client task
                                task = asyncio.create_task(imu_bridge_client_task())
                                imu_bridge_tasks[session_id] = task
                                print(f"🚀 Started IMU bridge WebSocket client task for {exercise} (session: {shared_session_id})")
                        
                        session['training_session_started'] = True
                        # Initialize session-level buffers for continuous data collection (rep sayılsın ya da sayılmasın)
                        session['session_landmarks'] = []  # All landmarks throughout session
                        session['session_imu_samples'] = []  # All IMU samples throughout session
                        session['session_start_time'] = time.time()
                        # Initialize rep-level buffers (will be reset after each rep is saved)
                        session['current_rep_landmarks'] = []  # Current rep's landmarks
                        session['current_rep_imu_samples'] = []  # Current rep's IMU samples
                        print(f"📊 Mode: {ml_mode} (ML Training mode - synchronized camera & IMU collection enabled for {exercise})")
                        print(f"   Shared session ID: {shared_session_id}")
                        print(f"   Session-level continuous collection enabled (all data, regardless of rep counting)")
                
                # Store workout configuration
                workout_config = data.get('workout_config', {})
                if workout_config:
                    if session:
                        session['workout_config'] = {
                            'numberOfSets': workout_config.get('numberOfSets', 3),
                            'repsPerSet': workout_config.get('repsPerSet', 10),
                            'restTimeSeconds': workout_config.get('restTimeSeconds', 60)
                        }
                        session['current_set'] = 1
                        session['current_rep_in_set'] = 0
                        session['total_reps_in_session'] = 0
                
                await websocket.send_json({'type': 'ready'})
                continue
            
            # Handle dataset collection start/stop
            if data.get('type') == 'start_collection':
                session = sessions.get(session_id)
                if session and dataset_collector and DATASET_COLLECTION_ENABLED:
                    try:
                        user_id = data.get('user_id', 'default')
                        dataset_collector.start_session(exercise, user_id=user_id)
                        session['dataset_collection_enabled'] = True
                        session['collected_reps_count'] = 0
                        await websocket.send_json({
                            'type': 'dataset_collection_status',
                            'status': 'collecting',
                            'collected_reps': 0
                        })
                        print(f"✅ Dataset collection started for {exercise}")
                    except Exception as e:
                        print(f"⚠️  Failed to start dataset collection: {e}")
                        await websocket.send_json({
                            'type': 'dataset_collection_status',
                            'status': 'error',
                            'error': str(e)
                        })
                else:
                    await websocket.send_json({
                        'type': 'dataset_collection_status',
                        'status': 'error',
                        'error': 'Dataset collector not available'
                    })
                continue
            
            if data.get('type') == 'stop_collection':
                session = sessions.get(session_id)
                if session and dataset_collector:
                    try:
                        # Save current session
                        if dataset_collector.is_collecting:
                            auto_label = data.get('auto_label_perfect', True)
                            dataset_collector.save_session(auto_label_perfect=auto_label)
                            collected_count = session.get('collected_reps_count', 0)
                            session['dataset_collection_enabled'] = False
                            await websocket.send_json({
                                'type': 'dataset_collection_status',
                                'status': 'saved',
                                'collected_reps': collected_count,
                                'message': f'Dataset saved: {collected_count} reps collected'
                            })
                            print(f"💾 Dataset saved: {collected_count} reps")
                        else:
                            await websocket.send_json({
                                'type': 'dataset_collection_status',
                                'status': 'idle',
                                'collected_reps': 0
                            })
                    except Exception as e:
                        print(f"⚠️  Failed to stop/save dataset collection: {e}")
                        await websocket.send_json({
                            'type': 'dataset_collection_status',
                            'status': 'error',
                            'error': str(e)
                        })
                continue
            
            if data.get('type') == 'pose':
                landmarks = data.get('landmarks', [])
                session = sessions[session_id]
                form_analyzer = session['form_analyzer']
                rep_counter = session['rep_counter']
                
                # State machine
                config = EXERCISE_CONFIG.get(exercise, {})
                required_landmarks = config.get('required_landmarks', [11, 12, 13, 14, 15, 16])
                calibration_msg = config.get('calibration_message', 'Show your body')
                
                if session['state'] == 'detecting':
                    # Check exercise-specific landmarks (use lenient threshold for initial detection)
                    all_visible, visible_count, missing = check_required_landmarks(
                        landmarks, required_landmarks, threshold=0.5, min_visibility_ratio=0.75
                    )
                    
                    if all_visible:
                        session['state'] = 'calibrating'
                        await websocket.send_json({
                            'type': 'state',
                            'state': 'calibrating',
                            'message': 'Calibration starting... Hold still!'
                        })
                    else:
                        # Send detailed feedback about what's missing
                        missing_str = ', '.join(missing[:3])  # Show max 3 missing
                        await websocket.send_json({
                            'type': 'visibility',
                            'visible_count': visible_count,
                            'required_count': len(required_landmarks),
                            'missing': missing_str,
                            'message': calibration_msg
                        })
                
                elif session['state'] == 'calibrating':
                    completed, timed_out = form_analyzer.calibrate(landmarks)
                    
                    if timed_out:
                        # Reset and go back to detecting
                        session['state'] = 'detecting'
                        await websocket.send_json({
                            'type': 'state',
                            'state': 'detecting',
                            'message': 'Calibration timeout. Please hold still and try again.'
                        })
                    elif completed:
                        print("✅ Calibration complete! Starting countdown...")
                        # IMPORTANT: Change state IMMEDIATELY to prevent re-entering calibration
                        session['state'] = 'countdown'
                        # Start countdown in background task to avoid blocking
                        asyncio.create_task(
                            countdown_task(websocket, session_id)
                        )
                    else:
                        await websocket.send_json({
                            'type': 'calibration_progress',
                            'progress': len(form_analyzer.calibration_frames) / FormAnalyzer.CALIBRATION_FRAMES
                        })
                
                elif session['state'] == 'countdown':
                    # Countdown state - don't process pose data during countdown
                    continue
                
                elif session['state'] == 'resting':
                    # Don't process pose during rest period - handled by rest_countdown_task
                    # This effectively pauses data collection for MLTRAINCAMERA and MLTRAINIMU
                    # Collectors remain active (is_collecting=True) but no new data is added
                    # Data collection will automatically resume when state returns to 'tracking'
                    continue
                
                elif session['state'] == 'tracking':
                    # Store landmarks and IMU data for dataset collection
                    ml_mode = session.get('ml_mode', 'usage')
                    current_time = time.time()
                    
                    # Debug: verify we're in tracking state
                    tracking_frame_count = session.get('tracking_frame_count', 0)
                    session['tracking_frame_count'] = tracking_frame_count + 1
                    if tracking_frame_count == 0:
                        print(f"🏁 TRACKING STATE ACTIVE! Exercise: {exercise}")
                    
                    if ml_mode == 'train':
                        # Training mode: collect to separate training collectors (exercise-specific)
                        # Store landmarks for camera training collector with 20Hz throttling (50ms = 0.05s)
                        camera_collector = camera_training_collectors.get(exercise)
                        if camera_collector and camera_collector.is_collecting:
                            last_sample_time = session.get('last_camera_sample_time')
                            # Throttle to 20Hz: only add if 50ms (0.05s) has passed since last sample
                            if last_sample_time is None or (current_time - last_sample_time) >= 0.05:
                                # Track rep number for this frame
                                # Rep number is current_rep_number + 1 if we're in 'up' phase (rep in progress)
                                # Otherwise 0 (no rep in progress)
                                frame_rep_number = 0
                                if rep_counter.phase == 'up' and rep_counter.count >= 0:
                                    # Rep is in progress, rep number is the next rep (will be completed)
                                    frame_rep_number = rep_counter.count + 1
                                
                                # Add to rep-level buffer (for rep-based collection)
                                session['current_rep_landmarks'].append(landmarks)
                                # Also add to session-level buffer (for continuous collection) with rep_number
                                if 'session_landmarks' not in session:
                                    session['session_landmarks'] = []
                                session['session_landmarks'].append({
                                    'timestamp': current_time,
                                    'landmarks': landmarks,
                                    'rep_number': frame_rep_number  # Mark which rep this frame belongs to
                                })
                                session['last_camera_sample_time'] = current_time
                                if len(session['current_rep_landmarks']) > 200:  # Increased for longer reps at 20Hz
                                    session['current_rep_landmarks'].pop(0)
                        
                        # In train mode, IMU data comes from gymbud_imu_bridge WebSocket client (bypass frontend throttling)
                        # Skip frontend IMU data processing in train mode
                        if ml_mode != 'train':
                            # Usage mode: process IMU data from frontend (not used in train mode)
                            pass  # IMU data not used in usage mode for training collection
                    elif session.get('dataset_collection_enabled') and dataset_collector and dataset_collector.is_collecting:
                        # Usage mode: collect to regular dataset collector with 20Hz throttling
                        last_sample_time = session.get('last_camera_sample_time')
                        # Throttle to 20Hz: only add if 50ms (0.05s) has passed since last sample
                        if last_sample_time is None or (current_time - last_sample_time) >= 0.05:
                            session['current_rep_landmarks'].append(landmarks)
                            session['last_camera_sample_time'] = current_time
                            if len(session['current_rep_landmarks']) > 200:  # Increased for longer reps at 20Hz
                                session['current_rep_landmarks'].pop(0)
                        
                        # Store IMU data if provided with 20Hz throttling per node
                        imu_data = data.get('imu_data') or data.get('imu')
                        if imu_data:
                            if 'last_imu_node_sample_time' not in session:
                                session['last_imu_node_sample_time'] = {}
                            
                            # Check each IMU node separately for 20Hz throttling
                            nodes_to_add = {}
                            for node_name in ['left_wrist', 'right_wrist', 'chest']:
                                if node_name in imu_data and imu_data[node_name]:
                                    last_node_time = session['last_imu_node_sample_time'].get(node_name)
                                    if last_node_time is None or (current_time - last_node_time) >= 0.05:
                                        nodes_to_add[node_name] = imu_data[node_name]
                                        session['last_imu_node_sample_time'][node_name] = current_time
                            
                            if nodes_to_add:
                                throttled_imu_data = {k: v for k, v in nodes_to_add.items()}
                                if 'timestamp' in imu_data:
                                    throttled_imu_data['timestamp'] = imu_data['timestamp']
                                session['current_rep_imu'].append(throttled_imu_data)
                                if len(session['current_rep_imu']) > 200:
                                    session['current_rep_imu'].pop(0)
                    
                    # Calculate angle
                    config = EXERCISE_CONFIG.get(exercise, {})
                    joints = config.get('joints', {}).get('left', (11, 13, 15))
                    
                    try:
                        # Use left arm joints (shoulder, elbow, wrist)
                        a = [landmarks[joints[0]]['x'], landmarks[joints[0]]['y']]
                        b = [landmarks[joints[1]]['x'], landmarks[joints[1]]['y']]
                        c = [landmarks[joints[2]]['x'], landmarks[joints[2]]['y']]
                        angle = calculate_angle(a, b, c)
                        
                        # Validate angle is reasonable (0-180°)
                        if angle < 0 or angle > 180 or np.isnan(angle):
                            print(f"⚠️ Invalid angle calculated: {angle}, setting to 0")
                            angle = 0
                    except (IndexError, KeyError, TypeError) as e:
                        print(f"⚠️ Angle calculation error: {e}")
                        angle = 0
                    
                    # Check form
                    form_result = form_analyzer.check_form(landmarks)
                    session['last_form_score'] = form_result['score']
                    session['last_form_result'] = form_result  # Store full form result for summary
                    
                    # Update rep counter (pass landmarks for motion validation)
                    rep_result = rep_counter.update(angle, form_result['score'], landmarks)
                    
                    # Track current rep number for marking frames
                    current_rep_number = rep_counter.count
                    
                    
                    # Store issues for session summary ONLY when rep is completed (not every frame!)
                    if rep_result:
                        session['all_issues'].extend(form_result['issues'])
                    
                    # Calculate average form score from all reps
                    avg_form = 0
                    if len(session.get('reps_data', [])) > 0:
                        all_form_scores = [r.get('form_score', 0) for r in session['reps_data']]
                        avg_form = sum(all_form_scores) / len(all_form_scores)
                    
                    # Debug: log every 10 frames for troubleshooting (more frequent)
                    frame_count = len(session.get('current_rep_landmarks', []))
                    if frame_count > 0 and frame_count % 10 == 0:
                        print(f"🔍 Frame {frame_count}: phase={rep_counter.phase}, angle={angle:.1f}°, form={form_result['score']:.0f}%, rep_count={rep_counter.count}, set={session.get('current_set', 1)}, rep_in_set={session.get('current_rep_in_set', 0)}")
                    # Also log first few frames to see if tracking is working
                    if frame_count <= 3:
                        print(f"🔍 TRACKING START: Frame {frame_count}, angle={angle:.1f}°, phase={rep_counter.phase}, rep_count={rep_counter.count}")
                    
                    # Debug: log rep completion
                    if rep_result:
                        print(f"✅ REP #{rep_result.get('rep', 0)} COMPLETED! Valid: {rep_result.get('is_valid', False)}, Score: {rep_result.get('form_score', 0):.1f}%, Angle: {angle:.1f}°")
                        print(f"   Range: {rep_result.get('min_angle', 0):.1f}° to {rep_result.get('max_angle', 0):.1f}°")
                        print(f"   Set: {session.get('current_set', 1)}, Rep in set: {session.get('current_rep_in_set', 0)}/{workout_config.get('repsPerSet', 10)}")
                    
                    # Add regional scores to rep_result if rep completed
                    if rep_result:
                        # ML Inference: Predict regional scores using trained model
                        ml_inference = session.get('ml_inference')
                        ml_prediction = None
                        baseline_similarity = None
                        hybrid_score = None
                        ml_regional_scores = None
                        
                        # Rule-based regional scores (fallback if ML not available)
                        rule_based_regional_scores = form_result.get('regional_scores', {
                            'arms': form_result['score'],
                            'legs': form_result['score'],
                            'core': form_result['score'],
                            'head': form_result['score']
                        })
                        
                        if ml_inference and ml_inference.has_camera_model():
                            try:
                                # Get landmarks sequence for current rep
                                rep_landmarks = session.get('current_rep_landmarks', [])
                                if len(rep_landmarks) > 0:
                                    # Predict using ML model (returns Dict[str, float] with regional scores)
                                    ml_prediction = ml_inference.predict_camera(rep_landmarks)
                                    
                                    # Extract regional scores from ML prediction
                                    if isinstance(ml_prediction, dict) and 'arms' in ml_prediction:
                                        ml_regional_scores = {
                                            'arms': ml_prediction.get('arms', 0.0),
                                            'legs': ml_prediction.get('legs', 0.0),
                                            'core': ml_prediction.get('core', 0.0),
                                            'head': ml_prediction.get('head', 0.0)
                                        }
                                    
                                    # Calculate baseline similarity (regional) if baselines are available
                                    baselines = session.get('baselines', {})
                                    if baselines and ml_regional_scores:
                                        # Use regional scores for similarity calculation
                                        baseline_similarity_dict, _ = calculate_baseline_similarity(
                                            current_features=None,
                                            baselines=baselines,
                                            current_regional_scores=ml_regional_scores
                                        )
                                        baseline_similarity = baseline_similarity_dict  # Dict[str, float] with regional similarities
                                        
                                        # Calculate hybrid score (ML + Baseline) - regional
                                        hybrid_score = calculate_hybrid_correction_score(
                                            ml_prediction,
                                            baseline_similarity,
                                            ml_weight=0.6,
                                            baseline_weight=0.4
                                        )
                                        
                                        # Use hybrid_score as final regional scores (ML + Baseline)
                                        if isinstance(hybrid_score, dict) and 'arms' in hybrid_score:
                                            rep_result['regional_scores'] = hybrid_score
                                        else:
                                            rep_result['regional_scores'] = ml_regional_scores
                                    elif ml_regional_scores:
                                        # ML prediction available but no baselines - use ML prediction
                                        rep_result['regional_scores'] = ml_regional_scores
                                    else:
                                        # Fallback to rule-based
                                        rep_result['regional_scores'] = rule_based_regional_scores
                                    
                                    if ml_regional_scores:
                                        print(f"🤖 ML Regional Scores: Arms={ml_regional_scores['arms']:.1f}%, Legs={ml_regional_scores['legs']:.1f}%, Core={ml_regional_scores['core']:.1f}%, Head={ml_regional_scores['head']:.1f}%")
                                        if baseline_similarity:
                                            print(f"   Baseline Similarity: Arms={baseline_similarity.get('arms', 0):.1f}%, Legs={baseline_similarity.get('legs', 0):.1f}%, Core={baseline_similarity.get('core', 0):.1f}%, Head={baseline_similarity.get('head', 0):.1f}%")
                                        if hybrid_score and isinstance(hybrid_score, dict):
                                            print(f"   Hybrid Scores: Arms={hybrid_score.get('arms', 0):.1f}%, Legs={hybrid_score.get('legs', 0):.1f}%, Core={hybrid_score.get('core', 0):.1f}%, Head={hybrid_score.get('head', 0):.1f}%")
                            except Exception as e:
                                print(f"⚠️  ML inference error: {e}")
                                import traceback
                                traceback.print_exc()
                                # Fallback to rule-based
                                rep_result['regional_scores'] = rule_based_regional_scores
                        else:
                            # ML not available - use rule-based regional scores
                            rep_result['regional_scores'] = rule_based_regional_scores
                        
                        # Add ML scores to rep_result
                        rep_result['ml_prediction'] = ml_prediction
                        rep_result['baseline_similarity'] = baseline_similarity
                        rep_result['hybrid_score'] = hybrid_score
                        
                        # Save rep to dataset (both usage and train modes record data)
                        ml_mode = session.get('ml_mode', 'usage')
                        rep_number = rep_result.get('rep', rep_counter.count)
                        
                        if ml_mode == 'train':
                            # Training mode: save to separate collectors (MLTRAINCAMERA and MLTRAINIMU)
                            # Get camera rep timestamp first (will be used for both camera and IMU collectors)
                            camera_rep_timestamp = time.time()
                            
                            # Save to camera training collector (exercise-specific)
                            camera_collector = camera_training_collectors.get(exercise)
                            camera_rep_sample = None
                            if camera_collector and camera_collector.is_collecting:
                                try:
                                    camera_rep_sample = camera_collector.add_rep_sample(
                                        exercise=exercise,
                                        rep_number=rep_number,
                                        landmarks_sequence=session['current_rep_landmarks'].copy(),
                                        imu_sequence=None,  # Don't save IMU in camera collector
                                        regional_scores=form_result.get('regional_scores'),
                                        regional_issues=form_result.get('regional_issues'),
                                        min_angle=rep_result.get('min_angle'),
                                        max_angle=rep_result.get('max_angle'),
                                        user_id='default'
                                    )
                                    # Use the camera rep sample's timestamp (more accurate, matches camera data)
                                    camera_rep_timestamp = camera_rep_sample.timestamp
                                    print(f"📹 Saved rep #{rep_number} to MLTRAINCAMERA/{exercise}/ (timestamp: {camera_rep_timestamp})")
                                except Exception as e:
                                    print(f"⚠️  Camera training collection error: {e}")
                            
                            # Save to IMU training collector (exercise-specific)
                            # Use camera rep timestamp to ensure synchronization
                            imu_collector = imu_training_collectors.get(exercise)
                            if imu_collector and imu_collector.is_collecting:
                                try:
                                    # Get IMU samples from session
                                    imu_samples_seq = session.get('current_rep_imu_samples', [])
                                    
                                    if len(imu_samples_seq) > 0:
                                        # Remove rep_number from samples before adding to collector
                                        imu_data_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in imu_samples_seq]
                                        imu_collector.add_rep_sequence(
                                            rep_number=rep_number,
                                            imu_sequence=imu_data_seq,
                                            rep_start_time=camera_rep_timestamp  # Use camera rep timestamp for synchronization
                                        )
                                        print(f"📡 Saved rep #{rep_number} to MLTRAINIMU/{exercise}/ ({len(imu_data_seq)} IMU samples from gymbud_imu_bridge)")
                                    else:
                                        # Use session-level IMU data if rep-level is empty (fallback)
                                        session_imu_samples = session.get('session_imu_samples', [])
                                        # Filter session-level samples by rep_number
                                        rep_imu_samples = [s for s in session_imu_samples if s.get('rep_number') == rep_number]
                                        if len(rep_imu_samples) > 0:
                                            # Extract IMU data from session-level samples (remove rep_number field for consistency)
                                            rep_imu_data = [{k: v for k, v in s.items() if k != 'rep_number'} for s in rep_imu_samples]
                                            imu_collector.add_rep_sequence(
                                                rep_number=rep_number,
                                                imu_sequence=rep_imu_data,
                                                rep_start_time=camera_rep_timestamp  # Use camera rep timestamp for synchronization
                                            )
                                            print(f"📡 Saved rep #{rep_number} to MLTRAINIMU/{exercise}/ ({len(rep_imu_data)} IMU samples from session-level data)")
                                        else:
                                            print(f"⚠️  Rep #{rep_number}: No IMU samples found")
                                        # Still add empty rep to maintain consistency
                                        imu_collector.add_rep_sequence(
                                            rep_number=rep_number,
                                            imu_sequence=[],
                                                rep_start_time=camera_rep_timestamp  # Use camera rep timestamp for synchronization
                                        )
                                    
                                    # Clear rep-level buffer after saving
                                    session['current_rep_imu_samples'] = []
                                except Exception as e:
                                    print(f"⚠️  IMU training collection error: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            # Update collected reps count
                            session['collected_reps_count'] = session.get('collected_reps_count', 0) + 1
                            
                            # Reset for next rep
                            session['current_rep_landmarks'] = []
                            session['current_rep_imu_samples'] = []
                            
                            # Send status update
                            await websocket.send_json({
                                'type': 'dataset_collection_status',
                                'status': 'collecting',
                                'collected_reps': session['collected_reps_count']
                            })
                        elif session.get('dataset_collection_enabled') and dataset_collector and dataset_collector.is_collecting:
                            # Usage mode: save to regular dataset collector
                            try:
                                landmarks_seq = session['current_rep_landmarks'].copy() if session.get('current_rep_landmarks') else []
                                imu_seq = session['current_rep_imu'].copy() if session.get('current_rep_imu') else []
                                
                                if len(landmarks_seq) > 0:
                                    dataset_collector.add_rep_sample(
                                        exercise=exercise,
                                        rep_number=rep_number,
                                        landmarks_sequence=landmarks_seq,
                                        imu_sequence=imu_seq if len(imu_seq) > 0 else None,
                                        regional_scores=form_result.get('regional_scores'),
                                        regional_issues=form_result.get('regional_issues'),
                                        min_angle=rep_result.get('min_angle'),
                                        max_angle=rep_result.get('max_angle'),
                                        user_id='default'
                                    )
                                    # Update collected reps count
                                    session['collected_reps_count'] = session.get('collected_reps_count', 0) + 1
                                    print(f"💾 Usage mode: Saved rep #{rep_number} to dataset/ ({len(landmarks_seq)} landmarks frames)")
                                    
                                    # Send status update
                                    await websocket.send_json({
                                        'type': 'dataset_collection_status',
                                        'status': 'collecting',
                                        'collected_reps': session['collected_reps_count']
                                    })
                                else:
                                    print(f"⚠️  Usage mode: No landmarks collected for rep #{rep_number}")
                                
                                # Reset landmarks and IMU for next rep
                                session['current_rep_landmarks'] = []
                                session['current_rep_imu'] = []
                            except Exception as e:
                                print(f"⚠️  Dataset collection error: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    # Send update with regional scores and set/rep info
                    workout_config = session.get('workout_config', {})
                    response = {
                        'type': 'update',
                        'angle': round(angle, 1),
                        'phase': rep_counter.phase,
                        'rep_count': rep_counter.count,
                        'valid_rep_count': rep_counter.valid_count,  # Only valid reps
                        'form_score': form_result['score'],
                        'avg_form': round(avg_form, 1),  # Average form across all reps
                        'issues': form_result['issues'],
                        'regional_scores': form_result.get('regional_scores', {
                            'arms': form_result['score'],
                            'legs': form_result['score'],
                            'core': form_result['score'],
                            'head': form_result['score']
                        }),
                        'current_set': session.get('current_set', 1),
                        'current_rep_in_set': session.get('current_rep_in_set', 0),
                        'reps_per_set': workout_config.get('repsPerSet', 10),
                        'number_of_sets': workout_config.get('numberOfSets', 3),
                    }
                    
                    # If rep completed, check set tracking and get AI feedback
                    if rep_result:
                        session['reps_data'].append(rep_result)
                        
                        # Get workout config values
                        rest_time = workout_config.get('restTimeSeconds', 60)
                        reps_per_set = workout_config.get('repsPerSet', 10)
                        number_of_sets = workout_config.get('numberOfSets', 3)
                        
                        # Update set tracking - count ALL reps (both valid and invalid)
                        # This ensures set completion even if form is poor
                        session['current_rep_in_set'] = session.get('current_rep_in_set', 0) + 1
                        session['total_reps_in_session'] = session.get('total_reps_in_session', 0) + 1
                        
                        # Re-read current state after increment
                        current_set = session.get('current_set', 1)
                        current_rep_in_set = session.get('current_rep_in_set', 0)
                        
                        print(f"🔢 Rep #{rep_result.get('rep', 0)} completed: Set {current_set}, Rep {current_rep_in_set}/{reps_per_set}, Valid: {rep_result.get('is_valid', False)}")
                        
                        # Update set/rep info in response
                        response['current_set'] = current_set
                        response['current_rep_in_set'] = current_rep_in_set
                        response['reps_per_set'] = reps_per_set
                        response['number_of_sets'] = number_of_sets
                        
                        # Check if set is complete
                        if current_rep_in_set >= reps_per_set:
                            # Set complete - check if workout is complete
                            if current_set >= number_of_sets:
                                # All sets completed - automatically end session
                                print(f"🏁 Workout complete! Set {current_set}/{number_of_sets}, Rep {current_rep_in_set}/{reps_per_set}")
                                session['state'] = 'finished'
                                
                                # Calculate session summary
                                ml_mode = session.get('ml_mode', 'usage')
                                collected_count = session.get('collected_reps_count', 0)
                                
                                # Generate session feedback
                                session_feedback = await get_session_feedback(
                            exercise,
                                    session['reps_data'],
                                    session['all_issues']
                                )
                                
                                # Calculate average regional scores
                                if session['reps_data']:
                                    all_regional_scores = [r.get('regional_scores', {}) for r in session['reps_data'] if r.get('regional_scores')]
                                    if all_regional_scores:
                                        avg_regional_scores = {
                                            'arms': sum(r.get('arms', 0) for r in all_regional_scores) / len(all_regional_scores),
                                            'legs': sum(r.get('legs', 0) for r in all_regional_scores) / len(all_regional_scores),
                                            'core': sum(r.get('core', 0) for r in all_regional_scores) / len(all_regional_scores),
                                            'head': sum(r.get('head', 0) for r in all_regional_scores) / len(all_regional_scores),
                                        }
                                    else:
                                        avg_regional_scores = {'arms': 0, 'legs': 0, 'core': 0, 'head': 0}
                                else:
                                    avg_regional_scores = {'arms': 0, 'legs': 0, 'core': 0, 'head': 0}
                                
                                # Generate regional feedbacks based on average regional scores
                                # Collect all regional issues from all reps
                                all_regional_issues = {}
                                for rep_data in session['reps_data']:
                                    regional_issues = rep_data.get('regional_issues', {})
                                    for region in ['arms', 'legs', 'core', 'head']:
                                        if region not in all_regional_issues:
                                            all_regional_issues[region] = []
                                        if region in regional_issues and regional_issues[region]:
                                            all_regional_issues[region].extend(regional_issues[region])
                                
                                # Get most common issue per region
                                regional_issues_summary = {}
                                for region in ['arms', 'legs', 'core', 'head']:
                                    if all_regional_issues.get(region):
                                        issue_counts = {}
                                        for issue in all_regional_issues[region]:
                                            issue_counts[issue] = issue_counts.get(issue, 0) + 1
                                        top_issue = max(issue_counts.items(), key=lambda x: x[1])[0] if issue_counts else None
                                        regional_issues_summary[region] = [top_issue] if top_issue else []
                                    else:
                                        regional_issues_summary[region] = []
                                
                                # Generate regional feedback using rule-based feedback
                                regional_feedbacks = {}
                                for region in ['arms', 'legs', 'core', 'head']:
                                    region_score = avg_regional_scores.get(region, 100)
                                    region_issues = regional_issues_summary.get(region, [])
                                    regional_feedbacks[region] = get_rule_based_regional_feedback(
                                        exercise, region, region_score, region_issues,
                                        rep_num=len(session['reps_data']), min_angle=None, max_angle=None
                                    )
                                
                                # Send session summary
                                summary_data = {
                                    'type': 'session_summary',
                                    'total_reps': len(session['reps_data']),
                                    'avg_form': round(
                                        sum(r['form_score'] for r in session['reps_data']) / len(session['reps_data'])
                                        if session['reps_data'] else 0, 1
                                    ),
                                    'regional_scores': avg_regional_scores,
                                    'regional_feedback': regional_feedbacks,  # Add regional feedback
                                    'feedback': session_feedback,
                                    'workout_complete': True,
                                    'message': 'Workout completed automatically! All sets and reps finished.'
                                }
                                
                                # Add rep completion info to response before sending
                                response['rep_completed'] = rep_result
                                response['rep_valid'] = rep_result.get('is_valid', True)
                                response['rep_feedback'] = rep_result.get('feedback', '')
                                response['workout_complete'] = True
                                # Add ML inference scores to response
                                response['ml_prediction'] = rep_result.get('ml_prediction')
                                response['baseline_similarity'] = rep_result.get('baseline_similarity')
                                response['hybrid_score'] = rep_result.get('hybrid_score')
                                
                                # Workout tamamlandığında otomatik kayıt yap (train mode için)
                                ml_mode = session.get('ml_mode', 'usage')
                                if ml_mode == 'train':
                                    try:
                                        camera_collector = camera_training_collectors.get(exercise)
                                        imu_collector = imu_training_collectors.get(exercise)
                                        collected_count = session.get('collected_reps_count', 0)
                                        
                                        # Add session-level continuous data (all data throughout session, rep sayılsın ya da sayılmasın)
                                        session_landmarks = session.get('session_landmarks', [])
                                        session_imu_samples = session.get('session_imu_samples', [])
                                        session_start_time = session.get('session_start_time', time.time())
                                        
                                        if camera_collector and camera_collector.is_collecting and len(session_landmarks) > 0:
                                            # Extract landmarks from session buffer
                                            landmarks_sequence = [item['landmarks'] for item in session_landmarks]
                                            try:
                                                camera_collector.add_rep_sample(
                                                    exercise=exercise,
                                                    rep_number=0,  # rep_number=0 means session-level continuous data
                                                    landmarks_sequence=landmarks_sequence,
                                                    imu_sequence=None,
                                                    user_id='default'
                                                )
                                                print(f"📹 Added session-level continuous camera data: {len(landmarks_sequence)} frames (rep_number=0)")
                                            except Exception as e:
                                                print(f"⚠️  Failed to add session-level camera data: {e}")
                                        
                                        if imu_collector and imu_collector.is_collecting:
                                            print(f"🔍 Debug (workout_complete): session_imu_samples size={len(session_imu_samples)}")
                                            if len(session_imu_samples) > 0:
                                                try:
                                                    # Remove rep_number from samples before adding to collector
                                                    imu_data_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in session_imu_samples]
                                                    imu_collector.add_rep_sequence(
                                                        rep_number=0,  # rep_number=0 means session-level continuous data
                                                        imu_sequence=imu_data_seq,
                                                        rep_start_time=session_start_time
                                                    )
                                                    print(f"📡 Added session-level continuous IMU data: {len(imu_data_seq)} samples (rep_number=0)")
                                                except Exception as e:
                                                    print(f"⚠️  Failed to add session-level IMU data: {e}")
                                                    import traceback
                                                    traceback.print_exc()
                                            else:
                                                print(f"⚠️  session_imu_samples is empty in workout_complete, cannot add session-level IMU data")
                                        
                                        # Save if collectors exist and have data (including session-level)
                                        if camera_collector and camera_collector.is_collecting:
                                            if len(camera_collector.current_samples) > 0:
                                                camera_session_id = camera_collector.current_session_id
                                                num_samples = len(camera_collector.current_samples)
                                                camera_collector.save_session(auto_label_perfect=True)
                                                print(f"💾 Saved camera training dataset: {num_samples} samples (reps + session-level) → MLTRAINCAMERA/{exercise}/ (session: {camera_session_id})")
                                                session['camera_session_id'] = camera_session_id
                                                # Note: save_session() already sets is_collecting = False
                                        
                                        if imu_collector and imu_collector.is_collecting:
                                            if len(imu_collector.current_samples) > 0:
                                                imu_session_id = imu_collector.current_session_id
                                                num_samples = len(imu_collector.current_samples)
                                                imu_collector.save_session()
                                                print(f"💾 Saved IMU training dataset: {num_samples} sequences (reps + session-level) → MLTRAINIMU/{exercise}/ (session: {imu_session_id})")
                                                session['imu_session_id'] = imu_session_id
                                            imu_collector.stop_session()
                                        
                                        # Check if data was saved successfully
                                        camera_session_id = session.get('camera_session_id')
                                        imu_session_id = session.get('imu_session_id')
                                        if camera_session_id or imu_session_id:
                                            print(f"✅ Training session completed: {collected_count} reps saved to training datasets")
                                    except Exception as e:
                                        print(f"⚠️  Failed to save training datasets when workout completed: {e}")
                                        import traceback
                                        traceback.print_exc()
                                
                                # Send final rep update
                                await websocket.send_json(response)
                                
                                # Send session summary
                                await websocket.send_json(summary_data)
                                print(f"📤 Sent automatic session_summary: total_reps={summary_data['total_reps']}, workout_complete=True")
                                
                                # Don't continue processing - session is finished
                                break
                            else:
                                # More sets to go - start rest period
                                session['state'] = 'resting'
                                await websocket.send_json({
                                    'type': 'state',
                                    'state': 'resting',
                                    'message': f'Set {current_set} complete! Rest time: {rest_time}s'
                                })
                                
                                # Start rest countdown
                                asyncio.create_task(
                                    rest_countdown_task(
                                        websocket,
                                        session_id,
                                        rest_time,
                                        current_set + 1
                                    )
                                )
                                
                                # Stop tracking for now (will resume after rest)
                                # Data collection (MLTRAINCAMERA and MLTRAINIMU) is paused during rest period
                                # Will automatically resume when rest_countdown_task sets state back to 'tracking'
                                response['set_complete'] = True
                        
                        # Add rep validation info to response
                        response['rep_completed'] = rep_result
                        response['rep_valid'] = rep_result.get('is_valid', True)
                        response['rep_feedback'] = rep_result.get('feedback', '')
                        # Add ML inference scores to response
                        response['ml_prediction'] = rep_result.get('ml_prediction')
                        response['baseline_similarity'] = rep_result.get('baseline_similarity')
                        response['hybrid_score'] = rep_result.get('hybrid_score')
                        
                        # Technical AI feedback with regional data (NON-BLOCKING)
                        # Don't await - send response immediately, feedback will come later
                        
                        # Determine fusion mode based on available ML models
                        fusion_mode = 'camera_only'  # Default
                        ml_inference = session.get('ml_inference')
                        if ml_inference:
                            has_camera = ml_inference.has_camera_model()
                            has_imu = ml_inference.has_imu_model()
                            if has_camera and has_imu:
                                fusion_mode = 'camera_primary'  # Sensor fusion
                            elif has_imu:
                                fusion_mode = 'imu_only'
                            elif has_camera:
                                fusion_mode = 'camera_only'
                        
                        # Get last landmark frame (for landmark-based feedback)
                        last_landmarks = None
                        rep_landmarks = session.get('current_rep_landmarks', [])
                        if len(rep_landmarks) > 0:
                            last_landmarks = rep_landmarks[-1]
                        
                        # Get initial positions from form analyzer (for landmark-based feedback)
                        initial_positions = None
                        form_analyzer = session.get('form_analyzer')
                        if form_analyzer and hasattr(form_analyzer, 'initial_positions'):
                            initial_positions = form_analyzer.initial_positions
                        
                        # Get IMU data (last sample from current rep)
                        imu_data_for_feedback = None
                        if fusion_mode in ['imu_only', 'camera_primary']:
                            rep_imu_samples = session.get('current_rep_imu_samples', [])
                            if len(rep_imu_samples) > 0:
                                # Get last IMU sample (remove rep_number field)
                                last_imu_sample = rep_imu_samples[-1].copy()
                                last_imu_sample.pop('rep_number', None)
                                last_imu_sample.pop('timestamp', None)
                                imu_data_for_feedback = last_imu_sample
                        
                        asyncio.create_task(
                            send_ai_feedback_async(
                                websocket,
                                exercise,
                                rep_result,
                                form_result['issues'],
                                form_result.get('regional_scores'),
                                form_result.get('regional_issues'),
                                ml_prediction=rep_result.get('ml_prediction'),
                                imu_data=imu_data_for_feedback,
                                landmarks=last_landmarks,
                                initial_positions=initial_positions,
                                fusion_mode=fusion_mode
                            )
                        )
                        # Don't include feedback in response - it will come as separate message
                        response['feedback'] = ''  # Empty for now
                        response['regional_feedback'] = {}  # Empty for now
                    else:
                        # No rep completed - still send update with current state
                        response['rep_completed'] = None
                        response['rep_valid'] = False
                    
                    # Always send update (even if no rep completed)
                    # Debug: log first few updates to verify tracking
                    if len(session.get('reps_data', [])) == 0 and rep_counter.count == 0:
                        update_count = session.get('update_count', 0)
                        session['update_count'] = update_count + 1
                        if update_count < 5:
                            print(f"📤 Sending update #{update_count+1}: rep_count={response['rep_count']}, angle={response['angle']}, phase={response['phase']}")
                    
                    await websocket.send_json(response)
            
            # Handle dataset collection save
            elif data.get('type') == 'save_dataset':
                if dataset_collector and DATASET_COLLECTION_ENABLED:
                    try:
                        auto_label = data.get('auto_label_perfect', False)
                        dataset_collector.save_session(auto_label_perfect=auto_label)
                        await websocket.send_json({
                            'type': 'dataset_saved',
                            'message': 'Dataset session saved successfully'
                        })
                    except Exception as e:
                        await websocket.send_json({
                            'type': 'dataset_error',
                            'error': str(e)
                        })
                else:
                    await websocket.send_json({
                        'type': 'dataset_error',
                        'error': 'Dataset collection not enabled'
                    })
            
            # Handle session end request
            elif data.get('type') == 'end_session':
                session = sessions.get(session_id)
                if session:
                    ml_mode = session.get('ml_mode', 'usage')
                    collected_count = session.get('collected_reps_count', 0)
                    
                    if ml_mode == 'train':
                        # Training mode: save both camera and IMU datasets (but don't mark as used yet)
                        try:
                            # Save session-level continuous data (all data throughout session, rep sayılsın ya da sayılmasın)
                            camera_collector = camera_training_collectors.get(exercise)
                            imu_collector = imu_training_collectors.get(exercise)
                            
                            # Add session-level continuous data as rep_number=0 (before saving)
                            session_landmarks = session.get('session_landmarks', [])
                            session_imu_samples = session.get('session_imu_samples', [])
                            session_start_time = session.get('session_start_time', time.time())
                            
                            if camera_collector and camera_collector.is_collecting and len(session_landmarks) > 0:
                                # Extract landmarks from session buffer (they're stored as {'timestamp': ..., 'landmarks': ...})
                                landmarks_sequence = [item['landmarks'] for item in session_landmarks]
                                try:
                                    camera_collector.add_rep_sample(
                                        exercise=exercise,
                                        rep_number=0,  # rep_number=0 means session-level continuous data
                                        landmarks_sequence=landmarks_sequence,
                                        imu_sequence=None,
                                        user_id='default'
                                    )
                                    print(f"📹 Added session-level continuous camera data: {len(landmarks_sequence)} frames (rep_number=0)")
                                except Exception as e:
                                    print(f"⚠️  Failed to add session-level camera data: {e}")
                            
                            if imu_collector and imu_collector.is_collecting:
                                print(f"🔍 Debug: session_imu_samples size={len(session_imu_samples)}")
                                if len(session_imu_samples) > 0:
                                    try:
                                        # Remove rep_number from samples before adding to collector
                                        imu_data_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in session_imu_samples]
                                        imu_collector.add_rep_sequence(
                                            rep_number=0,  # rep_number=0 means session-level continuous data
                                            imu_sequence=imu_data_seq,
                                            rep_start_time=session_start_time
                                        )
                                        print(f"📡 Added session-level continuous IMU data: {len(imu_data_seq)} samples (rep_number=0)")
                                    except Exception as e:
                                        print(f"⚠️  Failed to add session-level IMU data: {e}")
                                        import traceback
                                        traceback.print_exc()
                                else:
                                    print(f"⚠️  session_imu_samples is empty, cannot add session-level IMU data")
                            
                            # Now save all data (including rep-based and session-level continuous)
                            camera_session_id = None
                            if camera_collector and camera_collector.is_collecting:
                                camera_session_id = camera_collector.current_session_id
                                total_camera_samples = len(camera_collector.current_samples)
                                camera_collector.save_session(auto_label_perfect=True)
                                print(f"💾 Saved camera training dataset: {total_camera_samples} samples (reps + session-level) → MLTRAINCAMERA/{exercise}/ (session: {camera_session_id})")
                                
                            # Save IMU training dataset (exercise-specific)
                            imu_session_id = None
                            if imu_collector and imu_collector.is_collecting:
                                imu_session_id = imu_collector.current_session_id
                                total_imu_reps = len(imu_collector.current_samples)
                                imu_collector.save_session()
                                print(f"💾 Saved IMU training dataset: {total_imu_reps} sequences (reps + session-level) → MLTRAINIMU/{exercise}/ (session: {imu_session_id})")
                                
                            # Store session IDs for later tracking
                            session['camera_session_id'] = camera_session_id
                            session['imu_session_id'] = imu_session_id
                                
                            # Stop collectors (save_session already sets is_collecting=False for DatasetCollector)
                            if imu_collector:
                                imu_collector.stop_session()
                                
                            print(f"✅ Training session completed: {collected_count} reps + session-level continuous data saved to both datasets")
                            print(f"   Camera session ID: {camera_session_id}")
                            print(f"   IMU session ID: {imu_session_id}")
                            print(f"   Exercise: {exercise}")
                                # Note: We don't mark as used yet - user will decide in the dialog
                                
                        except Exception as e:
                            print(f"⚠️  Failed to save training datasets: {e}")
                    elif session.get('dataset_collection_enabled') and dataset_collector:
                        # Usage mode: Data is already collected via add_rep_sample during session
                        # Don't auto-save here - user will choose in dialog (save_only or skip)
                        print(f"💾 Usage mode: {collected_count} reps collected (already added to collector), waiting for user decision...")
                    # Calculate average regional scores
                    avg_regional_scores = {
                        'arms': 0,
                        'legs': 0,
                        'core': 0,
                        'head': 0
                    }
                    
                    if session['reps_data']:
                        # Sum all regional scores
                        total_regional = {'arms': 0, 'legs': 0, 'core': 0, 'head': 0}
                        count = 0
                        for rep in session['reps_data']:
                            if 'regional_scores' in rep:
                                for region in ['arms', 'legs', 'core', 'head']:
                                    total_regional[region] += rep['regional_scores'].get(region, 0)
                                count += 1
                        
                        # Calculate averages
                        if count > 0:
                            for region in ['arms', 'legs', 'core', 'head']:
                                avg_regional_scores[region] = round(total_regional[region] / count, 1)
                        else:
                            # Fallback to last form result
                            last_result = session.get('last_form_result', {})
                            avg_regional_scores = last_result.get('regional_scores', {
                                'arms': 0, 'legs': 0, 'core': 0, 'head': 0
                            })
                    else:
                        # No reps - use last form result if available
                        last_result = session.get('last_form_result', {})
                        avg_regional_scores = last_result.get('regional_scores', {
                            'arms': 0, 'legs': 0, 'core': 0, 'head': 0
                        })
                    
                    # Generate comprehensive session feedback
                    session_feedback = await get_session_feedback(
                        exercise,
                        session['reps_data'],
                        session['all_issues']
                    )
                    
                    # Generate regional feedbacks based on average regional scores
                    # Collect all regional issues from all reps
                    all_regional_issues = {}
                    for rep_data in session['reps_data']:
                        regional_issues = rep_data.get('regional_issues', {})
                        for region in ['arms', 'legs', 'core', 'head']:
                            if region not in all_regional_issues:
                                all_regional_issues[region] = []
                            if region in regional_issues and regional_issues[region]:
                                all_regional_issues[region].extend(regional_issues[region])
                    
                    # Get most common issue per region
                    regional_issues_summary = {}
                    for region in ['arms', 'legs', 'core', 'head']:
                        if all_regional_issues.get(region):
                            issue_counts = {}
                            for issue in all_regional_issues[region]:
                                issue_counts[issue] = issue_counts.get(issue, 0) + 1
                            top_issue = max(issue_counts.items(), key=lambda x: x[1])[0] if issue_counts else None
                            regional_issues_summary[region] = [top_issue] if top_issue else []
                        else:
                            regional_issues_summary[region] = []
                    
                    # Generate regional feedback using rule-based feedback
                    regional_feedbacks = {}
                    for region in ['arms', 'legs', 'core', 'head']:
                        region_score = avg_regional_scores.get(region, 100)
                        region_issues = regional_issues_summary.get(region, [])
                        regional_feedbacks[region] = get_rule_based_regional_feedback(
                            exercise, region, region_score, region_issues,
                            rep_num=len(session['reps_data']), min_angle=None, max_angle=None
                    )
                    
                    summary_data = {
                        'type': 'session_summary',
                        'total_reps': len(session['reps_data']),
                        'avg_form': round(
                            sum(r['form_score'] for r in session['reps_data']) / len(session['reps_data'])
                            if session['reps_data'] else 0, 1
                        ),
                        'regional_scores': avg_regional_scores,
                        'regional_feedback': regional_feedbacks,  # Add regional feedback
                        'feedback': session_feedback
                    }
                    print(f"📤 Sending session_summary: total_reps={summary_data['total_reps']}, avg_form={summary_data['avg_form']}, feedback_length={len(session_feedback) if session_feedback else 0}")
                    print(f"   Feedback preview: {session_feedback[:100] if session_feedback else 'None'}...")
                    await websocket.send_json(summary_data)
            
            # Handle training action (from dialog)
            elif data.get('type') == 'training_action':
                session = sessions.get(session_id)
                if not session:
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'error',
                        'message': 'Session not found'
                    })
                    continue
                
                action = data.get('action')
                ml_mode = session.get('ml_mode', 'usage')
                
                camera_session_id = session.get('camera_session_id')
                imu_session_id = session.get('imu_session_id')
                
                if action == 'save_only':
                    # Save data (usage mode: save to dataset/, train mode: data already saved to MLTRAIN*)
                    if ml_mode == 'usage':
                        # Save to regular dataset collector
                        if session.get('dataset_collection_enabled') and dataset_collector and dataset_collector.is_collecting:
                            try:
                                if len(dataset_collector.current_samples) == 0:
                                    print(f"⚠️ Usage mode: No samples to save (current_samples is empty)")
                                    await websocket.send_json({
                                        'type': 'training_status',
                                        'status': 'error',
                                        'message': 'Kaydedilecek veri yok. Rep tamamlandığında veriler toplandı mı kontrol edin.'
                                    })
                                    continue
                                else:
                                    dataset_collector.save_session(auto_label_perfect=True)
                                    print(f"💾 Usage mode: Saved session to dataset/ ({len(dataset_collector.current_samples)} samples)")
                            except Exception as e:
                                print(f"⚠️ Error saving usage mode session: {e}")
                                import traceback
                                traceback.print_exc()
                                await websocket.send_json({
                                    'type': 'training_status',
                                    'status': 'error',
                                    'message': f'Kayıt hatası: {str(e)}'
                                })
                                continue
                        else:
                            print(f"⚠️ Usage mode: Cannot save - collector not available or not collecting")
                            await websocket.send_json({
                                'type': 'training_status',
                                'status': 'error',
                                'message': 'Veri toplayıcı aktif değil. Veriler kaydedilemedi.'
                            })
                            continue
                    else:
                        # Train mode: Check if data was already saved, if not save now
                        camera_collector = camera_training_collectors.get(exercise)
                        imu_collector = imu_training_collectors.get(exercise)
                        
                        # Check if data was already saved (session_id stored means it was saved)
                        if not camera_session_id and camera_collector and camera_collector.is_collecting:
                            # Add session-level continuous data (all data throughout session, rep sayılsın ya da sayılmasın)
                            session_landmarks = session.get('session_landmarks', [])
                            session_imu_samples = session.get('session_imu_samples', [])
                            session_start_time = session.get('session_start_time', time.time())
                            
                            if len(session_landmarks) > 0:
                                # Extract landmarks from session buffer
                                landmarks_sequence = [item['landmarks'] for item in session_landmarks]
                                try:
                                    camera_collector.add_rep_sample(
                                        exercise=exercise,
                                        rep_number=0,  # rep_number=0 means session-level continuous data
                                        landmarks_sequence=landmarks_sequence,
                                        imu_sequence=None,
                                        user_id='default'
                                    )
                                    print(f"📹 Added session-level continuous camera data: {len(landmarks_sequence)} frames (rep_number=0)")
                                except Exception as e:
                                    print(f"⚠️  Failed to add session-level camera data: {e}")
                            
                            if len(session_imu_samples) > 0 and imu_collector and imu_collector.is_collecting:
                                try:
                                    # Remove rep_number from samples before adding to collector
                                    imu_data_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in session_imu_samples]
                                    imu_collector.add_rep_sequence(
                                        rep_number=0,  # rep_number=0 means session-level continuous data
                                        imu_sequence=imu_data_seq,
                                        rep_start_time=session_start_time
                                    )
                                    print(f"📡 Added session-level continuous IMU data: {len(imu_data_seq)} samples (rep_number=0)")
                                except Exception as e:
                                    print(f"⚠️  Failed to add session-level IMU data: {e}")
                            
                            # Data not saved yet, save it now (including session-level)
                            if len(camera_collector.current_samples) > 0:
                                try:
                                    camera_session_id = camera_collector.current_session_id
                                    num_samples = len(camera_collector.current_samples)
                                    camera_collector.save_session(auto_label_perfect=True)
                                    print(f"💾 Saved camera training dataset: {num_samples} samples (reps + session-level) → MLTRAINCAMERA/{exercise}/ (session: {camera_session_id})")
                                    session['camera_session_id'] = camera_session_id
                                except Exception as e:
                                    print(f"⚠️  Failed to save camera training dataset: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            if imu_collector and imu_collector.is_collecting:
                                if len(imu_collector.current_samples) > 0:
                                    try:
                                        imu_session_id = imu_collector.current_session_id
                                        num_samples = len(imu_collector.current_samples)
                                        imu_collector.save_session()
                                        print(f"💾 Saved IMU training dataset: {num_samples} sequences (reps + session-level) → MLTRAINIMU/{exercise}/ (session: {imu_session_id})")
                                        session['imu_session_id'] = imu_session_id
                                    except Exception as e:
                                        print(f"⚠️  Failed to save IMU training dataset: {e}")
                                        import traceback
                                        traceback.print_exc()
                            
                            # Stop collectors after saving (save_session already sets is_collecting=False for DatasetCollector)
                            if imu_collector:
                                imu_collector.stop_session()
                    
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'completed',
                        'message': 'Veriler kaydedildi. Daha sonra ML eğitimi için kullanabilirsiniz.'
                    })
                
                # Note: 'save_and_train' action removed - training is now done separately
                # Use train_ml_models.py script or Google Colab notebook for training
                # This keeps data collection fast and allows training on GPU (Colab/cloud)
                elif action == 'save_and_train':
                    # This action is no longer supported - redirect to save_only
                    print(f"⚠️  'save_and_train' action deprecated. Use train_ml_models.py or Colab notebook.")
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'completed',
                        'message': 'Veriler kaydedildi. Model eğitimi için train_ml_models.py veya TRAINING_GUIDE.md\'yi kullanın.'
                    })
                    continue
                
                elif action == 'skip':
                    # Skip: Don't save data (already collected during session, but mark as not to be used)
                    print(f"⏭️  User chose: Skip (data will not be saved)")
                    
                    if ml_mode == 'train':
                        # For train mode, data was collected but user wants to skip
                        # We could delete the session data here, but for now just acknowledge
                        camera_collector = camera_training_collectors.get(exercise)
                        imu_collector = imu_training_collectors.get(exercise)
                        if camera_collector:
                            camera_collector.current_samples = []  # Clear collected samples
                        if imu_collector:
                            imu_collector.current_samples = []  # Clear collected samples
                        print(f"   Cleared training data for {exercise}")
                    
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'completed',
                        'message': 'Veriler kaydedilmedi.'
                    })
                
                else:
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'error',
                        'message': f'Unknown action: {action}'
                    })
    
    except WebSocketDisconnect:
        if session_id in sessions:
            del sessions[session_id]
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()


@app.get("/")
async def root():
    return {"message": "Fitness AI Coach API", "status": "running"}

@app.post("/api/update_model/{exercise}")
async def update_model(exercise: str):
    """Update existing ML model using only unused datasets (exercise-specific)."""
    try:
        if not ML_TRAINING_ENABLED:
            return {
                "success": False,
                "message": "ML training not available (ml_trainer not found)"
            }
        
        # Check if model exists (exercise-specific path)
        from pathlib import Path
        model_dir = Path("models") / exercise / f"form_score_camera_random_forest"
        if not model_dir.exists():
            return {
                "success": False,
                "message": f"No existing model found for {exercise}. Use training mode to create a new model."
            }
        
        # Get unused sessions (exercise-specific)
        if not dataset_tracker:
            return {
                "success": False,
                "message": "Dataset tracker not available"
            }
        
        unused_camera_sessions = dataset_tracker.get_unused_camera_sessions("MLTRAINCAMERA", exercise=exercise)
        unused_imu_sessions = dataset_tracker.get_unused_imu_sessions("MLTRAINIMU", exercise=exercise)
        
        if len(unused_camera_sessions) == 0 and len(unused_imu_sessions) == 0:
            return {
                "success": False,
                "message": f"No unused training data found for {exercise}. All datasets have been used for training."
            }
        
        # Load unused datasets (exercise-specific)
        camera_collector = DC("MLTRAINCAMERA")
        camera_samples = camera_collector.load_dataset(exercise=exercise)  # Only load this exercise's data
        
        # Filter by unused session IDs (if tracker enabled)
        if dataset_tracker and unused_camera_sessions:
            # Filter samples by session (sessions are saved in exercise-specific folders now)
            camera_samples = [s for s in camera_samples 
                            if any(sid in str(getattr(s, 'session_id', '')) for sid in unused_camera_sessions)]
        
        if len(camera_samples) < 10:
            return {
                "success": False,
                "message": f"Not enough unused samples for {exercise} (need >=10, got {len(camera_samples)})"
            }
        
        # Auto-label
        for sample in camera_samples:
            if sample.expert_score is None and sample.regional_scores:
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                sample.expert_score = avg_score
                sample.is_perfect_form = (avg_score >= 90)
        
        # Extract features
        for sample in camera_samples:
            if sample.features is None:
                camera_collector.extract_features(sample)
        
        # Train model - MULTI-OUTPUT for regional scores
        predictor = FormScorePredictor(model_type="random_forest", multi_output=True)
        results = predictor.train(camera_samples, verbose=False, use_imu_features=False)
        
        # Save model (overwrite existing, exercise-specific) with extended metadata
        model_dir.mkdir(parents=True, exist_ok=True)
        predictor.save(
            str(model_dir),
            exercise=exercise,
            training_samples=len(camera_samples),
            performance_metrics=results
        )
        
        # Mark sessions as used (exercise-specific)
        for session_id in unused_camera_sessions:
            dataset_tracker.mark_camera_session_used(session_id)
        for session_id in unused_imu_sessions:
            dataset_tracker.mark_imu_session_used(session_id)
        
        return {
            "success": True,
            "message": f"Model updated successfully for {exercise}! Used {len(camera_samples)} samples from {len(unused_camera_sessions)} unused sessions."
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error updating model: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

