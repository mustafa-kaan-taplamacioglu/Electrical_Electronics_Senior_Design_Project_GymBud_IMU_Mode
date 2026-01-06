"""Form analyzer for exercise form analysis."""

import numpy as np
import time
from utils.pose_utils import check_required_landmarks, get_bone_angle_from_vertical, get_bone_angle_from_horizontal, get_angle_between_bones

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
    "tricep_extensions": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 160, "down": 60},
        # Upper body: Face (0-10), Upper Body (11-16), Hands (17-22) = 23 landmarks
        "required_landmarks": list(range(0, 23)),  # 0-22: Face + Upper Body + Hands
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "tricep_extensions": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 160, "down": 60},
        # Upper body: Face (0-10), Upper Body (11-16), Hands (17-22) = 23 landmarks
        "required_landmarks": list(range(0, 23)),  # 0-22: Face + Upper Body + Hands
        "calibration_message": "Face, upper body, and hands must be visible",
    },
    "dumbbell_rows": {
        "joints": {"left": (11, 13, 15), "right": (12, 14, 16)},
        "rep_threshold": {"up": 80, "down": 150},
        # SADECE SAĞ KOL VE GÖVDE: Sağ omuz (12), Sağ dirsek (14), Sağ bilek (16), Kalçalar (23-24)
        # Sol kol, bacaklar ve kafa ignore edilecek (default 0)
        "required_landmarks": [11, 12, 13, 14, 15, 16, 23, 24],  # Omuzlar, dirsekler, bilekler, kalçalar (sağ kol tracking için gerekli)
        "calibration_message": "Sağ kol, omuzlar ve kalçalar görünür olmalı (vücudun sağ tarafından 90° açıdan)",
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
            
            # Leg lengths (skip for dumbbell_rows - legs are not used for this exercise)
            if self.exercise != 'dumbbell_rows':
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
            else:
                # Dumbbell rows: Leg lengths are not used, set to None
                self.thigh_length = None
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
            
            # Knees (skip for dumbbell_rows - legs are not used)
            if self.exercise != 'dumbbell_rows':
                if avg[25].get('calibrated', False):
                    self.initial_positions['left_knee'] = {'x': avg[25]['x'], 'y': avg[25]['y']}
            if avg[26].get('calibrated', False):
                    self.initial_positions['right_knee'] = {'x': avg[26]['x'], 'y': avg[26]['y']}
            
            # Ankles (skip for dumbbell_rows - legs are not used)
            if self.exercise != 'dumbbell_rows':
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
        elif self.exercise == 'tricep_extensions':
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
            # SADECE KOLLAR VE GÖVDE DİKKATE ALINIR
            # Bacaklar ve kafa ignore edilir (default score = 100)
            
            # --- ARMS REGION ---
            # Elbow angles check (sadece dirsek açıları)
            left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
            right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
            
            # Eğer açılar geçerli değilse default skor ver
            if not np.isnan(left_elbow_angle) and not np.isnan(right_elbow_angle):
                if abs(left_elbow_angle - right_elbow_angle) > 25:
                    arms_issues.append('Dirsekler asimetrik')
                    arms_scores.append(70)
                else:
                    arms_scores.append(90)  # İyi form
            else:
                arms_scores.append(80)  # Default skor
            
            # --- CORE REGION ---
            # Sadece omuz seviyesi kontrolü (gövde eğikliği farketmez)
            try:
                shoulders_angle = get_bone_angle_from_horizontal(landmarks, 'shoulders')
                if shoulders_angle > 20:
                    core_issues.append('Omuzlar düz değil')
                    core_scores.append(65)
                else:
                    core_scores.append(90)  # İyi form
            except:
                core_scores.append(80)  # Default skor
            
            # --- HEAD REGION ---
            # Kafa ignore edilir (default score)
            head_scores.append(100)
            
            # --- LEGS REGION ---
            # Bacaklar ignore edilir (default score)
            legs_scores.append(100)
            
            # Combine all regional issues and scores
            issues.extend(arms_issues)
            issues.extend(core_issues)
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
            spine_center = init.get('spine_center')
            hip_shift = 0.0
            if spine_center is not None and self.hip_width is not None:
                hip_shift = abs((lm[23]['x'] + lm[24]['x']) / 2 - spine_center['x'])
            if spine_center is not None and self.hip_width is not None and hip_shift > self.hip_width * 0.1:
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
        if self.exercise in ['bicep_curls', 'lateral_shoulder_raises', 'tricep_extensions', 'dumbbell_shoulder_press']:
            # Upper body exercises: arms 50%, core 30%, head 10%, legs 10%
            final_score = (arms_score * 0.5 + core_score * 0.3 + head_score * 0.1 + legs_score * 0.1)
        elif self.exercise == 'squats':
            # Lower body exercises: legs 50%, core 40%, arms 5%, head 5%
            final_score = (legs_score * 0.5 + core_score * 0.4 + arms_score * 0.05 + head_score * 0.05)
        elif self.exercise == 'dumbbell_rows':
            # Back exercise: SADECE KOLLAR VE GÖVDE - arms 60%, core 40%
            # Bacaklar ve kafa ignore edilir (0% weight)
            final_score = (arms_score * 0.6 + core_score * 0.4)
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


