"""
Real-time Exercise Tracker
==========================
- Form quality feedback
- Rep counting
- Exercise detection

Usage:
    python realtime_exercise.py [exercise_name]
    
Example:
    python realtime_exercise.py bicep_curls
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import time

# Exercise configurations with detailed joint and landmark requirements
# MediaPipe landmarks:
# 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow
# 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip
# 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle

EXERCISE_CONFIG = {
    "bicep_curls": {
        # Dirsek açısı: omuz-dirsek-bilek
        "joints": {
            "left": (11, 13, 15),   # shoulder -> elbow -> wrist
            "right": (12, 14, 16)
        },
        "required_landmarks": {11, 12, 13, 14, 15, 16},  # shoulders, elbows, wrists
        "rep_threshold": {
            "contracted": 50,   # Kol bükülü (curl pozisyonu)
            "extended": 140     # Kol açık (başlangıç)
        },
        "rep_logic": {
            "start_phase": "extended",     # Rep başlangıcı: kol açık
            "count_on": "return_extended", # Rep sayılır: kol tekrar açılınca
            "direction": "contract_first"  # Önce büker sonra açar
        },
        "primary_angle": "elbow",
        "form_tips": [
            "Dirsekleri sabit tut",
            "Omuzları aşağıda tut",
            "Tam aç, tam kapat"
        ]
    },
    "squats": {
        # Diz açısı: kalça-diz-ayak bileği
        "joints": {
            "left": (23, 25, 27),   # hip -> knee -> ankle
            "right": (24, 26, 28)
        },
        "required_landmarks": {23, 24, 25, 26, 27, 28},  # hips, knees, ankles
        "rep_threshold": {
            "contracted": 90,   # Çömelmiş (diz bükük)
            "extended": 160     # Ayakta (diz açık)
        },
        "rep_logic": {
            "start_phase": "extended",     # Ayakta başla
            "count_on": "return_extended", # Ayağa kalkınca say
            "direction": "contract_first"  # Önce çömel sonra kalk
        },
        "primary_angle": "knee",
        "form_tips": [
            "Dizler ayak uçlarını geçmesin",
            "Sırtı düz tut",
            "Kalça geriye"
        ]
    },
    "lunges": {
        "joints": {
            "left": (23, 25, 27),
            "right": (24, 26, 28)
        },
        "required_landmarks": {23, 24, 25, 26, 27, 28},
        "rep_threshold": {
            "contracted": 90,
            "extended": 160
        },
        "rep_logic": {
            "start_phase": "extended",
            "count_on": "return_extended",
            "direction": "contract_first"
        },
        "primary_angle": "knee",
        "form_tips": [
            "Ön diz 90 derece",
            "Arka diz yere yakın",
            "Gövde dik"
        ]
    },
    "pushups": {
        "joints": {
            "left": (11, 13, 15),
            "right": (12, 14, 16)
        },
        "required_landmarks": {11, 12, 13, 14, 15, 16, 23, 24},
        "rep_threshold": {
            "contracted": 90,   # Aşağı inmiş (dirsek bükük)
            "extended": 160     # Yukarı (kollar açık)
        },
        "rep_logic": {
            "start_phase": "extended",     # Yukarıda başla
            "count_on": "return_extended", # Yukarı çıkınca say
            "direction": "contract_first"  # Önce in sonra çık
        },
        "primary_angle": "elbow",
        "form_tips": [
            "Vücut düz çizgi",
            "Göğüs yere yaklaşsın",
            "Core sıkı"
        ]
    },
    "lateral_shoulder_raises": {
        "joints": {
            "left": (13, 11, 23),   # elbow -> shoulder -> hip
            "right": (14, 12, 24)
        },
        "required_landmarks": {11, 12, 13, 14, 15, 16, 23, 24},
        "rep_threshold": {
            "contracted": 20,   # Kollar aşağıda
            "extended": 80      # Kollar kalkık
        },
        "rep_logic": {
            "start_phase": "contracted",    # Aşağıda başla
            "count_on": "return_contracted", # Aşağı inince say
            "direction": "extend_first"     # Önce kaldır sonra indir
        },
        "primary_angle": "shoulder",
        "form_tips": [
            "Kollar yandan kaldır",
            "Omuz hizasına kadar",
            "Omuzları silkme"
        ]
    },
    "tricep_extensions": {
        "joints": {
            "left": (11, 13, 15),
            "right": (12, 14, 16)
        },
        "required_landmarks": {11, 12, 13, 14, 15, 16},
        "rep_threshold": {
            "contracted": 60,   # Kol bükülü
            "extended": 160     # Kol açık
        },
        "rep_logic": {
            "start_phase": "contracted",    # Bükülü başla
            "count_on": "return_contracted", # Tekrar bükülünce say
            "direction": "extend_first"     # Önce aç sonra bük
        },
        "primary_angle": "elbow",
        "form_tips": [
            "Dirsek sabit",
            "Tam aç",
            "Yavaş indir"
        ]
    },
    "dumbbell_rows": {
        "joints": {
            "left": (11, 13, 15),
            "right": (12, 14, 16)
        },
        "required_landmarks": {11, 12, 13, 14, 15, 16, 23, 24},
        "rep_threshold": {
            "contracted": 50,   # Çekilmiş (dirsek bükük)
            "extended": 150     # Uzanmış
        },
        "rep_logic": {
            "start_phase": "extended",     # Uzanmış başla
            "count_on": "return_extended", # Tekrar uzanınca say
            "direction": "contract_first"  # Önce çek sonra bırak
        },
        "primary_angle": "elbow",
        "form_tips": [
            "Sırt düz",
            "Dirsek vücuda yakın",
            "Kürek kemiği sık"
        ]
    },
    "dumbbell_shoulder_press": {
        "joints": {
            "left": (11, 13, 15),
            "right": (12, 14, 16)
        },
        "required_landmarks": {11, 12, 13, 14, 15, 16, 23, 24},
        "rep_threshold": {
            "contracted": 90,   # Kollar omuz hizasında
            "extended": 160     # Kollar tam yukarı
        },
        "rep_logic": {
            "start_phase": "contracted",    # Omuz hizasında başla
            "count_on": "return_contracted", # Tekrar omuz hizasına inince say
            "direction": "extend_first"     # Önce yukarı it sonra indir
        },
        "primary_angle": "elbow",
        "form_tips": [
            "Core sıkı",
            "Arkaya yaslanma",
            "Tam yukarı it"
        ]
    }
}


def compute_angle(p1, p2, p3):
    """Compute angle at p2."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    
    if mag1 * mag2 == 0:
        return 0
    
    cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def get_landmark_coords(landmarks, idx, w, h):
    """Get landmark coordinates in pixels."""
    lm = landmarks.landmark[idx]
    return (lm.x * w, lm.y * h)


def calculate_form_score(current_angle, target_up, target_down, phase):
    """
    Calculate form score based on angle.
    Good form = angle is within the valid exercise range.
    """
    min_angle = min(target_up, target_down)
    max_angle = max(target_up, target_down)
    
    # Check if angle is within valid range
    if min_angle <= current_angle <= max_angle:
        # Perfect - within range
        # Extra points for hitting the extremes
        if current_angle <= min_angle + 15 or current_angle >= max_angle - 15:
            return 100  # At the peak positions
        else:
            return 85   # Good - in motion
    else:
        # Outside range - penalize
        if current_angle < min_angle:
            diff = min_angle - current_angle
        else:
            diff = current_angle - max_angle
        return max(0, 70 - diff * 2)


class FormAnalyzer:
    """Advanced form analysis for exercises."""
    
    # States
    STATE_DETECTING = "detecting"      # Looking for body
    STATE_CALIBRATING = "calibrating"  # Body found, calibrating
    STATE_READY = "ready"              # Ready to exercise
    STATE_TRACKING = "tracking"        # Exercise in progress
    
    def __init__(self):
        self.state = self.STATE_DETECTING
        self.calibration_frames = 0
        
        # Body proportions (for relative tolerance)
        self.shoulder_width = None
        self.torso_height = None
        
        # Initial positions (normalized by body proportions)
        self.initial_elbow_x = {"left": None, "right": None}
        self.initial_shoulder_y = {"left": None, "right": None}
        
        # Tracking
        self.visible_landmarks = set()
        self.current_exercise = None
        
        # Required landmarks per exercise
        self.exercise_landmarks = {
            "bicep_curls": {11, 12, 13, 14, 15, 16},  # shoulders, elbows, wrists
            "squats": {23, 24, 25, 26, 27, 28},        # hips, knees, ankles
            "lunges": {23, 24, 25, 26, 27, 28},        # hips, knees, ankles
            "pushups": {11, 12, 13, 14, 23, 24},       # shoulders, elbows, hips
            "lateral_shoulder_raises": {11, 12, 13, 14, 15, 16},  # shoulders, elbows, wrists
            "tricep_extensions": {11, 12, 13, 14, 15, 16},        # shoulders, elbows, wrists
            "dumbbell_rows": {11, 12, 13, 14, 23, 24},            # shoulders, elbows, hips
            "dumbbell_shoulder_press": {11, 12, 13, 14, 15, 16, 23, 24},  # upper body
        }
        self.required_landmarks = {11, 12, 23, 24}  # Default: shoulders and hips
        
    def set_exercise(self, exercise):
        """Set the current exercise and update required landmarks from config."""
        self.current_exercise = exercise
        # Use landmarks from EXERCISE_CONFIG if available
        if exercise in EXERCISE_CONFIG and "required_landmarks" in EXERCISE_CONFIG[exercise]:
            self.required_landmarks = EXERCISE_CONFIG[exercise]["required_landmarks"]
        elif exercise in self.exercise_landmarks:
            self.required_landmarks = self.exercise_landmarks[exercise]
        else:
            self.required_landmarks = {11, 12, 23, 24}  # Default
    
    def get_status_text(self):
        """Get current status for display."""
        if self.state == self.STATE_DETECTING:
            return "VUCUT ARANYOR...", (0, 255, 255)
        elif self.state == self.STATE_CALIBRATING:
            progress = int(self.calibration_frames / 30 * 100)
            return f"KALIBRE EDILIYOR... {progress}%", (0, 255, 255)
        elif self.state == self.STATE_READY:
            return "HAZIR! Harekete basla", (0, 255, 0)
        else:
            return "TAKIP EDILIYOR", (0, 255, 0)
    
    def check_body_visible(self, landmarks):
        """Check if required body parts are visible."""
        self.visible_landmarks = set()
        
        for idx in self.required_landmarks:
            lm = landmarks.landmark[idx]
            # Check visibility score
            if lm.visibility > 0.5:
                self.visible_landmarks.add(idx)
        
        return len(self.visible_landmarks) >= len(self.required_landmarks) * 0.7
    
    def calculate_body_proportions(self, landmarks, w, h):
        """Calculate body proportions for relative measurements."""
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        
        # Shoulder width in pixels
        self.shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
        
        # Torso height (shoulder to hip)
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        self.torso_height = abs(hip_center_y - shoulder_center_y) * h
        
        return self.shoulder_width > 50 and self.torso_height > 50
    
    def update(self, landmarks, w, h):
        """Update state machine. Returns True when calibration is complete."""
        
        # STATE: DETECTING
        if self.state == self.STATE_DETECTING:
            if self.check_body_visible(landmarks):
                if self.calculate_body_proportions(landmarks, w, h):
                    self.state = self.STATE_CALIBRATING
                    self.calibration_frames = 0
                    print("   ✅ Vücut algılandı! Kalibrasyon başlıyor...")
            return False
        
        # STATE: CALIBRATING
        elif self.state == self.STATE_CALIBRATING:
            if not self.check_body_visible(landmarks):
                # Lost body, go back to detecting
                self.state = self.STATE_DETECTING
                self.calibration_frames = 0
                return False
            
            left_elbow = landmarks.landmark[13]
            right_elbow = landmarks.landmark[14]
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
            
            if self.calibration_frames == 0:
                self.initial_elbow_x["left"] = left_elbow.x * w
                self.initial_elbow_x["right"] = right_elbow.x * w
                self.initial_shoulder_y["left"] = left_shoulder.y * h
                self.initial_shoulder_y["right"] = right_shoulder.y * h
            else:
                # Running average
                alpha = 0.85
                self.initial_elbow_x["left"] = alpha * self.initial_elbow_x["left"] + (1-alpha) * left_elbow.x * w
                self.initial_elbow_x["right"] = alpha * self.initial_elbow_x["right"] + (1-alpha) * right_elbow.x * w
                self.initial_shoulder_y["left"] = alpha * self.initial_shoulder_y["left"] + (1-alpha) * left_shoulder.y * h
                self.initial_shoulder_y["right"] = alpha * self.initial_shoulder_y["right"] + (1-alpha) * right_shoulder.y * h
            
            self.calibration_frames += 1
            
            if self.calibration_frames >= 30:
                self.state = self.STATE_READY
                print("   ✅ Form kalibre edildi! Harekete başlayabilirsin.")
                return True
            
            return False
        
        # STATE: READY or TRACKING
        else:
            if not self.check_body_visible(landmarks):
                return True  # Still calibrated, just lost tracking momentarily
            
            if self.state == self.STATE_READY:
                self.state = self.STATE_TRACKING
            
            return True
    
    @property
    def calibrated(self):
        return self.state in [self.STATE_READY, self.STATE_TRACKING]
    
    def check_bicep_curl_form(self, landmarks, w, h):
        """
        Check bicep curl form quality with RELATIVE tolerances.
        Checks both angles and relative node positions.
        Returns: (overall_score, issues_list)
        """
        issues = []
        scores = []
        
        if not self.calibrated or self.shoulder_width is None:
            return 100, []
        
        # Get all relevant landmarks
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_elbow = landmarks.landmark[13]
        right_elbow = landmarks.landmark[14]
        left_wrist = landmarks.landmark[15]
        right_wrist = landmarks.landmark[16]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        
        # Convert to pixel coordinates
        l_shoulder = (left_shoulder.x * w, left_shoulder.y * h)
        r_shoulder = (right_shoulder.x * w, right_shoulder.y * h)
        l_elbow = (left_elbow.x * w, left_elbow.y * h)
        r_elbow = (right_elbow.x * w, right_elbow.y * h)
        l_wrist = (left_wrist.x * w, left_wrist.y * h)
        r_wrist = (right_wrist.x * w, right_wrist.y * h)
        l_hip = (left_hip.x * w, left_hip.y * h)
        r_hip = (right_hip.x * w, right_hip.y * h)
        
        # RELATIVE tolerances based on body proportions
        elbow_drift_tolerance = self.shoulder_width * 0.25
        shoulder_rise_tolerance = self.torso_height * 0.10
        
        # ============================================
        # CHECK 1: Elbow X stability (shouldn't drift forward/backward)
        # ============================================
        left_elbow_drift = abs(l_elbow[0] - self.initial_elbow_x["left"])
        right_elbow_drift = abs(r_elbow[0] - self.initial_elbow_x["right"])
        
        if left_elbow_drift > elbow_drift_tolerance:
            severity = min(50, (left_elbow_drift / elbow_drift_tolerance - 1) * 30)
            issues.append("Sol dirsek one/arkaya kaciyor")
            scores.append(max(50, 100 - severity))
        else:
            scores.append(100)
            
        if right_elbow_drift > elbow_drift_tolerance:
            severity = min(50, (right_elbow_drift / elbow_drift_tolerance - 1) * 30)
            issues.append("Sag dirsek one/arkaya kaciyor")
            scores.append(max(50, 100 - severity))
        else:
            scores.append(100)
        
        # ============================================
        # CHECK 2: Shoulder stability (shouldn't rise)
        # ============================================
        left_shoulder_rise = self.initial_shoulder_y["left"] - l_shoulder[1]
        right_shoulder_rise = self.initial_shoulder_y["right"] - r_shoulder[1]
        
        if left_shoulder_rise > shoulder_rise_tolerance:
            severity = min(50, (left_shoulder_rise / shoulder_rise_tolerance - 1) * 30)
            issues.append("Sol omuz kalkiyor")
            scores.append(max(50, 100 - severity))
        else:
            scores.append(100)
            
        if right_shoulder_rise > shoulder_rise_tolerance:
            severity = min(50, (right_shoulder_rise / shoulder_rise_tolerance - 1) * 30)
            issues.append("Sag omuz kalkiyor")
            scores.append(max(50, 100 - severity))
        else:
            scores.append(100)
        
        # ============================================
        # CHECK 3: Elbow should be below shoulder (relative Y position)
        # This is a CRITICAL check - elbow above shoulder = very bad form
        # ============================================
        # Elbow Y should be greater than shoulder Y (lower on screen)
        # Allow small tolerance (5% of torso height)
        elbow_above_tolerance = self.torso_height * 0.05
        
        left_elbow_above = l_shoulder[1] - l_elbow[1]  # Positive = elbow above shoulder
        right_elbow_above = r_shoulder[1] - r_elbow[1]
        
        if left_elbow_above > elbow_above_tolerance:
            # How much above? More = worse
            severity = min(60, (left_elbow_above / self.torso_height) * 100)
            issues.append("Sol dirsek omuzun ustunde!")
            scores.append(max(20, 100 - severity))  # Can go as low as 20
        else:
            scores.append(100)
            
        if right_elbow_above > elbow_above_tolerance:
            severity = min(60, (right_elbow_above / self.torso_height) * 100)
            issues.append("Sag dirsek omuzun ustunde!")
            scores.append(max(20, 100 - severity))
        else:
            scores.append(100)
        
        # ============================================
        # CHECK 4: Elbow close to body (X position relative to hip)
        # Elbow X should be close to shoulder X (not flared out)
        # ============================================
        elbow_flare_tolerance = self.shoulder_width * 0.30
        
        left_elbow_flare = abs(l_elbow[0] - l_shoulder[0])
        right_elbow_flare = abs(r_elbow[0] - r_shoulder[0])
        
        if left_elbow_flare > elbow_flare_tolerance:
            severity = min(40, (left_elbow_flare / elbow_flare_tolerance - 1) * 25)
            issues.append("Sol dirsek vucuttan uzak")
            scores.append(max(60, 100 - severity))
        else:
            scores.append(100)
            
        if right_elbow_flare > elbow_flare_tolerance:
            severity = min(40, (right_elbow_flare / elbow_flare_tolerance - 1) * 25)
            issues.append("Sag dirsek vucuttan uzak")
            scores.append(max(60, 100 - severity))
        else:
            scores.append(100)
        
        # ============================================
        # CHECK 5: Upper arm angle (shoulder-elbow line should be ~vertical)
        # ============================================
        def get_angle_from_vertical(p1, p2):
            """Get angle of line from vertical (0 = perfectly vertical)"""
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            if dy < 1:
                return 90  # Horizontal
            return np.degrees(np.arctan(dx / dy))
        
        left_upper_arm_angle = get_angle_from_vertical(l_shoulder, l_elbow)
        right_upper_arm_angle = get_angle_from_vertical(r_shoulder, r_elbow)
        
        # Upper arm should be within 25 degrees of vertical
        upper_arm_tolerance = 30
        
        if left_upper_arm_angle > upper_arm_tolerance:
            severity = min(40, (left_upper_arm_angle - upper_arm_tolerance) * 1.5)
            issues.append(f"Sol ust kol acik ({int(left_upper_arm_angle)}°)")
            scores.append(max(60, 100 - severity))
        else:
            scores.append(100)
            
        if right_upper_arm_angle > upper_arm_tolerance:
            severity = min(40, (right_upper_arm_angle - upper_arm_tolerance) * 1.5)
            issues.append(f"Sag ust kol acik ({int(right_upper_arm_angle)}°)")
            scores.append(max(60, 100 - severity))
        else:
            scores.append(100)
        
        # ============================================
        # CHECK 6: Wrist-Hip distance (hand should move toward shoulder, not stay at hip)
        # During curl, wrist Y should decrease (move up)
        # ============================================
        # Calculate wrist to hip vertical distance (normalized by torso height)
        left_wrist_hip_dist = (l_hip[1] - l_wrist[1]) / self.torso_height
        right_wrist_hip_dist = (r_hip[1] - r_wrist[1]) / self.torso_height
        
        # When at rest, wrist is near hip (distance ~0)
        # When curled, wrist is near shoulder (distance ~1)
        # This is informational, not penalized - just tracked
        
        # ============================================
        # CHECK 7: Landmark visibility
        # ============================================
        missing = self.required_landmarks - self.visible_landmarks
        if len(missing) > 2:
            issues.append(f"Bazi noktalar gorunmuyor ({len(missing)})")
            scores.append(70)
        
        overall_score = np.mean(scores) if scores else 100
        return overall_score, issues
    
    def check_squats_form(self, landmarks, w, h):
        """
        Check squats form quality.
        Key checks: knees over toes, back straight, depth
        """
        issues = []
        scores = []
        
        if not self.calibrated or self.shoulder_width is None:
            return 100, []
        
        # Get landmarks
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        left_knee = landmarks.landmark[25]
        right_knee = landmarks.landmark[26]
        left_ankle = landmarks.landmark[27]
        right_ankle = landmarks.landmark[28]
        
        # Convert to pixels
        l_shoulder = (left_shoulder.x * w, left_shoulder.y * h)
        r_shoulder = (right_shoulder.x * w, right_shoulder.y * h)
        l_hip = (left_hip.x * w, left_hip.y * h)
        r_hip = (right_hip.x * w, right_hip.y * h)
        l_knee = (left_knee.x * w, left_knee.y * h)
        r_knee = (right_knee.x * w, right_knee.y * h)
        l_ankle = (left_ankle.x * w, left_ankle.y * h)
        r_ankle = (right_ankle.x * w, right_ankle.y * h)
        
        # CHECK 1: Knees shouldn't go too far forward (past toes)
        knee_forward_tolerance = self.shoulder_width * 0.3
        
        left_knee_forward = l_knee[0] - l_ankle[0]  # How far knee is in front of ankle
        right_knee_forward = r_knee[0] - r_ankle[0]
        
        # This depends on camera angle, so we check relative movement
        if abs(left_knee_forward) > knee_forward_tolerance * 2:
            issues.append("Sol diz cok onde")
            scores.append(70)
        else:
            scores.append(100)
            
        # CHECK 2: Knees should track over toes (not cave in)
        # Compare knee X to ankle X - they should be similar
        knee_cave_tolerance = self.shoulder_width * 0.15
        
        left_knee_cave = abs(l_knee[0] - l_ankle[0])
        right_knee_cave = abs(r_knee[0] - r_ankle[0])
        
        # CHECK 3: Hip should go back (hip behind heels in squat)
        # CHECK 4: Back angle - shoulders and hips should maintain alignment
        
        # Shoulder-hip alignment (back straightness)
        shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
        hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        
        # Back angle from vertical
        back_dx = abs(shoulder_center[0] - hip_center[0])
        back_dy = abs(shoulder_center[1] - hip_center[1])
        
        if back_dy > 10:
            back_angle = np.degrees(np.arctan(back_dx / back_dy))
            if back_angle > 45:
                issues.append(f"Sirt cok egik ({int(back_angle)}°)")
                scores.append(max(50, 100 - back_angle))
            else:
                scores.append(100)
        else:
            scores.append(100)
        
        # CHECK 5: Symmetry - both knees at similar depth
        knee_height_diff = abs(l_knee[1] - r_knee[1])
        if knee_height_diff > self.torso_height * 0.1:
            issues.append("Diz seviyeleri farkli")
            scores.append(80)
        else:
            scores.append(100)
        
        overall_score = np.mean(scores) if scores else 100
        return overall_score, issues
    
    def check_pushups_form(self, landmarks, w, h):
        """
        Check pushups form quality.
        Key checks: body alignment, depth, elbow position
        """
        issues = []
        scores = []
        
        if not self.calibrated or self.shoulder_width is None:
            return 100, []
        
        # Get landmarks
        nose = landmarks.landmark[0]
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_elbow = landmarks.landmark[13]
        right_elbow = landmarks.landmark[14]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        left_ankle = landmarks.landmark[27]
        right_ankle = landmarks.landmark[28]
        
        # Convert to pixels
        l_shoulder = (left_shoulder.x * w, left_shoulder.y * h)
        r_shoulder = (right_shoulder.x * w, right_shoulder.y * h)
        l_hip = (left_hip.x * w, left_hip.y * h)
        r_hip = (right_hip.x * w, right_hip.y * h)
        l_ankle = (left_ankle.x * w, left_ankle.y * h)
        r_ankle = (right_ankle.x * w, right_ankle.y * h)
        
        # CHECK 1: Body should be in a straight line (shoulder-hip-ankle alignment)
        shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
        hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        ankle_center = ((l_ankle[0] + r_ankle[0]) / 2, (l_ankle[1] + r_ankle[1]) / 2)
        
        # Check if hip sags or pikes (compare hip Y to line between shoulder and ankle)
        if abs(shoulder_center[0] - ankle_center[0]) > 10:
            expected_hip_y = shoulder_center[1] + (hip_center[0] - shoulder_center[0]) / (ankle_center[0] - shoulder_center[0]) * (ankle_center[1] - shoulder_center[1])
            hip_deviation = hip_center[1] - expected_hip_y
            
            deviation_tolerance = self.torso_height * 0.15
            if abs(hip_deviation) > deviation_tolerance:
                if hip_deviation > 0:
                    issues.append("Kalca dusuk (core sikistir)")
                else:
                    issues.append("Kalca yukarda")
                scores.append(max(60, 100 - abs(hip_deviation) / deviation_tolerance * 30))
            else:
                scores.append(100)
        else:
            scores.append(100)
        
        # CHECK 2: Elbows shouldn't flare out too much (check elbow-shoulder angle)
        # This is approximate since we can't see depth well
        scores.append(100)  # Placeholder
        
        overall_score = np.mean(scores) if scores else 100
        return overall_score, issues
    
    def check_lateral_raises_form(self, landmarks, w, h):
        """
        Check lateral shoulder raises form quality.
        Key checks: arms raising to sides, controlled movement
        """
        issues = []
        scores = []
        
        if not self.calibrated or self.shoulder_width is None:
            return 100, []
        
        # Get landmarks
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_elbow = landmarks.landmark[13]
        right_elbow = landmarks.landmark[14]
        left_wrist = landmarks.landmark[15]
        right_wrist = landmarks.landmark[16]
        
        # Convert to pixels
        l_shoulder = (left_shoulder.x * w, left_shoulder.y * h)
        r_shoulder = (right_shoulder.x * w, right_shoulder.y * h)
        l_elbow = (left_elbow.x * w, left_elbow.y * h)
        r_elbow = (right_elbow.x * w, right_elbow.y * h)
        l_wrist = (left_wrist.x * w, left_wrist.y * h)
        r_wrist = (right_wrist.x * w, right_wrist.y * h)
        
        # CHECK 1: Arms should raise to the SIDES (not front)
        # Wrist X should be far from shoulder X when raised
        
        # CHECK 2: Elbows should have slight bend (not locked)
        # CHECK 3: Shoulders shouldn't shrug up
        
        # Check shoulder rise
        left_shoulder_rise = self.initial_shoulder_y["left"] - l_shoulder[1]
        right_shoulder_rise = self.initial_shoulder_y["right"] - r_shoulder[1]
        
        shoulder_tolerance = self.torso_height * 0.08
        
        if left_shoulder_rise > shoulder_tolerance:
            issues.append("Sol omuz kalkiyor")
            scores.append(max(60, 100 - left_shoulder_rise / shoulder_tolerance * 30))
        else:
            scores.append(100)
            
        if right_shoulder_rise > shoulder_tolerance:
            issues.append("Sag omuz kalkiyor")
            scores.append(max(60, 100 - right_shoulder_rise / shoulder_tolerance * 30))
        else:
            scores.append(100)
        
        # CHECK 4: Arms should be symmetric
        left_arm_height = l_shoulder[1] - l_wrist[1]
        right_arm_height = r_shoulder[1] - r_wrist[1]
        
        height_diff = abs(left_arm_height - right_arm_height)
        if height_diff > self.torso_height * 0.15:
            issues.append("Kollar esit yukseklikte degil")
            scores.append(80)
        else:
            scores.append(100)
        
        overall_score = np.mean(scores) if scores else 100
        return overall_score, issues
    
    def check_tricep_extensions_form(self, landmarks, w, h):
        """
        Check tricep extensions form quality.
        Key checks: elbow stability, full extension
        """
        issues = []
        scores = []
        
        if not self.calibrated or self.shoulder_width is None:
            return 100, []
        
        # Similar to bicep curls - check elbow stability
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_elbow = landmarks.landmark[13]
        right_elbow = landmarks.landmark[14]
        
        l_shoulder = (left_shoulder.x * w, left_shoulder.y * h)
        r_shoulder = (right_shoulder.x * w, right_shoulder.y * h)
        l_elbow = (left_elbow.x * w, left_elbow.y * h)
        r_elbow = (right_elbow.x * w, right_elbow.y * h)
        
        # CHECK 1: Elbow position stability
        elbow_drift_tolerance = self.shoulder_width * 0.25
        
        if self.initial_elbow_x["left"]:
            left_elbow_drift = abs(l_elbow[0] - self.initial_elbow_x["left"])
            if left_elbow_drift > elbow_drift_tolerance:
                issues.append("Sol dirsek hareket ediyor")
                scores.append(max(60, 100 - left_elbow_drift / elbow_drift_tolerance * 30))
            else:
                scores.append(100)
        
        if self.initial_elbow_x["right"]:
            right_elbow_drift = abs(r_elbow[0] - self.initial_elbow_x["right"])
            if right_elbow_drift > elbow_drift_tolerance:
                issues.append("Sag dirsek hareket ediyor")
                scores.append(max(60, 100 - right_elbow_drift / elbow_drift_tolerance * 30))
            else:
                scores.append(100)
        
        overall_score = np.mean(scores) if scores else 100
        return overall_score, issues
    
    def check_dumbbell_rows_form(self, landmarks, w, h):
        """
        Check dumbbell rows form quality.
        Key checks: back position, elbow path
        """
        issues = []
        scores = []
        
        if not self.calibrated or self.shoulder_width is None:
            return 100, []
        
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        
        l_shoulder = (left_shoulder.x * w, left_shoulder.y * h)
        r_shoulder = (right_shoulder.x * w, right_shoulder.y * h)
        l_hip = (left_hip.x * w, left_hip.y * h)
        r_hip = (right_hip.x * w, right_hip.y * h)
        
        # CHECK 1: Back should be relatively flat (bent over position)
        shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
        hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        
        # Back angle
        back_dx = shoulder_center[0] - hip_center[0]
        back_dy = shoulder_center[1] - hip_center[1]
        
        if abs(back_dy) > 10:
            back_angle = np.degrees(np.arctan(abs(back_dx) / abs(back_dy)))
            # For rows, back should be angled but not too much
            if back_angle > 60:
                issues.append("Sirt cok dik")
                scores.append(80)
            else:
                scores.append(100)
        else:
            scores.append(100)
        
        overall_score = np.mean(scores) if scores else 100
        return overall_score, issues
    
    def check_shoulder_press_form(self, landmarks, w, h):
        """
        Check dumbbell shoulder press form quality.
        Key checks: core stability, arm path
        """
        issues = []
        scores = []
        
        if not self.calibrated or self.shoulder_width is None:
            return 100, []
        
        left_shoulder = landmarks.landmark[11]
        right_shoulder = landmarks.landmark[12]
        left_elbow = landmarks.landmark[13]
        right_elbow = landmarks.landmark[14]
        left_wrist = landmarks.landmark[15]
        right_wrist = landmarks.landmark[16]
        left_hip = landmarks.landmark[23]
        right_hip = landmarks.landmark[24]
        
        l_shoulder = (left_shoulder.x * w, left_shoulder.y * h)
        r_shoulder = (right_shoulder.x * w, right_shoulder.y * h)
        l_wrist = (left_wrist.x * w, left_wrist.y * h)
        r_wrist = (right_wrist.x * w, right_wrist.y * h)
        l_hip = (left_hip.x * w, left_hip.y * h)
        r_hip = (right_hip.x * w, right_hip.y * h)
        
        # CHECK 1: Core stability - torso shouldn't lean back too much
        shoulder_center = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
        hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
        
        lean_back = shoulder_center[0] - hip_center[0]  # Positive = leaning back
        lean_tolerance = self.shoulder_width * 0.3
        
        if lean_back > lean_tolerance:
            issues.append("Arkaya yaslanma")
            scores.append(max(60, 100 - lean_back / lean_tolerance * 30))
        else:
            scores.append(100)
        
        # CHECK 2: Arms should be symmetric
        left_wrist_height = l_shoulder[1] - l_wrist[1]
        right_wrist_height = r_shoulder[1] - r_wrist[1]
        
        height_diff = abs(left_wrist_height - right_wrist_height)
        if height_diff > self.torso_height * 0.15:
            issues.append("Kollar esit degil")
            scores.append(80)
        else:
            scores.append(100)
        
        overall_score = np.mean(scores) if scores else 100
        return overall_score, issues
    
    def check_form(self, exercise, landmarks, w, h):
        """
        Main form checking dispatcher.
        Routes to exercise-specific form check.
        """
        if exercise == "bicep_curls":
            return self.check_bicep_curl_form(landmarks, w, h)
        elif exercise == "squats":
            return self.check_squats_form(landmarks, w, h)
        elif exercise == "pushups":
            return self.check_pushups_form(landmarks, w, h)
        elif exercise == "lateral_shoulder_raises":
            return self.check_lateral_raises_form(landmarks, w, h)
        elif exercise == "tricep_extensions":
            return self.check_tricep_extensions_form(landmarks, w, h)
        elif exercise == "dumbbell_rows":
            return self.check_dumbbell_rows_form(landmarks, w, h)
        elif exercise == "dumbbell_shoulder_press":
            return self.check_shoulder_press_form(landmarks, w, h)
        elif exercise == "lunges":
            return self.check_squats_form(landmarks, w, h)  # Similar to squats
        else:
            return 100, []  # No specific checks


class RepCounter:
    """
    Counts exercise repetitions with exercise-specific logic.
    
    Rep Logic:
    - contracted: angle when muscle is contracted (smaller angle for curls, bent knee for squats)
    - extended: angle when muscle is extended (straight arm, straight leg)
    - direction: "contract_first" or "extend_first"
    - count_on: "return_extended" or "return_contracted"
    """
    
    def __init__(self, config):
        """
        Initialize with exercise config.
        
        Args:
            config: Exercise configuration dict with rep_threshold and rep_logic
        """
        thresholds = config.get("rep_threshold", {})
        self.contracted_threshold = thresholds.get("contracted", thresholds.get("up", 90))
        self.extended_threshold = thresholds.get("extended", thresholds.get("down", 160))
        
        rep_logic = config.get("rep_logic", {})
        self.start_phase = rep_logic.get("start_phase", "extended")
        self.count_on = rep_logic.get("count_on", "return_extended")
        self.direction = rep_logic.get("direction", "contract_first")
        
        self.count = 0
        self.phase = self.start_phase  # "extended" or "contracted"
        self.last_angle = None
        self.form_scores = []
        
        # Determine which threshold is smaller
        self.is_contracted_smaller = self.contracted_threshold < self.extended_threshold
        
    def get_phase_name(self):
        """Get human-readable phase name."""
        if self.phase == "extended":
            return "ACIK"
        else:
            return "BUKUK"
    
    def update(self, angle):
        """
        Update counter with new angle.
        Returns: (count, avg_form_score) when rep completed, else (None, None)
        """
        if self.last_angle is None:
            self.last_angle = angle
            return (None, None)
        
        completed_rep = False
        
        # Phase transition logic
        if self.phase == "extended":
            # Check if transitioning to contracted
            if self.is_contracted_smaller:
                # Angle decreases to contract (e.g., bicep curl)
                if angle <= self.contracted_threshold:
                    self.phase = "contracted"
                    if self.count_on == "return_contracted":
                        completed_rep = True
            else:
                # Angle increases to contract (unusual, but possible)
                if angle >= self.contracted_threshold:
                    self.phase = "contracted"
                    if self.count_on == "return_contracted":
                        completed_rep = True
                        
        elif self.phase == "contracted":
            # Check if transitioning to extended
            if self.is_contracted_smaller:
                # Angle increases to extend
                if angle >= self.extended_threshold:
                    self.phase = "extended"
                    if self.count_on == "return_extended":
                        completed_rep = True
            else:
                # Angle decreases to extend
                if angle <= self.extended_threshold:
                    self.phase = "extended"
                    if self.count_on == "return_extended":
                        completed_rep = True
        
        self.last_angle = angle
        
        if completed_rep:
            self.count += 1
            avg_score = np.mean(self.form_scores) if self.form_scores else 0
            self.form_scores = []
            return (self.count, avg_score)
        
        return (None, None)
    
    def add_form_score(self, score):
        self.form_scores.append(score)


def main():
    # Get exercise from command line
    exercise = "bicep_curls"
    if len(sys.argv) > 1:
        exercise = sys.argv[1]
    
    if exercise not in EXERCISE_CONFIG:
        print(f"❌ Unknown exercise: {exercise}")
        print(f"Available: {list(EXERCISE_CONFIG.keys())}")
        return
    
    config = EXERCISE_CONFIG[exercise]
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize rep counter with exercise-specific config
    counter = RepCounter(config)
    
    # Initialize form analyzer
    form_analyzer = FormAnalyzer()
    form_analyzer.set_exercise(exercise)  # Set required landmarks for this exercise
    
    # Stats
    all_form_scores = []
    start_time = time.time()
    current_issues = []
    
    print("\n" + "=" * 60)
    print(f"     REAL-TIME EXERCISE TRACKER: {exercise.upper()}")
    print("=" * 60)
    print(f"\n   Form Tips:")
    for tip in config["form_tips"]:
        print(f"      • {tip}")
    print(f"\n   📷 Kameraya tam görünecek şekilde durun")
    print(f"   ⏳ Sistem vücudunuzu algılayıp kalibre edecek")
    print(f"\n   Press 'q' to quit")
    print("=" * 60 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        # Default values
        angle = 0
        form_score = 0
        advanced_score = 100
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            
            # Update form analyzer state machine
            form_analyzer.update(landmarks, frame_width, frame_height)
            
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame, landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Only process angles, form, and reps AFTER calibration
            if form_analyzer.calibrated:
                # CHECK: Are required landmarks visible?
                body_visible = form_analyzer.check_body_visible(landmarks)
                
                if not body_visible:
                    # Body not visible - don't process, show warning
                    form_score = 0
                    current_issues = ["VUCUT GORUNMUYOR!"]
                    # Don't update rep counter when body not visible
                else:
                    # Body visible - proceed with tracking
                    
                    # Calculate angles for both sides
                    angles = []
                    for side, joints in config["joints"].items():
                        p1 = get_landmark_coords(landmarks, joints[0], frame_width, frame_height)
                        p2 = get_landmark_coords(landmarks, joints[1], frame_width, frame_height)
                        p3 = get_landmark_coords(landmarks, joints[2], frame_width, frame_height)
                        
                        angle_side = compute_angle(p1, p2, p3)
                        angles.append(angle_side)
                        
                        # Draw angle on frame
                        cv2.putText(frame, f"{int(angle_side)}°",
                                   (int(p2[0]) + 10, int(p2[1])),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Use average of both sides
                    angle = np.mean(angles)
                    
                    # Check symmetry
                    current_issues = []  # Reset issues each frame
                    if len(angles) == 2:
                        symmetry_diff = abs(angles[0] - angles[1])
                        if symmetry_diff > 20:
                            current_issues.append(f"Simetri bozuk ({int(symmetry_diff)}°)")
                    
                    # Advanced form check (position-based) for ALL exercises
                    advanced_score, advanced_issues = form_analyzer.check_form(
                        exercise, landmarks, frame_width, frame_height
                    )
                    current_issues.extend(advanced_issues)
                    
                    # Calculate angle-based score
                    thresholds = config["rep_threshold"]
                    contracted = thresholds.get("contracted", thresholds.get("up", 90))
                    extended = thresholds.get("extended", thresholds.get("down", 160))
                    angle_score = calculate_form_score(
                        angle,
                        contracted,
                        extended,
                        counter.phase
                    )
                    
                    # Combine scores based on issues
                    if len(advanced_issues) >= 2:
                        # Multiple issues = bad form, weight advanced checks more
                        form_score = advanced_score * 0.8
                    elif len(advanced_issues) == 1:
                        # One issue = partial penalty
                        form_score = (angle_score * 0.3 + advanced_score * 0.7)
                    else:
                        # No issues = combine equally
                        form_score = (angle_score + advanced_score) / 2
                    
                    counter.add_form_score(form_score)
                    
                    # Update rep counter (only when body visible)
                    result = counter.update(angle)
                    if result[0] is not None:
                        new_count, rep_score = result
                        all_form_scores.append(rep_score)
                        print(f"   Rep {new_count}: Form Score = {rep_score:.1f}%")
        
        # Draw UI
        # Background panel - made taller to show issues
        panel_height = 240 if current_issues else 200
        cv2.rectangle(frame, (10, 10), (420, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (420, panel_height), (255, 255, 255), 2)
        
        # Status text from form analyzer
        status_text, status_color = form_analyzer.get_status_text()
        cv2.putText(frame, status_text,
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Exercise name (always show)
        cv2.putText(frame, exercise.replace("_", " ").upper(),
                   (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Only show rep/form info if calibrated
        if form_analyzer.calibrated:
            # Check if body is visible (for display purposes)
            body_visible = len(form_analyzer.visible_landmarks) >= len(form_analyzer.required_landmarks) * 0.7
            
            if not body_visible:
                # Body NOT visible - show warning
                cv2.putText(frame, "VUCUT GORUNMUYOR!",
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Kameraya tam gorunun",
                           (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
                cv2.putText(frame, f"REPS: {counter.count}",
                           (280, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
                # Form bar at 0
                cv2.rectangle(frame, (20, 125), (370, 150), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, 125), (370, 150), (255, 255, 255), 2)
                cv2.putText(frame, "Form: 0%",
                           (160, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            else:
                # Body visible - show normal UI
                # Rep counter
                cv2.putText(frame, f"REPS: {counter.count}",
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # Current angle
                cv2.putText(frame, f"Angle: {int(angle)}°",
                           (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Phase indicator
                phase_color = (0, 255, 0) if counter.phase == "contracted" else (0, 165, 255)
                cv2.putText(frame, f"Phase: {counter.get_phase_name()}",
                           (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
                
                # Form score bar
                bar_width = int(min(form_score, 100) * 3.5)
                bar_color = (0, 255, 0) if form_score > 70 else (0, 165, 255) if form_score > 40 else (0, 0, 255)
                cv2.rectangle(frame, (20, 125), (20 + bar_width, 150), bar_color, -1)
                cv2.rectangle(frame, (20, 125), (370, 150), (255, 255, 255), 2)
                cv2.putText(frame, f"Form: {int(form_score)}%",
                           (160, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Show waiting message
            cv2.putText(frame, "Vucut algilanmasi bekleniyor...",
                       (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            cv2.putText(frame, "Kameraya tam gorunun",
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Average form score
        if all_form_scores:
            avg = np.mean(all_form_scores)
            cv2.putText(frame, f"Avg: {avg:.0f}%",
                       (300, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show form issues/warnings
        if current_issues and form_analyzer.calibrated:
            cv2.putText(frame, "UYARILAR:",
                       (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            for i, issue in enumerate(current_issues[:2]):  # Show max 2 issues
                cv2.putText(frame, f"• {issue}",
                           (20, 195 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        
        cv2.imshow(f"Exercise Tracker - {exercise}", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    
    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("                    WORKOUT SUMMARY")
    print("=" * 60)
    print(f"\n   Exercise:         {exercise}")
    print(f"   Duration:         {elapsed:.1f} seconds")
    print(f"   Total Reps:       {counter.count}")
    if all_form_scores:
        print(f"   Average Form:     {np.mean(all_form_scores):.1f}%")
        print(f"   Best Form:        {max(all_form_scores):.1f}%")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()

