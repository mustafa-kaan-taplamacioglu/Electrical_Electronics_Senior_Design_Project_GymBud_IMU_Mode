"""Rep counter for exercise repetition counting."""

import numpy as np
from typing import Optional
from services.form_analyzer import EXERCISE_CONFIG
from utils.pose_utils import get_bone_angle_from_horizontal, get_bone_angle_from_vertical, get_angle_between_bones

class RepCounter:
    """Counts exercise repetitions with form validation."""
    
    # Ultra-Strict minimum form score to count as valid rep
    MIN_FORM_SCORE = 70
    
    # Angle requirements per exercise (min_angle, max_angle, required_range_percent)
    ANGLE_REQUIREMENTS = {
        'bicep_curls': {'min': 40, 'max': 160, 'range_pct': 0.7},
        'squats': {'min': 75, 'max': 165, 'range_pct': 0.6},
        'lateral_shoulder_raises': {'min': 20, 'max': 80, 'range_pct': 0.7},
        'tricep_extensions': {'min': 60, 'max': 170, 'range_pct': 0.7},
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
        # State tracking for exercises that need extended/contracted position tracking
        self.reached_extended = False  # For shoulder press: tracks if we've reached extended position
        self.reached_contracted = False  # For rows: tracks if we've reached contracted position
        self.prev_angle = None  # Previous angle for hysteresis/history tracking
        
        # Dumbbell rows için state machine tracking (sadece sağ kol - vücudun sağ tarafından 90° açıdan)
        if exercise == 'dumbbell_rows':
            self.row_state_right = 'idle'  # idle, going_up, contracted, going_down
            self.row_prev_angle_right = None
            self.row_first_rep_started_right = False
            self.row_prev_rep_completed_right = False
            self.row_cooldown_frames = 0
    
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
        self.reached_extended = False  # Reset extended flag
        self.reached_contracted = False  # Reset contracted flag
        self.prev_angle = None  # Reset prev angle
        
        # Dumbbell rows state machine reset (sadece sağ kol)
        if self.exercise == 'dumbbell_rows':
            self.row_prev_rep_completed_right = False
            self.row_state_right = 'idle'
        
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
        
        elif self.exercise == 'tricep_extensions':
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
            # DUMBBELL ROW REP COUNTING (sadece sağ kol - vücudun sağ tarafından 90° açıdan)
            # State: idle -> going_up -> contracted -> going_down -> idle (rep complete!)
            if landmarks:
                # Calculate elbow angle for right arm only
                right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
                
                # Threshold'lar (Gerçek dumbbell row hareket açıklamasına göre)
                # PROPER FORM: "fully extended arm" (başlangıç) → "elbow passes torso" (contracted) → "straightening out arm" (başlangıç)
                # Başlangıç: Kol tamamen uzatılmış (fully extended) → açı ~170-180° (kol düz, aşağıda)
                # Contracted: Dirsek geriye çekilmiş, gövdeyi geçmiş → açı ~90-130° (dirsek bükülü, dumbbell vücuda yakın)
                EXTENDED_ANGLE = 165.0  # Kol tamamen uzatılmış (fully extended, ~165-180°) - bu değerin altına düşünce hareket başlar
                CONTRACTED_ANGLE_MIN = 100.0  # Kol yukarıda minimum açı (dirsek gövdeyi geçmiş, çok bükülü)
                CONTRACTED_ANGLE_MAX = 140.0  # Kol yukarıda maximum açı - bu değerin altına düşünce contracted state'ine geçer (elbow passes torso)
                ANGLE_HYSTERESIS = 10.0  # Gürültüyü azaltmak için
                
                # Helper function: State machine for right arm
                def detect_rep_state(elbow_angle, current_state, prev_angle):
                    """State machine: idle -> going_up -> contracted -> going_down -> idle (rep!)"""
                    if np.isnan(elbow_angle):
                        return current_state, False
                    
                    state = current_state
                    rep_completed = False
                    
                    if state == 'idle':
                        # Kol aşağı pozisyondan yukarı çekilmeye başladı
                        if elbow_angle < EXTENDED_ANGLE:
                            state = 'going_up'
                    elif state == 'going_up':
                        # Kol yukarı çekilmeye devam ediyor
                        if elbow_angle <= CONTRACTED_ANGLE_MAX:
                            state = 'contracted'
                        # Eğer açı tekrar artmaya başladıysa (yarım rep), IDLE'e dön
                        elif prev_angle is not None and not np.isnan(prev_angle) and elbow_angle > prev_angle + ANGLE_HYSTERESIS:
                            state = 'idle'
                    elif state == 'contracted':
                        # Kol yukarı pozisyonda
                        if elbow_angle > CONTRACTED_ANGLE_MAX + ANGLE_HYSTERESIS:
                            state = 'going_down'
                        elif elbow_angle < CONTRACTED_ANGLE_MIN:
                            state = 'going_up'
                    elif state == 'going_down':
                        # Kol aşağı indirilmeye devam ediyor
                        if elbow_angle >= EXTENDED_ANGLE - ANGLE_HYSTERESIS:
                            state = 'idle'
                            rep_completed = True  # Rep tamamlandı! (Tam döngü: aşağı -> yukarı -> aşağı)
                        # Eğer açı tekrar azalmaya başladıysa, contracted'a dön
                        elif prev_angle is not None and not np.isnan(prev_angle) and elbow_angle < prev_angle - ANGLE_HYSTERESIS:
                            state = 'contracted'
                    else:
                        state = 'idle'
                    
                    return state, rep_completed
                
                # Sağ kol için state machine (sadece sağ kol takip ediliyor)
                rep_completed_right = False
                if not np.isnan(right_elbow_angle):
                    old_state_right = self.row_state_right
                    self.row_state_right, rep_completed_right = detect_rep_state(
                        right_elbow_angle, self.row_state_right, self.row_prev_angle_right
                    )
                    
                    # Debug: State değişikliğini logla
                    if old_state_right != self.row_state_right:
                        print(f"🔄 RIGHT: {old_state_right} → {self.row_state_right} (angle: {right_elbow_angle:.1f}°)")
                    
                    # İlk rep başladı mı?
                    if old_state_right == 'idle' and self.row_state_right == 'going_up':
                        self.row_first_rep_started_right = True
                        print(f"🚀 RIGHT: First rep started!")
                    
                    if rep_completed_right:
                        print(f"✅ RIGHT: Rep completed! (angle: {right_elbow_angle:.1f}°)")
                    
                    self.row_prev_angle_right = right_elbow_angle
                
                # Display için phase güncelle (sağ kol state'ine göre)
                if self.row_state_right == 'idle':
                    self.phase = 'down'
                else:
                    self.phase = 'up'
                
                # Cooldown periodunu azalt
                if self.row_cooldown_frames > 0:
                    self.row_cooldown_frames -= 1
                
                # Rep sayımı: Sadece sağ kol için
                result = None
                right_rep_new = rep_completed_right and not self.row_prev_rep_completed_right and self.row_first_rep_started_right
                
                if right_rep_new and self.row_cooldown_frames == 0:
                    result = self.complete_rep()
                    self.row_cooldown_frames = 10
                    self.row_prev_rep_completed_right = True
                    print(f"✅ REP COUNTED! (RIGHT arm)")
                else:
                    self.row_prev_rep_completed_right = rep_completed_right
                
                return result
        
        elif self.exercise == 'dumbbell_shoulder_press':
            # DUMBBELL SHOULDER PRESS REP COUNTING (based on working detection code)
            # State machine: idle (down) -> going_up (up) -> extended (up) -> going_down (down) -> idle (rep complete)
            # Karşıdan yapılacak (front view)
            if landmarks:
                # Use average elbow angle (both arms)
                left_elbow_angle = get_angle_between_bones(landmarks, 'left_upper_arm', 'left_forearm')
                right_elbow_angle = get_angle_between_bones(landmarks, 'right_upper_arm', 'right_forearm')
                elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
                
                # Skip if angle is invalid
                if np.isnan(elbow_angle):
                    return None
                
                # Angle thresholds (from working detection code)
                BENT_ANGLE_MIN = 120.0  # Bent position minimum
                BENT_ANGLE_MAX = 140.0  # Bent position (idle/start) - small angle (120-140°)
                EXTENDED_ANGLE_MIN = 150.0  # Extended position (pressed up) - large angle (150-170°)
                ANGLE_HYSTERESIS = 5.0
                
                # State machine logic matching working detection code
                # phase 'down' = idle (bent, 120-140°)
                # phase 'up' = going_up or extended (pressed up, > 140°)
                # reached_extended = True when we've been in extended position (angle >= 150°)
                
                if self.phase == 'down':  # idle: bent position (120-140°)
                    self.reached_extended = False  # Reset when in idle
                    # Açı 140°'nin üstüne çıktıysa, going_up'a geç
                    if elbow_angle > BENT_ANGLE_MAX:
                        self.phase = 'up'
                    # Açı 120°'nin altındaysa, IDLE'de kal (çok bükülü)
                    # (No action needed, already in 'down' phase)
                elif self.phase == 'up':  # going_up or extended: pressed up (> 140°)
                    # Check if we've reached extended position (angle >= 150°)
                    if elbow_angle >= EXTENDED_ANGLE_MIN:
                        self.reached_extended = True  # Mark that we've reached extended position
                    
                    # Rep complete logic:
                    # If we've reached extended position AND angle has dropped below 140° (going_down -> idle)
                    if self.reached_extended and elbow_angle < BENT_ANGLE_MAX:
                        self.phase = 'down'
                        result = self.complete_rep()  # Rep complete! (Tam döngü: bükülü -> açık -> bükülü)
                    # If we haven't reached extended yet but angle dropped below 140°, reset to down (incomplete rep)
                    elif not self.reached_extended and elbow_angle < BENT_ANGLE_MAX:
                        self.phase = 'down'
                        self.reached_extended = False
                
                # Update prev_angle for next iteration
                self.prev_angle = elbow_angle
        
        else:
            # Generic fallback using angle thresholds
            if self.phase == 'down' and angle > up_threshold:
                self.phase = 'up'
            elif self.phase == 'up' and angle < down_threshold:
                self.phase = 'down'
                result = self.complete_rep()
        
        return result


