"""
Hybrid IMU Rep Detector
Combines proven rule-based gyroscope magnitude peak detection (primary) 
with ML form score validation (secondary) for robust rep counting.

Analysis from real bicep curl data (bicep_curls_20260103_230109):
- Gyro magnitude shows very clear peaks (200-300 range)
- Each peak represents one rep (~2.5-2.7 seconds per rep)
- ML form score is stable (67-72, ~5 point range) - not suitable for primary detection
- Best approach: Gyro peak detection (robust) + ML validation (quality check)
"""

import numpy as np
import time
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️  joblib not available - one-class classifier disabled")

# Import unit vectors utilities
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.quaternion_vectors import extract_orientation_vectors, quaternion_to_unit_vectors
    UNIT_VECTORS_AVAILABLE = True
except ImportError:
    UNIT_VECTORS_AVAILABLE = False
    print("⚠️  Unit vectors utilities not available")

try:
    from model_inference import ModelInference
    ML_INFERENCE_ENABLED = True
except ImportError:
    ML_INFERENCE_ENABLED = False
    print("⚠️  ML inference not available - using rule-based detection only")

try:
    from imu_feature_extractor import extract_imu_features
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTOR_AVAILABLE = False
    print("⚠️  imu_feature_extractor not available - one-class classifier disabled")

# Import ensemble model for form scoring and speed classification (bicep curls)
try:
    from services.bicep_curl_ensemble import (
        get_ensemble_model, 
        analyze_bicep_curl_rep, 
        classify_rep_speed,
        calculate_wrist_scores
    )
    BICEP_ENSEMBLE_AVAILABLE = True
except ImportError:
    BICEP_ENSEMBLE_AVAILABLE = False
    print("⚠️  Bicep curl ensemble model not available")

# Import multi-exercise ensemble model (lateral_shoulder_raises, squats, tricep_extensions)
try:
    from services.exercise_ensemble import (
        get_exercise_ensemble_model,
        analyze_exercise_rep,
        classify_exercise_rep_speed,
        calculate_exercise_wrist_scores,
        get_supported_exercises,
        get_exercise_config
    )
    MULTI_EXERCISE_ENSEMBLE_AVAILABLE = True
except ImportError:
    MULTI_EXERCISE_ENSEMBLE_AVAILABLE = False
    print("⚠️  Multi-exercise ensemble model not available")

# Import IMU-only form analyzer
try:
    from services.imu_form_analyzer import IMUFormAnalyzer
    IMU_FORM_ANALYZER_AVAILABLE = True
except ImportError:
    IMU_FORM_ANALYZER_AVAILABLE = False
    print("⚠️  IMU form analyzer not available")

ENSEMBLE_MODEL_AVAILABLE = BICEP_ENSEMBLE_AVAILABLE or MULTI_EXERCISE_ENSEMBLE_AVAILABLE

# Fallback function if ensemble not available
if not ENSEMBLE_MODEL_AVAILABLE:
    def calculate_wrist_scores(lw_pitch_range, rw_pitch_range, **kwargs):
        return {
            'regional_scores': {
                'left_wrist': min(100, lw_pitch_range / 1.3),
                'right_wrist': min(100, rw_pitch_range / 1.3),
                'arms': min(100, (lw_pitch_range + rw_pitch_range) / 2.6),
                'legs': 100.0, 'core': 100.0, 'head': 100.0
            },
            'regional_issues': {'left_wrist': [], 'right_wrist': [], 'arms': [], 'legs': [], 'core': [], 'head': []},
            'regional_feedback': {},
            'comparison_feedback': ''
        }
    
    def calculate_exercise_wrist_scores(exercise, lw_pitch_range, rw_pitch_range, **kwargs):
        return calculate_wrist_scores(lw_pitch_range, rw_pitch_range, **kwargs)


class HybridIMURepDetector:
    """
    Hybrid rep detector combining:
    1. Primary: Gyroscope magnitude peak detection (proven, rule-based)
    2. Secondary: ML form score validation (quality check, optional)
    """
    
    def __init__(self, exercise: str, ml_inference: Optional[ModelInference] = None):
        """
        Initialize hybrid rep detector.
        
        Args:
            exercise: Exercise name (e.g., "bicep_curls")
            ml_inference: ModelInference instance (optional, for form score validation)
        """
        self.exercise = exercise
        
        # Initialize ML inference (optional, for form score validation)
        if ml_inference:
            self.ml_inference = ml_inference
            self.has_ml_model = self.ml_inference.has_imu_model() if self.ml_inference else False
        else:
            self.ml_inference = ModelInference(exercise) if ML_INFERENCE_ENABLED else None
            self.has_ml_model = False
            if self.ml_inference:
                self.has_ml_model = self.ml_inference.has_imu_model()
        
        # PRIMARY: Gyroscope magnitude buffer (for peak detection)
        self.magnitude_buffer: List[float] = []
        self.timestamps: List[float] = []
        self.window_size = 90  # Sliding window size (from proven IMUPeriodicRepDetector)
        
        # ORIENTATION-BASED: Pitch/Roll/Yaw tracking (for direction-aware rep detection)
        # Separate tracking for left and right wrists for more robust detection
        self.left_wrist_pitch_buffer: List[float] = []  # Left wrist pitch history
        self.left_wrist_roll_buffer: List[float] = []   # Left wrist roll history
        self.left_wrist_yaw_buffer: List[float] = []    # Left wrist yaw history
        self.right_wrist_pitch_buffer: List[float] = []  # Right wrist pitch history
        self.right_wrist_roll_buffer: List[float] = []   # Right wrist roll history
        self.right_wrist_yaw_buffer: List[float] = []    # Right wrist yaw history
        
        # UNIT VECTORS tracking (for orientation-based rep detection)
        # Normal vector represents sensor's "up" direction - useful for detecting vertical motion
        self.left_wrist_normal_z_buffer: List[float] = []  # Left wrist normal vector Z component (vertical)
        self.right_wrist_normal_z_buffer: List[float] = []  # Right wrist normal vector Z component (vertical)
        self.left_wrist_tangent_y_buffer: List[float] = []  # Left wrist tangent vector Y component (forward)
        self.right_wrist_tangent_y_buffer: List[float] = []  # Right wrist tangent vector Y component (forward)
        
        # GYRO BUFFERS for each wrist (for LW/RW specific form analysis)
        self.left_wrist_gyro_buffer: List[float] = []   # Left wrist gyro magnitude history
        self.right_wrist_gyro_buffer: List[float] = []  # Right wrist gyro magnitude history
        
        # Averaged orientation buffers (for backward compatibility)
        self.pitch_buffer: List[float] = []  # Averaged pitch angle history (for bicep curl: up/down direction)
        self.roll_buffer: List[float] = []   # Averaged roll angle history (for wrist rotation)
        self.yaw_buffer: List[float] = []    # Averaged yaw angle history
        self.orientation_timestamps: List[float] = []  # Timestamps for orientation data
        
        # Direction tracking for bicep curls (using averaged values)
        self.current_direction = None  # 'up' or 'down'
        self.last_pitch = None
        self.last_roll = None
        self.last_yaw = None
        self.pitch_peak = None  # Maximum pitch (arm up position)
        self.pitch_valley = None  # Minimum pitch (arm down position)
        
        # SECONDARY: Form score history (for ML validation, optional)
        self.form_scores: List[float] = []
        self.score_timestamps: List[float] = []
        self.ml_window_size = 20  # Window size for ML form score calculation
        
        # ONE-CLASS CLASSIFIER: DISABLED (too strict, rejects valid bicep curls)
        # Using wrist synchronization + pitch range validation instead
        
        # IMU-ONLY FORM ANALYZER: For enhanced IMU-only mode analysis
        if IMU_FORM_ANALYZER_AVAILABLE and exercise == 'bicep_curls':
            self.imu_form_analyzer = IMUFormAnalyzer(exercise=exercise)
        else:
            self.imu_form_analyzer = None
        self.use_one_class_validation = False
        self.one_class_model = None
        self.one_class_scaler = None
        self.one_class_feature_names = None
        self.imu_buffer: List[Dict] = []  # Buffer for IMU sequences (for future use)
        
        # Rep tracking
        self.rep_count = 0
        self.last_rep_time = None
        self.last_rep_detection_time = None
        self.last_peak_index = -1
        
        # Bicep curl specific: 1 rep = 2 peaks (up → down → up again)
        # Track peak count to detect complete rep cycles
        self.peak_count = 0  # Count peaks, every 2 peaks = 1 rep
        self.last_peak_time = None
        
        # Rep duration tracking (for adaptive peak detection)
        self.rep_durations: List[float] = []
        self.sample_rate = 50.0  # Assume ~50 Hz IMU sampling rate
        
        # Peak detection parameters (optimized from real bicep curl data analysis)
        # Data: bicep_curls_20260103_230109 - 104 peaks for 100 reps (1.04 ratio, excellent!)
        # Peak distances: min=4, max=64, mean=20.7, median=21.0 samples
        # Peak values: Min: 58.0, Max: 312.6, Mean: 224.2, Median: 236.4
        # Signal median: 99.6, 75th percentile: 162.8
        # Optimal peak threshold: 130.2 (80% of 75th percentile)
        
        # Exercise-specific cooldowns - prevent false positives during fast movements
        exercise_cooldowns = {
            'bicep_curls': 0.6,              # Bicep curls need moderate cooldown
            'lateral_shoulder_raises': 0.8,   # Lateral raises need longer cooldown (full up-down cycle)
            'tricep_extensions': 0.6,         # Tricep extensions - moderate cooldown
            'squats': 1.0,                    # Squats are slower movements
            'dumbbell_shoulder_press': 0.7,   # Shoulder press - moderate cooldown
        }
        self.rep_cooldown = exercise_cooldowns.get(exercise, 0.6)  # Default 0.6s cooldown
        self.max_idle_time = 10.0  # Max idle time before reset (seconds) - more tolerant for pauses
        self.min_activity_threshold = 20.0  # Minimum magnitude for activity detection (to handle pauses)
        
        # ML validation parameters (secondary, optional)
        self.min_form_score = 50.0  # Minimum form score for valid rep (optional validation)
        self.use_ml_validation = False  # Whether to use ML validation (can enable if needed)
        
        # Debug tracking
        self.debug_enabled = True
        self.last_debug_time = None
        self.debug_interval = 2.0  # Print debug every 2 seconds
        
        # Ensemble model for form scoring and speed classification
        self.ensemble_model = None
        self.exercise_config = None
        
        # Load appropriate ensemble model based on exercise
        # Exercises with dedicated ensemble models
        multi_exercise_supported = [
            'lateral_shoulder_raises', 'squats', 'tricep_extensions',
            'dumbbell_shoulder_press'
        ]
        
        if exercise == 'bicep_curls' and BICEP_ENSEMBLE_AVAILABLE:
            try:
                self.ensemble_model = get_ensemble_model()
                print(f"   ✅ Bicep curl ensemble model loaded")
            except Exception as e:
                print(f"   ⚠️  Failed to load bicep curl ensemble model: {e}")
        elif MULTI_EXERCISE_ENSEMBLE_AVAILABLE and exercise in multi_exercise_supported:
            try:
                self.ensemble_model = get_exercise_ensemble_model(exercise)
                self.exercise_config = get_exercise_config(exercise)
                print(f"   ✅ Ensemble model loaded for {exercise}")
            except Exception as e:
                print(f"   ⚠️  Failed to load ensemble model for {exercise}: {e}")
        else:
            # Fallback: Try to load from multi-exercise ensemble if available
            if MULTI_EXERCISE_ENSEMBLE_AVAILABLE:
                try:
                    self.exercise_config = get_exercise_config(exercise)
                    if self.exercise_config:
                        self.ensemble_model = get_exercise_ensemble_model(exercise)
                        print(f"   ✅ Generic ensemble model loaded for {exercise}")
                except Exception as e:
                    print(f"   ℹ️  No ensemble model for {exercise}, using rule-based detection")
        
        # Rep analysis tracking
        self.last_rep_analysis: Optional[Dict] = None
        self.current_rep_start_time: Optional[float] = None
        self.current_rep_pitch_values: List[float] = []  # Averaged pitch
        self.current_rep_gyro_values: List[float] = []
        self.current_rep_samples = 0
        # Separate LW/RW tracking for more accurate range calculation
        self.current_rep_lw_pitch_values: List[float] = []  # Left wrist pitch during current rep
        self.current_rep_rw_pitch_values: List[float] = []  # Right wrist pitch during current rep
        self.current_rep_lw_roll_values: List[float] = []   # Left wrist roll during current rep
        self.current_rep_rw_roll_values: List[float] = []   # Right wrist roll during current rep
        
        print(f"✅ Hybrid IMU rep detector initialized for {exercise}")
        if self.has_ml_model:
            print(f"   - ML form score validation: {'ENABLED' if self.use_ml_validation else 'DISABLED (primary detection only)'}")
        else:
            print(f"   - ML form score validation: UNAVAILABLE (rule-based only)")
        if self.use_one_class_validation:
            print(f"   - One-Class classifier validation: ENABLED (bicep curl only)")
        else:
            print(f"   - One-Class classifier validation: DISABLED")
    
    def _load_one_class_classifier(self):
        """Load one-class classifier for bicep curl validation."""
        try:
            base_dir = Path(__file__).parent.parent
            model_file = base_dir / "bicep_curl_one_class_svm.joblib"
            scaler_file = base_dir / "bicep_curl_one_class_scaler.joblib"
            features_file = base_dir / "bicep_curl_one_class_features.json"
            
            if model_file.exists() and scaler_file.exists() and features_file.exists():
                self.one_class_model = joblib.load(model_file)
                self.one_class_scaler = joblib.load(scaler_file)
                with open(features_file, 'r') as f:
                    self.one_class_feature_names = json.load(f)
                self.use_one_class_validation = True
                print(f"   ✅ One-Class classifier loaded ({len(self.one_class_feature_names)} features)")
            else:
                print(f"   ⚠️  One-Class classifier files not found, skipping validation")
                self.use_one_class_validation = False
        except Exception as e:
            print(f"   ⚠️  Failed to load One-Class classifier: {e}")
            self.use_one_class_validation = False
    
    def _validate_bicep_curl_with_one_class(self, imu_sequence: List[dict]) -> bool:
        """
        Validate that IMU sequence is a bicep curl using one-class classifier.
        
        Args:
            imu_sequence: List of IMU samples
            
        Returns:
            True if sequence is classified as bicep curl, False otherwise
        """
        if not self.use_one_class_validation or not self.one_class_model:
            return True  # Skip validation if not available
        
        if len(imu_sequence) < 5:  # Minimum sequence length
            return True  # Too short, allow detection to proceed (lenient)
        
        try:
            # Extract features
            features = extract_imu_features(imu_sequence)
            if not features:
                return True  # Feature extraction failed, allow detection to proceed (lenient)
            
            # Convert to feature vector (same order as training)
            feature_vector = np.array([features.get(f, 0.0) for f in self.one_class_feature_names])
            
            # Reshape for prediction (1 sample, n_features)
            X = feature_vector.reshape(1, -1)
            
            # Scale
            X_scaled = self.one_class_scaler.transform(X)
            
            # Predict (1 = inlier/bicep curl, -1 = outlier/not bicep curl)
            prediction = self.one_class_model.predict(X_scaled)[0]
            
            is_valid = prediction == 1
            
            if self.debug_enabled and not is_valid:
                print(f"⚠️  One-Class classifier REJECTED sequence (not a bicep curl)")
            
            return is_valid
        except Exception as e:
            if self.debug_enabled:
                print(f"⚠️  One-Class classifier validation error: {e}")
            return True  # On error, allow detection to proceed
    
    def analyze_rep_with_ensemble(self, rep_duration: float) -> Dict:
        """
        Analyze a completed rep using the ensemble model.
        
        Args:
            rep_duration: Duration of the rep in seconds
            
        Returns:
            Dict with form_score, speed_class, feedback, issues, wrist scores, etc.
        """
        # Get pitch range from current rep (averaged)
        if self.current_rep_pitch_values:
            pitch_range = max(self.current_rep_pitch_values) - min(self.current_rep_pitch_values)
        else:
            # Fallback to buffer values
            if len(self.pitch_buffer) >= 10:
                recent_pitch = self.pitch_buffer[-30:] if len(self.pitch_buffer) >= 30 else self.pitch_buffer
                pitch_range = max(recent_pitch) - min(recent_pitch)
            else:
                pitch_range = 0.0
        
        # Get LW and RW pitch ranges separately
        lw_pitch_range = 0.0
        rw_pitch_range = 0.0
        lw_roll_range = 0.0
        rw_roll_range = 0.0
        sync_diff = 0.0
        
        # Calculate LW pitch range from current rep data
        # For tricep extensions: Use rep-specific buffer if available, otherwise use recent samples
        if self.exercise == 'tricep_extensions' and hasattr(self, 'current_rep_lw_pitch_values') and len(self.current_rep_lw_pitch_values) >= 3:
            # Use rep-specific LW pitch values (more accurate)
            lw_pitch_range = max(self.current_rep_lw_pitch_values) - min(self.current_rep_lw_pitch_values)
            if hasattr(self, 'current_rep_lw_roll_values') and len(self.current_rep_lw_roll_values) >= 3:
                lw_roll_range = max(self.current_rep_lw_roll_values) - min(self.current_rep_lw_roll_values)
        elif len(self.left_wrist_pitch_buffer) >= 3:
            # Fallback: Use recent samples (last 50 or all available)
            # For tricep extensions, use larger window to capture full rep
            window = min(100 if self.exercise == 'tricep_extensions' else 50, len(self.left_wrist_pitch_buffer))
            recent_lw_pitch = self.left_wrist_pitch_buffer[-window:]
            lw_pitch_range = max(recent_lw_pitch) - min(recent_lw_pitch)
            
            # LW roll range
            if len(self.left_wrist_roll_buffer) >= 3:
                recent_lw_roll = self.left_wrist_roll_buffer[-window:]
                lw_roll_range = max(recent_lw_roll) - min(recent_lw_roll)
        
        # Calculate RW pitch range from current rep data
        if self.exercise == 'tricep_extensions' and hasattr(self, 'current_rep_rw_pitch_values') and len(self.current_rep_rw_pitch_values) >= 3:
            # Use rep-specific RW pitch values (more accurate)
            rw_pitch_range = max(self.current_rep_rw_pitch_values) - min(self.current_rep_rw_pitch_values)
            if hasattr(self, 'current_rep_rw_roll_values') and len(self.current_rep_rw_roll_values) >= 3:
                rw_roll_range = max(self.current_rep_rw_roll_values) - min(self.current_rep_rw_roll_values)
        elif len(self.right_wrist_pitch_buffer) >= 3:
            # Fallback: Use recent samples (last 50 or all available)
            # For tricep extensions, use larger window to capture full rep
            window = min(100 if self.exercise == 'tricep_extensions' else 50, len(self.right_wrist_pitch_buffer))
            recent_rw_pitch = self.right_wrist_pitch_buffer[-window:]
            rw_pitch_range = max(recent_rw_pitch) - min(recent_rw_pitch)
            
            # RW roll range
            if len(self.right_wrist_roll_buffer) >= 3:
                recent_rw_roll = self.right_wrist_roll_buffer[-window:]
                rw_roll_range = max(recent_rw_roll) - min(recent_rw_roll)
        
        # Calculate sync difference (current positions)
        if len(self.left_wrist_pitch_buffer) > 0 and len(self.right_wrist_pitch_buffer) > 0:
            lw_current = self.left_wrist_pitch_buffer[-1]
            rw_current = self.right_wrist_pitch_buffer[-1]
            sync_diff = abs(lw_current - rw_current)
        
        # Get average gyro magnitude from current rep
        if self.current_rep_gyro_values:
            gyro_magnitude = np.mean(self.current_rep_gyro_values)
        else:
            # Fallback to buffer values
            if len(self.magnitude_buffer) >= 10:
                recent_mag = self.magnitude_buffer[-30:] if len(self.magnitude_buffer) >= 30 else self.magnitude_buffer
                gyro_magnitude = np.mean(recent_mag)
            else:
                gyro_magnitude = 100.0  # Default
        
        samples_count = self.current_rep_samples if self.current_rep_samples > 0 else 30
        
        # Calculate wrist-specific scores using appropriate function based on exercise
        if self.exercise == 'bicep_curls' and BICEP_ENSEMBLE_AVAILABLE:
            wrist_scores = calculate_wrist_scores(
                lw_pitch_range=lw_pitch_range,
                rw_pitch_range=rw_pitch_range,
                lw_roll_range=lw_roll_range,
                rw_roll_range=rw_roll_range,
                sync_diff=sync_diff
            )
        elif MULTI_EXERCISE_ENSEMBLE_AVAILABLE and self.exercise in ['lateral_shoulder_raises', 'squats', 'tricep_extensions']:
            wrist_scores = calculate_exercise_wrist_scores(
                exercise=self.exercise,
                lw_pitch_range=lw_pitch_range,
                rw_pitch_range=rw_pitch_range,
                lw_roll_range=lw_roll_range,
                rw_roll_range=rw_roll_range,
                sync_diff=sync_diff
            )
        else:
            # Fallback for other exercises
            wrist_scores = {
                'regional_scores': {
                    'left_wrist': min(100, lw_pitch_range / 1.3) if lw_pitch_range > 0 else 50,
                    'right_wrist': min(100, rw_pitch_range / 1.3) if rw_pitch_range > 0 else 50,
                    'arms': min(100, (lw_pitch_range + rw_pitch_range) / 2.6) if (lw_pitch_range + rw_pitch_range) > 0 else 50,
                    'legs': 100.0, 'core': 85.0, 'head': 90.0
                },
                'regional_issues': {'left_wrist': [], 'right_wrist': [], 'arms': [], 'legs': [], 'core': [], 'head': []},
                'regional_feedback': {},
                'comparison_feedback': ''
            }
        
        # FALLBACK: If LW/RW ranges are 0, calculate regional scores based on overall form score
        # This ensures regional scores are not all 0.0% when wrist data is unavailable
        if lw_pitch_range <= 0 and rw_pitch_range <= 0:
            # Calculate form score from pitch_range if available
            base_form_score = 70.0  # Default
            if pitch_range >= 90:
                base_form_score = 90.0
            elif pitch_range >= 70:
                base_form_score = 80.0
            elif pitch_range >= 50:
                base_form_score = 70.0
            elif pitch_range >= 30:
                base_form_score = 60.0
            else:
                base_form_score = 50.0
            
            # Apply speed penalty/bonus
            if rep_duration > 0:
                if 1.5 <= rep_duration <= 2.5:
                    base_form_score = min(100, base_form_score + 5)  # Ideal speed bonus
                elif rep_duration < 1.0:
                    base_form_score = max(0, base_form_score - 10)  # Too fast penalty
                elif rep_duration > 4.0:
                    base_form_score = max(0, base_form_score - 5)  # Too slow penalty
            
            wrist_scores['regional_scores'] = {
                'arms': base_form_score,
                'legs': 100.0 if self.exercise != 'squats' else base_form_score,
                'core': max(50, base_form_score - 5),
                'head': max(50, base_form_score)
            }
        
        # Calculate LW/RW gyro magnitudes separately
        lw_gyro_mag = gyro_magnitude / 2  # Default split
        rw_gyro_mag = gyro_magnitude / 2
        
        # Try to get actual separate gyro magnitudes from buffers
        if hasattr(self, 'left_wrist_gyro_buffer') and len(self.left_wrist_gyro_buffer) >= 5:
            lw_gyro_mag = np.mean(self.left_wrist_gyro_buffer[-30:] if len(self.left_wrist_gyro_buffer) >= 30 else self.left_wrist_gyro_buffer)
        if hasattr(self, 'right_wrist_gyro_buffer') and len(self.right_wrist_gyro_buffer) >= 5:
            rw_gyro_mag = np.mean(self.right_wrist_gyro_buffer[-30:] if len(self.right_wrist_gyro_buffer) >= 30 else self.right_wrist_gyro_buffer)
        
        # Use ensemble model if available - pass LW/RW data
        if self.ensemble_model and ENSEMBLE_MODEL_AVAILABLE:
            if self.exercise == 'bicep_curls' and BICEP_ENSEMBLE_AVAILABLE:
                analysis = analyze_bicep_curl_rep(
                    pitch_range=pitch_range,
                    gyro_magnitude=gyro_magnitude,
                    rep_duration=rep_duration,
                    samples_count=samples_count,
                    lw_pitch_range=lw_pitch_range if lw_pitch_range > 0 else None,
                    rw_pitch_range=rw_pitch_range if rw_pitch_range > 0 else None,
                    lw_gyro_mag=lw_gyro_mag if lw_gyro_mag > 0 else None,
                    rw_gyro_mag=rw_gyro_mag if rw_gyro_mag > 0 else None,
                    sync_diff=sync_diff
                )
            elif MULTI_EXERCISE_ENSEMBLE_AVAILABLE and self.exercise in ['lateral_shoulder_raises', 'squats', 'tricep_extensions']:
                # Use multi-exercise ensemble model
                analysis = analyze_exercise_rep(
                    exercise=self.exercise,
                    pitch_range=pitch_range,
                    roll_range=lw_roll_range if lw_roll_range > 0 else rw_roll_range,  # Use roll for lateral raises
                    gyro_magnitude=gyro_magnitude,
                    rep_duration=rep_duration,
                    samples_count=samples_count,
                    lw_pitch_range=lw_pitch_range if lw_pitch_range > 0 else None,
                    rw_pitch_range=rw_pitch_range if rw_pitch_range > 0 else None,
                    lw_roll_range=lw_roll_range,
                    rw_roll_range=rw_roll_range,
                    lw_gyro_mag=lw_gyro_mag if lw_gyro_mag > 0 else None,
                    rw_gyro_mag=rw_gyro_mag if rw_gyro_mag > 0 else None,
                    sync_diff=sync_diff
                )
            else:
                # Fallback analysis without specific ensemble model
                speed_info = self._classify_speed_simple(rep_duration)
                form_score = self._calculate_simple_form_score(pitch_range, rep_duration)
                
                analysis = {
                    'form_score': form_score,
                    'speed_class': speed_info['class'],
                    'speed_label': speed_info['label'],
                    'speed_emoji': speed_info['emoji'],
                    'speed_feedback': speed_info['feedback'],
                    'duration': round(rep_duration, 2),
                    'form_feedback': 'İyi form!' if form_score >= 70 else 'Form iyileştirilebilir.',
                    'issues': [],
                    'pitch_range': round(pitch_range, 1),
                    'gyro_magnitude': round(gyro_magnitude, 1)
                }
        else:
            # Fallback analysis without ensemble model
            speed_info = self._classify_speed_simple(rep_duration)
            form_score = self._calculate_simple_form_score(pitch_range, rep_duration)
            
            analysis = {
                'form_score': form_score,
                'speed_class': speed_info['class'],
                'speed_label': speed_info['label'],
                'speed_emoji': speed_info['emoji'],
                'speed_feedback': speed_info['feedback'],
                'duration': round(rep_duration, 2),
                'form_feedback': 'İyi form!' if form_score >= 70 else 'Form iyileştirilebilir.',
                'issues': [],
                'pitch_range': round(pitch_range, 1),
                'gyro_magnitude': round(gyro_magnitude, 1)
            }
        
        # Add wrist-specific data to analysis
        analysis['lw_pitch_range'] = round(lw_pitch_range, 1)
        analysis['rw_pitch_range'] = round(rw_pitch_range, 1)
        analysis['lw_roll_range'] = round(lw_roll_range, 1)
        analysis['rw_roll_range'] = round(rw_roll_range, 1)
        analysis['sync_diff'] = round(sync_diff, 1)
        analysis['regional_scores'] = wrist_scores.get('regional_scores', {})
        analysis['regional_issues'] = wrist_scores.get('regional_issues', {})
        analysis['regional_feedback'] = wrist_scores.get('regional_feedback', {})
        analysis['wrist_comparison'] = wrist_scores.get('comparison_feedback', '')
        
        # === IMU-ONLY MODE: Enhanced analysis using IMU form analyzer ===
        if self.imu_form_analyzer and self.exercise == 'bicep_curls':
            try:
                # Get IMU sequence from buffer (last rep's samples)
                # Use recent IMU buffer samples that correspond to this rep
                imu_sequence = []
                if hasattr(self, 'imu_buffer') and len(self.imu_buffer) > 0:
                    # Get samples from current rep (approximate - use recent samples)
                    # For more accuracy, we could track rep start/end in buffer
                    recent_samples = self.imu_buffer[-self.current_rep_samples:] if self.current_rep_samples > 0 else self.imu_buffer[-50:]
                    imu_sequence = recent_samples
                
                # If we have enough samples, use IMU form analyzer
                if len(imu_sequence) >= 5:
                    imu_analysis_result = self.imu_form_analyzer.analyze_bicep_curl_imu_only(
                        imu_sequence=imu_sequence,
                        rep_duration=rep_duration
                    )
                    
                    # Merge IMU-only analysis results
                    if imu_analysis_result.get('imu_analysis'):
                        analysis['imu_analysis'] = imu_analysis_result['imu_analysis']
                    
                    # Override form score with IMU-only analysis if it's more detailed
                    if imu_analysis_result.get('score') and len(imu_sequence) >= 10:
                        # Use IMU-only score if we have enough data
                        analysis['form_score'] = imu_analysis_result['score']
                        analysis['imu_only_score'] = True  # Flag to indicate IMU-only scoring
                    
                    # Merge regional scores (IMU-only gives more detailed arms analysis)
                    if imu_analysis_result.get('regional_scores'):
                        # Keep existing scores but enhance arms score
                        existing_arms = analysis.get('regional_scores', {}).get('arms', 0)
                        imu_arms = imu_analysis_result['regional_scores'].get('arms', 0)
                        # Use IMU score if it's more detailed (has imu_analysis)
                        if imu_analysis_result.get('imu_analysis'):
                            analysis['regional_scores']['arms'] = imu_arms
                    
                    # Merge issues
                    if imu_analysis_result.get('issues'):
                        existing_issues = analysis.get('issues', [])
                        analysis['issues'] = list(set(existing_issues + imu_analysis_result['issues']))
                    
                    # Merge regional issues
                    if imu_analysis_result.get('regional_issues'):
                        for region, region_issues in imu_analysis_result['regional_issues'].items():
                            if region_issues:
                                existing_region_issues = analysis.get('regional_issues', {}).get(region, [])
                                analysis.setdefault('regional_issues', {})[region] = list(set(existing_region_issues + region_issues))
            except Exception as e:
                if self.debug_enabled:
                    print(f"⚠️  IMU form analyzer error: {e}")
                # Continue with existing analysis if IMU analyzer fails
        
        # Store analysis
        self.last_rep_analysis = analysis
        
        # Reset current rep tracking
        self.current_rep_pitch_values = []
        self.current_rep_gyro_values = []
        self.current_rep_samples = 0
        # Reset separate LW/RW tracking
        self.current_rep_lw_pitch_values = []
        self.current_rep_rw_pitch_values = []
        self.current_rep_lw_roll_values = []
        self.current_rep_rw_roll_values = []
        
        return analysis
    
    def _classify_speed_simple(self, rep_duration: float) -> Dict:
        """Simple speed classification without ensemble model."""
        if rep_duration < 1.2:
            return {'class': 'very_fast', 'label': 'Very Fast', 'emoji': '🚀', 
                    'feedback': 'Too fast - slow down a bit.'}
        elif rep_duration < 1.8:
            return {'class': 'fast', 'label': 'Fast', 'emoji': '⚡', 
                    'feedback': 'Fast tempo - maintain form.'}
        elif rep_duration < 2.5:
            return {'class': 'medium', 'label': 'Medium', 'emoji': '✅', 
                    'feedback': 'Ideal tempo!'}
        elif rep_duration < 3.5:
            return {'class': 'slow', 'label': 'Slow', 'emoji': '🐢', 
                    'feedback': 'Slow and controlled.'}
        else:
            return {'class': 'very_slow', 'label': 'Very Slow', 'emoji': '🦥', 
                    'feedback': 'Try to speed up a bit.'}
    
    def _calculate_simple_form_score(self, pitch_range: float, rep_duration: float) -> float:
        """Calculate form score without ensemble model."""
        # Pitch range score (40%)
        if pitch_range >= 130:
            pitch_score = 100.0
        elif pitch_range >= 100:
            pitch_score = 70 + 30 * (pitch_range - 100) / 30
        elif pitch_range >= 60:
            pitch_score = 40 + 30 * (pitch_range - 60) / 40
        else:
            pitch_score = max(20, pitch_range / 60 * 40)
        
        # Speed score (30%)
        if 1.5 <= rep_duration <= 3.0:
            speed_score = 100.0
        elif 1.0 <= rep_duration < 1.5 or 3.0 < rep_duration <= 4.0:
            speed_score = 75.0
        else:
            speed_score = 50.0
        
        # Quality score (30%) - based on consistency
        quality_score = 75.0  # Default without more data
        
        return 0.4 * pitch_score + 0.3 * speed_score + 0.3 * quality_score
    
    def get_last_rep_analysis(self) -> Optional[Dict]:
        """Get the analysis of the last completed rep."""
        return self.last_rep_analysis
    
    def _calculate_gyro_magnitude(self, imu_sample: dict) -> Optional[float]:
        """
        Calculate gyroscope magnitude from IMU sample.
        Exercise-specific node selection.
        Also populates per-wrist gyro buffers for LW/RW analysis.
        """
        # All wrist-based exercises (bilateral upper body movements)
        wrist_based_exercises = [
            'bicep_curls', 'tricep_extensions', 'dumbbell_rows', 
            'lateral_shoulder_raises', 'dumbbell_shoulder_press', 'lateral_raises', 
            'front_raises', 'shoulder_press'
        ]
        
        if self.exercise in wrist_based_exercises:
            # Wrist-based exercises - use both wrists, average magnitude
            left_wrist = imu_sample.get('left_wrist', {})
            right_wrist = imu_sample.get('right_wrist', {})
            
            gyro_mags = []
            
            # Calculate and store LW gyro magnitude
            if left_wrist and isinstance(left_wrist, dict):
                gx = left_wrist.get('gx', 0) or 0
                gy = left_wrist.get('gy', 0) or 0
                gz = left_wrist.get('gz', 0) or 0
                lw_mag = np.sqrt(gx**2 + gy**2 + gz**2)
                gyro_mags.append(lw_mag)
                # Store in LW buffer
                self.left_wrist_gyro_buffer.append(lw_mag)
                if len(self.left_wrist_gyro_buffer) > 100:
                    self.left_wrist_gyro_buffer.pop(0)
            
            # Calculate and store RW gyro magnitude
            if right_wrist and isinstance(right_wrist, dict):
                gx = right_wrist.get('gx', 0) or 0
                gy = right_wrist.get('gy', 0) or 0
                gz = right_wrist.get('gz', 0) or 0
                rw_mag = np.sqrt(gx**2 + gy**2 + gz**2)
                gyro_mags.append(rw_mag)
                # Store in RW buffer
                self.right_wrist_gyro_buffer.append(rw_mag)
                if len(self.right_wrist_gyro_buffer) > 100:
                    self.right_wrist_gyro_buffer.pop(0)
            
            if len(gyro_mags) > 0:
                return np.mean(gyro_mags)
        elif self.exercise == 'squats':
            # Squats - use chest node
            chest = imu_sample.get('chest', {})
            if chest and isinstance(chest, dict):
                gx = chest.get('gx', 0) or 0
                gy = chest.get('gy', 0) or 0
                gz = chest.get('gz', 0) or 0
                return np.sqrt(gx**2 + gy**2 + gz**2)
        
        return None
    
    def _extract_orientation(self, imu_sample: dict) -> Optional[Dict[str, float]]:
        """
        Extract orientation (pitch/roll/yaw) and unit vectors from IMU sample.
        For bicep curls: Pitch indicates up/down motion (most important).
        Unit vectors (normal, tangent, binormal) represent sensor orientation in 3D space.
        
        Returns:
            Dict with 'pitch', 'roll', 'yaw', 'left_wrist', 'right_wrist', and optionally 'unit_vectors' or None if not available
        """
        # All wrist-based exercises (bilateral upper body movements)
        wrist_exercises = [
            'bicep_curls', 'tricep_extensions', 'dumbbell_rows', 'dumbbell_shoulder_press', 
            'lateral_raises', 'front_raises', 'lateral_shoulder_raises', 'tricep_extensions',
            'shoulder_press'
        ]
        
        # Extract orientation for all wrist-based exercises (bilateral movements)
        if self.exercise in wrist_exercises:
            # Wrist-based exercises - extract separate orientations for left and right wrists
            left_wrist = imu_sample.get('left_wrist', {})
            right_wrist = imu_sample.get('right_wrist', {})
            
            # Extract left wrist orientation (use 0.0 as default if value is None)
            left_pitch = None
            left_roll = None
            left_yaw = None
            if left_wrist and isinstance(left_wrist, dict):
                lp = left_wrist.get('pitch')
                lr = left_wrist.get('roll')
                ly = left_wrist.get('yaw')
                # If at least pitch exists, use this wrist's data
                if lp is not None:
                    left_pitch = float(lp)
                    left_roll = float(lr) if lr is not None else 0.0
                    left_yaw = float(ly) if ly is not None else 0.0
                # Debug: Log if pitch is missing but other values exist
                elif (lr is not None or ly is not None) and self.debug_enabled:
                    print(f"⚠️  LW pitch is None but roll={lr}, yaw={ly}")
            
            # Extract right wrist orientation (use 0.0 as default if value is None)
            right_pitch = None
            right_roll = None
            right_yaw = None
            if right_wrist and isinstance(right_wrist, dict):
                rp = right_wrist.get('pitch')
                rr = right_wrist.get('roll')
                ry = right_wrist.get('yaw')
                # If at least pitch exists, use this wrist's data
                if rp is not None:
                    right_pitch = float(rp)
                    right_roll = float(rr) if rr is not None else 0.0
                    right_yaw = float(ry) if ry is not None else 0.0
                # Debug: Log if pitch is missing but other values exist
                elif (rr is not None or ry is not None) and self.debug_enabled:
                    print(f"⚠️  RW pitch is None but roll={rr}, yaw={ry}")
            
            # Calculate averages (for backward compatibility)
            pitches = []
            rolls = []
            yaws = []
            if left_pitch is not None:
                pitches.append(left_pitch)
                rolls.append(left_roll)
                yaws.append(left_yaw)
            if right_pitch is not None:
                pitches.append(right_pitch)
                rolls.append(right_roll)
                yaws.append(right_yaw)
            
            if len(pitches) > 0:
                result = {
                    'pitch': np.mean(pitches),
                    'roll': np.mean(rolls),
                    'yaw': np.mean(yaws)
                }
                # Add separate wrist data - always add if pitch is available
                if left_pitch is not None:
                    result['left_wrist'] = {
                        'pitch': left_pitch,
                        'roll': left_roll,
                        'yaw': left_yaw
                    }
                    # Add unit vectors for left wrist
                    if UNIT_VECTORS_AVAILABLE:
                        lw_vectors = extract_orientation_vectors(imu_sample, 'left_wrist')
                        if lw_vectors:
                            result['left_wrist']['unit_vectors'] = {
                                'normal': lw_vectors['normal'].tolist(),
                                'tangent': lw_vectors['tangent'].tolist(),
                                'binormal': lw_vectors['binormal'].tolist()
                            }
                if right_pitch is not None:
                    result['right_wrist'] = {
                        'pitch': right_pitch,
                        'roll': right_roll,
                        'yaw': right_yaw
                    }
                    # Add unit vectors for right wrist
                    if UNIT_VECTORS_AVAILABLE:
                        rw_vectors = extract_orientation_vectors(imu_sample, 'right_wrist')
                        if rw_vectors:
                            result['right_wrist']['unit_vectors'] = {
                                'normal': rw_vectors['normal'].tolist(),
                                'tangent': rw_vectors['tangent'].tolist(),
                                'binormal': rw_vectors['binormal'].tolist()
                            }
                return result
        elif self.exercise == 'squats':
            # Squats - use chest node
            chest = imu_sample.get('chest', {})
            if chest and isinstance(chest, dict):
                pitch = chest.get('pitch')
                roll = chest.get('roll')
                yaw = chest.get('yaw')
                
                if pitch is not None and roll is not None and yaw is not None:
                    return {
                        'pitch': float(pitch),
                        'roll': float(roll),
                        'yaw': float(yaw)
                    }
        
        return None
    
    def _calculate_form_score(self, imu_sequence: List[dict]) -> Optional[float]:
        """
        Calculate form score using ML model (secondary validation, optional).
        Returns form score or None if ML model unavailable.
        """
        if not self.has_ml_model or not self.ml_inference:
            return None
        
        try:
            # Use ML inference to predict form score
            form_score = self.ml_inference.predict_imu(imu_sequence)
            return form_score
        except Exception as e:
            if self.debug_enabled:
                print(f"⚠️  ML form score calculation error: {e}")
            return None
    
    def _validate_bicep_curl_motion(self) -> bool:
        """
        Validate that the motion is a proper bicep curl (not lateral arm raise).
        
        Bicep curl requirements:
        - Pitch should change significantly (up/down motion)
        - Roll should change minimally (no lateral movement - elbows stay fixed)
        - Yaw should change minimally (no rotation - elbows stay fixed)
        
        NOTE: This validation is now more lenient - only rejects obvious lateral movements.
        Primary validation is done by one-class classifier.
        
        Returns:
            True if motion appears to be a valid bicep curl, False otherwise
        """
        if len(self.pitch_buffer) < 10 or len(self.roll_buffer) < 10 or len(self.yaw_buffer) < 10:
            return True  # Not enough data yet, allow detection to proceed
        
        # Get recent orientation values
        recent_pitch = self.pitch_buffer[-20:] if len(self.pitch_buffer) >= 20 else self.pitch_buffer
        recent_roll = self.roll_buffer[-20:] if len(self.roll_buffer) >= 20 else self.roll_buffer
        recent_yaw = self.yaw_buffer[-20:] if len(self.yaw_buffer) >= 20 else self.yaw_buffer
        
        if len(recent_pitch) < 10 or len(recent_roll) < 10 or len(recent_yaw) < 10:
            return True  # Not enough data
        
        # Calculate ranges (how much each angle changed)
        pitch_range = abs(max(recent_pitch) - min(recent_pitch))
        roll_range = abs(max(recent_roll) - min(recent_roll))
        yaw_range = abs(max(recent_yaw) - min(recent_yaw))
        
        # Bicep curl: Pitch should be dominant, roll/yaw should be minimal
        # Made VERY lenient: Only reject obvious lateral movements (roll/yaw >> pitch)
        min_pitch_range = 8.0  # Minimum 8° pitch change for valid curl (more lenient)
        max_roll_range = 180.0  # Maximum 180° roll change (very lenient - only catch extreme cases)
        max_yaw_range = 180.0  # Maximum 180° yaw change (very lenient)
        
        # Only reject if roll/yaw is MUCH larger than pitch (obvious lateral movement)
        # Allow roll/yaw up to 2x pitch (very lenient)
        is_valid = (
            pitch_range >= min_pitch_range and  # Pitch change is significant
            roll_range <= max_roll_range and    # Roll change is within acceptable range
            yaw_range <= max_yaw_range and      # Yaw change is within acceptable range
            pitch_range > roll_range * 0.3 or roll_range < 100.0  # Very lenient: allow if pitch > 30% of roll OR roll < 100°
        )
        
        if self.debug_enabled and not is_valid:
            print(f"⚠️  Invalid bicep curl motion detected: "
                  f"pitch_range={pitch_range:.1f}°, roll_range={roll_range:.1f}°, yaw_range={yaw_range:.1f}°")
        
        return is_valid
    
    def _detect_rep_from_orientation(self, timestamp: float) -> bool:
        """
        ORIENTATION-BASED REP DETECTION for bicep curls.
        Uses pitch angle changes from BOTH left and right wrists to detect up → down → up pattern = 1 rep.
        Validates motion is proper bicep curl by checking synchronized movement of both wrists.
        
        Bicep curl pattern:
        - Arm down (start): Low pitch
        - Arm up (curl): High pitch (peak)
        - Arm down (return): Low pitch (valley)
        - Arm up (next curl): High pitch (peak) = 1 rep completed
        
        Direction tracking:
        - Pitch increases (positive change) = arm moving up
        - Pitch decreases (negative change) = arm moving down
        - Both wrists should move synchronously (similar pitch changes)
        - 1 rep = down → up → down → up (complete cycle)
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if rep detected, False otherwise
        """
        # Use both left and right wrist orientations for more robust detection
        use_separate_wrists = (len(self.left_wrist_pitch_buffer) >= 10 and 
                               len(self.right_wrist_pitch_buffer) >= 10)
        
        if use_separate_wrists:
            # Use separate wrist buffers for more robust detection
            lw_recent_pitch = self.left_wrist_pitch_buffer[-20:] if len(self.left_wrist_pitch_buffer) >= 20 else self.left_wrist_pitch_buffer
            rw_recent_pitch = self.right_wrist_pitch_buffer[-20:] if len(self.right_wrist_pitch_buffer) >= 20 else self.right_wrist_pitch_buffer
            
            if len(lw_recent_pitch) < 5 or len(rw_recent_pitch) < 5:
                return False
            
            # Average the two wrists for combined detection
            min_len = min(len(lw_recent_pitch), len(rw_recent_pitch))
            recent_pitch = [(lw_recent_pitch[i] + rw_recent_pitch[i]) / 2.0 
                           for i in range(min_len)]
            current_pitch = (lw_recent_pitch[-1] + rw_recent_pitch[-1]) / 2.0
            
            # Check synchronization: both wrists should have similar pitch changes
            # Use absolute pitch values (not just change) for better synchronization check
            lw_current_pitch = lw_recent_pitch[-1]
            rw_current_pitch = rw_recent_pitch[-1]
            lw_avg_pitch = np.mean(lw_recent_pitch)
            rw_avg_pitch = np.mean(rw_recent_pitch)
            
            # Check synchronization: both wrists should move together with proper bicep curl pattern
            # For a proper bicep curl, both wrists should have similar pitch values (within threshold)
            pitch_diff = abs(lw_current_pitch - rw_current_pitch)
            avg_pitch_diff = abs(lw_avg_pitch - rw_avg_pitch)
            
            # Synchronization thresholds - more lenient to account for natural variation
            # Use very lenient thresholds for exercises without specific training data
            if self.exercise in ['bicep_curls']:
                max_sync_diff = 50.0  # Stricter for bicep curls
                max_change_diff = 40.0
            else:
                max_sync_diff = 90.0  # More lenient for other exercises
                max_change_diff = 70.0  # More lenient
            min_change_threshold = 3.0  # Lower threshold for better detection
            
            # Check if both wrists are moving in similar direction
            lw_pitch_change = lw_recent_pitch[-1] - lw_recent_pitch[0] if len(lw_recent_pitch) > 1 else 0
            rw_pitch_change = rw_recent_pitch[-1] - rw_recent_pitch[0] if len(rw_recent_pitch) > 1 else 0
            change_diff = abs(lw_pitch_change - rw_pitch_change)
            
            # Only check direction for significant movements (avoid noise rejection)
            if abs(lw_pitch_change) > min_change_threshold and abs(rw_pitch_change) > min_change_threshold:
                same_direction = (lw_pitch_change * rw_pitch_change >= 0)  # Same sign = same direction
            else:
                same_direction = True  # Small movements - don't reject based on direction
            
            # Wrist synchronization check - exercise-specific criteria
            # For tricep_extensions: Check if wrists are moving (not position matching)
            # This is because IMU orientation can cause opposite signs for the same motion
            if self.exercise == 'tricep_extensions':
                # Tricep extensions: Check that wrists have movement
                # Due to IMU orientation, LW and RW may show OPPOSITE signs for the same motion
                lw_pitch_range = max(lw_recent_pitch) - min(lw_recent_pitch) if len(lw_recent_pitch) >= 3 else 0
                rw_pitch_range = max(rw_recent_pitch) - min(rw_recent_pitch) if len(rw_recent_pitch) >= 3 else 0
                
                # More lenient check for tricep extensions:
                # Option 1: At least one wrist has significant movement (>= 40°) - for single-arm variations
                # Option 2: Combined average is sufficient (>= 25°)
                # Option 3: Both wrists have some movement (>= 15° each)
                min_single_wrist = 40.0  # At least one wrist needs this
                min_combined_avg = 25.0   # Average of both wrists
                min_both_wrists = 15.0    # Minimum for both-wrist check
                
                combined_avg = (lw_pitch_range + rw_pitch_range) / 2.0
                
                is_synchronized_motion = (
                    # At least one wrist has significant movement
                    (lw_pitch_range >= min_single_wrist or rw_pitch_range >= min_single_wrist) or
                    # OR combined average is good
                    (combined_avg >= min_combined_avg) or
                    # OR both wrists have at least minimal movement
                    (lw_pitch_range >= min_both_wrists and rw_pitch_range >= min_both_wrists)
                )
                
                # For tricep extensions, only reject if there's truly NO movement
                # Let the final pitch_range check (130°) be the main filter
                if not is_synchronized_motion:
                    # Only reject if both wrists have very little movement
                    if lw_pitch_range < 10.0 and rw_pitch_range < 10.0:
                        if self.debug_enabled:
                            print(f"⚠️  Tricep ext: No significant movement: LW={lw_pitch_range:.1f}°, RW={rw_pitch_range:.1f}° - REJECTING")
                        return False
                    # Otherwise, allow it to proceed to pitch_range check
                    is_synchronized_motion = True
            else:
                # Other exercises: Standard synchronization check
                # 1. Current pitch difference <= threshold (arms at similar position)
                # 2. Either: change_diff is acceptable OR same_direction is True
                is_synchronized_motion = (
                    pitch_diff <= max_sync_diff and  # Current positions are similar
                    (change_diff <= max_change_diff or same_direction)  # Either small change diff OR same direction
                )
                
                # Only reject movements with LARGE differences (clearly different movements)
                if pitch_diff > max_sync_diff or (change_diff > max_change_diff and not same_direction):
                    # Wrists clearly not synchronized - reject this movement
                    if self.debug_enabled and pitch_diff > max_sync_diff:
                        print(f"⚠️  Wrists not synchronized: LW={lw_current_pitch:.1f}°, RW={rw_current_pitch:.1f}°, diff={pitch_diff:.1f}° - REJECTING")
                    is_synchronized_motion = False
                    return False
        else:
            # Fallback to averaged orientation (backward compatibility)
            if len(self.pitch_buffer) < 10:
                return False
            recent_pitch = self.pitch_buffer[-20:] if len(self.pitch_buffer) >= 20 else self.pitch_buffer
            if len(recent_pitch) < 5:
                return False
            current_pitch = recent_pitch[-1]
            # For fallback, assume synchronized (can't check with separate wrists)
            is_synchronized_motion = True
        
        pitch_array = np.array(recent_pitch)
        
        # Also get current roll and yaw for validation (use averaged if available)
        current_roll = self.roll_buffer[-1] if len(self.roll_buffer) > 0 else None
        current_yaw = self.yaw_buffer[-1] if len(self.yaw_buffer) > 0 else None
        
        # Track direction changes
        if self.last_pitch is None:
            self.last_pitch = current_pitch
            # Initialize direction based on first pitch value
            # If pitch is high, we might be starting at "up" position
            if current_pitch > np.median(recent_pitch):
                self.current_direction = 'up'
            else:
                self.current_direction = 'down'
            return False
        
        pitch_change = current_pitch - self.last_pitch
        pitch_change_threshold = 3.0  # Minimum 3 degrees change to detect direction (more sensitive)
        
        # Detect direction based on pitch change
        if pitch_change > pitch_change_threshold:
            # Moving up (pitch increasing - arm going up)
            new_direction = 'up'
        elif pitch_change < -pitch_change_threshold:
            # Moving down (pitch decreasing - arm going down)
            new_direction = 'down'
        else:
            # No significant change - keep current direction
            new_direction = self.current_direction if self.current_direction else 'down'
        
        # Detect rep completion: down → up transition after up → down
        # Pattern: up → down → up = 1 rep completed (back to up position)
        rep_detected = False
        
        if self.current_direction == 'down' and new_direction == 'up':
            # Down → Up transition: This completes a rep cycle
            # We need to verify we had a previous up → down cycle
            if self.pitch_peak is not None and self.pitch_valley is not None:
                # Verify we had a significant up → down → up cycle
                pitch_range = abs(self.pitch_peak - self.pitch_valley)
                min_pitch_range = 80.0  # Minimum 80 degrees range for valid rep (ensures full bicep curl motion - stricter)
                
                # Check time since last rep (cooldown)
                time_since_last_rep = timestamp - self.last_rep_detection_time if self.last_rep_detection_time else float('inf')
                
                # Validation: Wrist synchronization + pitch range
                # One-Class classifier disabled (too strict, rejected valid bicep curls)
                # Relying on:
                # 1. Wrist synchronization (both wrists move together - proper bicep curl)
                # 2. Pitch range (significant up/down movement)
                
                # Get exercise-specific minimum pitch range
                # First check exercise_config, then use exercise-specific defaults
                if self.exercise_config and 'min_pitch_range' in self.exercise_config:
                    exercise_min_pitch_range = self.exercise_config['min_pitch_range']
                elif self.exercise == 'bicep_curls':
                    exercise_min_pitch_range = 80.0  # Bicep curls need significant movement
                elif self.exercise == 'lateral_shoulder_raises':
                    exercise_min_pitch_range = 50.0  # More lenient for lateral raises
                elif self.exercise == 'lateral_raises':
                    exercise_min_pitch_range = 50.0  # Alias
                elif self.exercise == 'squats':
                    exercise_min_pitch_range = 30.0  # Chest pitch for squats
                elif self.exercise == 'tricep_extensions':
                    exercise_min_pitch_range = 130.0  # Triceps extensions need VERY high ROM (150-175° in training data)
                elif self.exercise == 'dumbbell_shoulder_press':
                    exercise_min_pitch_range = 40.0  # Shoulder press - up/down motion
                elif self.exercise == 'shoulder_press':
                    exercise_min_pitch_range = 40.0  # Alias
                elif self.exercise == 'dumbbell_rows':
                    exercise_min_pitch_range = 40.0  # Rows
                elif self.exercise == 'front_raises':
                    exercise_min_pitch_range = 50.0  # Front raises
                else:
                    exercise_min_pitch_range = 40.0  # Default lenient threshold
                
                # Accept rep if: pitch range is sufficient, cooldown passed, and synchronized
                if pitch_range >= exercise_min_pitch_range and time_since_last_rep >= self.rep_cooldown and is_synchronized_motion:
                    rep_detected = True
                    if self.debug_enabled:
                        # Get LW/RW ranges for debug output
                        lw_range_debug = 0.0
                        rw_range_debug = 0.0
                        if hasattr(self, 'current_rep_lw_pitch_values') and len(self.current_rep_lw_pitch_values) >= 3:
                            lw_range_debug = max(self.current_rep_lw_pitch_values) - min(self.current_rep_lw_pitch_values)
                        elif len(self.left_wrist_pitch_buffer) >= 3:
                            window = min(100 if self.exercise == 'tricep_extensions' else 50, len(self.left_wrist_pitch_buffer))
                            lw_range_debug = max(self.left_wrist_pitch_buffer[-window:]) - min(self.left_wrist_pitch_buffer[-window:])
                        if hasattr(self, 'current_rep_rw_pitch_values') and len(self.current_rep_rw_pitch_values) >= 3:
                            rw_range_debug = max(self.current_rep_rw_pitch_values) - min(self.current_rep_rw_pitch_values)
                        elif len(self.right_wrist_pitch_buffer) >= 3:
                            window = min(100 if self.exercise == 'tricep_extensions' else 50, len(self.right_wrist_pitch_buffer))
                            rw_range_debug = max(self.right_wrist_pitch_buffer[-window:]) - min(self.right_wrist_pitch_buffer[-window:])
                        
                        print(f"🎯 Rep detected via ORIENTATION: pitch_range={pitch_range:.1f}° "
                              f"(peak={self.pitch_peak:.1f}°, valley={self.pitch_valley:.1f}°) "
                              f"LW_range={lw_range_debug:.1f}°, RW_range={rw_range_debug:.1f}°")
                
                # Reset for next cycle
                self.pitch_peak = current_pitch
                self.pitch_valley = None
            elif self.pitch_peak is not None:
                # We had a peak but no valley yet - might be starting new rep
                self.pitch_peak = max(self.pitch_peak, current_pitch)
        
        elif self.current_direction == 'up' and new_direction == 'down':
            # Up → Down transition: Record peak (maximum pitch)
            if self.pitch_peak is None or current_pitch > self.pitch_peak:
                self.pitch_peak = max(recent_pitch) if recent_pitch else current_pitch
            # Don't set valley yet - wait for down → up to complete cycle
        
        elif new_direction == 'down':
            # Continuing down: Track valley (minimum pitch)
            if self.pitch_valley is None or current_pitch < self.pitch_valley:
                self.pitch_valley = min(recent_pitch) if recent_pitch else current_pitch
        
        elif new_direction == 'up':
            # Continuing up: Track peak (maximum pitch)
            if self.pitch_peak is None or current_pitch > self.pitch_peak:
                self.pitch_peak = max(recent_pitch) if recent_pitch else current_pitch
        
        # Update tracking
        self.last_pitch = current_pitch
        self.current_direction = new_direction
        
        return rep_detected
    
    def _detect_peak(self, timestamp: float) -> bool:
        """
        PRIMARY DETECTION: Gyroscope magnitude peak detection.
        Based on proven IMUPeriodicRepDetector logic, optimized for bicep curls.
        
        Strategy:
        1. Track gyroscope magnitude in sliding window
        2. Find local peaks above adaptive threshold (based on 75th percentile)
        3. Ensure minimum distance between peaks (based on rep durations)
        4. Count each peak as one rep
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if rep detected, False otherwise
        """
        if len(self.magnitude_buffer) < self.window_size:
            return False
        
        # Get recent signal window
        signal = np.array(self.magnitude_buffer[-self.window_size:])
        
        # Calculate adaptive thresholds (based on signal statistics)
        signal_median = np.median(signal)
        signal_75th = np.percentile(signal, 75)
        signal_90th = np.percentile(signal, 90)
        
        # Peak threshold: Based on real bicep curl data analysis
        # Data: bicep_curls_20260103_230109
        # Peak values: Min: 58.0, Max: 312.6, Mean: 224.2, Median: 236.4
        # Signal median: 99.6, 75th percentile: 162.8
        # Optimal peak threshold: 130.2 (80% of 75th percentile)
        # Use more conservative threshold to avoid false positives
        if len(self.rep_durations) > 0:
            avg_duration = np.median(self.rep_durations)
            duration_factor = 1.0 / max(0.1, min(3.0, avg_duration))
            peak_threshold_factor = 0.80 + (0.05 * duration_factor)  # 0.80-0.85 range (more conservative)
        else:
            # Initial detection: use 80% of 75th percentile (conservative)
            peak_threshold_factor = 0.80
        
        peak_threshold = signal_75th * peak_threshold_factor
        # Minimum peak height: Based on real data (peak values typically 200-300)
        # Be more conservative to avoid counting small noise peaks
        signal_mean = np.mean(signal)
        # Minimum peak should be at least 100 (based on plot data: peaks are 200-300)
        # Or at least 1.2x median (more conservative than before)
        min_peak_height = max(100.0, signal_median * 1.2, signal_mean * 1.1)  # Much more conservative
        
        # Initialize variables (will be set if valid peak found)
        rep_valid = False
        latest_peak_idx = None
        latest_peak_value = None
        latest_peak_rel_idx = None
        latest_peak_time = None
        
        # Find peaks in recent window (last 150 samples = ~6-7 reps, more tolerant)
        recent_window = min(150, len(signal))
        recent_signal = signal[-recent_window:]
        
        if len(recent_signal) < 7:  # Need at least 7 samples for peak detection
            return False
        
        # Local peak detection (wider neighborhood: ±3 samples)
        # Based on real data: peak distances min=4, max=64, mean=20.7, median=21.0 samples
        peaks_found = []
        for i in range(3, len(recent_signal) - 3):
            # Check if local maximum (wider neighborhood: ±3 samples)
            is_local_max = (recent_signal[i] > recent_signal[i-1] and 
                           recent_signal[i] > recent_signal[i+1] and
                           recent_signal[i] > recent_signal[i-2] and
                           recent_signal[i] > recent_signal[i+2] and
                           recent_signal[i] > recent_signal[i-3] and
                           recent_signal[i] > recent_signal[i+3])
            
            if is_local_max:
                peak_value = recent_signal[i]
                # Check threshold and minimum height (more conservative)
                if peak_value > peak_threshold and peak_value > min_peak_height:
                    # Calculate absolute index in buffer
                    peak_idx = len(self.magnitude_buffer) - recent_window + i
                    peaks_found.append((peak_idx, peak_value, i))  # Include relative index
        
        # Filter peaks: ensure minimum distance between peaks (based on real data: median ~21 samples)
        # This prevents counting multiple peaks from the same rep
        if len(peaks_found) > 1:
            filtered_peaks = []
            min_peak_distance_samples = 15  # Minimum 15 samples between peaks (~0.3s at 50Hz)
            for peak_idx, peak_value, rel_idx in peaks_found:
                if len(filtered_peaks) == 0:
                    filtered_peaks.append((peak_idx, peak_value, rel_idx))
                else:
                    # Check distance from last filtered peak
                    last_peak_idx = filtered_peaks[-1][0]
                    distance = abs(peak_idx - last_peak_idx)
                    if distance >= min_peak_distance_samples:
                        filtered_peaks.append((peak_idx, peak_value, rel_idx))
            peaks_found = filtered_peaks
        
        # Process latest peak
        if len(peaks_found) > 0:
            latest_peak_idx, latest_peak_value, latest_peak_rel_idx = peaks_found[-1]
            latest_peak_time = self.timestamps[latest_peak_idx] if latest_peak_idx < len(self.timestamps) else timestamp
            
            # Adaptive minimum distance between peaks (based on rep durations)
            # Real data: peak distances min=4, max=64, mean=20.7, median=21.0 samples
            # For bicep curl: 1 rep = 2 peaks, so minimum distance should be ~10-15 samples
            if len(self.rep_durations) > 0:
                avg_duration_samples = int(np.median(self.rep_durations) * self.sample_rate)
                # Minimum distance: at least 50% of average rep duration (more conservative)
                min_peak_distance = max(int(avg_duration_samples * 0.5), 10)  # 50% of avg duration, min 10
            else:
                min_peak_distance = 10  # Default: 10 samples (~0.2s at 50Hz) - more conservative
            
            # Check if this is a new peak (not already counted)
            # Use relative index check (more robust for buffer wrap-around)
            if self.last_peak_index >= 0:
                # Calculate relative positions in buffer
                buffer_len = len(self.magnitude_buffer)
                last_peak_rel_pos = self.last_peak_index / buffer_len if buffer_len > 0 else 0
                latest_peak_rel_pos = latest_peak_idx / buffer_len if buffer_len > 0 else 0
                
                # Check if latest peak is after last peak (considering wrap-around)
                index_diff = latest_peak_idx - self.last_peak_index
                
                # If index difference is small or negative, might be same peak
                # But if peak is in recent window's end and last was at beginning, it's new
                if index_diff <= 0:
                    # Check if this is a wrap-around case or truly a new peak after pause
                    if latest_peak_rel_idx > recent_window * 0.8:  # Peak is near end of recent window
                        # This might be a new peak after pause - check time difference
                        if self.last_rep_detection_time:
                            time_diff = latest_peak_time - self.last_rep_detection_time
                            if time_diff > self.rep_cooldown * 2:  # At least 2x cooldown (pause detected)
                                # Likely new rep after pause, allow it
                                if self.debug_enabled:
                                    print(f"🔄 New peak after pause detected (time_diff={time_diff:.2f}s)")
                            else:
                                # Too close, likely same peak
                                return False
                    else:
                        # Same peak, ignore
                        return False
                # If index difference is positive but too small, check time
                elif index_diff < min_peak_distance:
                    # Check time-based distance instead
                    if self.last_rep_detection_time:
                        time_diff = latest_peak_time - self.last_rep_detection_time
                        if time_diff < self.rep_cooldown:
                            return False
            
            # Check cooldown period (time-based, more reliable)
            if self.last_rep_detection_time is not None:
                time_since_last_rep = timestamp - self.last_rep_detection_time
                if time_since_last_rep < self.rep_cooldown:
                    return False
            
            # Check for idle period (don't reset, just log - allow detection after pause)
            # Use last_peak_time if available, otherwise last_rep_detection_time
            last_activity_time = self.last_peak_time if self.last_peak_time else self.last_rep_detection_time
            if last_activity_time is not None:
                time_since_last_activity = timestamp - last_activity_time
                if time_since_last_activity > self.max_idle_time:
                    # Reset peak tracking if pause too long (likely lost sync)
                    if self.last_peak_index >= 0 and self.debug_enabled:
                        print(f"⏸️  Long pause detected ({time_since_last_activity:.1f}s) - resetting peak tracking")
                    self.last_peak_index = -1
                    self.peak_count = 0  # Reset peak count on long pause (resync)
                    # Don't return False - continue to detect the peak (might be new rep start)
            
            # SECONDARY: ML form score validation (optional)
            rep_valid = True
            if self.use_ml_validation and len(self.form_scores) > 0:
                # Get recent form score
                recent_form_score = self.form_scores[-1] if self.form_scores else None
                if recent_form_score is not None and recent_form_score < self.min_form_score:
                    rep_valid = False
                    if self.debug_enabled:
                        print(f"⚠️  Rep rejected by ML validation (form_score={recent_form_score:.1f} < {self.min_form_score})")
            
        # Valid peak detected! (Only used if orientation detection didn't detect rep)
        # Only process if we have a valid peak and all peak variables are set
        if rep_valid and latest_peak_idx is not None:
            # Track peak for fallback/confirmation (but don't count as rep yet)
            # Orientation detection is primary, peak detection is fallback
            self.peak_count += 1
            self.last_peak_index = latest_peak_idx
            self.last_peak_time = latest_peak_time or timestamp
            
            # For peak-only detection: Every 2 peaks = 1 rep (fallback mode)
            # But orientation detection takes priority, so only count if orientation not available
            if not hasattr(self, 'pitch_buffer') or len(self.pitch_buffer) < 10:
                # No orientation data available, use peak-based detection
                if self.peak_count % 2 == 0:
                    # Complete rep detected (2 peaks = 1 rep)
                    rep_duration = None
                    if self.last_rep_detection_time and self.last_peak_time:
                        rep_duration = self.last_peak_time - self.last_rep_detection_time
                        if rep_duration > 0 and rep_duration < self.max_idle_time:
                            self.rep_durations.append(rep_duration)
                            if len(self.rep_durations) > 20:
                                self.rep_durations.pop(0)
                    
                    self.last_rep_detection_time = self.last_peak_time
                    self.rep_count += 1
                    self.last_rep_time = self.last_peak_time
                    
                    if self.debug_enabled:
                        duration_str = f" (duration={rep_duration:.2f}s)" if rep_duration and rep_duration > 0 else ""
                        print(f"✅ Rep #{self.rep_count} detected via PEAK (fallback) at buffer[{latest_peak_idx}] (peak #{self.peak_count}){duration_str}")
                    
                    return True
                else:
                    if self.debug_enabled:
                        print(f"📈 Peak #{self.peak_count} detected at buffer[{latest_peak_idx}] (need 1 more for rep - peak mode)")
            
            # Peak detected but orientation takes priority (just track, don't count yet)
            # Orientation detection will handle rep counting
            return False
        
        return False
    
    def process_imu_sample(self, imu_sample: dict, timestamp: float) -> Optional[Dict]:
        """
        Process IMU sample and detect rep if available.
        
        Args:
            imu_sample: IMU data dictionary
            timestamp: Sample timestamp
            
        Returns:
            Dict with rep info if rep detected, None otherwise
        """
        # PRIMARY: Calculate gyroscope magnitude
        gyro_mag = self._calculate_gyro_magnitude(imu_sample)
        if gyro_mag is not None:
            self.magnitude_buffer.append(gyro_mag)
            self.timestamps.append(timestamp)
            
            # Keep buffer limited but larger to handle pauses/delays
            max_buffer_size = self.window_size * 3  # Increase buffer size to handle pauses
            if len(self.magnitude_buffer) > max_buffer_size:
                # Remove oldest entries, but keep enough for peak detection
                remove_count = len(self.magnitude_buffer) - max_buffer_size
                self.magnitude_buffer = self.magnitude_buffer[remove_count:]
                self.timestamps = self.timestamps[remove_count:]
                # Adjust last_peak_index if buffer was trimmed
                if self.last_peak_index >= 0:
                    self.last_peak_index = max(0, self.last_peak_index - remove_count)
        
        # ORIENTATION-BASED: Extract and track pitch/roll/yaw (separate for LW and RW)
        orientation = self._extract_orientation(imu_sample)
        if orientation is not None:
            # Store averaged values (for backward compatibility)
            self.pitch_buffer.append(orientation['pitch'])
            self.roll_buffer.append(orientation['roll'])
            self.yaw_buffer.append(orientation['yaw'])
            self.orientation_timestamps.append(timestamp)
            
            # Store separate left and right wrist orientations
            if 'left_wrist' in orientation:
                lw = orientation['left_wrist']
                self.left_wrist_pitch_buffer.append(lw['pitch'])
                self.left_wrist_roll_buffer.append(lw['roll'])
                self.left_wrist_yaw_buffer.append(lw['yaw'])
                
                # Store unit vectors for left wrist (if available)
                if 'unit_vectors' in lw and UNIT_VECTORS_AVAILABLE:
                    uv = lw['unit_vectors']
                    # Normal Z component (vertical direction) - useful for up/down detection
                    if 'normal' in uv and len(uv['normal']) >= 3:
                        self.left_wrist_normal_z_buffer.append(float(uv['normal'][2]))
                    # Tangent Y component (forward direction) - useful for movement direction
                    if 'tangent' in uv and len(uv['tangent']) >= 2:
                        self.left_wrist_tangent_y_buffer.append(float(uv['tangent'][1]))
            
            if 'right_wrist' in orientation:
                rw = orientation['right_wrist']
                self.right_wrist_pitch_buffer.append(rw['pitch'])
                self.right_wrist_roll_buffer.append(rw['roll'])
                self.right_wrist_yaw_buffer.append(rw['yaw'])
                
                # Store unit vectors for right wrist (if available)
                if 'unit_vectors' in rw and UNIT_VECTORS_AVAILABLE:
                    uv = rw['unit_vectors']
                    # Normal Z component (vertical direction) - useful for up/down detection
                    if 'normal' in uv and len(uv['normal']) >= 3:
                        self.right_wrist_normal_z_buffer.append(float(uv['normal'][2]))
                    # Tangent Y component (forward direction) - useful for movement direction
                    if 'tangent' in uv and len(uv['tangent']) >= 2:
                        self.right_wrist_tangent_y_buffer.append(float(uv['tangent'][1]))
            
            # Keep orientation buffers limited
            max_orientation_buffer = 100  # Keep last 100 orientation samples
            if len(self.pitch_buffer) > max_orientation_buffer:
                self.pitch_buffer.pop(0)
                self.roll_buffer.pop(0)
                self.yaw_buffer.pop(0)
                self.orientation_timestamps.pop(0)
            
            # Keep separate wrist buffers limited
            if len(self.left_wrist_pitch_buffer) > max_orientation_buffer:
                self.left_wrist_pitch_buffer.pop(0)
                self.left_wrist_roll_buffer.pop(0)
                self.left_wrist_yaw_buffer.pop(0)
            if len(self.right_wrist_pitch_buffer) > max_orientation_buffer:
                self.right_wrist_pitch_buffer.pop(0)
                self.right_wrist_roll_buffer.pop(0)
                self.right_wrist_yaw_buffer.pop(0)
            
            # Keep unit vectors buffers limited
            if len(self.left_wrist_normal_z_buffer) > max_orientation_buffer:
                self.left_wrist_normal_z_buffer.pop(0)
                self.left_wrist_tangent_y_buffer.pop(0)
            if len(self.right_wrist_normal_z_buffer) > max_orientation_buffer:
                self.right_wrist_normal_z_buffer.pop(0)
                self.right_wrist_tangent_y_buffer.pop(0)
        
        # ONE-CLASS VALIDATION: Store IMU samples in buffer (for validation)
        if self.use_one_class_validation:
            self.imu_buffer.append(imu_sample)
            # Keep buffer limited (last 50 samples for validation)
            max_imu_buffer = 50
            if len(self.imu_buffer) > max_imu_buffer:
                self.imu_buffer.pop(0)
        
        # SECONDARY: Calculate ML form score (if ML available and validation enabled)
        if self.use_ml_validation and self.has_ml_model:
            # Add to IMU buffer for ML prediction
            if 'imu_buffer' not in self.__dict__:
                self.imu_buffer: List[dict] = []
            
            self.imu_buffer.append(imu_sample)
            if len(self.imu_buffer) > self.ml_window_size * 2:
                self.imu_buffer.pop(0)
            
            # Calculate form score for recent window
            if len(self.imu_buffer) >= self.ml_window_size:
                window_sequence = self.imu_buffer[-self.ml_window_size:]
                form_score = self._calculate_form_score(window_sequence)
                
                if form_score is not None:
                    self.form_scores.append(form_score)
                    self.score_timestamps.append(timestamp)
                    
                    # Keep form score history limited
                    if len(self.form_scores) > self.ml_window_size * 2:
                        self.form_scores.pop(0)
                        self.score_timestamps.pop(0)
        
        # PRIMARY: Try orientation-based detection first (more accurate for direction)
        # Support multiple exercises with orientation-based detection
        orientation_rep_detected = False
        supported_orientation_exercises = [
            'bicep_curls', 'lateral_shoulder_raises', 'squats', 'tricep_extensions',
            'dumbbell_shoulder_press', 'lateral_raises', 'front_raises', 'shoulder_press',
            'dumbbell_rows', 'tricep_extensions'
        ]
        if self.exercise in supported_orientation_exercises and len(self.pitch_buffer) >= 10:
            orientation_rep_detected = self._detect_rep_from_orientation(timestamp)
        
        # FALLBACK: Detect rep from gyroscope magnitude peak (if orientation not available)
        peak_rep_detected = False
        if not orientation_rep_detected:
            peak_rep_detected = self._detect_peak(timestamp)
        
        rep_detected = orientation_rep_detected or peak_rep_detected
        
        # Track current rep data for ensemble analysis
        if orientation is not None:
            self.current_rep_pitch_values.append(orientation['pitch'])
            self.current_rep_samples += 1
            
            # Track separate LW/RW pitch values for accurate range calculation
            if 'left_wrist' in orientation:
                lw = orientation['left_wrist']
                self.current_rep_lw_pitch_values.append(lw['pitch'])
                if 'roll' in lw:
                    self.current_rep_lw_roll_values.append(lw['roll'])
            if 'right_wrist' in orientation:
                rw = orientation['right_wrist']
                self.current_rep_rw_pitch_values.append(rw['pitch'])
                if 'roll' in rw:
                    self.current_rep_rw_roll_values.append(rw['roll'])
        if gyro_mag is not None:
            self.current_rep_gyro_values.append(gyro_mag)
        
        # Track rep start time
        if self.current_rep_start_time is None and len(self.current_rep_pitch_values) > 0:
            self.current_rep_start_time = timestamp
        
        if rep_detected:
            # Update rep count and tracking
            rep_duration = None
            if orientation_rep_detected:
                # Orientation-based rep detected
                if self.last_rep_detection_time and timestamp:
                    rep_duration = timestamp - self.last_rep_detection_time
                    if rep_duration > 0 and rep_duration < self.max_idle_time:
                        self.rep_durations.append(rep_duration)
                        if len(self.rep_durations) > 20:
                            self.rep_durations.pop(0)
                
                self.last_rep_detection_time = timestamp
                self.rep_count += 1
                self.last_rep_time = timestamp
                
                # Reset peak counter when using orientation (keep in sync)
                self.peak_count = 0
            
            # Calculate rep duration from start time
            if rep_duration is None and self.current_rep_start_time:
                rep_duration = timestamp - self.current_rep_start_time
            if rep_duration is None:
                rep_duration = 2.0  # Default duration if not tracked
            
            # IMPORTANT: For tricep extensions, ensure LW/RW buffers are populated
            # If current_rep buffers are empty, use recent buffer data
            if self.exercise == 'tricep_extensions':
                if len(self.current_rep_lw_pitch_values) == 0 and len(self.left_wrist_pitch_buffer) >= 10:
                    # Use recent LW buffer data (last 100 samples for full rep)
                    window = min(100, len(self.left_wrist_pitch_buffer))
                    self.current_rep_lw_pitch_values = self.left_wrist_pitch_buffer[-window:].copy()
                if len(self.current_rep_rw_pitch_values) == 0 and len(self.right_wrist_pitch_buffer) >= 10:
                    # Use recent RW buffer data (last 100 samples for full rep)
                    window = min(100, len(self.right_wrist_pitch_buffer))
                    self.current_rep_rw_pitch_values = self.right_wrist_pitch_buffer[-window:].copy()
            
            # ENSEMBLE ANALYSIS: Get form score and speed classification
            rep_analysis = self.analyze_rep_with_ensemble(rep_duration)
            form_score = rep_analysis.get('form_score', None)
            
            # Reset rep start time for next rep
            self.current_rep_start_time = timestamp
            
            if self.debug_enabled and orientation_rep_detected:
                speed_emoji = rep_analysis.get('speed_emoji', '')
                speed_label = rep_analysis.get('speed_label', '')
                duration_str = f" (duration={rep_duration:.2f}s, {speed_emoji}{speed_label})"
                print(f"✅ Rep #{self.rep_count} detected via ORIENTATION{duration_str}")
            
            # Peak-based rep handling is done in _detect_peak method
            result = {
                'rep': self.rep_count,
                'timestamp': timestamp,
                'form_score': form_score,
                'method': 'hybrid_peak_detection',
                'ml_validated': self.use_ml_validation and form_score is not None,
                # Ensemble analysis results
                'speed_class': rep_analysis.get('speed_class'),
                'speed_label': rep_analysis.get('speed_label'),
                'speed_emoji': rep_analysis.get('speed_emoji'),
                'speed_feedback': rep_analysis.get('speed_feedback'),
                'duration': rep_analysis.get('duration'),
                'form_feedback': rep_analysis.get('form_feedback'),
                'issues': rep_analysis.get('issues', []),
                'pitch_range': rep_analysis.get('pitch_range'),
                'gyro_magnitude': rep_analysis.get('gyro_magnitude'),
                # LW/RW pitch ranges for ROM feedback
                'lw_pitch_range': rep_analysis.get('lw_pitch_range', 0),
                'rw_pitch_range': rep_analysis.get('rw_pitch_range', 0),
                # IMU analysis for detailed feedback
                'imu_analysis': rep_analysis.get('imu_analysis', None)
            }
            
            if self.debug_enabled:
                form_score_str = f"{form_score:.1f}" if form_score is not None else "N/A"
                print(f"✅ Rep #{self.rep_count} detected (gyro peak, form_score={form_score_str})")
            
            return result
        
        # Debug output (periodic)
        if self.debug_enabled:
            current_time = time.time()
            if self.last_debug_time is None or (current_time - self.last_debug_time) >= self.debug_interval:
                form_score_str = f"{self.form_scores[-1]:.1f}" if len(self.form_scores) > 0 else "N/A"
                lw_buf = len(self.left_wrist_pitch_buffer)
                rw_buf = len(self.right_wrist_pitch_buffer)
                print(f"🔍 Hybrid Detector: buffer={len(self.magnitude_buffer)}, "
                      f"reps={self.rep_count}, LW={lw_buf}, RW={rw_buf}, form={form_score_str}")
                self.last_debug_time = current_time
        
        return None
    
    def add_imu_sample(self, imu_sample: dict, timestamp: Optional[float] = None) -> Optional[Dict]:
        """
        Alias for process_imu_sample (compatibility with other detectors).
        
        Args:
            imu_sample: IMU data dictionary (may contain 'timestamp' key)
            timestamp: Sample timestamp (optional, extracted from imu_sample or uses current time)
            
        Returns:
            Dict with rep info if rep detected, None otherwise
        """
        if timestamp is None:
            # Try to extract timestamp from IMU sample
            timestamp = imu_sample.get('timestamp', time.time())
        return self.process_imu_sample(imu_sample, timestamp)

