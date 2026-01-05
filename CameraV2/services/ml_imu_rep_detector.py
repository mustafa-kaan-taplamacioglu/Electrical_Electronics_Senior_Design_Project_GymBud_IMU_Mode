"""
ML-based IMU Rep Detector
Uses trained ML model and saved bicep curl IMU data patterns for rep detection.
"""

import numpy as np
import time
from typing import Optional, Dict, List
from pathlib import Path
import json

try:
    from model_inference import ModelInference
    from imu_feature_extractor import extract_imu_features
    ML_INFERENCE_ENABLED = True
except ImportError:
    ML_INFERENCE_ENABLED = False
    print("⚠️  ML inference not available - falling back to rule-based detection")


class MLIMURepDetector:
    """
    ML-based IMU rep detector using trained model and saved patterns.
    Uses form score predictions from ML model to detect rep patterns.
    """
    
    def __init__(self, exercise: str, ml_inference: Optional[ModelInference] = None):
        """
        Initialize ML-based rep detector.
        
        Args:
            exercise: Exercise name (e.g., "bicep_curls")
            ml_inference: ModelInference instance (optional, will create if not provided)
        """
        self.exercise = exercise
        
        # Initialize ML inference
        if ml_inference:
            self.ml_inference = ml_inference
        else:
            self.ml_inference = ModelInference(exercise) if ML_INFERENCE_ENABLED else None
        
        # Check if IMU model is available
        self.has_ml_model = False
        if self.ml_inference:
            self.has_ml_model = self.ml_inference.has_imu_model()
            if self.has_ml_model:
                self.ml_inference.load_imu_model()
                print(f"✅ ML-based rep detector initialized with IMU model for {exercise}")
            else:
                print(f"⚠️  No IMU model found for {exercise} - will use pattern matching from saved data")
        
        # IMU sequence buffer (sliding window)
        self.imu_buffer: List[Dict] = []
        self.timestamps: List[float] = []
        
        # Form score history (for pattern detection)
        self.form_scores: List[float] = []
        self.score_timestamps: List[float] = []
        
        # Rep tracking
        self.rep_count = 0
        self.last_rep_time = None
        self.last_rep_detection_time = None
        
        # Pattern matching from saved data
        self.saved_patterns: List[List[Dict]] = []
        self._load_saved_patterns()
        
        # Detection parameters (optimized for bicep curls - periodic detection)
        self.min_window_size = 15  # Minimum samples for ML prediction (longer for better pattern detection)
        self.window_size = 25  # Sliding window size for form score calculation (larger for full rep cycle)
        self.rep_cooldown = 0.8  # Minimum time between reps (seconds) - bicep curl is ~1-2s per rep
        self.min_rep_duration = 0.6  # Minimum time for a complete rep cycle (seconds)
        self.max_idle_time = 5.0  # Max idle time before reset (seconds)
        
        # Pattern detection thresholds (adjusted for bicep curls - periodic pattern)
        self.form_score_peak_threshold = 0.45  # Form score must drop by this ratio to detect rep completion (conservative)
        self.form_score_rise_threshold = 0.3  # Form score must rise above this to start new rep
        self.min_score_range = 5.0  # Minimum score range for rep detection (to filter noise - higher for accuracy)
        self.min_absolute_drop = 5.0  # Minimum absolute form score drop for rep (conservative)
        
        # Debug tracking
        self.debug_enabled = True
        self.last_debug_time = None
        
        # State tracking
        self.is_tracking = False
        self.current_rep_start_time = None
        self.last_form_score = None
        self.form_score_peak = None
        self.form_score_valley = None
    
    def _load_saved_patterns(self):
        """Load saved bicep curl IMU patterns from training data."""
        if not self.has_ml_model:
            # Try to load patterns from saved data
            dataset_dir = Path("MLTRAINIMU") / self.exercise
            if not dataset_dir.exists():
                print(f"⚠️  No saved IMU patterns found for {self.exercise}")
                return
            
            # Load all sessions
            for session_dir in dataset_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                imu_samples_path = session_dir / "imu_samples.json"
                if not imu_samples_path.exists():
                    continue
                
                try:
                    with open(imu_samples_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract rep sequences (rep_number > 0)
                    if 'samples' in data:
                        for rep_data in data['samples']:
                            if rep_data.get('rep_number', 0) > 0 and 'samples' in rep_data:
                                imu_sequence = rep_data['samples']
                                if len(imu_sequence) > 0:
                                    self.saved_patterns.append(imu_sequence)
                    
                    print(f"✅ Loaded {len(self.saved_patterns)} rep patterns from {session_dir.name}")
                except Exception as e:
                    print(f"⚠️  Failed to load patterns from {session_dir.name}: {e}")
            
            if len(self.saved_patterns) > 0:
                print(f"✅ Loaded {len(self.saved_patterns)} saved IMU patterns for pattern matching")
    
    def _calculate_form_score(self, imu_sequence: List[Dict]) -> Optional[float]:
        """
        Calculate form score for IMU sequence using ML model.
        
        Args:
            imu_sequence: List of IMU samples
            
        Returns:
            Form score (0-100) or None if model not available
        """
        if not self.has_ml_model or not self.ml_inference:
            return None
        
        if len(imu_sequence) < self.min_window_size:
            return None
        
        try:
            # Use ML model to predict form score
            score = self.ml_inference.predict_imu(imu_sequence)
            return score
        except Exception as e:
            print(f"⚠️  Form score prediction error: {e}")
            return None
    
    def _detect_pattern_similarity(self, current_sequence: List[Dict]) -> float:
        """
        Calculate similarity between current sequence and saved patterns.
        
        Args:
            current_sequence: Current IMU sequence
            
        Returns:
            Similarity score (0-1) or 0 if no patterns available
        """
        if len(self.saved_patterns) == 0 or len(current_sequence) == 0:
            return 0.0
        
        # Extract features from current sequence
        try:
            current_features = extract_imu_features(current_sequence)
            if not current_features:
                return 0.0
        except Exception as e:
            print(f"⚠️  Feature extraction error: {e}")
            return 0.0
        
        # Calculate similarity with saved patterns
        max_similarity = 0.0
        
        for pattern in self.saved_patterns:
            if len(pattern) == 0:
                continue
            
            try:
                pattern_features = extract_imu_features(pattern)
                if not pattern_features:
                    continue
                
                # Calculate cosine similarity or feature distance
                # Simple approach: compare key features
                similarity = self._calculate_feature_similarity(current_features, pattern_features)
                max_similarity = max(max_similarity, similarity)
            except Exception as e:
                continue
        
        return max_similarity
    
    def _calculate_feature_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """
        Calculate similarity between two feature dictionaries.
        
        Args:
            features1: First feature dict
            features2: Second feature dict
            
        Returns:
            Similarity score (0-1)
        """
        if not features1 or not features2:
            return 0.0
        
        # Get common keys
        common_keys = set(features1.keys()) & set(features2.keys())
        if len(common_keys) == 0:
            return 0.0
        
        # Calculate normalized Euclidean distance
        differences = []
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            # Normalize by range (assume features are in reasonable ranges)
            diff = abs(val1 - val2) / (1.0 + abs(val1) + abs(val2))
            differences.append(diff)
        
        avg_diff = np.mean(differences) if differences else 1.0
        similarity = 1.0 / (1.0 + avg_diff)  # Convert distance to similarity
        
        return similarity
    
    def _detect_rep_from_form_score_pattern(self, timestamp: float) -> bool:
        """
        Detect rep completion from form score pattern (periodic bicep curl pattern).
        
        Strategy for bicep curls:
        1. Valley (start) → Peak (up, max contraction) → Valley (down, rep completes)
        2. Must have BOTH: significant rise (valley-to-peak) AND significant drop (peak-to-valley)
        3. Minimum rep duration check to filter noise
        4. Cooldown check to prevent multiple detections
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if rep detected, False otherwise
        """
        if len(self.form_scores) < 8:  # Need at least 8 scores for pattern detection
            return False
        
        # Get recent form scores with timestamps
        recent_scores = self.form_scores[-min(self.window_size, len(self.form_scores)):]
        recent_timestamps = self.score_timestamps[-min(self.window_size, len(self.form_scores)):]
        
        if len(recent_scores) < 8:
            return False
        
        # Calculate score range
        score_range = max(recent_scores) - min(recent_scores)
        if score_range < self.min_score_range:
            return False  # Not enough variation for a rep
        
        # Find peak and valley indices
        recent_scores_array = np.array(recent_scores)
        peak_idx = np.argmax(recent_scores_array)
        valley_before_idx = np.argmin(recent_scores_array[:peak_idx]) if peak_idx > 0 else None
        valley_after_idx = peak_idx + 1 + np.argmin(recent_scores_array[peak_idx + 1:]) if peak_idx < len(recent_scores) - 1 else None
        
        peak_score = recent_scores[peak_idx]
        current_score = recent_scores[-1]
        
        # Must have valley BEFORE peak (rep started with rise)
        if valley_before_idx is None or valley_before_idx >= peak_idx:
            return False  # No clear valley before peak
        
        # Must have valley AFTER peak (rep completed with drop)
        if valley_after_idx is None or valley_after_idx <= peak_idx:
            return False  # No clear valley after peak
        
        valley_before_score = recent_scores[valley_before_idx]
        valley_after_score = recent_scores[valley_after_idx] if valley_after_idx < len(recent_scores) else current_score
        
        # Check for complete valley-to-peak-to-valley pattern
        rise = peak_score - valley_before_score
        drop = peak_score - valley_after_score
        
        # Both rise AND drop must be significant (complete rep cycle)
        min_rise = score_range * 0.3  # At least 30% of range for rise
        min_drop = score_range * self.form_score_peak_threshold  # At least threshold% for drop
        
        if rise < min_rise or drop < min_drop:
            return False  # Incomplete pattern
        
        # Also check absolute drop (conservative)
        if drop < self.min_absolute_drop:
            return False  # Drop too small
        
        # Check minimum rep duration (valley before to valley after)
        if valley_before_idx < len(recent_timestamps) and valley_after_idx < len(recent_timestamps):
            rep_duration = recent_timestamps[valley_after_idx] - recent_timestamps[valley_before_idx]
            if rep_duration < self.min_rep_duration:
                return False  # Rep too fast (likely noise)
        
        # Check cooldown (time since last rep)
        if self.last_rep_detection_time is not None:
            time_since_last = timestamp - self.last_rep_detection_time
            if time_since_last < self.rep_cooldown:
                return False  # Too soon after last rep
        
        # All checks passed - valid rep detected
        if self.debug_enabled:
            print(f"🔍 Rep pattern detected: valley_before={valley_before_score:.1f}, peak={peak_score:.1f}, "
                  f"valley_after={valley_after_score:.1f}, rise={rise:.1f}, drop={drop:.1f}, range={score_range:.1f}")
        return True
    
    def add_imu_sample(self, imu_sample: Dict) -> Optional[Dict]:
        """
        Add IMU sample and detect rep using ML model.
        
        Args:
            imu_sample: IMU sample dict (left_wrist, right_wrist, chest)
            
        Returns:
            Rep completion dict or None
        """
        timestamp = imu_sample.get('timestamp', time.time())
        
        # Add to buffer
        self.imu_buffer.append(imu_sample)
        self.timestamps.append(timestamp)
        
        # Keep buffer size limited
        max_buffer_size = self.window_size * 3
        if len(self.imu_buffer) > max_buffer_size:
            self.imu_buffer.pop(0)
            self.timestamps.pop(0)
        
        # Need minimum samples for analysis
        if len(self.imu_buffer) < self.min_window_size:
            return None
        
        # Cooldown check
        if self.last_rep_detection_time is not None:
            time_since_last_rep = timestamp - self.last_rep_detection_time
            if time_since_last_rep < self.rep_cooldown:
                return None
            
            # Reset if idle too long
            if time_since_last_rep > self.max_idle_time:
                self.is_tracking = False
                self.form_score_peak = None
                self.form_score_valley = None
        
        # Calculate form score for current window
        window_sequence = self.imu_buffer[-self.window_size:]
        form_score = self._calculate_form_score(window_sequence)
        
        rep_detected = False
        detection_method = None
        
        # Debug logging (every 1 second)
        if self.debug_enabled and (self.last_debug_time is None or timestamp - self.last_debug_time > 1.0):
            self.last_debug_time = timestamp
            form_score_str = f"{form_score:.1f}" if form_score is not None else "None"
            print(f"🔍 ML Detector Debug: buffer={len(self.imu_buffer)}, form_scores={len(self.form_scores)}, "
                  f"last_form_score={form_score_str}, "
                  f"last_rep_time={self.last_rep_detection_time}, saved_patterns={len(self.saved_patterns)}")
        
        if form_score is not None:
            # Add to form score history
            self.form_scores.append(form_score)
            self.score_timestamps.append(timestamp)
            
            # Keep history limited
            if len(self.form_scores) > self.window_size * 3:  # Keep more history for better pattern detection
                self.form_scores.pop(0)
                self.score_timestamps.pop(0)
            
            # Detect rep from form score pattern
            if len(self.form_scores) >= 5:
                rep_detected = self._detect_rep_from_form_score_pattern(timestamp)
                if rep_detected:
                    detection_method = 'ml_pattern'
        
        # Fallback 1: Simpler pattern detection - only if main pattern detection fails
        # This is more conservative and requires larger drops
        if not rep_detected and len(self.form_scores) >= 10:
            recent_scores = self.form_scores[-10:]
            score_range = max(recent_scores) - min(recent_scores)
            
            if score_range >= self.min_score_range:
                # Find peak and check for significant drop after
                peak_idx = np.argmax(recent_scores)
                if peak_idx < len(recent_scores) - 3:  # Peak not at the very end
                    peak_score = recent_scores[peak_idx]
                    current_score = recent_scores[-1]
                    score_drop = peak_score - current_score
                    
                    # Must have both: relative drop >= threshold AND absolute drop >= minimum
                    drop_ratio = score_drop / score_range if score_range > 0 else 0
                    if drop_ratio >= self.form_score_peak_threshold and score_drop >= self.min_absolute_drop:
                        # Check cooldown and minimum duration
                        if self.last_rep_detection_time is None or \
                           (timestamp - self.last_rep_detection_time) >= self.rep_cooldown:
                            rep_detected = True
                            detection_method = 'simple_drop'
        
        # Fallback 2: Pattern matching with saved data (only as last resort)
        # This is disabled by default - only use ML pattern detection
        # if not rep_detected and len(self.saved_patterns) > 0 and len(self.imu_buffer) >= 30:
        #     current_sequence = self.imu_buffer[-min(40, len(self.imu_buffer)):]  # Last 40 samples
        #     similarity = self._detect_pattern_similarity(current_sequence)
        #     
        #     # Higher similarity threshold for accuracy (conservative)
        #     if similarity > 0.75:  # 75% similarity threshold (conservative)
        #         if self.last_rep_detection_time is None or \
        #            (timestamp - self.last_rep_detection_time) >= self.rep_cooldown:
        #             rep_detected = True
        #             detection_method = 'pattern_matching'
        
        if rep_detected:
            self.rep_count += 1
            self.last_rep_detection_time = timestamp
            self.last_rep_time = timestamp
            self.is_tracking = True
            
            # Reset pattern tracking
            self.form_score_peak = None
            self.form_score_valley = None
            
            result = {
                'rep': self.rep_count,
                'timestamp': timestamp,
                'form_score': form_score,
                'method': detection_method or ('ml_model' if form_score is not None else 'pattern_matching')
            }
            form_score_str = f"{form_score:.1f}" if form_score is not None else "None"
            print(f"✅ ML Rep #{self.rep_count} detected via {result['method']} (form_score={form_score_str})")
            return result
        
        return None
    
    def reset(self):
        """Reset detector state."""
        self.rep_count = 0
        self.last_rep_time = None
        self.last_rep_detection_time = None
        self.is_tracking = False
        self.form_scores = []
        self.score_timestamps = []
        self.imu_buffer = []
        self.timestamps = []
        self.form_score_peak = None
        self.form_score_valley = None

