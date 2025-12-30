"""
Dataset Collection System for Exercise Form Analysis
Collects real-time rep data with labels for ML training
"""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle
from imu_feature_extractor import extract_imu_features


@dataclass
class RepSample:
    """Single rep sample with all features and labels."""
    # Metadata
    timestamp: float
    exercise: str
    rep_number: int
    # Raw pose data (landmarks over time) - must be before default arguments
    landmarks_sequence: List[List[Dict]]  # List of frames, each frame has 33 landmarks
    # Raw IMU data (IMU samples over time)
    imu_sequence: Optional[List[Dict]] = None  # List of IMU samples, each with left_wrist, right_wrist data
    user_id: str = "default"
    
    # Extracted features
    features: Dict[str, float] = None  # Camera-based features
    imu_features: Dict[str, float] = None  # IMU-based features
    
    # Labels (ground truth)
    expert_score: Optional[float] = None  # Expert-rated form score (0-100)
    user_feedback: Optional[str] = None  # User feedback ("perfect", "good", "bad")
    is_perfect_form: Optional[bool] = None  # Boolean label
    
    # Regional scores (from current system)
    regional_scores: Dict[str, float] = None
    regional_issues: Dict[str, List[str]] = None
    
    # Angles
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    range_of_motion: Optional[float] = None


class DatasetCollector:
    """Collects and manages exercise form dataset."""
    
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        
        self.current_session_id = None
        self.current_exercise = None  # Track current exercise
        self.current_samples: List[RepSample] = []
        self.is_collecting = False
    
    def start_session(self, exercise: str, user_id: str = "default") -> str:
        """Start a new data collection session."""
        session_id = f"{exercise}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session_id = session_id
        self.current_exercise = exercise  # Store exercise name
        self.current_samples = []
        self.is_collecting = True
        
        print(f"📊 Started data collection session: {session_id}")
        return session_id
    
    def add_rep_sample(
        self,
        exercise: str,
        rep_number: int,
        landmarks_sequence: List[List[Dict]],
        imu_sequence: List[Dict] = None,
        regional_scores: Dict[str, float] = None,
        regional_issues: Dict[str, List[str]] = None,
        min_angle: float = None,
        max_angle: float = None,
        user_id: str = "default"
    ) -> RepSample:
        """Add a rep sample to current session."""
        if not self.is_collecting:
            raise ValueError("No active collection session. Call start_session() first.")
        
        sample = RepSample(
            timestamp=datetime.now().timestamp(),
            exercise=exercise,
            rep_number=rep_number,
            user_id=user_id,
            landmarks_sequence=landmarks_sequence,
            imu_sequence=imu_sequence or [],
            regional_scores=regional_scores or {},
            regional_issues=regional_issues or {},
            min_angle=min_angle,
            max_angle=max_angle,
            range_of_motion=max_angle - min_angle if (min_angle and max_angle) else None
        )
        
        self.current_samples.append(sample)
        print(f"✅ Added rep #{rep_number} to session (total: {len(self.current_samples)}) [Landmarks: {len(landmarks_sequence)} frames, IMU: {len(imu_sequence) if imu_sequence else 0} samples]")
        
        return sample
    
    def label_sample(
        self,
        sample_index: int,
        expert_score: float = None,
        user_feedback: str = None,
        is_perfect_form: bool = None
    ):
        """Label a sample with expert/user feedback."""
        if sample_index >= len(self.current_samples):
            raise IndexError(f"Sample index {sample_index} out of range")
        
        sample = self.current_samples[sample_index]
        if expert_score is not None:
            sample.expert_score = expert_score
        if user_feedback is not None:
            sample.user_feedback = user_feedback
        if is_perfect_form is not None:
            sample.is_perfect_form = is_perfect_form
        
        print(f"🏷️  Labeled sample #{sample_index}: score={expert_score}, perfect={is_perfect_form}")
    
    def extract_camera_features(self, sample: RepSample, fps: float = 20.0) -> Dict[str, float]:
        """
        Extract camera-based features from landmarks sequence.
        (Renamed from extract_features for clarity)
        """
        return self.extract_features(sample, fps)
    
    def extract_features(self, sample: RepSample, fps: float = 20.0) -> Dict[str, float]:
        """Extract features from landmarks sequence using feature extractor with normalization."""
        from exercise_embeddings.feature_extractor import extract_all_features
        from exercise_embeddings.joint_mapping import (
            convert_mediapipe_to_common,
            normalize_pose_to_relative,
            normalize_pose_scale
        )
        
        # Convert landmarks to numpy array format
        # landmarks_sequence: List[List[Dict]] -> (coords, frames, joints)
        frames = len(sample.landmarks_sequence)
        if frames == 0:
            return {}
        
        # Map MediaPipe landmarks to common format
        # MediaPipe has 33 landmarks, we need to map to common format
        landmarks_array = []
        for frame_landmarks in sample.landmarks_sequence:
            # Extract x, y, z coordinates for each landmark
            frame_data = []
            for lm in frame_landmarks:
                frame_data.append([lm.get('x', 0), lm.get('y', 0), lm.get('z', 0) if 'z' in lm else 0])
            landmarks_array.append(frame_data)
        
        # Convert to (coords, frames, joints) format
        # Shape: (coords=3, frames, joints=33) for MediaPipe
        landmarks_np = np.array(landmarks_array).transpose(2, 0, 1)  # (coords, frames, joints)
        
        # NORMALIZATION PIPELINE
        # 1. Convert to common joint format (33 -> 13 joints)
        landmarks_common = convert_mediapipe_to_common(landmarks_np)  # (coords, frames, 13)
        
        # 2. Pelvis-center normalization (position invariance)
        # Pelvis = midpoint of left_hip (idx 7) and right_hip (idx 8)
        # Calculate pelvis center for each frame
        left_hip = landmarks_common[:, :, 7:8]  # (coords, frames, 1)
        right_hip = landmarks_common[:, :, 8:9]  # (coords, frames, 1)
        pelvis_center = (left_hip + right_hip) / 2  # (coords, frames, 1)
        
        # Subtract pelvis center from all joints
        normalized_pose = landmarks_common - pelvis_center
        
        # 3. Shoulder width normalization (scale invariance)
        # Left shoulder: idx 1, Right shoulder: idx 2 in common format
        normalized_pose = normalize_pose_scale(normalized_pose, left_shoulder_idx=1, right_shoulder_idx=2)
        
        # Extract features from normalized pose
        # Note: Angles are already size-invariant, but normalization helps with position/scale variations
        features = extract_all_features(normalized_pose, fps=fps)
        
        sample.features = features
        return features
    
    def save_session(self, auto_label_perfect: bool = False):
        """Save current session to disk."""
        if not self.current_session_id:
            raise ValueError("No active session to save")
        
        # Create exercise-specific folder structure
        # Format: {dataset_dir}/{exercise}/{session_id}/
        if self.current_exercise:
            exercise_dir = self.dataset_dir / self.current_exercise
            exercise_dir.mkdir(exist_ok=True)
            session_dir = exercise_dir / self.current_session_id
        else:
            # Fallback to old structure (for backwards compatibility)
            session_dir = self.dataset_dir / self.current_session_id
        
        session_dir.mkdir(exist_ok=True)
        
        # Extract features for all samples
        print("🔧 Extracting camera features...")
        for sample in self.current_samples:
            if sample.features is None:
                self.extract_camera_features(sample)
        
        # Extract IMU features for all samples
        print("🔧 Extracting IMU features...")
        for sample in self.current_samples:
            if sample.imu_features is None and sample.imu_sequence:
                sample.imu_features = self.extract_imu_features(sample)
        
        # Auto-label perfect reps if requested
        if auto_label_perfect:
            for sample in self.current_samples:
                if sample.regional_scores:
                    avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                    if avg_score >= 90:
                        sample.is_perfect_form = True
                        sample.expert_score = avg_score
        
        # Save samples as JSON
        samples_data = [asdict(sample) for sample in self.current_samples]
        
        # Convert numpy arrays to lists for JSON serialization
        for sample_data in samples_data:
            if 'landmarks_sequence' in sample_data:
                # Keep landmarks as-is (already dicts)
                pass
        
        with open(session_dir / "samples.json", "w") as f:
            json.dump(samples_data, f, indent=2, default=str)
        
        # Save summary CSV
        summary_path = session_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rep_number", "expert_score", "is_perfect", "user_feedback",
                "arms_score", "legs_score", "core_score", "head_score",
                "min_angle", "max_angle", "range_of_motion"
            ])
            
            for sample in self.current_samples:
                writer.writerow([
                    sample.rep_number,
                    sample.expert_score or "",
                    sample.is_perfect_form or "",
                    sample.user_feedback or "",
                    sample.regional_scores.get('arms', '') if sample.regional_scores else '',
                    sample.regional_scores.get('legs', '') if sample.regional_scores else '',
                    sample.regional_scores.get('core', '') if sample.regional_scores else '',
                    sample.regional_scores.get('head', '') if sample.regional_scores else '',
                    sample.min_angle or '',
                    sample.max_angle or '',
                    sample.range_of_motion or ''
                ])
        
        print(f"💾 Saved session to {session_dir}")
        print(f"   - {len(self.current_samples)} samples")
        print(f"   - {summary_path}")
        
        # Reset session
        self.current_session_id = None
        self.current_exercise = None
        self.current_samples = []
        self.is_collecting = False
    
    def load_dataset(self, exercise: str = None) -> List[RepSample]:
        """
        Load all samples from dataset directory.
        
        Args:
            exercise: If provided, only load samples for this exercise.
                     If None, load from all exercises.
        """
        all_samples = []
        
        # If exercise is specified, only look in that exercise's folder
        if exercise:
            exercise_dir = self.dataset_dir / exercise
            if not exercise_dir.exists():
                print(f"⚠️  No dataset folder found for exercise: {exercise}")
                return []
            
            # Load from exercise-specific folder
            for session_dir in exercise_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                samples_file = session_dir / "samples.json"
                if not samples_file.exists():
                    continue
                
                with open(samples_file, "r") as f:
                    samples_data = json.load(f)
                
                for sample_data in samples_data:
                    # Reconstruct RepSample
                    sample = RepSample(**sample_data)
                    all_samples.append(sample)
        else:
            # Load from all exercises (backwards compatibility + multi-exercise support)
            for exercise_dir in self.dataset_dir.iterdir():
                if not exercise_dir.is_dir():
                    continue
                
                # Check if it's an exercise folder (contains session subdirectories)
                for session_dir in exercise_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    
                    samples_file = session_dir / "samples.json"
                    if not samples_file.exists():
                        continue
                    
                    with open(samples_file, "r") as f:
                        samples_data = json.load(f)
                    
                    for sample_data in samples_data:
                        # Reconstruct RepSample
                        sample = RepSample(**sample_data)
                        all_samples.append(sample)
        
        print(f"📂 Loaded {len(all_samples)} samples from dataset" + (f" (exercise: {exercise})" if exercise else ""))
        return all_samples
    
    def get_perfect_form_baselines(self, exercise: str) -> Dict[str, Dict[str, float]]:
        """Calculate baseline values from perfect form samples."""
        all_samples = self.load_dataset()
        
        # Filter perfect form samples for this exercise
        perfect_samples = [
            s for s in all_samples
            if s.exercise == exercise and s.is_perfect_form == True
        ]
        
        if len(perfect_samples) == 0:
            print(f"⚠️  No perfect form samples found for {exercise}")
            return {}
        
        print(f"📊 Calculating baselines from {len(perfect_samples)} perfect form samples")
        
        # Extract features for all perfect samples
        for sample in perfect_samples:
            if sample.features is None:
                self.extract_features(sample)
        
        # Calculate statistics for each feature
        feature_names = list(perfect_samples[0].features.keys())
        baselines = {}
        
        for feature_name in feature_names:
            values = [s.features[feature_name] for s in perfect_samples if feature_name in s.features]
            if values:
                baselines[feature_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # Also calculate regional score baselines
        if perfect_samples[0].regional_scores:
            regional_baselines = {}
            for region in ['arms', 'legs', 'core', 'head']:
                scores = [s.regional_scores.get(region, 0) for s in perfect_samples if s.regional_scores]
                if scores:
                    regional_baselines[region] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores))
                    }
            baselines['regional_scores'] = regional_baselines
        
        return baselines


if __name__ == "__main__":
    # Example usage
    collector = DatasetCollector("dataset")
    
    # Start session
    session_id = collector.start_session("bicep_curls", user_id="test_user")
    
    # Simulate adding a rep (in real usage, this comes from WebSocket)
    # sample = collector.add_rep_sample(...)
    
    # Label it
    # collector.label_sample(0, expert_score=95, is_perfect_form=True)
    
    # Save
    # collector.save_session(auto_label_perfect=True)
    
    print("✅ Dataset collector ready!")

