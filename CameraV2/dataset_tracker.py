"""
Dataset Tracking System
Tracks which datasets have been used for training and which are unused
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime


class DatasetTracker:
    """Tracks dataset usage for training."""
    
    def __init__(self, tracker_file: str = "dataset_tracker.json"):
        self.tracker_file = Path(tracker_file)
        self.tracker_data = self._load_tracker()
    
    def _load_tracker(self) -> Dict:
        """Load tracker data from file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Failed to load tracker: {e}")
                return {}
        return {
            'used_camera_sessions': [],
            'used_imu_sessions': [],
            'last_updated': None
        }
    
    def _save_tracker(self):
        """Save tracker data to file."""
        self.tracker_data['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.tracker_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save tracker: {e}")
    
    def mark_camera_session_used(self, session_id: str):
        """Mark a camera session as used for training."""
        if 'used_camera_sessions' not in self.tracker_data:
            self.tracker_data['used_camera_sessions'] = []
        
        if session_id not in self.tracker_data['used_camera_sessions']:
            self.tracker_data['used_camera_sessions'].append(session_id)
            self._save_tracker()
            print(f"✅ Marked camera session {session_id} as used")
    
    def mark_imu_session_used(self, session_id: str):
        """Mark an IMU session as used for training."""
        if 'used_imu_sessions' not in self.tracker_data:
            self.tracker_data['used_imu_sessions'] = []
        
        if session_id not in self.tracker_data['used_imu_sessions']:
            self.tracker_data['used_imu_sessions'].append(session_id)
            self._save_tracker()
            print(f"✅ Marked IMU session {session_id} as used")
    
    def mark_session_pair_used(self, camera_session_id: str, imu_session_id: str):
        """Mark both camera and IMU sessions as used (they are paired)."""
        self.mark_camera_session_used(camera_session_id)
        self.mark_imu_session_used(imu_session_id)
    
    def is_camera_session_used(self, session_id: str) -> bool:
        """Check if a camera session has been used for training."""
        return session_id in self.tracker_data.get('used_camera_sessions', [])
    
    def is_imu_session_used(self, session_id: str) -> bool:
        """Check if an IMU session has been used for training."""
        return session_id in self.tracker_data.get('used_imu_sessions', [])
    
    def get_unused_camera_sessions(self, dataset_dir: str = "MLTRAINCAMERA", exercise: str = None) -> List[str]:
        """
        Get list of unused camera session IDs.
        
        Args:
            dataset_dir: Dataset directory path
            exercise: If provided, only get sessions for this exercise
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            return []
        
        used_sessions = set(self.tracker_data.get('used_camera_sessions', []))
        all_sessions = []
        
        if exercise:
            # Only get sessions for this exercise (from exercise-specific folder)
            exercise_dir = dataset_path / exercise
            if exercise_dir.exists():
                all_sessions = [d.name for d in exercise_dir.iterdir() if d.is_dir()]
        else:
            # Get sessions from all exercises (for backwards compatibility)
            for exercise_dir in dataset_path.iterdir():
                if not exercise_dir.is_dir():
                    continue
                # Check if it's an exercise folder (contains session subdirectories)
                for session_dir in exercise_dir.iterdir():
                    if session_dir.is_dir():
                        all_sessions.append(session_dir.name)
        
        unused_sessions = [s for s in all_sessions if s not in used_sessions]
        
        return sorted(unused_sessions)
    
    def get_unused_imu_sessions(self, dataset_dir: str = "MLTRAINIMU", exercise: str = None) -> List[str]:
        """
        Get list of unused IMU session IDs.
        
        Args:
            dataset_dir: Dataset directory path
            exercise: If provided, only get sessions for this exercise
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            return []
        
        used_sessions = set(self.tracker_data.get('used_imu_sessions', []))
        all_sessions = []
        
        if exercise:
            # Only get sessions for this exercise (from exercise-specific folder)
            exercise_dir = dataset_path / exercise
            if exercise_dir.exists():
                all_sessions = [d.name for d in exercise_dir.iterdir() if d.is_dir()]
        else:
            # Get sessions from all exercises (for backwards compatibility)
            for exercise_dir in dataset_path.iterdir():
                if not exercise_dir.is_dir():
                    continue
                # Check if it's an exercise folder (contains session subdirectories)
                for session_dir in exercise_dir.iterdir():
                    if session_dir.is_dir():
                        all_sessions.append(session_dir.name)
        
        unused_sessions = [s for s in all_sessions if s not in used_sessions]
        
        return sorted(unused_sessions)
    
    def get_unused_session_pairs(self, camera_dir: str = "MLTRAINCAMERA", 
                                 imu_dir: str = "MLTRAINIMU", exercise: str = None) -> List[tuple]:
        """
        Get pairs of unused camera and IMU sessions (same session ID).
        
        Args:
            camera_dir: Camera dataset directory
            imu_dir: IMU dataset directory
            exercise: If provided, only get pairs for this exercise
        """
        unused_camera = set(self.get_unused_camera_sessions(camera_dir, exercise=exercise))
        unused_imu = set(self.get_unused_imu_sessions(imu_dir, exercise=exercise))
        
        # Find matching session IDs (they should have same session ID)
        pairs = []
        for session_id in unused_camera:
            if session_id in unused_imu:
                pairs.append((session_id, session_id))
        
        return pairs
    
    def get_all_used_sessions(self) -> Dict[str, List[str]]:
        """Get all used sessions."""
        return {
            'camera': self.tracker_data.get('used_camera_sessions', []),
            'imu': self.tracker_data.get('used_imu_sessions', [])
        }
    
    def reset_tracker(self):
        """Reset tracker (mark all sessions as unused)."""
        self.tracker_data = {
            'used_camera_sessions': [],
            'used_imu_sessions': [],
            'last_updated': datetime.now().isoformat()
        }
        self._save_tracker()
        print("🔄 Tracker reset - all sessions marked as unused")

