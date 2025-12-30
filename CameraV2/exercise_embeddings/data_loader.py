"""
Data Loader for MM-Fit Dataset
Loads pose data and labels, filters by included exercises.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import MMFIT_PATH, INCLUDED_EXERCISES


@dataclass
class RepetitionSegment:
    """Single exercise repetition segment."""
    workout_id: str
    exercise: str
    start_frame: int
    end_frame: int
    rep_count: int
    pose_2d: np.ndarray  # (2, frames, joints)
    pose_3d: Optional[np.ndarray] = None  # (3, frames, joints)


class MMFitDataLoader:
    """Loads and processes MM-Fit dataset."""
    
    def __init__(self, mmfit_path: Path = MMFIT_PATH):
        self.mmfit_path = Path(mmfit_path)
        self.workout_ids = self._get_workout_ids()
        
    def _get_workout_ids(self) -> List[str]:
        """Get all workout folder IDs."""
        return sorted([
            d.name for d in self.mmfit_path.iterdir() 
            if d.is_dir() and d.name.startswith('w')
        ])
    
    def load_workout(self, workout_id: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load pose data and labels for a single workout."""
        workout_path = self.mmfit_path / workout_id
        
        pose_2d = np.load(workout_path / f"{workout_id}_pose_2d.npy")
        pose_3d = np.load(workout_path / f"{workout_id}_pose_3d.npy")
        
        labels = pd.read_csv(
            workout_path / f"{workout_id}_labels.csv",
            header=None,
            names=["start_frame", "end_frame", "rep_count", "exercise"]
        )
        
        return pose_2d, pose_3d, labels
    
    def get_repetition_segments(
        self, 
        exercises: List[str] = None
    ) -> List[RepetitionSegment]:
        """
        Extract all repetition segments from dataset.
        
        Args:
            exercises: List of exercise names to include. 
                      If None, uses INCLUDED_EXERCISES.
        
        Returns:
            List of RepetitionSegment objects.
        """
        if exercises is None:
            exercises = INCLUDED_EXERCISES
            
        segments = []
        
        for workout_id in self.workout_ids:
            try:
                pose_2d, pose_3d, labels = self.load_workout(workout_id)
            except Exception as e:
                print(f"Warning: Could not load {workout_id}: {e}")
                continue
            
            # Filter by included exercises
            filtered_labels = labels[labels['exercise'].isin(exercises)]
            
            for _, row in filtered_labels.iterrows():
                start = int(row['start_frame'])
                end = int(row['end_frame'])
                
                # Extract pose frames for this segment
                # MM-Fit format: (coords, frames, joints)
                segment_2d = pose_2d[:, start:end+1, :]
                segment_3d = pose_3d[:, start:end+1, :] if pose_3d is not None else None
                
                segment = RepetitionSegment(
                    workout_id=workout_id,
                    exercise=row['exercise'],
                    start_frame=start,
                    end_frame=end,
                    rep_count=int(row['rep_count']),
                    pose_2d=segment_2d,
                    pose_3d=segment_3d
                )
                segments.append(segment)
        
        return segments
    
    def get_segments_by_exercise(
        self, 
        exercises: List[str] = None
    ) -> Dict[str, List[RepetitionSegment]]:
        """
        Get segments grouped by exercise type.
        
        Returns:
            Dictionary mapping exercise name to list of segments.
        """
        segments = self.get_repetition_segments(exercises)
        
        grouped = {}
        for segment in segments:
            if segment.exercise not in grouped:
                grouped[segment.exercise] = []
            grouped[segment.exercise].append(segment)
        
        return grouped


def load_mediapipe_recording(
    pose_2d_path: str,
    pose_3d_path: str = None,
    labels_path: str = None
) -> List[RepetitionSegment]:
    """
    Load a MediaPipe recording in MM-Fit format.
    
    Args:
        pose_2d_path: Path to pose_2d.npy file
        pose_3d_path: Path to pose_3d.npy file (optional)
        labels_path: Path to labels.csv file (optional)
    
    Returns:
        List of RepetitionSegment objects
    """
    pose_2d = np.load(pose_2d_path)
    pose_3d = np.load(pose_3d_path) if pose_3d_path else None
    
    if labels_path:
        labels = pd.read_csv(
            labels_path, header=None,
            names=["start_frame", "end_frame", "rep_count", "exercise"]
        )
    else:
        # Treat entire recording as one segment
        labels = pd.DataFrame([{
            "start_frame": 0,
            "end_frame": pose_2d.shape[1] - 1,
            "rep_count": 0,
            "exercise": "unknown"
        }])
    
    segments = []
    for _, row in labels.iterrows():
        start = int(row['start_frame'])
        end = int(row['end_frame'])
        
        segment = RepetitionSegment(
            workout_id="recording",
            exercise=row['exercise'],
            start_frame=start,
            end_frame=end,
            rep_count=int(row['rep_count']),
            pose_2d=pose_2d[:, start:end+1, :],
            pose_3d=pose_3d[:, start:end+1, :] if pose_3d is not None else None
        )
        segments.append(segment)
    
    return segments

