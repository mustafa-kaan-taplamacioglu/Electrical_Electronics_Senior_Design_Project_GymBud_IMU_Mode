"""
Model Loading and Inference System
Loads and uses trained ML models for real-time form score prediction
Supports Camera-only, IMU-only, and Sensor Fusion modes
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
from ml_trainer import FormScorePredictor
from dataset_collector import DatasetCollector
from imu_feature_extractor import extract_imu_features
import json


class ModelInference:
    """Manages ML model loading and inference based on sensor fusion mode."""
    
    def __init__(self, exercise: str):
        """
        Initialize model inference system.
        
        Args:
            exercise: Exercise name (e.g., "bicep_curls")
        """
        self.exercise = exercise
        self.models_dir = Path("models")
        
        # Model instances (loaded lazily)
        self.camera_model: Optional[FormScorePredictor] = None
        self.imu_model: Optional[FormScorePredictor] = None
        
        # Feature extractors
        self.collector = DatasetCollector("dataset")
        
        # Model paths
        self.camera_model_path = None
        self.imu_model_path = None
        
        # Find available models
        self._find_models()
    
    def _find_models(self):
        """Find available camera and IMU models for the exercise."""
        # Look for camera model
        camera_patterns = [
            self.models_dir / f"form_score_{self.exercise}_camera_random_forest",
            self.models_dir / f"form_score_{self.exercise}_camera_gradient_boosting",
            self.models_dir / f"form_score_{self.exercise}_camera_ridge",
        ]
        
        for pattern in camera_patterns:
            if pattern.exists() and (pattern / "model.pkl").exists():
                self.camera_model_path = pattern
                break
        
        # Look for IMU model
        imu_patterns = [
            self.models_dir / f"form_score_{self.exercise}_imu_random_forest",
            self.models_dir / f"form_score_{self.exercise}_imu_gradient_boosting",
            self.models_dir / f"form_score_{self.exercise}_imu_ridge",
        ]
        
        for pattern in imu_patterns:
            if pattern.exists() and (pattern / "model.pkl").exists():
                self.imu_model_path = pattern
                break
    
    def load_camera_model(self) -> bool:
        """Load camera model if available."""
        if self.camera_model_path and self.camera_model is None:
            try:
                self.camera_model = FormScorePredictor.load(str(self.camera_model_path))
                print(f"✅ Loaded camera model: {self.camera_model_path}")
                return True
            except Exception as e:
                print(f"⚠️  Failed to load camera model: {e}")
                return False
        return self.camera_model is not None
    
    def load_imu_model(self) -> bool:
        """Load IMU model if available."""
        if self.imu_model_path and self.imu_model is None:
            try:
                self.imu_model = FormScorePredictor.load(str(self.imu_model_path))
                print(f"✅ Loaded IMU model: {self.imu_model_path}")
                return True
            except Exception as e:
                print(f"⚠️  Failed to load IMU model: {e}")
                return False
        return self.imu_model is not None
    
    def predict_camera(self, landmarks_sequence: list) -> Optional[float]:
        """
        Predict form score using camera model.
        
        Args:
            landmarks_sequence: List of landmark frames (List[List[Dict]])
            
        Returns:
            Predicted form score (0-100) or None if model not available
        """
        if not self.load_camera_model():
            return None
        
        # Create a temporary RepSample for feature extraction
        from dataset_collector import RepSample
        from datetime import datetime
        temp_sample = RepSample(
            timestamp=datetime.now().timestamp(),
            exercise=self.exercise,
            rep_number=0,
            landmarks_sequence=landmarks_sequence
        )
        
        # Extract camera features
        features = self.collector.extract_camera_features(temp_sample)
        
        if not features:
            return None
        
        # Predict
        try:
            score = self.camera_model.predict(features)
            return float(score)
        except Exception as e:
            print(f"⚠️  Camera model prediction error: {e}")
            return None
    
    def predict_imu(self, imu_sequence: list) -> Optional[float]:
        """
        Predict form score using IMU model.
        
        Args:
            imu_sequence: List of IMU samples
            
        Returns:
            Predicted form score (0-100) or None if model not available
        """
        if not self.load_imu_model():
            return None
        
        # Extract IMU features
        features = extract_imu_features(imu_sequence)
        
        if not features:
            return None
        
        # Predict
        try:
            score = self.imu_model.predict(features)
            return float(score)
        except Exception as e:
            print(f"⚠️  IMU model prediction error: {e}")
            return None
    
    def predict_fusion(self, landmarks_sequence: list, imu_sequence: list, 
                      camera_weight: float = 0.5, imu_weight: float = 0.5) -> Optional[float]:
        """
        Predict form score using both camera and IMU models (sensor fusion).
        
        Args:
            landmarks_sequence: List of landmark frames
            imu_sequence: List of IMU samples
            camera_weight: Weight for camera model prediction (default: 0.5)
            imu_weight: Weight for IMU model prediction (default: 0.5)
            
        Returns:
            Weighted average form score (0-100) or None if no models available
        """
        camera_score = self.predict_camera(landmarks_sequence)
        imu_score = self.predict_imu(imu_sequence)
        
        # Weighted fusion
        if camera_score is not None and imu_score is not None:
            # Both models available - weighted average
            total_weight = camera_weight + imu_weight
            fused_score = (camera_score * camera_weight + imu_score * imu_weight) / total_weight
            return float(fused_score)
        elif camera_score is not None:
            # Only camera available
            return camera_score
        elif imu_score is not None:
            # Only IMU available
            return imu_score
        else:
            # No models available
            return None
    
    def predict(self, mode: str, landmarks_sequence: list = None, 
                imu_sequence: list = None) -> Optional[float]:
        """
        Predict form score based on sensor fusion mode.
        
        Args:
            mode: "camera_only", "imu_only", or "camera_primary" (sensor fusion)
            landmarks_sequence: List of landmark frames (for camera modes)
            imu_sequence: List of IMU samples (for IMU modes)
            
        Returns:
            Predicted form score (0-100) or None
        """
        if mode == "camera_only":
            if landmarks_sequence is None:
                return None
            return self.predict_camera(landmarks_sequence)
        
        elif mode == "imu_only":
            if imu_sequence is None:
                return None
            return self.predict_imu(imu_sequence)
        
        elif mode == "camera_primary":
            # Sensor fusion mode - use both models
            # Default weights: 60% camera, 40% IMU
            return self.predict_fusion(
                landmarks_sequence or [],
                imu_sequence or [],
                camera_weight=0.6,
                imu_weight=0.4
            )
        
        else:
            print(f"⚠️  Unknown mode: {mode}")
            return None
    
    def has_camera_model(self) -> bool:
        """Check if camera model is available."""
        return self.camera_model_path is not None
    
    def has_imu_model(self) -> bool:
        """Check if IMU model is available."""
        return self.imu_model_path is not None

