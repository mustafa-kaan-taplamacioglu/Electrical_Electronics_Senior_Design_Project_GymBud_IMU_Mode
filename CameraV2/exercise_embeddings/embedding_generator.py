"""
Embedding Generator
Converts pose sequences into fixed-length embedding vectors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from .data_loader import RepetitionSegment
from .joint_mapping import (
    convert_mmfit_to_common,
    convert_mediapipe_to_common,
    normalize_pose_to_relative,
    normalize_pose_scale
)
from .feature_extractor import extract_all_features, features_to_vector
from .config import EMBEDDING_DIM


class EmbeddingGenerator:
    """Generates fixed-length embeddings from pose sequences."""
    
    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.scaler = StandardScaler()
        self.feature_order = None
        self.is_fitted = False
        
    def preprocess_pose(
        self,
        pose_data: np.ndarray,
        source: str = "mmfit"  # "mmfit" or "mediapipe"
    ) -> np.ndarray:
        """
        Preprocess pose data: convert to common joints, normalize.
        
        Args:
            pose_data: Raw pose data, shape (coords, frames, joints)
            source: Data source ("mmfit" or "mediapipe")
        
        Returns:
            Preprocessed pose data
        """
        # Convert to common joint representation
        if source == "mmfit":
            common_pose = convert_mmfit_to_common(pose_data)
        else:
            common_pose = convert_mediapipe_to_common(pose_data)
        
        # Normalize: remove absolute position (center on nose)
        normalized = normalize_pose_to_relative(common_pose, reference_joint=0)
        
        # Normalize: scale by shoulder width
        scaled = normalize_pose_scale(normalized, left_shoulder_idx=1, right_shoulder_idx=2)
        
        return scaled
    
    def extract_features(
        self,
        pose_data: np.ndarray,
        fps: float = 30.0
    ) -> Dict[str, float]:
        """
        Extract kinematic features from preprocessed pose.
        
        Args:
            pose_data: Preprocessed pose data
            fps: Frames per second
        
        Returns:
            Feature dictionary
        """
        return extract_all_features(pose_data, fps)
    
    def fit(
        self,
        segments: List[RepetitionSegment],
        source: str = "mmfit"
    ):
        """
        Fit the embedding generator on training data.
        Learns feature scaling parameters.
        
        Args:
            segments: List of repetition segments
            source: Data source type
        """
        # Extract features from all segments
        all_features = []
        
        for segment in segments:
            preprocessed = self.preprocess_pose(segment.pose_2d, source)
            features = self.extract_features(preprocessed)
            all_features.append(features)
        
        if not all_features:
            raise ValueError("No segments provided for fitting")
        
        # Determine feature order (union of all features)
        all_keys = set()
        for f in all_features:
            all_keys.update(f.keys())
        self.feature_order = sorted(all_keys)
        
        # Convert to vectors
        vectors = np.array([
            features_to_vector(f, self.feature_order) 
            for f in all_features
        ])
        
        # Handle NaN/Inf values
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit scaler
        self.scaler.fit(vectors)
        self.is_fitted = True
        
        print(f"Fitted on {len(segments)} segments")
        print(f"Feature dimension: {len(self.feature_order)}")
    
    def generate_embedding(
        self,
        segment: RepetitionSegment,
        source: str = "mmfit"
    ) -> np.ndarray:
        """
        Generate embedding vector for a single segment.
        
        Args:
            segment: Repetition segment
            source: Data source type
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if not self.is_fitted:
            raise ValueError("Generator not fitted. Call fit() first.")
        
        # Preprocess
        preprocessed = self.preprocess_pose(segment.pose_2d, source)
        
        # Extract features
        features = self.extract_features(preprocessed)
        
        # Convert to vector
        vector = features_to_vector(features, self.feature_order)
        
        # Handle NaN/Inf
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        scaled = self.scaler.transform(vector.reshape(1, -1))[0]
        
        # If we have more features than embedding_dim, reduce via PCA-like compression
        # For now, we just truncate or pad
        if len(scaled) > self.embedding_dim:
            embedding = scaled[:self.embedding_dim]
        elif len(scaled) < self.embedding_dim:
            embedding = np.zeros(self.embedding_dim)
            embedding[:len(scaled)] = scaled
        else:
            embedding = scaled
        
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def generate_embeddings_batch(
        self,
        segments: List[RepetitionSegment],
        source: str = "mmfit"
    ) -> np.ndarray:
        """
        Generate embeddings for multiple segments.
        
        Args:
            segments: List of repetition segments
            source: Data source type
        
        Returns:
            Embedding matrix of shape (n_segments, embedding_dim)
        """
        embeddings = []
        for segment in segments:
            try:
                emb = self.generate_embedding(segment, source)
                embeddings.append(emb)
            except Exception as e:
                print(f"Warning: Failed to generate embedding: {e}")
                embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)
    
    def save(self, path: str):
        """Save the fitted generator to disk."""
        data = {
            "embedding_dim": self.embedding_dim,
            "feature_order": self.feature_order,
            "scaler": self.scaler,
            "is_fitted": self.is_fitted
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved generator to {path}")
    
    @classmethod
    def load(cls, path: str) -> "EmbeddingGenerator":
        """Load a fitted generator from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        generator = cls(embedding_dim=data["embedding_dim"])
        generator.feature_order = data["feature_order"]
        generator.scaler = data["scaler"]
        generator.is_fitted = data["is_fitted"]
        
        print(f"Loaded generator from {path}")
        return generator

