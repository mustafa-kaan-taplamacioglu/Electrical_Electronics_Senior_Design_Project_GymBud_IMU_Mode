"""
Exercise Embeddings Pipeline
Main interface for building and querying the vector database.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import INCLUDED_EXERCISES, MMFIT_PATH, EMBEDDING_DIM
from .data_loader import MMFitDataLoader, RepetitionSegment, load_mediapipe_recording
from .embedding_generator import EmbeddingGenerator
from .vector_db import ExerciseVectorDB, SearchResult


class ExerciseEmbeddingPipeline:
    """
    Main pipeline for exercise motion analysis.
    Handles building, saving, loading, and querying the vector database.
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        mmfit_path: str = MMFIT_PATH
    ):
        self.embedding_dim = embedding_dim
        self.mmfit_path = Path(mmfit_path)
        
        self.data_loader = MMFitDataLoader(self.mmfit_path)
        self.generator = EmbeddingGenerator(embedding_dim)
        self.vector_db = ExerciseVectorDB(embedding_dim)
        
        self.is_built = False
    
    def build(
        self,
        exercises: List[str] = None,
        verbose: bool = True
    ):
        """
        Build the vector database from MM-Fit dataset.
        
        Args:
            exercises: List of exercises to include. Uses INCLUDED_EXERCISES if None.
            verbose: Print progress
        """
        if exercises is None:
            exercises = INCLUDED_EXERCISES
        
        if verbose:
            print("=" * 60)
            print("Building Exercise Embedding Pipeline")
            print("=" * 60)
            print(f"\nIncluded exercises: {exercises}")
        
        # Load segments grouped by exercise
        if verbose:
            print("\n Loading data...")
        
        segments_by_exercise = self.data_loader.get_segments_by_exercise(exercises)
        
        # Collect all segments for fitting
        all_segments = []
        for exercise, segments in segments_by_exercise.items():
            all_segments.extend(segments)
            if verbose:
                print(f"   {exercise}: {len(segments)} segments")
        
        if verbose:
            print(f"\n   Total: {len(all_segments)} segments")
        
        # Fit the embedding generator
        if verbose:
            print("\n Fitting embedding generator...")
        
        self.generator.fit(all_segments, source="mmfit")
        
        # Generate embeddings and add to vector database
        if verbose:
            print("\n Generating embeddings...")
        
        for exercise, segments in segments_by_exercise.items():
            embeddings = self.generator.generate_embeddings_batch(segments, source="mmfit")
            
            # Create metadata
            metadata_list = [
                {
                    "workout_id": seg.workout_id,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "rep_count": seg.rep_count
                }
                for seg in segments
            ]
            
            self.vector_db.add_embeddings(exercise, embeddings, metadata_list)
        
        self.is_built = True
        
        if verbose:
            print("\n" + "=" * 60)
            print("Pipeline built successfully!")
            print("=" * 60)
            print(f"\nDatabase stats:")
            for exercise, count in self.vector_db.get_exercise_stats().items():
                print(f"   {exercise}: {count} embeddings")
    
    def query_mediapipe_recording(
        self,
        pose_2d_path: str,
        pose_3d_path: str = None,
        k: int = 5
    ) -> Dict:
        """
        Query the database with a MediaPipe recording.
        
        Args:
            pose_2d_path: Path to pose_2d.npy file
            pose_3d_path: Path to pose_3d.npy file (optional)
            k: Number of similar matches to return
        
        Returns:
            Query results dictionary
        """
        if not self.is_built:
            raise ValueError("Pipeline not built. Call build() first.")
        
        # Load recording
        segments = load_mediapipe_recording(pose_2d_path, pose_3d_path)
        
        if not segments:
            return {"error": "No segments found in recording"}
        
        # Use first segment (entire recording)
        segment = segments[0]
        
        # Generate embedding
        embedding = self.generator.generate_embedding(segment, source="mediapipe")
        
        # Find most similar exercise
        best_exercise, avg_sim, top_results = self.vector_db.find_most_similar_exercise(
            embedding, k=k
        )
        
        # Get results for all exercises
        all_results = self.vector_db.search_all_exercises(embedding, k=k)
        
        return {
            "predicted_exercise": best_exercise,
            "confidence": avg_sim,
            "top_matches": [
                {
                    "exercise": r.exercise,
                    "similarity": r.similarity,
                    "workout_id": r.workout_id
                }
                for r in top_results
            ],
            "all_exercise_scores": {
                ex: np.mean([r.similarity for r in results]) if results else 0.0
                for ex, results in all_results.items()
            },
            "embedding": embedding.tolist()
        }
    
    def query_embedding(
        self,
        embedding: np.ndarray,
        k: int = 5
    ) -> Dict:
        """
        Query the database with a pre-computed embedding.
        
        Args:
            embedding: Query embedding vector
            k: Number of similar matches to return
        
        Returns:
            Query results dictionary
        """
        if not self.is_built:
            raise ValueError("Pipeline not built. Call build() first.")
        
        # Normalize if needed
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Find most similar exercise
        best_exercise, avg_sim, top_results = self.vector_db.find_most_similar_exercise(
            embedding, k=k
        )
        
        return {
            "predicted_exercise": best_exercise,
            "confidence": avg_sim,
            "top_matches": [
                {
                    "exercise": r.exercise,
                    "similarity": r.similarity,
                    "workout_id": r.workout_id
                }
                for r in top_results
            ]
        }
    
    def compare_segments(
        self,
        segment1: RepetitionSegment,
        segment2: RepetitionSegment,
        source: str = "mediapipe"
    ) -> float:
        """
        Compare similarity between two segments.
        
        Args:
            segment1, segment2: Segments to compare
            source: Data source type
        
        Returns:
            Cosine similarity score
        """
        emb1 = self.generator.generate_embedding(segment1, source)
        emb2 = self.generator.generate_embedding(segment2, source)
        
        # Cosine similarity (embeddings are already L2-normalized)
        return float(np.dot(emb1, emb2))
    
    def save(self, path: str):
        """Save the entire pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.generator.save(str(path / "generator.pkl"))
        self.vector_db.save(str(path / "vector_db"))
        
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ExerciseEmbeddingPipeline":
        """Load a pipeline from disk."""
        path = Path(path)
        
        pipeline = cls()
        pipeline.generator = EmbeddingGenerator.load(str(path / "generator.pkl"))
        pipeline.vector_db = ExerciseVectorDB.load(str(path / "vector_db"))
        pipeline.is_built = True
        
        print(f"Pipeline loaded from {path}")
        return pipeline


def build_and_save_pipeline(
    output_path: str = "exercise_pipeline",
    exercises: List[str] = None
) -> ExerciseEmbeddingPipeline:
    """
    Convenience function to build and save a pipeline.
    
    Args:
        output_path: Path to save the pipeline
        exercises: List of exercises to include
    
    Returns:
        Built pipeline
    """
    pipeline = ExerciseEmbeddingPipeline()
    pipeline.build(exercises)
    pipeline.save(output_path)
    return pipeline

