"""
Vector Database for Exercise Embeddings
Uses FAISS for efficient similarity search.
"""

import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class SearchResult:
    """Result from similarity search."""
    exercise: str
    similarity: float
    segment_id: int
    workout_id: str


class ExerciseVectorDB:
    """
    Vector database for exercise motion embeddings.
    Stores embeddings grouped by exercise type.
    Uses cosine similarity for search.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.indices: Dict[str, faiss.IndexFlatIP] = {}  # IP = Inner Product (cosine after L2 norm)
        self.metadata: Dict[str, List[Dict]] = {}  # Store segment metadata
        self.global_index: Optional[faiss.IndexFlatIP] = None
        self.global_metadata: List[Dict] = []
        
    def _create_index(self) -> faiss.IndexFlatIP:
        """Create a new FAISS index for cosine similarity."""
        # IndexFlatIP uses inner product, which equals cosine similarity for L2-normalized vectors
        return faiss.IndexFlatIP(self.embedding_dim)
    
    def add_embeddings(
        self,
        exercise: str,
        embeddings: np.ndarray,
        metadata_list: List[Dict] = None
    ):
        """
        Add embeddings for an exercise type.
        
        Args:
            exercise: Exercise name
            embeddings: Embedding matrix, shape (n, embedding_dim)
            metadata_list: Optional list of metadata dicts for each embedding
        """
        if exercise not in self.indices:
            self.indices[exercise] = self._create_index()
            self.metadata[exercise] = []
        
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Add to exercise-specific index
        self.indices[exercise].add(embeddings)
        
        # Store metadata
        if metadata_list is None:
            metadata_list = [{"id": i} for i in range(len(embeddings))]
        
        for meta in metadata_list:
            meta["exercise"] = exercise
        
        self.metadata[exercise].extend(metadata_list)
        
        print(f"Added {len(embeddings)} embeddings for '{exercise}' (total: {self.indices[exercise].ntotal})")
    
    def build_global_index(self):
        """
        Build a single global index containing all embeddings.
        Useful for finding similar movements across all exercises.
        """
        all_embeddings = []
        self.global_metadata = []
        
        for exercise, index in self.indices.items():
            n = index.ntotal
            if n > 0:
                # Reconstruct embeddings from index
                embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)
                for i in range(n):
                    embeddings[i] = faiss.rev_swig_ptr(index.get_xb(), n * self.embedding_dim).reshape(n, -1)[i]
                
                all_embeddings.append(embeddings)
                self.global_metadata.extend(self.metadata[exercise])
        
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings).astype(np.float32)
            self.global_index = self._create_index()
            self.global_index.add(all_embeddings)
            print(f"Built global index with {self.global_index.ntotal} embeddings")
    
    def search_exercise(
        self,
        query: np.ndarray,
        exercise: str,
        k: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar embeddings within a specific exercise type.
        
        Args:
            query: Query embedding, shape (embedding_dim,)
            exercise: Exercise type to search in
            k: Number of results to return
        
        Returns:
            List of SearchResult objects
        """
        if exercise not in self.indices:
            return []
        
        index = self.indices[exercise]
        if index.ntotal == 0:
            return []
        
        k = min(k, index.ntotal)
        
        # Prepare query
        query = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
        
        # Search
        similarities, indices = index.search(query, k)
        
        # Build results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0:
                meta = self.metadata[exercise][idx]
                results.append(SearchResult(
                    exercise=exercise,
                    similarity=float(sim),
                    segment_id=idx,
                    workout_id=meta.get("workout_id", "unknown")
                ))
        
        return results
    
    def search_all_exercises(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """
        Search for similar embeddings across all exercise types.
        
        Args:
            query: Query embedding, shape (embedding_dim,)
            k: Number of results per exercise
        
        Returns:
            Dict mapping exercise name to list of results
        """
        results = {}
        for exercise in self.indices:
            results[exercise] = self.search_exercise(query, exercise, k)
        return results
    
    def find_most_similar_exercise(
        self,
        query: np.ndarray,
        k: int = 5
    ) -> Tuple[str, float, List[SearchResult]]:
        """
        Find which exercise type the query is most similar to.
        
        Args:
            query: Query embedding
            k: Number of neighbors to consider
        
        Returns:
            Tuple of (exercise_name, avg_similarity, top_results)
        """
        all_results = self.search_all_exercises(query, k)
        
        # Calculate average similarity for each exercise
        exercise_scores = {}
        for exercise, results in all_results.items():
            if results:
                avg_sim = np.mean([r.similarity for r in results])
                exercise_scores[exercise] = (avg_sim, results)
        
        if not exercise_scores:
            return ("unknown", 0.0, [])
        
        # Find best matching exercise
        best_exercise = max(exercise_scores, key=lambda x: exercise_scores[x][0])
        avg_sim, results = exercise_scores[best_exercise]
        
        return (best_exercise, avg_sim, results)
    
    def get_exercise_stats(self) -> Dict[str, int]:
        """Get count of embeddings per exercise."""
        return {
            exercise: index.ntotal 
            for exercise, index in self.indices.items()
        }
    
    def save(self, path: str):
        """Save the vector database to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save each index
        for exercise, index in self.indices.items():
            faiss.write_index(index, str(path / f"{exercise}.index"))
        
        # Save metadata
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump({
                "embedding_dim": self.embedding_dim,
                "metadata": self.metadata,
                "exercises": list(self.indices.keys())
            }, f)
        
        print(f"Saved vector database to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ExerciseVectorDB":
        """Load a vector database from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
        
        db = cls(embedding_dim=data["embedding_dim"])
        db.metadata = data["metadata"]
        
        # Load indices
        for exercise in data["exercises"]:
            index_path = path / f"{exercise}.index"
            if index_path.exists():
                db.indices[exercise] = faiss.read_index(str(index_path))
        
        print(f"Loaded vector database from {path}")
        return db

