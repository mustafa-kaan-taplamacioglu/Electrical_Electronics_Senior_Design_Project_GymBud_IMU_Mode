# Exercise Embeddings Pipeline
# Vector database for fitness movement analysis

from .config import INCLUDED_EXERCISES, EMBEDDING_DIM
from .data_loader import MMFitDataLoader, RepetitionSegment, load_mediapipe_recording
from .joint_mapping import convert_mmfit_to_common, convert_mediapipe_to_common
from .feature_extractor import extract_all_features
from .embedding_generator import EmbeddingGenerator
from .vector_db import ExerciseVectorDB, SearchResult
from .pipeline import ExerciseEmbeddingPipeline, build_and_save_pipeline

__all__ = [
    "INCLUDED_EXERCISES",
    "EMBEDDING_DIM",
    "MMFitDataLoader",
    "RepetitionSegment",
    "load_mediapipe_recording",
    "convert_mmfit_to_common",
    "convert_mediapipe_to_common",
    "extract_all_features",
    "EmbeddingGenerator",
    "ExerciseVectorDB",
    "SearchResult",
    "ExerciseEmbeddingPipeline",
    "build_and_save_pipeline"
]

