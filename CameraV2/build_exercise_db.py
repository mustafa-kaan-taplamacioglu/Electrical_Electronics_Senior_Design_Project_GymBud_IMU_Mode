"""
Build Exercise Vector Database
==============================
Builds the exercise embedding pipeline from MM-Fit dataset.

Usage:
    python build_exercise_db.py
"""

import sys
sys.path.insert(0, ".")

from exercise_embeddings.pipeline import ExerciseEmbeddingPipeline
from exercise_embeddings.config import INCLUDED_EXERCISES


def main():
    print("\n" + "=" * 70)
    print("           EXERCISE VECTOR DATABASE BUILDER")
    print("=" * 70)
    
    print(f"\n📋 Included exercises (excluding Cardio & Core):")
    for i, ex in enumerate(INCLUDED_EXERCISES, 1):
        print(f"   {i}. {ex}")
    
    # Build pipeline
    print("\n" + "-" * 70)
    pipeline = ExerciseEmbeddingPipeline()
    pipeline.build(INCLUDED_EXERCISES, verbose=True)
    
    # Save pipeline
    print("\n" + "-" * 70)
    pipeline.save("exercise_pipeline")
    
    # Print summary
    print("\n" + "=" * 70)
    print("                     SUMMARY")
    print("=" * 70)
    
    stats = pipeline.vector_db.get_exercise_stats()
    total = sum(stats.values())
    
    print(f"\n📊 Database Statistics:")
    print(f"   Total embeddings: {total}")
    print(f"   Embedding dimension: {pipeline.embedding_dim}")
    print(f"\n   Per exercise:")
    for ex, count in sorted(stats.items()):
        print(f"      {ex:30} {count:4} embeddings")
    
    print("\n✅ Pipeline saved to: exercise_pipeline/")
    print("\n   Files:")
    print("      - exercise_pipeline/generator.pkl")
    print("      - exercise_pipeline/vector_db/")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()

