"""
Query Exercise Vector Database
==============================
Query the database with a new MediaPipe recording.

Usage:
    python query_exercise.py [pose_2d_path] [pose_3d_path]
    
Example:
    python query_exercise.py recording_pose_2d.npy recording_pose_3d.npy
"""

import sys
import json
import numpy as np

sys.path.insert(0, ".")

from exercise_embeddings.pipeline import ExerciseEmbeddingPipeline


def main():
    # Default paths
    pose_2d_path = "recording_pose_2d.npy"
    pose_3d_path = "recording_pose_3d.npy"
    
    # Override from command line
    if len(sys.argv) > 1:
        pose_2d_path = sys.argv[1]
    if len(sys.argv) > 2:
        pose_3d_path = sys.argv[2]
    
    print("\n" + "=" * 70)
    print("           EXERCISE SIMILARITY QUERY")
    print("=" * 70)
    
    # Load pipeline
    print("\n📂 Loading pipeline...")
    try:
        pipeline = ExerciseEmbeddingPipeline.load("exercise_pipeline")
    except FileNotFoundError:
        print("❌ Error: Pipeline not found. Run build_exercise_db.py first.")
        return
    
    # Load and display recording info
    print(f"\n📹 Query recording:")
    print(f"   2D Pose: {pose_2d_path}")
    print(f"   3D Pose: {pose_3d_path}")
    
    pose_2d = np.load(pose_2d_path)
    print(f"   Shape: {pose_2d.shape}")
    print(f"   Frames: {pose_2d.shape[1]}")
    
    # Query
    print("\n🔍 Querying database...")
    results = pipeline.query_mediapipe_recording(pose_2d_path, pose_3d_path, k=5)
    
    # Display results
    print("\n" + "=" * 70)
    print("                     RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 PREDICTED EXERCISE: {results['predicted_exercise'].upper()}")
    print(f"   Confidence: {results['confidence']:.4f}")
    
    print(f"\n📊 Top Matches:")
    for i, match in enumerate(results['top_matches'], 1):
        print(f"   {i}. {match['exercise']:30} sim: {match['similarity']:.4f}  (from {match['workout_id']})")
    
    print(f"\n📈 All Exercise Scores:")
    sorted_scores = sorted(
        results['all_exercise_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for ex, score in sorted_scores:
        bar = "█" * int(score * 40)
        print(f"   {ex:30} {score:.4f} {bar}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Save results to JSON
    output_file = "query_results.json"
    with open(output_file, "w") as f:
        # Remove embedding from saved results (too long)
        save_results = {k: v for k, v in results.items() if k != "embedding"}
        json.dump(save_results, f, indent=2)
    print(f"📄 Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()

