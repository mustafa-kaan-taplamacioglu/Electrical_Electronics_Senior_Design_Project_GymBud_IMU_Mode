"""
ML training and model update handlers
"""

import asyncio
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional

from handlers.state import sessions, dataset_tracker

# Import ML training enabled flag
try:
    from ml_trainer import FormScorePredictor, BaselineCalculator
    ML_TRAINING_ENABLED = True
except ImportError:
    ML_TRAINING_ENABLED = False

try:
    from dataset_tracker import DatasetTracker
    DATASET_TRACKER_ENABLED = True
except ImportError:
    DATASET_TRACKER_ENABLED = False


async def train_ml_model_async(
    exercise: str,
    websocket: WebSocket,
    camera_session_id: str = None,
    imu_session_id: str = None
):
    """Train ML model using collected datasets (both camera and IMU)."""
    if not ML_TRAINING_ENABLED:
        error_msg = f"⚠️  ML training disabled"
        print(error_msg)
        try:
            await websocket.send_json({
                'type': 'training_status',
                'status': 'error',
                'message': error_msg
            })
        except (RuntimeError, WebSocketDisconnect, AttributeError):
            pass
        return

    try:
        print(f"🤖 Starting ML model training for {exercise}...")
        
        from dataset_collector import DatasetCollector as DC
        from ml_trainer import FormScorePredictor
        
        # Load datasets
        camera_collector = DC("MLTRAINCAMERA")
        all_camera_samples = camera_collector.load_dataset()
        
        # Filter by exercise and unused sessions (if tracker enabled)
        camera_samples = [s for s in all_camera_samples if s.exercise == exercise]
        
        # Only use unused sessions if tracker is enabled
        if DATASET_TRACKER_ENABLED and dataset_tracker:
            # TODO: Implement proper session filtering
            pass
        
        if len(camera_samples) < 10:
            error_msg = f"⚠️  Not enough samples: {len(camera_samples)} < 10"
            print(error_msg)
            try:
                await websocket.send_json({
                    'type': 'training_status',
                    'status': 'error',
                    'message': error_msg
                })
            except (RuntimeError, WebSocketDisconnect, AttributeError):
                pass
            return
        
        # Auto-label if not labeled
        labeled_camera_samples = [s for s in camera_samples if s.expert_score is not None or s.is_perfect_form is not None]
        if len(labeled_camera_samples) == 0:
            for sample in camera_samples:
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                sample.expert_score = avg_score
                sample.is_perfect_form = (avg_score >= 90)
            labeled_camera_samples = camera_samples

        # Train model (run in thread to avoid blocking)
        def train_model():
            predictor = FormScorePredictor(model_type="random_forest")
            
            # Extract camera features if not already extracted
            for sample in labeled_camera_samples:
                if sample.features is None:
                    from camera_feature_extractor import CameraFeatureExtractor
                    extractor = CameraFeatureExtractor()
                    sample.features = extractor.extract_features(sample.landmarks_sequence)
            
            # Train camera model
            results = predictor.train(labeled_camera_samples, verbose=False, use_imu_features=False)
            
            # Save model with extended metadata
            model_dir = Path("models") / f"form_score_{exercise}_random_forest"
            model_dir.mkdir(parents=True, exist_ok=True)
            predictor.save(
                str(model_dir),
                exercise=exercise,
                training_samples=len(labeled_camera_samples),
                performance_metrics=results
            )
            
            # Calculate baselines if perfect samples exist
            perfect_samples = [s for s in labeled_camera_samples if s.is_perfect_form == True]
            if perfect_samples:
                from ml_trainer import BaselineCalculator
                calc = BaselineCalculator()
                baselines = calc.calculate_baselines(perfect_samples)
                
                # Save baselines
                baseline_file = model_dir / "baselines.json"
                import json
                with open(baseline_file, 'w') as f:
                    json.dump(baselines, f, indent=2, default=str)
            
            return (results, len(labeled_camera_samples))
        
        # Run training in thread pool
        training_result = await asyncio.to_thread(train_model)
        results, sample_count = training_result
        
        # Format performance metrics for display
        metrics_text = ""
        if results:
            metrics_text += f"   Test R²: {results.get('test_r2', 0):.3f}\n"
            metrics_text += f"   Test MAE: {results.get('test_mae', 0):.2f}\n"
            metrics_text += f"   Train R²: {results.get('train_r2', 0):.3f}"
        
        # Send success notification with performance metrics
        try:
            if websocket.client_state.name == 'CONNECTED':
                await websocket.send_json({
                    'type': 'training_status',
                    'status': 'completed',
                    'message': f'✅ ML model training completed! Model saved to models/form_score_{exercise}_random_forest/ ({sample_count} samples){metrics_text}',
                    'performance_metrics': results,
                    'sample_count': sample_count,
                    'model_path': f'models/form_score_{exercise}_random_forest/'
                })
        except (RuntimeError, WebSocketDisconnect, AttributeError):
            pass
        
        print(f"✅ ML model training completed for {exercise}")
        print(f"   - Model: models/form_score_{exercise}_random_forest/")
        print(f"   - Samples used: {sample_count}")
        
    except Exception as e:
        error_msg = f"⚠️  ML training failed: {str(e)}"
        print(error_msg)
        
        # Send error notification
        try:
            if websocket.client_state.name == 'CONNECTED':
                await websocket.send_json({
                    'type': 'training_status',
                    'status': 'error',
                    'message': error_msg
                })
        except (RuntimeError, WebSocketDisconnect, AttributeError):
            pass


async def update_model(exercise: str):
    """Update existing ML model using only unused datasets (exercise-specific)."""
    try:
        if not ML_TRAINING_ENABLED:
            return {
                "success": False,
                "message": "ML training disabled"
            }
        
        from pathlib import Path
        model_dir = Path("models") / f"form_score_{exercise}_random_forest"
        if not model_dir.exists():
            return {
                "success": False,
                "message": f"Model not found for {exercise}"
            }
        
        if not DATASET_TRACKER_ENABLED or dataset_tracker is None:
            return {
                "success": False,
                "message": "Dataset tracker not available"
            }
        
        # Get unused sessions for this exercise
        unused_camera_sessions = dataset_tracker.get_unused_camera_sessions(exercise)
        unused_imu_sessions = dataset_tracker.get_unused_imu_sessions(exercise)
        
        if not unused_camera_sessions:
            return {
                "success": False,
                "message": f"No unused training data found for {exercise}"
            }
        
        # Load samples from unused sessions
        from dataset_collector import DatasetCollector as DC
        camera_collector = DC("MLTRAINCAMERA")
        all_camera_samples = camera_collector.load_dataset()
        
        # Filter samples by exercise and unused sessions
        camera_samples = [
            s for s in all_camera_samples
            if s.exercise == exercise and getattr(s, 'session_id', None) in unused_camera_sessions
        ]
        
        if len(camera_samples) < 10:
            return {
                "success": False,
                "message": f"Not enough unused samples: {len(camera_samples)} < 10"
            }
        
        # Auto-label if not labeled
        for sample in camera_samples:
            if sample.expert_score is None:
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores) if sample.regional_scores else 100
                sample.expert_score = avg_score
                sample.is_perfect_form = (avg_score >= 90)
        
        # Extract features
        for sample in camera_samples:
            if sample.features is None:
                from camera_feature_extractor import CameraFeatureExtractor
                extractor = CameraFeatureExtractor()
                sample.features = extractor.extract_features(sample.landmarks_sequence)
        
        # Train model - MULTI-OUTPUT for regional scores
        from ml_trainer import FormScorePredictor
        predictor = FormScorePredictor(model_type="random_forest", multi_output=True)
        results = predictor.train(camera_samples, verbose=False, use_imu_features=False)
        
        # Save model (overwrite existing, exercise-specific) with extended metadata
        model_dir.mkdir(parents=True, exist_ok=True)
        predictor.save(
            str(model_dir),
            exercise=exercise,
            training_samples=len(camera_samples),
            performance_metrics=results
        )
        
        # Mark sessions as used (exercise-specific)
        for session_id in unused_camera_sessions:
            dataset_tracker.mark_camera_session_used(session_id)
        for session_id in unused_imu_sessions:
            dataset_tracker.mark_imu_session_used(session_id)
        
        return {
            "success": True,
            "message": f"Model updated successfully for {exercise}! Used {len(camera_samples)} samples from {len(unused_camera_sessions)} unused sessions."
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error updating model: {str(e)}"
        }

