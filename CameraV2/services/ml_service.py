"""ML service for model training."""

import asyncio
from fastapi import WebSocket

# ML Training imports (optional)
try:
    from ml_trainer import FormScorePredictor, BaselineCalculator
    from dataset_collector import DatasetCollector as DC
    ML_TRAINING_ENABLED = True
except ImportError:
    ML_TRAINING_ENABLED = False

async def train_ml_model_async(
    exercise: str,
    websocket: WebSocket,
    camera_session_id: str = None,
    imu_session_id: str = None
):
    """Train ML model using collected datasets (both camera and IMU)."""
    if not ML_TRAINING_ENABLED:
        error_msg = "ML training not available (ml_trainer not found)"
        print(f"⚠️  {error_msg}")
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_json({
                'type': 'training_status',
                'status': 'error',
                'message': error_msg
            })
        return
    
    try:
        print(f"🤖 Starting ML model training for {exercise}...")
        
        # Load datasets
        camera_collector = DC("MLTRAINCAMERA")
        all_camera_samples = camera_collector.load_dataset()
        
        # Filter by exercise and unused sessions (if tracker enabled)
        camera_samples = [s for s in all_camera_samples if s.exercise == exercise]
        
        # Note: dataset_tracker is passed from api_server if needed
        # For now, use all samples (tracker integration can be added later)
        # TODO: Implement proper session filtering with dataset_tracker
        
        if len(camera_samples) < 10:
            raise ValueError(f"Not enough camera samples for training (need >=10, got {len(camera_samples)})")
        
        # Auto-label if not labeled
        labeled_camera_samples = [s for s in camera_samples if s.expert_score is not None or s.is_perfect_form is not None]
        if len(labeled_camera_samples) == 0:
            print("   Auto-labeling camera samples based on regional scores...")
            for sample in camera_samples:
                if sample.regional_scores:
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
                    camera_collector.extract_features(sample)
            
            # Train camera model
            results = predictor.train(labeled_camera_samples, verbose=False, use_imu_features=False)
            
            # Save model with extended metadata
            from pathlib import Path
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
                baselines = BaselineCalculator.calculate_baselines(perfect_samples)
                
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
            metrics_text = f"\n📊 Model Performance:\n"
            metrics_text += f"   Test R²: {results.get('test_r2', 0):.3f}\n"
            metrics_text += f"   Test MAE: {results.get('test_mae', 0):.2f}\n"
            metrics_text += f"   Train R²: {results.get('train_r2', 0):.3f}"
        
        # Send success notification with performance metrics
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_json({
                'type': 'training_status',
                'status': 'completed',
                'message': f'✅ ML model training completed! Model saved to models/form_score_{exercise}_random_forest/ ({sample_count} samples){metrics_text}',
                'performance_metrics': results,
                'sample_count': sample_count,
                'model_path': f'models/form_score_{exercise}_random_forest/'
            })
        
        print(f"✅ ML model training completed for {exercise}")
        print(f"   - Model: models/form_score_{exercise}_random_forest/")
        print(f"   - Samples used: {sample_count}")
        
    except Exception as e:
        error_msg = f"⚠️  ML training failed: {str(e)}"
        print(error_msg)
        
        # Send error notification
        if websocket.client_state.name == 'CONNECTED':
            await websocket.send_json({
                'type': 'training_status',
                'status': 'error',
                'message': error_msg
            })


