"""
FastAPI Backend for Fitness AI Coach - Refactored Version
==========================================================
Modular structure with separated handlers.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from openai import OpenAI

# Import optional dependencies
try:
    from dataset_collector import DatasetCollector
    DATASET_COLLECTION_ENABLED = True
except ImportError:
    DATASET_COLLECTION_ENABLED = False

try:
    from imu_dataset_collector import IMUDatasetCollector
    IMU_DATASET_COLLECTION_ENABLED = True
except ImportError:
    IMU_DATASET_COLLECTION_ENABLED = False
    print("⚠️  IMU dataset collection disabled (imu_dataset_collector not found)")

try:
    from ml_trainer import FormScorePredictor, BaselineCalculator
    ML_TRAINING_ENABLED = True
except ImportError:
    ML_TRAINING_ENABLED = False
    print("⚠️  ML training disabled (ml_trainer not found)")

try:
    from model_inference import ModelInference
    from ml_inference_helper import calculate_baseline_similarity, calculate_hybrid_correction_score, load_baselines
    ML_INFERENCE_ENABLED = True
except ImportError as e:
    ML_INFERENCE_ENABLED = False
    print(f"⚠️  ML inference disabled: {e}")

try:
    from dataset_tracker import DatasetTracker
    DATASET_TRACKER_ENABLED = True
except ImportError:
    DATASET_TRACKER_ENABLED = False
    print("⚠️  Dataset tracker disabled (dataset_tracker not found)")

# Import modularized services
import sys
sys.path.insert(0, '.')

from utils.pose_utils import (
    check_required_landmarks,
    calculate_angle,
    get_bone_vector,
    get_bone_length,
    get_bone_angle_from_vertical,
    get_bone_angle_from_horizontal,
    get_angle_between_bones,
    BONES
)
from services.form_analyzer import FormAnalyzer
from services.rep_counter import RepCounter
from services.imu_rep_detector import IMUPeriodicRepDetector
from services.feedback_service import (
    select_feedback_category,
    get_smart_feedback,
    get_rule_based_regional_feedback,
    get_regional_ai_feedback,
    get_rule_based_overall_feedback,
    EXERCISE_FEEDBACK_LIBRARY
)
from services.ai_service import (
    get_ai_feedback,
    send_ai_feedback_async,
    init_openai_client
)

# Import handlers
from handlers.config import EXERCISE_CONFIG
from handlers.state import (
    sessions,
    connected_clients,
    openai_client,
    camera_training_collectors,
    imu_training_collectors,
    imu_bridge_tasks,
    dataset_tracker,
    dataset_collector,
    DATASET_COLLECTION_ENABLED,
    DATASET_TRACKER_ENABLED,
    IMU_DATASET_COLLECTION_ENABLED
)
from handlers.utils import check_required_landmarks as utils_check_required_landmarks, calculate_angle as utils_calculate_angle
from handlers.ml_handlers import train_ml_model_async, update_model
from handlers.session_handlers import get_session_feedback, countdown_task, rest_countdown_task

# Initialize FastAPI app
app = FastAPI(title="Fitness AI Coach API")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client - import from websocket_handler
from handlers.websocket_handler import init_openai

# Import WebSocket handler
from handlers.websocket_handler import websocket_endpoint

# Simple route handlers
@app.get("/")
async def root():
    return {"message": "Fitness AI Coach API", "status": "running"}

@app.post("/api/update_model/{exercise}")
async def update_model_endpoint(exercise: str):
    """Update existing ML model using only unused datasets (exercise-specific)."""
    return await update_model(exercise)


# Register WebSocket route
app.websocket("/ws/{exercise}")(websocket_endpoint)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

