"""
WebSocket handler for real-time pose analysis
"""

from fastapi import WebSocket, WebSocketDisconnect
import json
import numpy as np
from typing import Optional, Dict
import asyncio
import time
from datetime import datetime
import websockets
import copy

# Optional imports
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

try:
    from ml_trainer import FormScorePredictor, BaselineCalculator
    from dataset_collector import DatasetCollector as DC
    ML_TRAINING_ENABLED = True
except ImportError:
    ML_TRAINING_ENABLED = False

try:
    from model_inference import ModelInference
    from ml_inference_helper import calculate_baseline_similarity, calculate_hybrid_correction_score, load_baselines
    from imu_feature_extractor import extract_imu_features
    import numpy as np
    ML_INFERENCE_ENABLED = True
except ImportError as e:
    ML_INFERENCE_ENABLED = False

try:
    from dataset_tracker import DatasetTracker
    DATASET_TRACKER_ENABLED = True
except ImportError:
    DATASET_TRACKER_ENABLED = False

# Import services
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
from services.form_analyzer import FormAnalyzer, EXERCISE_CONFIG
from services.rep_counter import RepCounter
from services.imu_rep_detector import IMUPeriodicRepDetector
from services.ml_imu_rep_detector import MLIMURepDetector
from services.hybrid_imu_rep_detector import HybridIMURepDetector
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
from handlers.state import (
    sessions,
    camera_training_collectors,
    imu_training_collectors,
    imu_bridge_tasks,
    dataset_tracker,
    dataset_collector,
    openai_client,
    DATASET_COLLECTION_ENABLED as STATE_DATASET_COLLECTION_ENABLED,
    IMU_DATASET_COLLECTION_ENABLED as STATE_IMU_DATASET_COLLECTION_ENABLED
)
from handlers.ml_handlers import train_ml_model_async
from handlers.session_handlers import get_session_feedback, countdown_task, rest_countdown_task

# Use state module's flags (already imported from handlers.state)


def init_openai(api_key: str):
    """Initialize OpenAI client."""
    from handlers.state import openai_client as state_openai_client
    import handlers.state as state_module
    from openai import OpenAI
    state_module.openai_client = OpenAI(api_key=api_key)
    init_openai_client(state_module.openai_client)


# WebSocket endpoint function
async def websocket_endpoint(websocket: WebSocket, exercise: str):
    """WebSocket endpoint for real-time pose analysis."""
    # Accept WebSocket connection first
    try:
        await websocket.accept()
        print(f"✅ WebSocket connection accepted for {exercise}")
    except Exception as e:
        print(f"⚠️  Failed to accept WebSocket connection: {e}")
        return
    
    # Create session
    session_id = id(websocket)
    
    # Initialize ML inference if enabled
    ml_inference_instance = None
    baselines_dict = {}
    if ML_INFERENCE_ENABLED:
        try:
            ml_inference_instance = ModelInference(exercise)
            baselines_dict = load_baselines(exercise)
            print(f"🤖 ML Inference enabled for {exercise}")
        except Exception as e:
            print(f"⚠️  Failed to initialize ML inference: {e}")
            ml_inference_instance = None

    sessions[session_id] = {
        'exercise': exercise,  # Store exercise name for IMU bridge task filtering
        'websocket': websocket,  # Store websocket for IMU bridge task to send updates
        'form_analyzer': FormAnalyzer(exercise),
        'rep_counter': RepCounter(exercise),
        'state': 'detecting',
        'reps_data': [],       # Store all rep data
        'all_issues': [],      # Store all issues for session summary
        'landmarks_history': [],  # Store landmarks sequence for dataset collection
        'current_rep_landmarks': [],  # Landmarks for current rep
        'current_rep_imu': [],  # IMU data for current rep (list of IMU samples) - for usage mode
        'current_rep_imu_samples': [],  # IMU samples for training mode (separate from usage mode)
        'dataset_collection_enabled': False,  # Track if collection is active
        'collected_reps_count': 0,  # Count collected reps
        'last_camera_sample_time': None,  # For 20Hz throttling (50ms interval)
        'last_imu_sample_time': {},  # For 20Hz throttling per node (50ms interval per node)
        'ml_mode': 'usage',  # 'usage' (basic mode + data recording) or 'train' (ML training only)
        'training_session_started': False,  # Track if training collectors are started
        # ML Inference (for real-time predictions)
        'ml_inference': ml_inference_instance,
        'baselines': baselines_dict,
        # Workout configuration
        'workout_config': {
            'numberOfSets': 3,
            'repsPerSet': 10,
            'restTimeSeconds': 60
        },
        # Set tracking
        'current_set': 1,
        'current_rep_in_set': 0,
        'total_reps_in_session': 0,
    }
    
    try:
        while True:
            # Check WebSocket state before receiving
            try:
                ws_state = getattr(websocket, 'client_state', None)
                if ws_state and hasattr(ws_state, 'name') and ws_state.name != 'CONNECTED':
                    print(f"⚠️  WebSocket not connected, exiting loop (state: {ws_state.name if ws_state else 'unknown'})")
                    break
            except AttributeError:
                pass  # client_state not available, continue
            
            try:
                data = await websocket.receive_json()
            except (RuntimeError, WebSocketDisconnect) as e:
                print(f"⚠️  WebSocket receive error: {e}")
                break
            except Exception as e:
                print(f"⚠️  Unexpected error receiving message: {e}")
                import traceback
                traceback.print_exc()
                break
            
                # Initialize OpenAI (optional - if provided, will be used for enhanced feedback)
            if data.get('type') == 'init':
                if data.get('api_key'):
                    init_openai(data['api_key'])
                    print("✅ OpenAI API initialized (will be used for enhanced feedback)")
                else:
                    print("ℹ️  No API key provided - using rule-based feedback only")
                
                # Store ML mode (default: usage - basic mode with data recording)
                ml_mode = data.get('ml_mode', 'usage')
                fusion_mode = data.get('fusion_mode')  # Get fusion_mode from frontend
                print(f"📡 Received fusion_mode from frontend: {fusion_mode}")
                session = sessions.get(session_id)
                if session:
                    # Store fusion_mode in session
                    if fusion_mode:
                        session['fusion_mode'] = fusion_mode
                        print(f"✅ Fusion mode set to: {fusion_mode} (body detect: {'DISABLED' if fusion_mode == 'imu_only' else 'ENABLED'})")
                    else:
                        # Auto-detect fusion_mode based on available ML models
                        ml_inference = session.get('ml_inference')
                        if ml_inference:
                            has_camera = ml_inference.has_camera_model()
                            has_imu = ml_inference.has_imu_model()
                            if has_camera and has_imu:
                                session['fusion_mode'] = 'camera_primary'
                            elif has_imu:
                                session['fusion_mode'] = 'imu_only'
                            else:
                                session['fusion_mode'] = 'camera_only'
                        else:
                            session['fusion_mode'] = 'camera_only'

                    # Initialize Hybrid IMU rep detector for ALL modes (gyro peak + ML validation)
                    # This enables IMU-based rep detection even in fusion/camera modes
                    session['imu_periodic_detector'] = HybridIMURepDetector(exercise, ml_inference_instance)
                    print(f"✅ Hybrid IMU rep detector initialized for {exercise} (gyro peak + ML validation)")
                    
                    # IMU-only mode: Start countdown immediately
                    if fusion_mode == 'imu_only':
                        # Start countdown for IMU-only mode (like camera mode)
                        session['state'] = 'countdown'
                        print(f"🚀 IMU-only mode: Starting countdown before tracking")
                        try:
                            await websocket.send_json({
                                'type': 'state',
                                'state': 'countdown',
                                'message': 'IMU-only mode: Get ready! Starting countdown...'
                            })
                            # Start countdown in background task
                            asyncio.create_task(countdown_task(websocket, session_id))
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            pass
                    
                    # Usage mode: data recording enabled automatically
                    # Train mode: dataset collection enabled automatically
                    if ml_mode == 'usage':
                        session['dataset_collection_enabled'] = True  # Usage mode records data
                        # Usage mode: automatically start dataset collector
                        try:
                            user_id = data.get('user_id', 'default')
                            if dataset_collector:
                                dataset_collector.start_session(exercise, user_id=user_id)
                                session['collected_reps_count'] = 0
                                print(f"✅ Usage mode: Dataset collector started for {exercise}")
                        except Exception as e:
                            print(f"⚠️  Failed to start dataset collector in usage mode: {e}")
                        print(f"📊 Mode: {ml_mode} (Usage mode - data recording enabled)")
                    
                    elif ml_mode == 'train':
                        # Train mode: start separate collectors for camera and IMU (exercise-specific)
                        session['dataset_collection_enabled'] = True  # Train mode collects data

                        # Create a shared session_id to ensure synchronization
                        # Both camera and IMU collectors will use the same session_id
                        shared_session_id = f"{exercise}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        session['training_session_id'] = shared_session_id

                        # Get or create exercise-specific collectors
                        if exercise not in camera_training_collectors:
                            camera_training_collectors[exercise] = DatasetCollector("MLTRAINCAMERA")
                            camera_training_collectors[exercise].start_session(exercise)
                            # Override session_id to ensure sync (in case they were created in different seconds)
                            camera_training_collectors[exercise].current_session_id = shared_session_id
                            print(f"📹 Camera training collector started for {exercise} (session: {shared_session_id})")

                        if exercise not in imu_training_collectors:
                            imu_training_collectors[exercise] = IMUDatasetCollector("MLTRAINIMU")
                            imu_training_collectors[exercise].start_session(exercise)
                            # Override session_id to ensure sync (in case they were created in different seconds)
                            imu_training_collectors[exercise].current_session_id = shared_session_id
                            print(f"📡 IMU training collector started for {exercise} (session: {shared_session_id})")

                            # Connect to gymbud_imu_bridge WebSocket to receive IMU data directly (bypass frontend throttling)
                            # IMPORTANT: Task is exercise-level, but rep detection is session-specific
                            # Store this session_id for IMU bridge task to use
                            session['imu_bridge_session_id'] = session_id

                            # Check if task already exists for this exercise
                            if exercise not in imu_bridge_tasks:
                                async def imu_bridge_client_task():
                                    """Connect to gymbud_imu_bridge WebSocket and receive IMU data directly."""
                                    IMU_BRIDGE_WS_URL = "ws://localhost:8765"

                                    while True:
                                        try:
                                            async with websockets.connect(IMU_BRIDGE_WS_URL) as ws:
                                                print(f"✅ Connected to gymbud_imu_bridge WebSocket for {exercise}")

                                                async for message in ws:
                                                    try:
                                                        raw_data = json.loads(message)
                                                        
                                                        # Handle different data formats from gymbud_imu_bridge
                                                        # Format 1: Direct dict with 'nodes' key
                                                        # Format 2: List of messages (shouldn't happen but handle it)
                                                        if isinstance(raw_data, list):
                                                            # If it's a list, process each item
                                                            for item in raw_data:
                                                                if isinstance(item, dict) and 'nodes' in item:
                                                                    data = item
                                                                    break
                                                            else:
                                                                continue  # Skip if no valid data found
                                                        elif isinstance(raw_data, dict):
                                                            data = raw_data
                                                        else:
                                                            continue  # Skip invalid format
                                                        
                                                        # Verify data has 'nodes' key
                                                        if 'nodes' not in data or not isinstance(data.get('nodes'), dict):
                                                            continue
                                                        
                                                        # Find the MOST RECENT active IMU-only session for this exercise
                                                        # IMPORTANT: Only process for ONE session to prevent double counting
                                                        target_session_id = None
                                                        target_session = None
                                                        latest_start_time = 0

                                                        for active_session_id, active_session in list(sessions.items()):
                                                            if active_session.get('exercise') != exercise:
                                                                continue
                                                            fusion_mode = active_session.get('fusion_mode', 'camera_only')
                                                            if fusion_mode not in ['imu_only', 'camera_primary']:
                                                                continue
                                                            active_session['training_session_started'] = True

                                                            # Find the session with the most recent start time
                                                            # Use session_id as tiebreaker (higher = more recent)
                                                            session_start = active_session.get('session_start_time', 0)
                                                            session_id_numeric = int(str(active_session_id)[-10:]) if str(active_session_id)[-10:].isdigit() else 0
                                                            comparison_value = (session_start, session_id_numeric)
                                                            latest_comparison = (latest_start_time, int(str(target_session_id)[-10:]) if target_session_id and str(target_session_id)[-10:].isdigit() else 0)

                                                            if comparison_value > latest_comparison:
                                                                latest_start_time = session_start
                                                                target_session_id = active_session_id
                                                                target_session = active_session

                                                        # If no active session found, skip this IMU sample
                                                        if not target_session_id:
                                                            continue

                                                                # CRITICAL: Only process IMU data for the MOST RECENT session
                                                                # This prevents double counting when multiple sessions exist
                                                        
                                                        # Check session timeout (30 minutes inactivity)
                                                        session_start_time = target_session.get('session_start_time', time.time())
                                                        last_activity_time = target_session.get('last_activity_time', session_start_time)
                                                        current_time = time.time()
                                                        SESSION_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
                                                        
                                                        # Update last activity time
                                                        target_session['last_activity_time'] = current_time
                                                        
                                                        # Check timeout
                                                        if current_time - last_activity_time > SESSION_TIMEOUT_SECONDS:
                                                            print(f"⏱️  Session timeout detected for {exercise} (inactive for {int(current_time - last_activity_time)}s)")
                                                            target_session['state'] = 'finished'
                                                            # Send timeout message
                                                            session_websocket = target_session.get('websocket')
                                                            if session_websocket:
                                                                try:
                                                                    ws_state = getattr(session_websocket, 'client_state', None)
                                                                    if ws_state and hasattr(ws_state, 'name') and ws_state.name == 'CONNECTED':
                                                                        await session_websocket.send_json({
                                                                            'type': 'session_summary',
                                                                            'total_reps': target_session.get('total_reps_in_session', 0),
                                                                            'avg_form': 0,
                                                                            'feedback': 'Session timeout: No activity detected for 30 minutes.',
                                                                            'workout_complete': False,
                                                                            'timeout': True
                                                                        })
                                                                except Exception:
                                                                    pass
                                                            continue  # Skip processing this sample
                                                        active_session_id = target_session_id
                                                        active_session = target_session

                                                        # Create IMU sample data from gymbud_imu_bridge format
                                                        timestamp = data.get('timestamp', time.time())
                                                        imu_sample_data = {
                                                            'timestamp': timestamp,
                                                            'rep_number': 0  # Will be updated after rep detection
                                                        }

                                                        # Add each node's data (convert from nested to flat format)
                                                        nodes_data = data.get('nodes', {})
                                                        if not isinstance(nodes_data, dict):
                                                            continue  # Skip if nodes is not a dict
                                                        
                                                        for node_name in ['left_wrist', 'right_wrist', 'chest']:
                                                            if node_name not in nodes_data:
                                                                continue
                                                            node_data = nodes_data[node_name]
                                                            if not isinstance(node_data, dict):
                                                                continue
                                                            accel = node_data.get('accel', {})
                                                            gyro = node_data.get('gyro', {})
                                                            quaternion = node_data.get('quaternion', {})
                                                            euler = node_data.get('euler', {})
                                                            
                                                            # Ensure all are dicts
                                                            if not all(isinstance(x, dict) for x in [accel, gyro, quaternion, euler]):
                                                                continue
                                                            
                                                            imu_sample_data[node_name] = {
                                                                'ax': accel.get('x'),
                                                                'ay': accel.get('y'),
                                                                'az': accel.get('z'),
                                                                'gx': gyro.get('x'),
                                                                'gy': gyro.get('y'),
                                                                'gz': gyro.get('z'),
                                                                'qw': quaternion.get('w'),
                                                                'qx': quaternion.get('x'),
                                                                'qy': quaternion.get('y'),
                                                                'qz': quaternion.get('z'),
                                                                'roll': euler.get('roll'),
                                                                'pitch': euler.get('pitch'),
                                                                'yaw': euler.get('yaw')
                                                            }

                                                        # For IMU-only mode: Use IMUPeriodicRepDetector for rep detection
                                                        fusion_mode = active_session.get('fusion_mode', 'camera_only')
                                                        
                                                        # Skip processing if workout is complete (state is 'finished')
                                                        if active_session.get('state') == 'finished':
                                                            # Workout completed, skip rep detection
                                                            continue
                                                        
                                                        # Get IMU detector and collector
                                                        imu_detector = active_session.get('imu_periodic_detector')
                                                        imu_collector = imu_training_collectors.get(exercise) if IMU_DATASET_COLLECTION_ENABLED else None
                                                        
                                                        # Initialize rep buffer if needed
                                                        if 'current_rep_imu_samples' not in active_session:
                                                            active_session['current_rep_imu_samples'] = []

                                                        # Add current sample to rep buffer BEFORE detection (so detector has enough data)
                                                        active_session['current_rep_imu_samples'].append(copy.deepcopy(imu_sample_data))

                                                                        # Add sample to detector
                                                        timestamp = imu_sample_data.get('timestamp', time.time())
                                                        rep_result = imu_detector.process_imu_sample(imu_sample_data, timestamp)
                                                        
                                                        # Initialize workout_complete flag (will be set to True if workout is done)
                                                        workout_complete = False
                                                        current_rep_in_set = active_session.get('current_rep_in_set', 0)
                                                        current_set = active_session.get('current_set', 1)
                                                        reps_per_set = active_session.get('workout_config', {}).get('repsPerSet', 10)
                                                        number_of_sets = active_session.get('workout_config', {}).get('numberOfSets', 1)

                                                        # Rep detected!
                                                        if rep_result and rep_result.get('rep', 0) > 0:
                                                            rep_number = rep_result.get('rep', 0)
                                                            print(f"✅ IMU Rep #{rep_number} detected in train mode (session: {active_session_id})!")

                                                            # Get form score from ensemble model (IMU-only mode)
                                                            # Ensemble model provides better form scoring based on pitch range, speed, and movement quality
                                                            ensemble_form_score = rep_result.get('form_score', None)
                                                            ensemble_regional_scores = rep_result.get('regional_scores', {})
                                                            ensemble_issues = rep_result.get('issues', [])
                                                            ensemble_form_feedback = rep_result.get('form_feedback', '')
                                                            
                                                            # Use ensemble form score if available, otherwise fallback to ML model
                                                            form_score = 100.0  # Default
                                                            form_result = {'score': form_score, 'issues': [], 'regional_scores': {}, 'regional_issues': {}}
                                                            
                                                            if ensemble_form_score is not None:
                                                                # Use ensemble model's form score (preferred)
                                                                form_score = max(0.0, min(100.0, float(ensemble_form_score)))
                                                                form_result['score'] = form_score
                                                                form_result['issues'] = ensemble_issues
                                                                form_result['regional_scores'] = ensemble_regional_scores
                                                                form_result['regional_issues'] = rep_result.get('regional_issues', {})
                                                                
                                                                # Get LW/RW specific pitch ranges for feedback
                                                                lw_pitch_range = rep_result.get('lw_pitch_range', 0)
                                                                rw_pitch_range = rep_result.get('rw_pitch_range', 0)
                                                                
                                                                print(f"📊 Ensemble form score: {form_score:.1f}% | LW range: {lw_pitch_range:.1f}° | RW range: {rw_pitch_range:.1f}°")
                                                                
                                                                # Generate LW/RW specific feedback
                                                                if lw_pitch_range > 0 and rw_pitch_range > 0:
                                                                    pitch_diff = abs(lw_pitch_range - rw_pitch_range)
                                                                    if pitch_diff > 20:
                                                                        if lw_pitch_range > rw_pitch_range:
                                                                            form_result['regional_issues']['arms'] = form_result['regional_issues'].get('arms', []) + [f'Left arm has wider range of motion ({lw_pitch_range:.0f}° vs {rw_pitch_range:.0f}°)']
                                                                        else:
                                                                            form_result['regional_issues']['arms'] = form_result['regional_issues'].get('arms', []) + [f'Right arm has wider range of motion ({rw_pitch_range:.0f}° vs {lw_pitch_range:.0f}°)']
                                                            elif ml_inference_instance and ml_inference_instance.has_imu_model():
                                                                # Fallback to ML model
                                                                try:
                                                                    rep_imu_sequence = active_session.get('current_rep_imu_samples', [])
                                                                    if len(rep_imu_sequence) >= 10:
                                                                        clean_imu_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in rep_imu_sequence]
                                                                        predicted_score = ml_inference_instance.predict_imu(clean_imu_seq)
                                                                        if predicted_score is not None:
                                                                            form_score = max(0.0, min(100.0, float(predicted_score)))
                                                                            form_result['score'] = form_score
                                                                            print(f"📊 ML form score (fallback): {form_score:.1f}%")
                                                                except Exception as e:
                                                                    print(f"⚠️  Failed to calculate ML form score: {e}")
                                                            
                                                            # Update session form score
                                                            active_session['latest_form_score'] = form_score
                                                            active_session['last_form_score'] = form_score

                                                            # Update session rep counts (increment, don't use detector's rep_number directly)
                                                            # Detector's rep_number is absolute count from detector start, but we need session-level count
                                                            prev_total = active_session.get('total_reps_in_session', 0)
                                                            # Always increment by 1 - detector may reset or give inconsistent numbers
                                                            active_session['total_reps_in_session'] = prev_total + 1
                                                            active_session['collected_reps_count'] = active_session['total_reps_in_session']
                                                        
                                                            # Store rep data with form score, regional scores, and speed
                                                            rep_data = {
                                                                'rep': active_session['total_reps_in_session'],
                                                                'form_score': form_score,
                                                                'timestamp': timestamp,
                                                                'imu_samples': len(active_session.get('current_rep_imu_samples', [])),
                                                                'regional_scores': form_result.get('regional_scores', {}),
                                                                'regional_issues': form_result.get('regional_issues', {}),
                                                                'speed_class': rep_result.get('speed_class', 'medium'),
                                                                'speed_label': rep_result.get('speed_label', ''),
                                                                'duration': rep_result.get('duration', 0),
                                                                'lw_pitch_range': rep_result.get('lw_pitch_range', 0),
                                                                'rw_pitch_range': rep_result.get('rw_pitch_range', 0),
                                                                'imu_analysis': rep_result.get('imu_analysis', None)  # Add IMU analysis for session feedback
                                                            }
                                                            if 'reps_data' not in active_session:
                                                                active_session['reps_data'] = []
                                                            active_session['reps_data'].append(rep_data)

                                                            # Update current_rep_in_set and current_set
                                                            current_rep_in_set = active_session.get('current_rep_in_set', 0) + 1
                                                            current_set = active_session.get('current_set', 1)
                                                            reps_per_set = active_session.get('workout_config', {}).get('repsPerSet', 10)
                                                            number_of_sets = active_session.get('workout_config', {}).get('numberOfSets', 1)

                                                            # Check if set is complete (before updating)
                                                            set_complete = current_rep_in_set >= reps_per_set
                                                            workout_complete = set_complete and current_set >= number_of_sets
                                                            
                                                            if set_complete and not workout_complete:
                                                                # Set complete but more sets to go
                                                                current_set += 1
                                                                current_rep_in_set = 0
                                                            elif set_complete:
                                                                # Last set complete - workout is complete (keep rep count for display)
                                                                current_rep_in_set = reps_per_set  # Keep at max for display
                                                                workout_complete = True  # Ensure workout_complete is True
                                                                # Don't increment set - we're done

                                                            active_session['current_rep_in_set'] = current_rep_in_set
                                                            active_session['current_set'] = current_set

                                                            # Add current rep's IMU samples to collector (before resetting buffer)
                                                            if imu_collector and imu_collector.is_collecting:
                                                                try:
                                                                    # Remove rep_number field before adding
                                                                    rep_imu_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in active_session['current_rep_imu_samples']]
                                                                    imu_collector.add_rep_sequence(
                                                                        rep_number=rep_number,
                                                                        imu_sequence=rep_imu_seq,
                                                                        rep_start_time=timestamp
                                                                    )
                                                                    print(f"📡 Added IMU rep #{rep_number} to training collector ({len(rep_imu_seq)} samples)")
                                                                except Exception as e:
                                                                    print(f"⚠️  Failed to add IMU rep to collector: {e}")

                                                            # Send update to frontend (with safe WebSocket check)
                                                            session_websocket = active_session.get('websocket')
                                                            if session_websocket:
                                                                try:
                                                                    # Check WebSocket state safely
                                                                    ws_state = getattr(session_websocket, 'client_state', None)
                                                                    if ws_state and hasattr(ws_state, 'name') and ws_state.name == 'CONNECTED':
                                                                        # Get ensemble analysis from rep_result (speed, form, feedback, LW/RW)
                                                                        speed_emoji = rep_result.get('speed_emoji', '')
                                                                        speed_label = rep_result.get('speed_label', '')
                                                                        speed_class = rep_result.get('speed_class', 'medium')
                                                                        speed_feedback = rep_result.get('speed_feedback', '')
                                                                        rep_duration = rep_result.get('duration', 0)
                                                                        rep_form_feedback = rep_result.get('form_feedback', '')
                                                                        rep_issues = rep_result.get('issues', [])
                                                                        
                                                                        # Get LW/RW specific data from ensemble analysis
                                                                        regional_scores = rep_result.get('regional_scores', {'arms': form_score, 'legs': 100.0, 'core': 85.0, 'head': 90.0})
                                                                        lw_data = rep_result.get('left_wrist', {})
                                                                        rw_data = rep_result.get('right_wrist', {})
                                                                        sync_score = rep_result.get('sync_score', 100.0)
                                                                        lw_pitch_range = rep_result.get('lw_pitch_range', 0)
                                                                        rw_pitch_range = rep_result.get('rw_pitch_range', 0)
                                                                        
                                                                        # Generate comprehensive feedback message with form score and speed
                                                                        score_emoji = "🎉" if form_score >= 85 else "👍" if form_score >= 70 else "💪" if form_score >= 50 else "⚠️"
                                                                        feedback_msg = f"{score_emoji} Rep #{active_session['total_reps_in_session']} - Form: %{form_score:.0f}"
                                                                        if speed_emoji and speed_label:
                                                                            feedback_msg += f" | {speed_emoji} {speed_label}"
                                                                        if rep_duration and rep_duration > 0:
                                                                            feedback_msg += f" ({rep_duration:.1f}s)"
                                                                        
                                                                        await session_websocket.send_json({
                                                                            'type': 'update',
                                                                            'rep_count': active_session['total_reps_in_session'],
                                                                            'current_set': current_set,
                                                                            'current_rep_in_set': current_rep_in_set,
                                                                            'reps_per_set': reps_per_set,
                                                                            'number_of_sets': active_session.get('workout_config', {}).get('numberOfSets', 3),
                                                                            'feedback': feedback_msg,
                                                                            'angle': active_session.get('latest_angle', 0),
                                                                            'form_score': form_score,
                                                                            'avg_form': form_score,
                                                                            'phase': 'up',
                                                                            'speed_class': speed_class,
                                                                            'speed_label': speed_label,
                                                                            'speed_emoji': speed_emoji,
                                                                            'speed_feedback': speed_feedback,
                                                                            'rep_duration': rep_duration,
                                                                            'form_feedback': rep_form_feedback,
                                                                            'rep_issues': rep_issues,
                                                                            # Regional scores from ensemble analysis
                                                                            'regional_scores': regional_scores,
                                                                            # LW/RW specific data
                                                                            'left_wrist': lw_data,
                                                                            'right_wrist': rw_data,
                                                                            'sync_score': sync_score,
                                                                            'lw_pitch_range': lw_pitch_range,
                                                                            'rw_pitch_range': rw_pitch_range,
                                                                            'rep_completed': {
                                                                                'rep': active_session['total_reps_in_session'],
                                                                                'form_score': form_score,
                                                                                'is_valid': True,
                                                                                'regional_scores': regional_scores,
                                                                                'left_wrist': lw_data,
                                                                                'right_wrist': rw_data,
                                                                                'speed_class': speed_class,
                                                                                'speed_label': speed_label,
                                                                                'speed_emoji': speed_emoji,
                                                                                'duration': rep_duration
                                                                            }
                                                                        })
                                                                        lw_score = lw_data.get('score', '-') if lw_data else '-'
                                                                        rw_score = rw_data.get('score', '-') if rw_data else '-'
                                                                        print(f"📤 Sent rep #{active_session['total_reps_in_session']} update (form={form_score:.1f}%, LW={lw_score}, RW={rw_score}, speed={speed_label})")
                                                                    else:
                                                                        # WebSocket is not connected, remove it from session
                                                                        active_session['websocket'] = None
                                                                        print(f"⚠️  WebSocket not connected for session {active_session_id}, removed from session")
                                                                except AttributeError:
                                                                    # Try alternative method to check connection
                                                                    try:
                                                                        # Get ensemble data (use same variables defined above)
                                                                        speed_emoji = rep_result.get('speed_emoji', '') if rep_result else ''
                                                                        speed_label = rep_result.get('speed_label', '') if rep_result else ''
                                                                        speed_class = rep_result.get('speed_class', 'medium') if rep_result else 'medium'
                                                                        rep_duration = rep_result.get('duration', 0) if rep_result else 0
                                                                        regional_scores = rep_result.get('regional_scores', {'arms': form_score, 'legs': 100.0, 'core': 85.0, 'head': 90.0}) if rep_result else {}
                                                                        lw_data = rep_result.get('left_wrist', {}) if rep_result else {}
                                                                        rw_data = rep_result.get('right_wrist', {}) if rep_result else {}
                                                                        
                                                                        score_emoji = "🎉" if form_score >= 85 else "👍" if form_score >= 70 else "💪" if form_score >= 50 else "⚠️"
                                                                        feedback_msg = f"{score_emoji} Rep #{active_session['total_reps_in_session']} - Form: %{form_score:.0f}"
                                                                        if speed_emoji and speed_label:
                                                                            feedback_msg += f" | {speed_emoji} {speed_label}"
                                                                        
                                                                        await session_websocket.send_json({
                                                                            'type': 'update',
                                                                            'rep_count': active_session['total_reps_in_session'],
                                                                            'current_set': current_set,
                                                                            'current_rep_in_set': current_rep_in_set,
                                                                            'reps_per_set': reps_per_set,
                                                                            'number_of_sets': active_session.get('workout_config', {}).get('numberOfSets', 3),
                                                                            'feedback': feedback_msg,
                                                                            'angle': active_session.get('latest_angle', 0),
                                                                            'form_score': form_score,
                                                                            'avg_form': form_score,
                                                                            'phase': 'up',  # Default phase for IMU-only mode
                                                                            'speed_class': speed_class,
                                                                            'speed_label': speed_label,
                                                                            'speed_emoji': speed_emoji,
                                                                            'rep_duration': rep_duration,
                                                                            'regional_scores': regional_scores,
                                                                            'left_wrist': lw_data,
                                                                            'right_wrist': rw_data,
                                                                            'rep_completed': {
                                                                                'rep': active_session['total_reps_in_session'],
                                                                                'form_score': form_score,
                                                                                'is_valid': True,
                                                                                'regional_scores': regional_scores,
                                                                                'speed_class': speed_class,
                                                                                'speed_label': speed_label,
                                                                                'speed_emoji': speed_emoji,
                                                                                'duration': rep_duration
                                                                            }
                                                                        })
                                                                        print(f"📤 Sent rep #{active_session['total_reps_in_session']} update to frontend (form_score={form_score:.1f}%, speed={speed_label})")
                                                                    except Exception:
                                                                        active_session['websocket'] = None
                                                                        print(f"⚠️  Failed to send rep update, removed websocket from session")
                                                                except (RuntimeError, WebSocketDisconnect, AttributeError) as e:
                                                                    # WebSocket is closed, remove it from session
                                                                    active_session['websocket'] = None
                                                                    print(f"⚠️  Failed to send rep update to frontend (WebSocket closed): {e}")

                                                            # Send AI feedback for completed rep (IMU-only mode)
                                                            if openai_client:
                                                                try:
                                                                    # Get IMU data for feedback
                                                                    rep_imu_samples = active_session.get('current_rep_imu_samples', [])
                                                                    imu_data_for_feedback = None
                                                                    if rep_imu_samples:
                                                                        last_imu_sample = rep_imu_samples[-1].copy()
                                                                        last_imu_sample.pop('rep_number', None)
                                                                        last_imu_sample.pop('timestamp', None)
                                                                        imu_data_for_feedback = last_imu_sample
                                                                    
                                                                    # Create rep_result for feedback (include speed and LW/RW data)
                                                                    rep_result_for_feedback = {
                                                                        'rep': active_session['total_reps_in_session'],
                                                                        'form_score': form_score,
                                                                        'is_valid': True,
                                                                        'ml_prediction': {'score': form_score} if form_score < 100 else None,
                                                                        'speed_class': rep_result.get('speed_class', 'medium'),
                                                                        'speed_label': rep_result.get('speed_label', ''),
                                                                        'speed_emoji': rep_result.get('speed_emoji', ''),
                                                                        'duration': rep_result.get('duration', 0),
                                                                        'form_feedback': rep_result.get('form_feedback', ''),
                                                                        'issues': rep_result.get('issues', []),
                                                                        'lw_pitch_range': rep_result.get('lw_pitch_range', 0),
                                                                        'rw_pitch_range': rep_result.get('rw_pitch_range', 0)
                                                                    }
                                                                    
                                                                    asyncio.create_task(
                                                                        send_ai_feedback_async(
                                                                            session_websocket,
                                                                            exercise,
                                                                            rep_result_for_feedback,
                                                                            form_result.get('issues', []),
                                                                            form_result.get('regional_scores', {}),
                                                                            form_result.get('regional_issues', {}),
                                                                            ml_prediction=rep_result_for_feedback.get('ml_prediction'),
                                                                            imu_data=imu_data_for_feedback,
                                                                            landmarks=None,
                                                                            initial_positions=None,
                                                                            fusion_mode='imu_only'
                                                                        )
                                                                    )
                                                                except Exception as e:
                                                                    print(f"⚠️  Failed to send AI feedback: {e}")

                                                            # Reset rep buffer for next rep
                                                            active_session['current_rep_imu_samples'] = []

                                                            # Update rep number in imu_sample_data for session-level collection
                                                            imu_sample_data['rep_number'] = rep_number
                                                        
                                                        # Check if workout is complete (all sets done) - IMU-only mode
                                                        # IMPORTANT: Check this AFTER all rep processing to ensure session summary is sent
                                                        if workout_complete:
                                                            # Prevent further rep detection by setting state to finished
                                                            active_session['state'] = 'finished'
                                                            # All sets completed - automatically end session
                                                            print(f"🏁 IMU-only Workout complete! Set {current_set}/{number_of_sets}, Rep {current_rep_in_set}/{reps_per_set}")
                                                            active_session['state'] = 'finished'
                                                            
                                                            # Calculate session summary
                                                            ml_mode = active_session.get('ml_mode', 'usage')
                                                            collected_count = active_session.get('collected_reps_count', 0)
                                                            
                                                            # Generate session feedback
                                                            session_feedback = await get_session_feedback(
                                                                exercise,
                                                                active_session.get('reps_data', []),
                                                                active_session.get('all_issues', [])
                                                            )
                                                            
                                                            # Calculate average regional scores
                                                            if active_session.get('reps_data'):
                                                                all_regional_scores = [r.get('regional_scores', {}) for r in active_session['reps_data'] if r.get('regional_scores')]
                                                                if all_regional_scores:
                                                                    avg_regional_scores = {
                                                                        'arms': sum(r.get('arms', 0) for r in all_regional_scores) / len(all_regional_scores),
                                                                        'legs': sum(r.get('legs', 0) for r in all_regional_scores) / len(all_regional_scores),
                                                                        'core': sum(r.get('core', 0) for r in all_regional_scores) / len(all_regional_scores),
                                                                        'head': sum(r.get('head', 0) for r in all_regional_scores) / len(all_regional_scores),
                                                                    }
                                                                else:
                                                                    avg_regional_scores = {'arms': 0, 'legs': 0, 'core': 0, 'head': 0}
                                                            else:
                                                                avg_regional_scores = {'arms': 0, 'legs': 0, 'core': 0, 'head': 0}
                                                            
                                                            # Generate regional feedbacks
                                                            all_regional_issues = {}
                                                            for rep_data in active_session.get('reps_data', []):
                                                                regional_issues = rep_data.get('regional_issues', {})
                                                                for region in ['arms', 'legs', 'core', 'head']:
                                                                    if region not in all_regional_issues:
                                                                        all_regional_issues[region] = []
                                                                    if regional_issues.get(region):
                                                                        all_regional_issues[region].extend(regional_issues[region])
                                                            
                                                            regional_issues_summary = {}
                                                            for region in ['arms', 'legs', 'core', 'head']:
                                                                if all_regional_issues.get(region):
                                                                    issue_counts = {}
                                                                    for issue in all_regional_issues[region]:
                                                                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
                                                                    top_issue = max(issue_counts.items(), key=lambda x: x[1])[0] if issue_counts else None
                                                                    regional_issues_summary[region] = [top_issue] if top_issue else []
                                                                else:
                                                                    regional_issues_summary[region] = []
                                                            
                                                            # Calculate avg_form for feedback fallback
                                                            avg_form = round(
                                                                sum(r['form_score'] for r in active_session['reps_data']) / len(active_session['reps_data'])
                                                                if active_session.get('reps_data') else 0, 1
                                                            )
                                                            
                                                            regional_feedbacks = {}
                                                            for region in ['arms', 'legs', 'core', 'head']:
                                                                region_score = avg_regional_scores.get(region, 0)
                                                                region_issues = regional_issues_summary.get(region, [])
                                                                regional_feedbacks[region] = get_rule_based_regional_feedback(
                                                                    exercise, region, region_score, region_issues,
                                                                    rep_num=len(active_session.get('reps_data', [])), min_angle=None, max_angle=None,
                                                                    fallback_score=avg_form
                                                                )
                                                            
                                                            # Send session summary
                                                            total_reps = active_session.get('total_reps_in_session', len(active_session.get('reps_data', [])))
                                                            summary_data = {
                                                                'type': 'session_summary',
                                                                'total_reps': total_reps,
                                                                'avg_form': avg_form,
                                                                'regional_scores': avg_regional_scores,
                                                                'regional_feedback': regional_feedbacks,
                                                                'feedback': session_feedback,
                                                                'workout_complete': True,
                                                                'message': 'Workout completed automatically! All sets and reps finished.',
                                                                # Add ml_mode and flags for training dialog
                                                                'ml_mode': ml_mode,
                                                                'collected_count': collected_count,
                                                                'show_training_dialog': True
                                                            }
                                                            
                                                            # IMU-only mode: DON'T auto-save - let user decide via dialog
                                                            if ml_mode == 'train':
                                                                imu_collector = imu_training_collectors.get(exercise)
                                                                imu_samples = len(imu_collector.current_samples) if imu_collector else 0
                                                                session_imu_count = len(active_session.get('session_imu_samples', []))
                                                                print(f"📊 IMU-only Train mode data ready (NOT saved yet - waiting for user decision):")
                                                                print(f"   IMU collector: {imu_samples} rep sequences + {session_imu_count} session samples")
                                                            
                                                            # Send session summary
                                                            if session_websocket:
                                                                try:
                                                                    ws_state = getattr(session_websocket, 'client_state', None)
                                                                    if ws_state and hasattr(ws_state, 'name') and ws_state.name == 'CONNECTED':
                                                                        await session_websocket.send_json(summary_data)
                                                                        print(f"📤 Sent IMU-only session_summary: total_reps={summary_data['total_reps']}, workout_complete=True")
                                                                except (RuntimeError, WebSocketDisconnect, AttributeError) as e:
                                                                    print(f"⚠️  Failed to send session summary: {e}")
                                                            
                                                            # Don't continue processing - session is finished
                                                            continue
                                                        elif current_rep_in_set >= reps_per_set and current_set < number_of_sets:
                                                            # More sets to go - start rest period
                                                            active_session['state'] = 'resting'
                                                            rest_time = active_session.get('workout_config', {}).get('restTimeSeconds', 60)
                                                            if session_websocket:
                                                                try:
                                                                    ws_state = getattr(session_websocket, 'client_state', None)
                                                                    if ws_state and hasattr(ws_state, 'name') and ws_state.name == 'CONNECTED':
                                                                        await session_websocket.send_json({
                                                                            'type': 'state',
                                                                            'state': 'resting',
                                                                            'message': f'Set {current_set} complete! Rest time: {rest_time}s'
                                                                        })
                                                                        # Start rest countdown
                                                                        asyncio.create_task(
                                                                            rest_countdown_task(
                                                                                session_websocket,
                                                                                active_session_id,
                                                                                rest_time,
                                                                                current_set + 1
                                                                            )
                                                                        )
                                                                except (RuntimeError, WebSocketDisconnect, AttributeError):
                                                                    pass

                                                        # Add to session-level buffer (continuous collection)
                                                        if 'session_imu_samples' not in active_session:
                                                            active_session['session_imu_samples'] = []
                                                        active_session['session_imu_samples'].append(copy.deepcopy(imu_sample_data))
                                                        
                                                        continue
                                                    except Exception as e:
                                                        print(f"⚠️  Error processing IMU data from gymbud_imu_bridge: {e}")
                                        except websockets.exceptions.ConnectionClosedOK:
                                            print(f"🔌 Disconnected from gymbud_imu_bridge WebSocket for {exercise}. Reconnecting...")
                                            await asyncio.sleep(1)
                                        except Exception as e:
                                            print(f"❌ Failed to connect to gymbud_imu_bridge WebSocket for {exercise}: {e}. Retrying in 1 second...")
                                            await asyncio.sleep(1)

                                # Start the WebSocket client task (exercise-level, not session-level)
                                task = asyncio.create_task(imu_bridge_client_task())
                                imu_bridge_tasks[exercise] = task
                                print(f"🚀 Started IMU bridge WebSocket client task for {exercise} (shared across all sessions)")
                            else:
                                print(f"ℹ️  IMU bridge task already running for {exercise}, reusing existing task")

                        session['training_session_started'] = True
                        # Initialize session-level buffers for continuous data collection (rep sayılsın ya da sayılmasın)
                        session['session_landmarks'] = []  # All landmarks throughout session
                        session['session_imu_samples'] = []  # All IMU samples throughout session
                        session['session_start_time'] = time.time()
                        # Initialize rep-level buffers (will be reset after each rep is saved)
                        session['current_rep_landmarks'] = []  # Current rep's landmarks
                        session['current_rep_imu_samples'] = []  # Current rep's IMU samples
                        print(f"📊 Mode: {ml_mode} (ML Training mode - synchronized camera & IMU collection enabled for {exercise})")
                        print(f"   Shared session ID: {shared_session_id}")
                        print(f"   Session-level continuous collection enabled (all data, regardless of rep counting)")

                    # Store workout configuration
                    workout_config = data.get('workout_config', {})
                    if workout_config:
                        session['workout_config'] = {
                            'numberOfSets': workout_config.get('numberOfSets', 3),
                            'repsPerSet': workout_config.get('repsPerSet', 10),
                            'restTimeSeconds': workout_config.get('restTimeSeconds', 60)
                        }
                        session['current_set'] = 1
                        session['current_rep_in_set'] = 0
                        session['total_reps_in_session'] = 0

                    # Only send 'ready' if not in IMU-only mode (already sent state message above)
                    if session and session.get('fusion_mode') != 'imu_only':
                        try:
                            await websocket.send_json({'type': 'ready'})
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            pass
                    continue
            
            # Handle dataset collection start/stop
            if data.get('type') == 'start_collection':
                try:
                    user_id = data.get('user_id', 'default')
                    if dataset_collector:
                        dataset_collector.start_session(exercise, user_id=user_id)
                        session['dataset_collection_enabled'] = True
                        session['collected_reps_count'] = 0
                        try:
                            await websocket.send_json({
                                'type': 'dataset_collection_status',
                                'status': 'collecting',
                                'collected_reps': 0
                            })
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            pass
                        print(f"✅ Dataset collection started for {exercise}")
                except Exception as e:
                    print(f"⚠️  Failed to start dataset collection: {e}")
                    try:
                        await websocket.send_json({
                            'type': 'dataset_collection_status',
                            'status': 'error',
                            'error': str(e)
                        })
                    except (RuntimeError, WebSocketDisconnect, AttributeError):
                        pass
                else:
                    try:
                        await websocket.send_json({
                            'type': 'dataset_collection_status',
                            'status': 'error',
                            'error': 'Dataset collector not available'
                        })
                    except (RuntimeError, WebSocketDisconnect, AttributeError):
                        pass
                continue

            if data.get('type') == 'stop_collection':
                try:
                    # Save current session
                    auto_label = data.get('auto_label_perfect', True)
                    if dataset_collector:
                        dataset_collector.save_session(auto_label_perfect=auto_label)
                        collected_count = session.get('collected_reps_count', 0)
                        session['dataset_collection_enabled'] = False

                        # Safely send status update
                        try:
                            await websocket.send_json({
                                'type': 'dataset_collection_status',
                                'status': 'saved',
                                'collected_reps': collected_count,
                                'message': f'Dataset saved: {collected_count} reps collected'
                            })
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            # WebSocket closed, silently ignore
                            pass
                        print(f"💾 Dataset saved: {collected_count} reps")
                    else:
                        # Safely send idle status
                        try:
                            await websocket.send_json({
                                'type': 'dataset_collection_status',
                                'status': 'idle',
                                'collected_reps': 0
                            })
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            # WebSocket closed, silently ignore
                            pass
                except Exception as e:
                    print(f"⚠️  Failed to stop/save dataset collection: {e}")
                    # Safely send error status
                    try:
                        await websocket.send_json({
                            'type': 'dataset_collection_status',
                            'status': 'error',
                            'error': str(e)
                        })
                    except (RuntimeError, WebSocketDisconnect, AttributeError):
                        # WebSocket closed, silently ignore
                        pass
                continue

            if data.get('type') == 'pose':
                session = sessions[session_id]
                fusion_mode = session.get('fusion_mode', 'camera_only')
                
                # IMU-only mode: Skip pose processing (rep detection handled by IMU bridge WebSocket)
                if fusion_mode == 'imu_only':
                    continue
                
                # Extract landmarks from pose data
                landmarks = data.get('pose', [])
                if not landmarks:
                    # No landmarks data, skip processing but still save for training if enabled
                    if session.get('dataset_collection_enabled') and ml_mode == 'train':
                        # Still collect IMU data even without body detection
                        pass
                    if fusion_mode in ['camera_primary', 'fusion']:
                        print(f"⚠️  Fusion mode ({fusion_mode}): No landmarks received in pose data")
                    continue
                
                form_analyzer = session['form_analyzer']
                rep_counter = session['rep_counter']
                
                # State machine
                config = EXERCISE_CONFIG.get(exercise, {})
                required_landmarks = config.get('required_landmarks', [11, 12, 13, 14, 15, 16])
                calibration_msg = config.get('calibration_message', 'Show your body')
                
                if session['state'] == 'detecting':
                    # Check exercise-specific landmarks (use lenient threshold for initial detection)
                    all_visible, visible_count, missing = check_required_landmarks(
                        landmarks, required_landmarks, threshold=0.5, min_visibility_ratio=0.75
                    )
                    
                    if all_visible:
                        session['state'] = 'calibrating'
                        try:
                            if websocket.client_state.name == 'CONNECTED':
                                await websocket.send_json({
                                    'type': 'state',
                                    'state': 'calibrating',
                                    'message': 'Calibration starting... Hold still!'
                                })
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            pass
                    else:
                        # Send detailed feedback about what's missing
                        missing_str = ', '.join(missing[:3])  # Show max 3 missing
                        try:
                            if websocket.client_state.name == 'CONNECTED':
                                await websocket.send_json({
                                    'type': 'visibility',
                                    'visible_count': visible_count,
                                    'required_count': len(required_landmarks),
                                    'missing': missing_str,
                                    'message': calibration_msg
                                })
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            pass
                
                elif session['state'] == 'calibrating':
                    completed, timed_out = form_analyzer.calibrate(landmarks)
                    
                    if timed_out:
                        # Reset and go back to detecting
                        session['state'] = 'detecting'
                        try:
                            if websocket.client_state.name == 'CONNECTED':
                                await websocket.send_json({
                                    'type': 'state',
                                    'state': 'detecting',
                                    'message': 'Calibration timeout. Please hold still and try again.'
                                })
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            pass
                    elif completed:
                        print("✅ Calibration complete! Starting countdown...")
                        # IMPORTANT: Change state IMMEDIATELY to prevent re-entering calibration
                        session['state'] = 'countdown'
                        # Start countdown in background task to avoid blocking
                        asyncio.create_task(
                            countdown_task(websocket, session_id)
                        )
                    else:
                        try:
                            if websocket.client_state.name == 'CONNECTED':
                                await websocket.send_json({
                                    'type': 'calibration_progress',
                                    'progress': len(form_analyzer.calibration_frames) / FormAnalyzer.CALIBRATION_FRAMES
                                })
                        except (RuntimeError, WebSocketDisconnect, AttributeError):
                            pass
                
                elif session['state'] == 'countdown':
                    # Countdown state - don't process pose data during countdown
                    continue
                
                elif session['state'] == 'resting':
                    # Don't process pose during rest period - handled by rest_countdown_task
                    # This effectively pauses data collection for MLTRAINCAMERA and MLTRAINIMU
                    # Collectors remain active (is_collecting=True) but no new data is added
                    # Data collection will automatically resume when state returns to 'tracking'
                    continue
                
                elif session['state'] == 'tracking':
                    # Store landmarks and IMU data for dataset collection
                    ml_mode = session.get('ml_mode', 'usage')
                    current_time = time.time()
                    
                    # Debug: verify we're in tracking state
                    tracking_frame_count = session.get('tracking_frame_count', 0)
                    session['tracking_frame_count'] = tracking_frame_count + 1
                    
                        # Training mode: collect to separate training collectors (exercise-specific)
                        # Store landmarks for camera training collector with 20Hz throttling (50ms = 0.05s)
                    if ml_mode == 'train':
                        if camera_collector and camera_collector.is_collecting:
                            # Throttle to 20Hz: only add if 50ms (0.05s) has passed since last sample
                            last_sample_time = session.get('last_camera_sample_time')
                            if last_sample_time is None or (current_time - last_sample_time) >= 0.05:
                                # Track rep number for this frame
                                # Rep number is current_rep_number + 1 if we're in 'up' phase (rep in progress)
                                # Otherwise 0 (no rep in progress)
                                frame_rep_number = 0
                                if rep_counter.phase == 'up':
                                    # Rep is in progress, rep number is the next rep (will be completed)
                                    frame_rep_number = rep_counter.count + 1

                                # Add to rep-level buffer (for rep-based collection)
                                session['current_rep_landmarks'].append(landmarks)
                                
                                # Also add to session-level buffer (for continuous collection) with rep_number
                                if 'session_landmarks' not in session:
                                    session['session_landmarks'] = []
                                session['session_landmarks'].append({
                                    'timestamp': current_time,
                                    'landmarks': landmarks,
                                    'rep_number': frame_rep_number  # Mark which rep this frame belongs to
                                })
                                session['last_camera_sample_time'] = current_time
                                if len(session['current_rep_landmarks']) > 200:  # Increased for longer reps at 20Hz
                                    session['current_rep_landmarks'].pop(0)

                        # In train mode, IMU data comes from gymbud_imu_bridge WebSocket client (bypass frontend throttling)
                        # Skip frontend IMU data processing in train mode
                    elif session.get('dataset_collection_enabled') and dataset_collector and dataset_collector.is_collecting:
                        # Usage mode: collect to regular dataset collector with 20Hz throttling
                        last_sample_time = session.get('last_camera_sample_time')
                        # Throttle to 20Hz: only add if 50ms (0.05s) has passed since last sample
                        if last_sample_time is None or (current_time - last_sample_time) >= 0.05:
                            session['last_camera_sample_time'] = current_time
                            if len(session['current_rep_landmarks']) > 200:  # Increased for longer reps at 20Hz
                                session['current_rep_landmarks'].pop(0)
                        
                        # Store IMU data if provided with 20Hz throttling per node
                        imu_data = data.get('imu_data') or data.get('imu')
                        if imu_data:
                            # Check each IMU node separately for 20Hz throttling
                            if 'last_imu_node_sample_time' not in session:
                                session['last_imu_node_sample_time'] = {}
                            
                            nodes_to_add = {}
                            for node_name in ['left_wrist', 'right_wrist', 'chest']:
                                if node_name not in imu_data:
                                    continue
                                last_node_time = session['last_imu_node_sample_time'].get(node_name)
                                if last_node_time is None or (current_time - last_node_time) >= 0.05:
                                    nodes_to_add[node_name] = imu_data[node_name]
                                    session['last_imu_node_sample_time'][node_name] = current_time

                            if nodes_to_add:
                                throttled_imu_data = {k: v for k, v in nodes_to_add.items()}
                                if 'timestamp' in imu_data:
                                    throttled_imu_data['timestamp'] = imu_data['timestamp']
                                else:
                                    throttled_imu_data['timestamp'] = current_time
                                
                                if 'current_rep_imu' not in session:
                                    session['current_rep_imu'] = []
                                session['current_rep_imu'].append(throttled_imu_data)
                                if len(session['current_rep_imu']) > 200:  # Limit buffer size
                                    session['current_rep_imu'].pop(0)

                    # Calculate angle
                    # Check fusion mode for IMU-only mode angle calculation
                    fusion_mode = session.get('fusion_mode', 'camera_only')
                    
                    if fusion_mode == 'imu_only':
                        # IMU-only mode: Use IMU pitch values to calculate angle
                        rep_imu_samples = session.get('current_rep_imu', [])
                        if rep_imu_samples:
                            last_imu = rep_imu_samples[-1]
                            lw_pitch = last_imu.get('left_wrist', {}).get('euler', {}).get('pitch', 0)
                            rw_pitch = last_imu.get('right_wrist', {}).get('euler', {}).get('pitch', 0)
                            
                            # Calculate average pitch (or use max if one is missing)
                            if lw_pitch != 0 and rw_pitch != 0:
                                avg_pitch = (lw_pitch + rw_pitch) / 2
                            elif lw_pitch != 0:
                                avg_pitch = lw_pitch
                            elif rw_pitch != 0:
                                avg_pitch = rw_pitch
                            else:
                                avg_pitch = 0
                            
                            # Convert pitch to elbow angle for biceps curl
                            # Pitch range: -90° (arm down/extended) to +90° (arm up/contracted)
                            # Elbow angle: 180° (extended) to 30° (contracted)
                            # Mapping: pitch -90° → elbow 180°, pitch 0° → elbow 90°, pitch +90° → elbow 30°
                            if avg_pitch != 0:
                                # Linear mapping: pitch -90 to +90 → elbow 180 to 30
                                elbow_angle = 105 - (avg_pitch * 0.833)  # 150° range / 180° pitch range = 0.833
                                angle = max(30, min(180, elbow_angle))  # Clamp to reasonable range
                            else:
                                angle = 0
                        else:
                            angle = 0
                    else:
                        # Camera mode: Use landmark-based angle calculation
                        config = EXERCISE_CONFIG.get(exercise, {})
                        joints = config.get('joints', {}).get('left', (11, 13, 15))
                        
                        try:
                            # Use left arm joints (shoulder, elbow, wrist)
                            a = [landmarks[joints[0]]['x'], landmarks[joints[0]]['y']]
                            b = [landmarks[joints[1]]['x'], landmarks[joints[1]]['y']]
                            c = [landmarks[joints[2]]['x'], landmarks[joints[2]]['y']]
                            angle = calculate_angle(a, b, c)
                            
                            # Validate angle is reasonable (0-180°)
                            if angle < 0 or angle > 180 or np.isnan(angle):
                                angle = 0
                        except (IndexError, KeyError, TypeError) as e:
                            print(f"⚠️ Angle calculation error: {e}")
                            angle = 0
                    
                    # Check form
                    form_result = form_analyzer.check_form(landmarks)
                    session['last_form_score'] = form_result['score']
                    session['last_form_result'] = form_result  # Store full form result for summary
                    
                    # Update rep counter (pass landmarks for motion validation)
                    rep_result = rep_counter.update(angle, form_result['score'], landmarks)
                    
                    # Track current rep number for marking frames
                    current_rep_number = rep_counter.count
                    
                    
                    # Store issues for session summary ONLY when rep is completed (not every frame!)
                    if rep_result:
                        # Rep completed - handle rep completion
                        workout_config = session.get('workout_config', {})
                        print(f"   Range: {rep_result.get('min_angle', 0):.1f}° to {rep_result.get('max_angle', 0):.1f}°")
                        print(f"   Set: {session.get('current_set', 1)}, Rep in set: {session.get('current_rep_in_set', 0)}/{workout_config.get('repsPerSet', 10)}")
                    
                    # Debug: log every 10 frames for troubleshooting (more frequent)
                    frame_count = len(session.get('current_rep_landmarks', []))
                    if frame_count > 0 and frame_count % 10 == 0:
                    # Also log first few frames to see if tracking is working
                        print(f"🔍 TRACKING: Frame {frame_count}, angle={angle:.1f}°, phase={rep_counter.phase}, rep_count={rep_counter.count}")
                    
                    # Add regional scores to rep_result if rep completed
                    if rep_result:
                        ml_prediction = None
                        baseline_similarity = None
                        hybrid_score = None
                        ml_regional_scores = None
                        
                        # Rule-based regional scores (fallback if ML not available)
                        rule_based_regional_scores = form_result.get('regional_scores', {
                            'arms': form_result['score'],
                            'legs': form_result['score'],
                            'core': form_result['score'],
                            'head': form_result['score']
                        })
                        
                        ml_prediction = None
                        baseline_similarity = None
                        hybrid_score = None
                        
                        if ml_inference and ml_inference.has_camera_model():
                            try:
                                # Get landmarks sequence for current rep
                                rep_landmarks = session.get('current_rep_landmarks', [])
                                # Predict using ML model (returns Dict[str, float] with regional scores)
                                ml_prediction = ml_inference.predict_camera(rep_landmarks)

                                # Extract regional scores from ML prediction
                                ml_regional_scores = {
                                    'arms': ml_prediction.get('arms', 0.0),
                                    'legs': ml_prediction.get('legs', 0.0),
                                    'core': ml_prediction.get('core', 0.0),
                                    'head': ml_prediction.get('head', 0.0)
                                }

                                # Calculate baseline similarity (regional) if baselines are available
                                baselines = session.get('baselines', {})
                                # Use regional scores for similarity calculation
                                baseline_similarity_dict, _ = calculate_baseline_similarity(
                                    current_features=None,
                                    baselines=baselines,
                                    current_regional_scores=ml_regional_scores
                                )
                                baseline_similarity = baseline_similarity_dict  # Dict[str, float] with regional similarities

                                # Calculate hybrid score (ML + Baseline) - regional
                                hybrid_score = calculate_hybrid_correction_score(
                                    ml_prediction,
                                    baseline_similarity,
                                    ml_weight=0.6,
                                    baseline_weight=0.4
                                )

                                # Use hybrid_score as final regional scores (ML + Baseline)
                                if hybrid_score:
                                    rep_result['regional_scores'] = hybrid_score
                                elif ml_regional_scores:
                                    # ML prediction available but no baselines - use ML prediction
                                    rep_result['regional_scores'] = ml_regional_scores
                                else:
                                    # Fallback to rule-based
                                    rep_result['regional_scores'] = rule_based_regional_scores

                                print(f"🤖 ML Regional Scores: Arms={ml_regional_scores['arms']:.1f}%, Legs={ml_regional_scores['legs']:.1f}%, Core={ml_regional_scores['core']:.1f}%, Head={ml_regional_scores['head']:.1f}%")
                                print(f"   Baseline Similarity: Arms={baseline_similarity.get('arms', 0):.1f}%, Legs={baseline_similarity.get('legs', 0):.1f}%, Core={baseline_similarity.get('core', 0):.1f}%, Head={baseline_similarity.get('head', 0):.1f}%")
                                print(f"   Hybrid Scores: Arms={hybrid_score.get('arms', 0):.1f}%, Legs={hybrid_score.get('legs', 0):.1f}%, Core={hybrid_score.get('core', 0):.1f}%, Head={hybrid_score.get('head', 0):.1f}%")
                            except Exception as e:
                                print(f"⚠️  ML inference error: {e}")
                                import traceback
                                traceback.print_exc()
                                # Fallback to rule-based
                                rep_result['regional_scores'] = rule_based_regional_scores
                        else:
                            # ML not available - use rule-based regional scores
                            rep_result['regional_scores'] = rule_based_regional_scores
                        
                        # Add ML scores to rep_result
                        rep_result['ml_prediction'] = ml_prediction
                        rep_result['baseline_similarity'] = baseline_similarity
                        rep_result['hybrid_score'] = hybrid_score
                        
                        # Save rep to dataset (both usage and train modes record data)
                        ml_mode = session.get('ml_mode', 'usage')
                        rep_number = rep_result.get('rep', rep_counter.count)
                        
                            # Training mode: save to separate collectors (MLTRAINCAMERA and MLTRAINIMU)
                            # Get camera rep timestamp first (will be used for both camera and IMU collectors)
                        if ml_mode == 'train':
                            # Save to camera training collector (exercise-specific)
                            camera_collector = camera_training_collectors.get(exercise)
                            camera_rep_sample = None
                            camera_rep_timestamp = None
                            if camera_collector and camera_collector.is_collecting:
                                try:
                                    camera_rep_sample = camera_collector.add_rep_sample(
                                        exercise=exercise,
                                        rep_number=rep_number,
                                        landmarks_sequence=session['current_rep_landmarks'].copy(),
                                        imu_sequence=None,  # Don't save IMU in camera collector
                                        regional_scores=form_result.get('regional_scores'),
                                        regional_issues=form_result.get('regional_issues'),
                                        min_angle=rep_result.get('min_angle'),
                                        max_angle=rep_result.get('max_angle'),
                                        user_id='default'
                                    )
                                    # Use the camera rep sample's timestamp (more accurate, matches camera data)
                                    camera_rep_timestamp = camera_rep_sample.timestamp
                                    print(f"📹 Saved rep #{rep_number} to MLTRAINCAMERA/{exercise}/ (timestamp: {camera_rep_timestamp})")
                                except Exception as e:
                                    print(f"⚠️  Camera training collection error: {e}")

                            # Save to IMU training collector (exercise-specific)
                            # Use camera rep timestamp to ensure synchronization
                            imu_collector = imu_training_collectors.get(exercise)
                            if imu_collector and imu_collector.is_collecting and camera_rep_timestamp:
                                try:
                                    # Get IMU samples from session
                                    imu_samples_seq = session.get('current_rep_imu_samples', [])

                                    # Remove rep_number from samples before adding to collector
                                    imu_data_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in imu_samples_seq]
                                    
                                    if imu_data_seq:
                                        imu_collector.add_rep_sequence(
                                            rep_number=rep_number,
                                            imu_sequence=imu_data_seq,
                                            rep_start_time=camera_rep_timestamp  # Use camera rep timestamp for synchronization
                                        )
                                        print(f"📡 Saved rep #{rep_number} to MLTRAINIMU/{exercise}/ ({len(imu_data_seq)} IMU samples from gymbud_imu_bridge)")
                                    else:
                                        # Use session-level IMU data if rep-level is empty (fallback)
                                        session_imu_samples = session.get('session_imu_samples', [])
                                        # Filter session-level samples by rep_number
                                        rep_imu_samples = [s for s in session_imu_samples if s.get('rep_number') == rep_number]
                                        # Extract IMU data from session-level samples (remove rep_number field for consistency)
                                        rep_imu_data = [{k: v for k, v in s.items() if k != 'rep_number'} for s in rep_imu_samples]
                                        
                                        if rep_imu_data:
                                            imu_collector.add_rep_sequence(
                                                rep_number=rep_number,
                                                imu_sequence=rep_imu_data,
                                                rep_start_time=camera_rep_timestamp  # Use camera rep timestamp for synchronization
                                            )
                                            print(f"📡 Saved rep #{rep_number} to MLTRAINIMU/{exercise}/ ({len(rep_imu_data)} IMU samples from session-level data)")
                                        else:
                                            print(f"⚠️  Rep #{rep_number}: No IMU samples found")
                                    # Clear rep-level buffer after saving
                                    session['current_rep_imu_samples'] = []
                                except Exception as e:
                                    print(f"⚠️  IMU training collection error: {e}")
                                    import traceback
                                    traceback.print_exc()

                            # Update collected reps count
                            session['collected_reps_count'] = session.get('collected_reps_count', 0) + 1
                            
                            # Reset for next rep
                            session['current_rep_landmarks'] = []
                            session['current_rep_imu_samples'] = []
                            
                            # Send status update
                            try:
                                if websocket.client_state.name == 'CONNECTED':
                                    await websocket.send_json({
                                        'type': 'dataset_collection_status',
                                        'status': 'collecting',
                                        'collected_reps': session['collected_reps_count']
                                    })
                            except (RuntimeError, WebSocketDisconnect, AttributeError):
                                pass
                        elif session.get('dataset_collection_enabled') and dataset_collector and dataset_collector.is_collecting:
                            # Usage mode: save to regular dataset collector
                            try:
                                landmarks_seq = session['current_rep_landmarks'].copy() if session.get('current_rep_landmarks') else []
                                imu_seq = session['current_rep_imu'].copy() if session.get('current_rep_imu') else []
                                
                                if len(landmarks_seq) > 0:
                                    dataset_collector.add_rep_sample(
                                        exercise=exercise,
                                        rep_number=rep_number,
                                        landmarks_sequence=landmarks_seq,
                                        imu_sequence=imu_seq if len(imu_seq) > 0 else None,
                                        regional_scores=form_result.get('regional_scores'),
                                        regional_issues=form_result.get('regional_issues'),
                                        min_angle=rep_result.get('min_angle'),
                                        max_angle=rep_result.get('max_angle'),
                                        user_id='default'
                                    )
                                    # Update collected reps count
                                    session['collected_reps_count'] = session.get('collected_reps_count', 0) + 1
                                    print(f"💾 Usage mode: Saved rep #{rep_number} to dataset/ ({len(landmarks_seq)} landmarks frames)")
                                    
                                    # Send status update
                                    await websocket.send_json({
                                        'type': 'dataset_collection_status',
                                        'status': 'collecting',
                                        'collected_reps': session['collected_reps_count']
                                    })
                                else:
                                    print(f"⚠️  Usage mode: No landmarks collected for rep #{rep_number}")
                                
                                # Reset landmarks and IMU for next rep
                                session['current_rep_landmarks'] = []
                                session['current_rep_imu'] = []
                            except Exception as e:
                                print(f"⚠️  Dataset collection error: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    # Send update with regional scores and set/rep info
                    workout_config = session.get('workout_config', {})
                    response = {
                        'type': 'update',
                        'angle': round(angle, 1),
                        'phase': rep_counter.phase,
                        'rep_count': rep_counter.count,
                        'valid_rep_count': rep_counter.valid_count,  # Only valid reps
                        'form_score': form_result['score'],
                        'avg_form': round(avg_form, 1),  # Average form across all reps
                        'issues': form_result['issues'],
                        'regional_scores': form_result.get('regional_scores', {
                            'arms': form_result['score'],
                            'legs': form_result['score'],
                            'core': form_result['score'],
                            'head': form_result['score']
                        }),
                        'current_set': session.get('current_set', 1),
                        'current_rep_in_set': session.get('current_rep_in_set', 0),
                        'reps_per_set': workout_config.get('repsPerSet', 10),
                        'number_of_sets': workout_config.get('numberOfSets', 3),
                    }
                    
                    # If rep completed, check set tracking and get AI feedback
                    if rep_result:
                        
                        # Get workout config values
                        rest_time = workout_config.get('restTimeSeconds', 60)
                        reps_per_set = workout_config.get('repsPerSet', 10)
                        number_of_sets = workout_config.get('numberOfSets', 3)
                        
                        # Update set tracking - count ALL reps (both valid and invalid)
                        # This ensures set completion even if form is poor
                        session['current_rep_in_set'] = session.get('current_rep_in_set', 0) + 1
                        session['total_reps_in_session'] = session.get('total_reps_in_session', 0) + 1
                        
                        # Re-read current state after increment
                        current_set = session.get('current_set', 1)
                        current_rep_in_set = session.get('current_rep_in_set', 0)
                        
                        print(f"🔢 Rep #{rep_result.get('rep', 0)} completed: Set {current_set}, Rep {current_rep_in_set}/{reps_per_set}, Valid: {rep_result.get('is_valid', False)}")
                        
                        # Store rep data for session summary (camera mode)
                        if 'reps_data' not in session:
                            session['reps_data'] = []
                        
                        # Create rep_data from rep_result
                        rep_data = {
                            'rep': rep_result.get('rep', session['total_reps_in_session']),
                            'form_score': rep_result.get('form_score', form_result['score']),
                            'timestamp': time.time(),
                            'regional_scores': rep_result.get('regional_scores', form_result.get('regional_scores', {})),
                            'regional_issues': rep_result.get('regional_issues', form_result.get('regional_issues', {})),
                            'is_valid': rep_result.get('is_valid', True),
                            'min_angle': rep_result.get('min_angle'),
                            'max_angle': rep_result.get('max_angle'),
                            'lw_pitch_range': rep_result.get('lw_pitch_range', 0),
                            'rw_pitch_range': rep_result.get('rw_pitch_range', 0),
                            'imu_analysis': rep_result.get('imu_analysis', None)  # Add IMU analysis if available
                        }
                        session['reps_data'].append(rep_data)
                        
                        # Update set/rep info in response
                        response['current_set'] = current_set
                        response['current_rep_in_set'] = current_rep_in_set
                        response['reps_per_set'] = reps_per_set
                        response['number_of_sets'] = number_of_sets
                        
                        # Check if set is complete
                        if current_rep_in_set >= reps_per_set:
                            # Set complete - check if workout is complete
                            if current_set >= number_of_sets:
                                # All sets completed - automatically end session
                                print(f"🏁 Workout complete! Set {current_set}/{number_of_sets}, Rep {current_rep_in_set}/{reps_per_set}")
                                session['state'] = 'finished'
                                
                                # Calculate session summary
                                ml_mode = session.get('ml_mode', 'usage')
                                collected_count = session.get('collected_reps_count', 0)
                                
                                # Generate session feedback
                                session_feedback = await get_session_feedback(
                            exercise,
                                    session['reps_data'],
                                    session['all_issues']
                                )
                                
                                # Calculate average regional scores
                                if session['reps_data']:
                                    all_regional_scores = [r.get('regional_scores', {}) for r in session['reps_data'] if r.get('regional_scores')]
                                    if all_regional_scores:
                                        avg_regional_scores = {
                                    'arms': sum(r.get('arms', 0) for r in all_regional_scores) / len(all_regional_scores),
                                    'legs': sum(r.get('legs', 0) for r in all_regional_scores) / len(all_regional_scores),
                                    'core': sum(r.get('core', 0) for r in all_regional_scores) / len(all_regional_scores),
                                    'head': sum(r.get('head', 0) for r in all_regional_scores) / len(all_regional_scores),
                                }
                                else:
                                    avg_regional_scores = {'arms': 0, 'legs': 0, 'core': 0, 'head': 0}
                            else:
                                avg_regional_scores = {'arms': 0, 'legs': 0, 'core': 0, 'head': 0}
                                
                            # Generate regional feedbacks based on average regional scores
                            # Collect all regional issues from all reps
                            all_regional_issues = {}
                            for rep_data in session.get('reps_data', []):
                                regional_issues = rep_data.get('regional_issues', {})
                                for region in ['arms', 'legs', 'core', 'head']:
                                    if region not in all_regional_issues:
                                        all_regional_issues[region] = []
                                    if regional_issues.get(region):
                                        all_regional_issues[region].extend(regional_issues[region])

                            # Get most common issue per region
                            regional_issues_summary = {}
                            for region in ['arms', 'legs', 'core', 'head']:
                                if all_regional_issues.get(region):
                                    issue_counts = {}
                                    for issue in all_regional_issues[region]:
                                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
                                    top_issue = max(issue_counts.items(), key=lambda x: x[1])[0] if issue_counts else None
                                    regional_issues_summary[region] = [top_issue] if top_issue else []
                                else:
                                    regional_issues_summary[region] = []
                            
                            # Calculate avg_form for feedback fallback
                            avg_form = round(
                                sum(r['form_score'] for r in session['reps_data']) / len(session['reps_data'])
                                if session['reps_data'] else 0, 1
                            )
                            
                            # Generate regional feedback using rule-based feedback
                            regional_feedbacks = {}
                            for region in ['arms', 'legs', 'core', 'head']:
                                region_score = avg_regional_scores.get(region, 0)
                                region_issues = regional_issues_summary.get(region, [])
                                regional_feedbacks[region] = get_rule_based_regional_feedback(
                                    exercise, region, region_score, region_issues,
                                    rep_num=len(session['reps_data']), min_angle=None, max_angle=None,
                                    fallback_score=avg_form
                                )
                            
                            # Send session summary
                            # Use total_reps_in_session for IMU-only mode (reps_data may be empty)
                            total_reps = session.get('total_reps_in_session', len(session.get('reps_data', [])))
                            
                            # Prepare rep list with scores and speeds
                            rep_list = []
                            for rep_data in session.get('reps_data', []):
                                rep_list.append({
                                    'rep_number': rep_data.get('rep', 0),
                                    'form_score': round(rep_data.get('form_score', 0), 1),
                                    'duration': round(rep_data.get('duration', 0), 2),
                                    'speed_class': rep_data.get('speed_class', 'medium'),
                                    'speed_label': rep_data.get('speed_label', 'Medium'),
                                    'speed_emoji': rep_data.get('speed_emoji', '✅'),
                                    'is_valid': rep_data.get('is_valid', True),
                                    'regional_scores': rep_data.get('regional_scores', {}),
                                    'issues': rep_data.get('issues', [])
                                })
                            
                            summary_data = {
                                'type': 'session_summary',
                                'total_reps': total_reps,
                                'avg_form': avg_form,
                                'regional_scores': avg_regional_scores,
                                'regional_feedback': regional_feedbacks,
                                'feedback': session_feedback,
                                'rep_list': rep_list,
                                'workout_complete': True,
                                'message': 'Workout completed automatically! All sets and reps finished.'
                            }
                            
                            # Add rep completion info to response before sending
                            response['rep_completed'] = rep_result
                            response['rep_valid'] = rep_result.get('is_valid', True)
                            response['rep_feedback'] = rep_result.get('feedback', '')
                            response['workout_complete'] = True
                            # Add ML inference scores to response
                            response['ml_prediction'] = rep_result.get('ml_prediction')
                            response['baseline_similarity'] = rep_result.get('baseline_similarity')
                            response['hybrid_score'] = rep_result.get('hybrid_score')
                            
                            # Workout tamamlandığında veriyi henüz kaydetME - kullanıcıya sor
                            # Data is NOT saved here - user will decide via training dialog
                            ml_mode = session.get('ml_mode', 'usage')
                            collected_count = session.get('collected_reps_count', 0)
                            
                            # Add ml_mode and collected_count to summary for dialog
                            summary_data['ml_mode'] = ml_mode
                            summary_data['collected_count'] = collected_count
                            summary_data['show_training_dialog'] = True  # Always show dialog
                            
                            # Log data status
                            if ml_mode == 'train':
                                camera_collector = camera_training_collectors.get(exercise)
                                imu_collector = imu_training_collectors.get(exercise)
                                camera_samples = len(camera_collector.current_samples) if camera_collector else 0
                                imu_samples = len(imu_collector.current_samples) if imu_collector else 0
                                session_landmarks_count = len(session.get('session_landmarks', []))
                                session_imu_count = len(session.get('session_imu_samples', []))
                                print(f"📊 Train mode data ready (NOT saved yet - waiting for user decision):")
                                print(f"   Camera collector: {camera_samples} rep samples + {session_landmarks_count} session frames")
                                print(f"   IMU collector: {imu_samples} rep sequences + {session_imu_count} session samples")

                            # Send final rep update
                            try:
                                if websocket.client_state.name == 'CONNECTED':
                                    await websocket.send_json(response)
                            except (RuntimeError, WebSocketDisconnect, AttributeError):
                                pass
                            
                            # Send session summary
                            try:
                                if websocket.client_state.name == 'CONNECTED':
                                    await websocket.send_json(summary_data)
                                    print(f"📤 Sent automatic session_summary: total_reps={summary_data['total_reps']}, workout_complete=True")
                            except (RuntimeError, WebSocketDisconnect, AttributeError):
                                pass
                            
                            # Don't continue processing - session is finished (handled above in workout complete check)
                            continue
                        else:
                                # More sets to go - start rest period
                                session['state'] = 'resting'
                                rest_time = session.get('workout_config', {}).get('restTimeSeconds', 60)
                                await websocket.send_json({
                                    'type': 'state',
                                    'state': 'resting',
                                    'message': f'Set {current_set} complete! Rest time: {rest_time}s'
                                })
                                
                                # Start rest countdown
                                asyncio.create_task(
                                    rest_countdown_task(
                                        websocket,
                                        session_id,
                                        rest_time,
                                        current_set + 1
                                    )
                                    )
                                
                                # Stop tracking for now (will resume after rest)
                                # Data collection (MLTRAINCAMERA and MLTRAINIMU) is paused during rest period
                                # Will automatically resume when rest_countdown_task sets state back to 'tracking'
                                response['set_complete'] = True
                        
                        # Add rep validation info to response
                        response['rep_completed'] = rep_result
                        response['rep_valid'] = rep_result.get('is_valid', True)
                        response['rep_feedback'] = rep_result.get('feedback', '')
                        # Add ML inference scores to response
                        response['ml_prediction'] = rep_result.get('ml_prediction')
                        response['baseline_similarity'] = rep_result.get('baseline_similarity')
                        response['hybrid_score'] = rep_result.get('hybrid_score')
                        
                        # Technical AI feedback with regional data (NON-BLOCKING)
                        # Don't await - send response immediately, feedback will come later
                        
                        # Determine fusion mode based on available ML models
                        fusion_mode = 'camera_only'  # Default
                        ml_inference = session.get('ml_inference')
                        if ml_inference:
                            has_camera = ml_inference.has_camera_model()
                            has_imu = ml_inference.has_imu_model()
                            if has_camera and has_imu:
                                fusion_mode = 'camera_primary'  # Sensor fusion
                            elif has_imu:
                                fusion_mode = 'imu_only'
                            elif has_camera:
                                fusion_mode = 'camera_only'

                        # Get last landmark frame (for landmark-based feedback)
                        last_landmarks = None
                        rep_landmarks = session.get('current_rep_landmarks', [])
                        if len(rep_landmarks) > 0:
                            last_landmarks = rep_landmarks[-1]
                        
                        # Get initial positions from form analyzer (for landmark-based feedback)
                        initial_positions = None
                        form_analyzer = session.get('form_analyzer')
                        if form_analyzer and hasattr(form_analyzer, 'initial_positions'):
                            initial_positions = form_analyzer.initial_positions
                        
                        # Get IMU data (last sample from current rep)
                        imu_data_for_feedback = None
                        if fusion_mode in ['imu_only', 'camera_primary']:
                            rep_imu_samples = session.get('current_rep_imu', [])
                            if rep_imu_samples:
                                # Get last IMU sample (remove rep_number field)
                                last_imu_sample = rep_imu_samples[-1].copy()
                                last_imu_sample.pop('rep_number', None)
                                last_imu_sample.pop('timestamp', None)
                                imu_data_for_feedback = last_imu_sample

                        asyncio.create_task(
                            send_ai_feedback_async(
                                websocket,
                                exercise,
                                rep_result,
                                form_result['issues'],
                                form_result.get('regional_scores'),
                                form_result.get('regional_issues'),
                                ml_prediction=rep_result.get('ml_prediction'),
                                imu_data=imu_data_for_feedback,
                                landmarks=last_landmarks,
                                initial_positions=initial_positions,
                                fusion_mode=fusion_mode
                            )
                            )
                        # Don't include feedback in response - it will come as separate message
                        response['feedback'] = ''  # Empty for now
                        response['regional_feedback'] = {}  # Empty for now
                    else:
                        # No rep completed - still send update with current state
                        response['rep_completed'] = None
                        response['rep_valid'] = False
                    
                    # Always send update (even if no rep completed)
                    # Debug: log first few updates to verify tracking
                    update_count = session.get('update_count', 0)
                    if len(session.get('reps_data', [])) == 0 and rep_counter.count == 0:
                        update_count = update_count + 1
                        session['update_count'] = update_count
                        print(f"📤 Sending update #{update_count}: rep_count={response['rep_count']}, angle={response['angle']}, phase={response['phase']}")

                    await websocket.send_json(response)
            
            # Handle dataset collection save
            elif data.get('type') == 'save_dataset':
                if dataset_collector and DATASET_COLLECTION_ENABLED:
                    auto_label = data.get('auto_label_perfect', False)
                    try:
                        dataset_collector.save_session(auto_label_perfect=auto_label)
                        await websocket.send_json({
                            'type': 'dataset_saved',
                            'message': 'Dataset session saved successfully'
                        })
                    except Exception as e:
                        await websocket.send_json({
                            'type': 'dataset_error',
                            'error': str(e)
                        })
                else:
                    await websocket.send_json({
                        'type': 'dataset_error',
                        'error': 'Dataset collection not enabled'
                    })
            
            # Handle session end request
            elif data.get('type') == 'end_session':
                session = sessions.get(session_id)
                if session:
                    ml_mode = session.get('ml_mode', 'usage')
                    collected_count = session.get('collected_reps_count', 0)
                    
                    # Training mode: save both camera and IMU datasets (but don't mark as used yet)
                    if ml_mode == 'train':
                        # Save session-level continuous data (all data throughout session, rep sayılsın ya da sayılmasın)
                        camera_collector = camera_training_collectors.get(exercise)
                        imu_collector = imu_training_collectors.get(exercise)
                        
                        # Add session-level continuous data as rep_number=0 (before saving)
                        session_landmarks = session.get('session_landmarks', [])
                        session_imu_samples = session.get('session_imu_samples', [])
                        session_start_time = session.get('session_start_time', time.time())
                        
                        # Extract landmarks from session buffer (they're stored as {'timestamp': ..., 'landmarks': ...})
                        landmarks_sequence = [item['landmarks'] for item in session_landmarks]
                        
                        if camera_collector and camera_collector.is_collecting and len(landmarks_sequence) > 0:
                            try:
                                camera_collector.add_rep_sample(
                                    exercise=exercise,
                                    rep_number=0,  # rep_number=0 means session-level continuous data
                                    landmarks_sequence=landmarks_sequence,
                                    imu_sequence=None,
                                    user_id='default'
                                )
                                print(f"📹 Added session-level continuous camera data: {len(landmarks_sequence)} frames (rep_number=0)")
                            except Exception as e:
                                print(f"⚠️  Failed to add session-level camera data: {e}")

                            if imu_collector and imu_collector.is_collecting:
                                try:
                                    # Remove rep_number from samples before adding to collector
                                    imu_data_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in session_imu_samples]
                                    imu_collector.add_rep_sequence(
                                        rep_number=0,  # rep_number=0 means session-level continuous data
                                        imu_sequence=imu_data_seq,
                                        rep_start_time=session_start_time
                                    )
                                    print(f"📡 Added session-level continuous IMU data: {len(imu_data_seq)} samples (rep_number=0)")
                                except Exception as e:
                                    print(f"⚠️  Failed to add session-level IMU data: {e}")
                                    import traceback
                                    traceback.print_exc()
                            else:
                                print(f"⚠️  session_imu_samples is empty, cannot add session-level IMU data")

                            # Now save all data (including rep-based and session-level continuous)
                        try:
                            camera_session_id = None
                            if camera_collector and camera_collector.is_collecting:
                                total_camera_samples = len(camera_collector.current_samples)
                                camera_collector.save_session(auto_label_perfect=True)
                                camera_session_id = camera_collector.current_session_id
                                print(f"💾 Saved camera training dataset: {total_camera_samples} samples (reps + session-level) → MLTRAINCAMERA/{exercise}/ (session: {camera_session_id})")
                                
                            # Save IMU training dataset (exercise-specific)
                            imu_session_id = None
                            if imu_collector and imu_collector.is_collecting:
                                total_imu_reps = len(imu_collector.current_samples)
                                imu_collector.save_session()
                                imu_session_id = imu_collector.current_session_id
                                print(f"💾 Saved IMU training dataset: {total_imu_reps} sequences (reps + session-level) → MLTRAINIMU/{exercise}/ (session: {imu_session_id})")
                                
                            # Store session IDs for later tracking
                            session['camera_session_id'] = camera_session_id
                            session['imu_session_id'] = imu_session_id
                            
                            # Stop collectors (save_session already sets is_collecting=False for DatasetCollector)
                            if imu_collector:
                                imu_collector.stop_session()
                                
                            print(f"✅ Training session completed: {collected_count} reps + session-level continuous data saved to both datasets")
                            print(f"   Camera session ID: {camera_session_id}")
                            print(f"   IMU session ID: {imu_session_id}")
                            print(f"   Exercise: {exercise}")
                                # Note: We don't mark as used yet - user will decide in the dialog
                        except Exception as e:
                            print(f"⚠️  Failed to save training datasets: {e}")
                            import traceback
                            traceback.print_exc()
                    elif session.get('dataset_collection_enabled') and dataset_collector:
                        # Usage mode: Data is already collected via add_rep_sample during session
                        # Don't auto-save here - user will choose in dialog (save_only or skip)
                        print(f"💾 Usage mode: {collected_count} reps collected (already added to collector), waiting for user decision...")
                    # Calculate average regional scores
                    avg_regional_scores = {
                        'arms': 0,
                        'legs': 0,
                        'core': 0,
                        'head': 0
                    }
                    
                        # Sum all regional scores
                        # Sum all regional scores
                    if session['reps_data']:
                        count = 0
                        for rep in session['reps_data']:
                            if 'regional_scores' in rep:
                                total_regional[region] += rep['regional_scores'].get(region, 0)
                                count += 1

                        # Calculate averages
                        if count > 0:
                            avg_regional_scores[region] = round(total_regional[region] / count, 1)
                        else:
                            # Fallback to last form result
                            last_result = session.get('last_form_result', {})
                            avg_regional_scores = last_result.get('regional_scores', {
                                'arms': 0, 'legs': 0, 'core': 0, 'head': 0
                            })
                    else:
                        # No reps - use last form result if available
                        last_result = session.get('last_form_result', {})
                        avg_regional_scores = last_result.get('regional_scores', {
                            'arms': 0, 'legs': 0, 'core': 0, 'head': 0
                        })
                    
                    # Generate comprehensive session feedback
                    session_feedback = await get_session_feedback(
                        exercise,
                        session['reps_data'],
                        session['all_issues']
                    )
                    
                    # Generate regional feedbacks based on average regional scores
                    # Collect all regional issues from all reps
                    all_regional_issues = {}
                    for rep_data in session['reps_data']:
                        regional_issues = rep_data.get('regional_issues', {})
                        for region in ['arms', 'legs', 'core', 'head']:
                            if region not in all_regional_issues:
                                all_regional_issues[region] = []
                            if regional_issues.get(region):
                                all_regional_issues[region].extend(regional_issues[region])

                    # Get most common issue per region
                    regional_issues_summary = {}
                    for region in ['arms', 'legs', 'core', 'head']:
                        if all_regional_issues.get(region):
                            issue_counts = {}
                            for issue in all_regional_issues[region]:
                                issue_counts[issue] = issue_counts.get(issue, 0) + 1
                            top_issue = max(issue_counts.items(), key=lambda x: x[1])[0] if issue_counts else None
                            regional_issues_summary[region] = [top_issue] if top_issue else []
                        else:
                            regional_issues_summary[region] = []
                    
                    # Use total_reps_in_session for IMU-only mode (reps_data may be empty)
                    total_reps = session.get('total_reps_in_session', len(session.get('reps_data', [])))
                    
                    # Calculate avg_form from reps_data (IMU-only mode also has reps_data now)
                    avg_form = 0.0
                    if session.get('reps_data'):
                        avg_form = round(
                            sum(r.get('form_score', 100) for r in session['reps_data']) / len(session['reps_data']), 1
                        )
                    elif session.get('latest_form_score'):
                        # Fallback: use latest form score if no reps_data
                        avg_form = round(float(session['latest_form_score']), 1)
                    
                    # Generate regional feedback using rule-based feedback
                    regional_feedbacks = {}
                    for region in ['arms', 'legs', 'core', 'head']:
                        region_score = avg_regional_scores.get(region, 0)
                        region_issues = regional_issues_summary.get(region, [])
                        regional_feedbacks[region] = get_rule_based_regional_feedback(
                            exercise, region, region_score, region_issues,
                            rep_num=len(session['reps_data']), min_angle=None, max_angle=None,
                            fallback_score=avg_form
                        )
                    
                    summary_data = {
                        'type': 'session_summary',
                        'total_reps': total_reps,
                        'avg_form': avg_form,
                        'regional_scores': avg_regional_scores,
                        'regional_feedback': regional_feedbacks,  # Add regional feedback
                        'feedback': session_feedback,
                        'workout_complete': True  # Ensure workout_complete is set for dialog
                    }
                    print(f"📤 Sending session_summary: total_reps={summary_data['total_reps']}, avg_form={summary_data['avg_form']}, feedback_length={len(session_feedback) if session_feedback else 0}")
                    print(f"   Feedback preview: {session_feedback[:100] if session_feedback else 'None'}...")
                    try:
                        if websocket.client_state.name == 'CONNECTED':
                            await websocket.send_json(summary_data)
                            print(f"📤 Sent automatic session_summary: total_reps={summary_data['total_reps']}, workout_complete=True")
                    except (RuntimeError, WebSocketDisconnect, AttributeError):
                        pass
            
            # Handle training action (from dialog)
            elif data.get('type') == 'training_action':
                session = sessions.get(session_id)
                if not session:
                    try:
                        if websocket.client_state.name == 'CONNECTED':
                            await websocket.send_json({
                                'type': 'training_status',
                                'status': 'error',
                                'message': 'Session not found'
                            })
                    except (RuntimeError, WebSocketDisconnect, AttributeError):
                        pass
                    continue
                
                action = data.get('action')
                ml_mode = session.get('ml_mode', 'usage')
                
                camera_session_id = session.get('camera_session_id')
                imu_session_id = session.get('imu_session_id')
                
                # Save data (usage mode: save to dataset/, train mode: data already saved to MLTRAIN*)
                if action == 'save_only':
                    # Save to regular dataset collector
                    if session.get('dataset_collection_enabled') and dataset_collector and dataset_collector.is_collecting:
                        try:
                            if len(dataset_collector.current_samples) == 0:
                                print(f"⚠️ Usage mode: No samples to save (current_samples is empty)")
                                await websocket.send_json({
                                    'type': 'training_status',
                                    'status': 'error',
                                    'message': 'Kaydedilecek veri yok. Rep tamamlandığında veriler toplandı mı kontrol edin.'
                                })
                                continue
                            else:
                                dataset_collector.save_session(auto_label_perfect=True)
                                print(f"💾 Usage mode: Saved session to dataset/ ({len(dataset_collector.current_samples)} samples)")
                        except Exception as e:
                            print(f"⚠️ Error saving usage mode session: {e}")
                            import traceback
                            traceback.print_exc()
                            await websocket.send_json({
                                'type': 'training_status',
                                'status': 'error',
                                'message': f'Kayıt hatası: {str(e)}'
                            })
                            continue
                    elif ml_mode == 'usage':
                        # Usage mode but collector not available
                        print(f"⚠️ Usage mode: Cannot save - collector not available or not collecting")
                        await websocket.send_json({
                            'type': 'training_status',
                            'status': 'error',
                            'message': 'Veri toplayıcı aktif değil. Veriler kaydedilemedi.'
                        })
                        continue
                    else:
                        # Train mode: Check if data was already saved, if not save now
                        camera_collector = camera_training_collectors.get(exercise)
                        imu_collector = imu_training_collectors.get(exercise)
                        
                        # Check if data was already saved (session_id stored means it was saved)
                        # Add session-level continuous data (all data throughout session, rep sayılsın ya da sayılmasın)
                        if not camera_session_id and camera_collector and camera_collector.is_collecting:
                            session_landmarks = session.get('session_landmarks', [])
                            session_imu_samples = session.get('session_imu_samples', [])
                            session_start_time = session.get('session_start_time', time.time())
                            
                            # Extract landmarks from session buffer
                            if session_landmarks:
                                landmarks_sequence = [item['landmarks'] for item in session_landmarks]
                                if len(landmarks_sequence) > 0:
                                    try:
                                        camera_collector.add_rep_sample(
                                            exercise=exercise,
                                            rep_number=0,  # rep_number=0 means session-level continuous data
                                            landmarks_sequence=landmarks_sequence,
                                            imu_sequence=None,
                                            user_id='default'
                                        )
                                        print(f"📹 Added session-level continuous camera data: {len(landmarks_sequence)} frames (rep_number=0)")
                                    except Exception as e:
                                        print(f"⚠️  Failed to add session-level camera data: {e}")

                            if len(session_imu_samples) > 0 and imu_collector and imu_collector.is_collecting:
                                try:
                                    # Remove rep_number from samples before adding to collector
                                    imu_data_seq = [{k: v for k, v in s.items() if k != 'rep_number'} for s in session_imu_samples]
                                    imu_collector.add_rep_sequence(
                                        rep_number=0,  # rep_number=0 means session-level continuous data
                                        imu_sequence=imu_data_seq,
                                        rep_start_time=session_start_time
                                    )
                                    print(f"📡 Added session-level continuous IMU data: {len(imu_data_seq)} samples (rep_number=0)")
                                except Exception as e:
                                    print(f"⚠️  Failed to add session-level IMU data: {e}")

                            # Data not saved yet, save it now (including session-level)
                            try:
                                if len(camera_collector.current_samples) > 0:
                                    camera_collector.save_session(auto_label_perfect=True)
                                    camera_session_id = camera_collector.current_session_id
                                    num_samples = len(camera_collector.current_samples)
                                    print(f"💾 Saved camera training dataset: {num_samples} samples (reps + session-level) → MLTRAINCAMERA/{exercise}/ (session: {camera_session_id})")
                                    session['camera_session_id'] = camera_session_id
                            except Exception as e:
                                print(f"⚠️  Failed to save camera training dataset: {e}")
                                import traceback
                                traceback.print_exc()

                            try:
                                if imu_collector and imu_collector.is_collecting and len(imu_collector.current_samples) > 0:
                                    imu_collector.save_session()
                                    imu_session_id = imu_collector.current_session_id
                                    num_samples = len(imu_collector.current_samples)
                                    print(f"💾 Saved IMU training dataset: {num_samples} sequences (reps + session-level) → MLTRAINIMU/{exercise}/ (session: {imu_session_id})")
                                    session['imu_session_id'] = imu_session_id
                            except Exception as e:
                                print(f"⚠️  Failed to save IMU training dataset: {e}")
                                import traceback
                                traceback.print_exc()

                            # Stop collectors after saving (save_session already sets is_collecting=False for DatasetCollector)
                            if imu_collector:
                                imu_collector.stop_session()
                    
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'completed',
                        'message': 'Veriler kaydedildi. Daha sonra ML eğitimi için kullanabilirsiniz.'
                    })
                
                # Note: 'save_and_train' action removed - training is now done separately
                # Use train_ml_models.py script or Google Colab notebook for training
                # This keeps data collection fast and allows training on GPU (Colab/cloud)
                elif action == 'save_and_train':
                    # This action is no longer supported - redirect to save_only
                    print(f"⚠️  'save_and_train' action deprecated. Use train_ml_models.py or Colab notebook.")
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'completed',
                        'message': 'Veriler kaydedildi. Model eğitimi için train_ml_models.py veya TRAINING_GUIDE.md\'yi kullanın.'
                    })
                    continue
                
                elif action == 'skip':
                    # Skip: Don't save data (already collected during session, but mark as not to be used)
                    print(f"⏭️  User chose: Skip (data will not be saved)")
                    
                    if ml_mode == 'train':
                        # For train mode, data was collected but user wants to skip
                        # Clear collectors' data
                        camera_collector = camera_training_collectors.get(exercise)
                        imu_collector = imu_training_collectors.get(exercise)
                        
                        if camera_collector and camera_collector.is_collecting:
                            camera_collector.current_samples = []  # Clear collected samples
                            camera_collector.stop_session()
                            print(f"   Cleared camera training data for {exercise}")
                        
                        if imu_collector and imu_collector.is_collecting:
                            imu_collector.current_samples = []  # Clear collected samples
                            imu_collector.stop_session()
                            print(f"   Cleared IMU training data for {exercise}")
                    
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'completed',
                        'message': 'Veriler kaydedilmedi.'
                    })
                
                else:
                    await websocket.send_json({
                        'type': 'training_status',
                        'status': 'error',
                        'message': f'Unknown action: {action}'
                    })
    
    except WebSocketDisconnect:
        print(f"🔌 WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if session_id in sessions:
            session = sessions[session_id]
            session['connected'] = False
            # Don't delete session - keep it for potential reconnection
