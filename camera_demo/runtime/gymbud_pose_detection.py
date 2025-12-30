#!/usr/bin/env python3
"""
GymBud - Real-Time Pose Detection

Enhancements:
- Focus on biceps curl analysis using essential landmarks
- Toggleable recording with rep-based JSON logging
- Landmark normalization plus elbow, shoulder, and torso metrics
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import base64


SNAPSHOT_DIR = Path(__file__).with_name("pose_snapshots")
REP_LOG_FILE = Path(__file__).with_name("gymbud_reps_log.jsonl")
MODEL_PATH = Path(__file__).with_name("biceps_form_model.joblib")
SCALER_PATH = Path(__file__).with_name("scaler.joblib")

WINDOW_TITLE = "GymBud - Real-Time Pose Detection"
TEXT_COLOR = (0, 255, 0)
COLOR_CORRECT = (0, 255, 0)
COLOR_PARTIAL = (0, 255, 255)
COLOR_LEG_DRIVE = (0, 165, 255)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

ESSENTIAL_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.RIGHT_HIP,
]
STATE_IDLE = "idle"
TOP_ANGLE_DEG = 150.0
BOTTOM_ANGLE_DEG = 70.0
ANGLE_HYSTERESIS = 5.0

LABEL_SCORES = {"Correct": 100.0, "Leg_Drive": 75.0, "Partial": 60.0}
FEATURE_ORDER = [
    "ROM",
    "min_elbow_angle",
    "max_elbow_angle",
    "rep_duration_seconds",
    "avg_speed",
    "shoulder_stability",
    "wrist_sway",
    "mean_elbow_angle",
    "angle_range",
    "peak_contraction_angle",
    "tempo_up_down_ratio",
]


def ensure_storage_paths() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC in degrees using vectors BA and BC."""
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return float("nan")
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))


def extract_relevant_landmarks(landmarks) -> dict:
    essential_data = {}
    for landmark_enum in ESSENTIAL_LANDMARKS:
        lm = landmarks[landmark_enum.value]
        essential_data[landmark_enum.name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        }
    return essential_data


# ============================================================================
# Joint Mapping Layer: MediaPipe → CSV Format Compatibility
# ============================================================================

# CSV column mapping (format: LShoulder_X20, LElbow_X21, LWrist_X22, etc.)
CSV_LANDMARK_COLUMNS = {
    "LEFT_SHOULDER": {"x": "LShoulder_X20", "y": "LShoulder_Y20", "z": "LShoulder_Z20"},
    "LEFT_ELBOW": {"x": "LElbow_X21", "y": "LElbow_Y21", "z": "LElbow_Z21"},
    "LEFT_WRIST": {"x": "LWrist_X22", "y": "LWrist_Y22", "z": "LWrist_Z22"},
    "LEFT_HIP": {"x": "LHip_X8", "y": "LHip_Y8", "z": "LHip_Z8"},
    "RIGHT_SHOULDER": {"x": "RShoulder_X17", "y": "RShoulder_Y17", "z": "RShoulder_Z17"},
    "RIGHT_ELBOW": {"x": "RElbow_X18", "y": "RElbow_Y18", "z": "RElbow_Z18"},
    "RIGHT_WRIST": {"x": "RWrist_X19", "y": "RWrist_Y19", "z": "RWrist_Z19"},
    "RIGHT_HIP": {"x": "RHip_X2", "y": "RHip_Y2", "z": "RHip_Z2"},
}


def map_mediapipe_to_csv_format(normalized_landmarks: dict) -> dict:
    """
    Convert MediaPipe normalized landmarks to CSV-compatible format.
    
    This mapping layer ensures compatibility between:
    - MediaPipe runtime format (LEFT_SHOULDER, LEFT_ELBOW, etc.)
    - CSV training format (LShoulder_X20, LElbow_X21, etc.)
    
    Args:
        normalized_landmarks: Dictionary with MediaPipe format keys
        
    Returns:
        Dictionary with CSV-compatible column names
    """
    csv_data = {}
    for mp_key, csv_cols in CSV_LANDMARK_COLUMNS.items():
        if mp_key in normalized_landmarks:
            csv_data[csv_cols["x"]] = normalized_landmarks[mp_key]["x"]
            csv_data[csv_cols["y"]] = normalized_landmarks[mp_key]["y"]
            csv_data[csv_cols["z"]] = normalized_landmarks[mp_key]["z"]
    return csv_data


def get_csv_coordinates(normalized_landmarks: dict, landmark_name: str, coord: str = "x") -> float:
    """
    Get CSV-format coordinate from normalized landmarks.
    
    Helper function to retrieve coordinates in CSV format for compatibility.
    
    Args:
        normalized_landmarks: Dictionary with MediaPipe format keys
        landmark_name: MediaPipe landmark name (e.g., "LEFT_SHOULDER")
        coord: Coordinate to retrieve ("x", "y", or "z")
        
    Returns:
        Coordinate value or 0.0 if not found
    """
    if landmark_name in CSV_LANDMARK_COLUMNS:
        csv_col = CSV_LANDMARK_COLUMNS[landmark_name][coord]
        mp_key = landmark_name
        if mp_key in normalized_landmarks:
            return normalized_landmarks[mp_key][coord]
    return 0.0


VISIBILITY_THRESHOLD = 0.5


def are_landmarks_visible_for_biceps(raw_landmarks: dict) -> bool:
    required_landmarks = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
    for landmark_name in required_landmarks:
        if landmark_name not in raw_landmarks:
            return False
        visibility = raw_landmarks[landmark_name].get("visibility", 0.0)
        if visibility < VISIBILITY_THRESHOLD:
            return False
    return True


def draw_landmark_labels(frame, pose_landmarks):
    """Draw landmark names on the frame for debug purposes."""
    if pose_landmarks is None:
        return
    
    height, width = frame.shape[:2]
    
    landmark_name_map = {
        mp_pose.PoseLandmark.LEFT_SHOULDER: "L_SHOULDER",
        mp_pose.PoseLandmark.LEFT_ELBOW: "L_ELBOW",
        mp_pose.PoseLandmark.LEFT_WRIST: "L_WRIST",
        mp_pose.PoseLandmark.LEFT_HIP: "L_HIP",
        mp_pose.PoseLandmark.RIGHT_SHOULDER: "R_SHOULDER",
        mp_pose.PoseLandmark.RIGHT_ELBOW: "R_ELBOW",
        mp_pose.PoseLandmark.RIGHT_WRIST: "R_WRIST",
        mp_pose.PoseLandmark.RIGHT_HIP: "R_HIP",
    }
    
    for landmark_enum in ESSENTIAL_LANDMARKS:
        landmark = pose_landmarks.landmark[landmark_enum.value]
        
        if landmark.visibility >= 0.3:
            x_px = int(landmark.x * width)
            y_px = int(landmark.y * height)
            
            label = landmark_name_map.get(landmark_enum, landmark_enum.name)
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = x_px - text_size[0] // 2
            text_y = y_px - 10
            
            bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
            
            if landmark_enum in [mp_pose.PoseLandmark.LEFT_SHOULDER, 
                               mp_pose.PoseLandmark.LEFT_ELBOW,
                               mp_pose.PoseLandmark.LEFT_WRIST]:
                bg_color = (0, 0, 255)
                text_color = (255, 255, 255)
            
            cv2.rectangle(
                frame,
                (text_x - 3, text_y - text_size[1] - 3),
                (text_x + text_size[0] + 3, text_y + 3),
                bg_color,
                -1
            )
            
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
                cv2.LINE_AA,
            )


def normalize_landmarks(raw_landmarks: dict) -> dict:
    pelvis_center = np.array(
        [
            (raw_landmarks["LEFT_HIP"]["x"] + raw_landmarks["RIGHT_HIP"]["x"]) / 2.0,
            (raw_landmarks["LEFT_HIP"]["y"] + raw_landmarks["RIGHT_HIP"]["y"]) / 2.0,
            (raw_landmarks["LEFT_HIP"]["z"] + raw_landmarks["RIGHT_HIP"]["z"]) / 2.0,
        ]
    )
    body_height = max(
        np.linalg.norm(np.array([lm["x"], lm["y"], lm["z"]]) - pelvis_center)
        for lm in raw_landmarks.values()
    )
    body_height = body_height if body_height > 1e-6 else 1e-6

    normalized = {}
    for name, values in raw_landmarks.items():
        point = np.array([values["x"], values["y"], values["z"]])
        normalized_point = (point - pelvis_center) / body_height
        normalized[name] = {
            "x": float(normalized_point[0]),
            "y": float(normalized_point[1]),
            "z": float(normalized_point[2]),
        }
    return normalized


def calculate_elbow_angle(normalized_landmarks: dict) -> float:
    try:
        shoulder = np.array(
            [
                normalized_landmarks["LEFT_SHOULDER"]["x"],
                normalized_landmarks["LEFT_SHOULDER"]["y"],
                normalized_landmarks["LEFT_SHOULDER"]["z"],
            ]
        )
        elbow = np.array(
            [
                normalized_landmarks["LEFT_ELBOW"]["x"],
                normalized_landmarks["LEFT_ELBOW"]["y"],
                normalized_landmarks["LEFT_ELBOW"]["z"],
            ]
        )
        wrist = np.array(
            [
                normalized_landmarks["LEFT_WRIST"]["x"],
                normalized_landmarks["LEFT_WRIST"]["y"],
                normalized_landmarks["LEFT_WRIST"]["z"],
            ]
        )
    except KeyError:
        return float("nan")
    return angle_between_points(shoulder, elbow, wrist)


def calculate_torso_angle(normalized_landmarks: dict) -> float:
    try:
        shoulder = np.array(
            [
                normalized_landmarks["LEFT_SHOULDER"]["x"],
                normalized_landmarks["LEFT_SHOULDER"]["y"],
                normalized_landmarks["LEFT_SHOULDER"]["z"],
            ]
        )
        hip = np.array(
            [
                normalized_landmarks["LEFT_HIP"]["x"],
                normalized_landmarks["LEFT_HIP"]["y"],
                normalized_landmarks["LEFT_HIP"]["z"],
            ]
        )
    except KeyError:
        return float("nan")
    torso_vector = shoulder - hip
    if np.linalg.norm(torso_vector) == 0:
        return float("nan")
    vertical_vector = np.array([0.0, -1.0, 0.0])
    cosine_angle = np.dot(torso_vector, vertical_vector) / (
        np.linalg.norm(torso_vector) * np.linalg.norm(vertical_vector)
    )
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def detect_rep_state(elbow_angle: float, current_state: str) -> tuple[str, bool]:
    if np.isnan(elbow_angle):
        return current_state, False

    state = current_state
    rep_completed = False

    if state == STATE_IDLE:
        if elbow_angle >= TOP_ANGLE_DEG:
            state = "going_up"
    elif state == "going_up":
        if elbow_angle <= BOTTOM_ANGLE_DEG:
            state = "top"
    elif state == "top":
        if elbow_angle > BOTTOM_ANGLE_DEG + ANGLE_HYSTERESIS:
            state = "going_down"
    elif state == "going_down":
        if elbow_angle >= TOP_ANGLE_DEG - ANGLE_HYSTERESIS:
            state = STATE_IDLE
            rep_completed = True
    else:
        state = STATE_IDLE

    return state, rep_completed


def save_rep_json(rep_number: int, frames: list[dict]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rep_path = SNAPSHOT_DIR / f"rep_{rep_number:03d}_{timestamp}.json"
    data = {"rep_number": rep_number, "frames": frames}
    with rep_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return rep_path


def draw_debug_panel(frame, features: dict):
    if features is None:
        return

    panel_x = 10
    panel_y = frame.shape[0] - 180
    panel_width = 250
    panel_height = 160
    padding = 10

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (30, 30, 30),
        -1,
    )
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (255, 255, 255),
        2,
    )
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    text_y = panel_y + 25
    cv2.putText(
        frame,
        "REP METRICS",
        (panel_x + padding, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    text_y += 25
    cv2.putText(
        frame,
        f"ROM: {features.get('ROM', 0):.1f}",
        (panel_x + padding, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    text_y += 20
    cv2.putText(
        frame,
        f"Tempo: {features.get('tempo_up_down_ratio', 0):.2f}",
        (panel_x + padding, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    text_y += 20
    cv2.putText(
        frame,
        f"Shoulder_Sway: {features.get('shoulder_stability', 0):.3f}",
        (panel_x + padding, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    text_y += 20
    cv2.putText(
        frame,
        f"Wrist_Sway: {features.get('wrist_sway', 0):.3f}",
        (panel_x + padding, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def extract_rep_features_runtime(rep_buffer: dict) -> dict:
    """Runtime body-invariant rep feature extraction (same as training)."""

    angles = rep_buffer.get('angles', [])
    if len(angles) == 0:
        return None

    angles = np.array(angles)
    min_elbow = float(np.min(angles))
    max_elbow = float(np.max(angles))
    rom = max_elbow - min_elbow

    # === BODY INVARIANT SWAY ===
    shoulder_y = rep_buffer.get('shoulder_y', [])
    wrist_x = rep_buffer.get('wrist_x', [])

    if len(shoulder_y) > 5:
        baseline_sh_y = np.mean(shoulder_y[:5])
        shoulder_stability = float(np.std([y - baseline_sh_y for y in shoulder_y]))
    else:
        shoulder_stability = float(np.std(shoulder_y)) if len(shoulder_y) else 0.0

    if len(wrist_x) > 5:
        baseline_wr_x = np.mean(wrist_x[:5])
        wrist_sway = float(np.std([x - baseline_wr_x for x in wrist_x]))
    else:
        wrist_sway = float(np.std(wrist_x)) if len(wrist_x) else 0.0

    # === TEMPO ===
    min_idx = int(np.argmin(angles))
    up_frames = angles[:min_idx]
    down_frames = angles[min_idx:]
    up_duration = len(up_frames) / 30.0
    down_duration = len(down_frames) / 30.0
    tempo_ratio = up_duration / max(down_duration, 1e-6)

    return {
        'ROM': rom,
        'min_elbow_angle': min_elbow,
        'max_elbow_angle': max_elbow,
        'rep_duration_seconds': len(angles)/30.0,
        'avg_speed': rom / max(len(angles)/30.0, 1e-6),
        'shoulder_stability': shoulder_stability,
        'wrist_sway': wrist_sway,
        'mean_elbow_angle': float(np.mean(angles)),
        'angle_range': rom,
        'peak_contraction_angle': min_elbow,
        'tempo_up_down_ratio': tempo_ratio
    }


def score_to_label(score: float) -> str:
    closest_label = "Correct"
    min_diff = float("inf")
    for label, target_score in LABEL_SCORES.items():
        diff = abs(score - target_score)
        if diff < min_diff:
            min_diff = diff
            closest_label = label
    return closest_label


def log_rep_record(rep_id: int, features: dict, score: float, label: str):
    record = {
        "timestamp": time.time(),
        "rep_id": rep_id,
        "features": features,
        "score": float(score),
        "label": label,
    }
    with REP_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def is_iphone_camera(camera_index):
    """Check if a camera index corresponds to an iPhone camera."""
    import platform
    import subprocess
    import json
    
    if platform.system() != "Darwin":
        return False
    
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'SPCameraDataType' in data and camera_index < len(data['SPCameraDataType']):
                cam = data['SPCameraDataType'][camera_index]
                name = cam.get('_name', '').lower()
                return 'iphone' in name
    except Exception:
        pass
    return False


def find_available_camera(camera_index=None):
    """Try different camera indices to find an available one, or use specified index."""
    # First, check if a camera index was selected via API
    if camera_index is None:
        try:
            with open('/tmp/gymbud_camera_index.txt', 'r') as f:
                camera_index = int(f.read().strip())
                print(f"Using selected camera index: {camera_index}")
        except (FileNotFoundError, ValueError):
            camera_index = None
    
    # If specific index provided, try that first (but skip if it's iPhone)
    if camera_index is not None:
        if is_iphone_camera(camera_index):
            print(f"⚠️ Camera {camera_index} is an iPhone camera - skipping to avoid notifications")
            camera_index = None
        else:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Camera found at index {camera_index}")
                    return cap, camera_index
                cap.release()
    
    # Otherwise, try different camera indices (skip iPhone cameras)
    for idx in range(5):
        if camera_index is not None and idx == camera_index:
            continue  # Already tried
        if is_iphone_camera(idx):
            print(f"⚠️ Skipping iPhone camera at index {idx}")
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera found at index {idx}")
                return cap, idx
            cap.release()
    return None, None


def draw_startup_screen():
    screen = np.zeros((600, 800, 3), dtype=np.uint8)
    screen[:] = (30, 30, 30)
    
    cv2.putText(screen, "GymBud", (250, 150), cv2.FONT_HERSHEY_TRIPLEX, 3.0, (0, 255, 0), 4)
    cv2.putText(screen, "Biceps Curl Form Analyzer", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    
    cv2.rectangle(screen, (250, 300), (550, 380), (0, 255, 0), -1)
    cv2.putText(screen, "Press 'S' to Start Camera", (265, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.rectangle(screen, (250, 420), (550, 500), (0, 0, 255), -1)
    cv2.putText(screen, "Press 'Q' to Quit", (290, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return screen


def show_startup_screen(model, scaler):
    cv2.namedWindow("GymBud - Startup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GymBud - Startup", 800, 600)
    cv2.moveWindow("GymBud - Startup", 200, 200)
    
    print("\n" + "="*60)
    print("GymBud - Biceps Curl Form Analyzer")
    print("="*60)
    print("GUI Window opened. Press 'S' to start camera, 'Q' to quit.")
    print("="*60 + "\n")
    
    while True:
        screen = draw_startup_screen()
        
        try:
            model_status = "Loaded" if model is not None else "Not Found"
            status_color = (0, 255, 0) if model_status == "Loaded" else (0, 165, 255)
            cv2.putText(screen, f"Model: {model_status}", (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        except:
            pass
        
        cv2.imshow("GymBud - Startup", screen)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            cv2.destroyWindow("GymBud - Startup")
            return False
        elif key == ord('s') or key == ord('S'):
            cv2.destroyWindow("GymBud - Startup")
            return True
    
    cv2.destroyWindow("GymBud - Startup")
    return False


def start_camera_session(model, scaler, bridge=None, camera_index=None):
    cap, camera_idx = find_available_camera(camera_index)
    if cap is None:
        print("ERROR: Unable to access camera. Ensure it is connected and permissions are granted.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # MJPEG stream server is now started by GymBudBridge.__init__()
    # We just need to update frames here

    prev_time = time.time()
    recording = False
    rep_state = STATE_IDLE
    current_rep_frames = []
    shoulder_baseline_y = None
    torso_baseline_angle = None

    current_rep_id = 0
    total_reps_completed = 0
    in_rep = False
    rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}
    last_score = None
    last_label = None
    last_features = None

    # Create OpenCV window only if not in bridge mode (bridge runs in thread, can't create window)
    show_window = (bridge is None)
    if show_window:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, 1280, 720)
        cv2.moveWindow(WINDOW_TITLE, 100, 100)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            
            # Update MJPEG stream (if bridge mode) - BEFORE any overlays
            if bridge is not None:
                try:
                    # Import path: gymbud_pose_detection.py is in root, backend/ is sibling
                    import sys
                    from pathlib import Path
                    backend_path = Path(__file__).parent / 'backend'
                    if str(backend_path) not in sys.path:
                        sys.path.insert(0, str(backend_path))
                    from camera_stream import update_frame
                    update_frame(frame)  # Update stream with clean frame
                except ImportError as e:
                    # Only log once to avoid spam
                    if not hasattr(start_camera_session, '_stream_import_error_logged'):
                        print(f"⚠️ MJPEG stream import failed: {e}")
                        start_camera_session._stream_import_error_logged = True
                except Exception as e:
                    # Only log once to avoid spam
                    if not hasattr(start_camera_session, '_stream_error_logged'):
                        print(f"⚠️ MJPEG stream update failed: {e}")
                        start_camera_session._stream_error_logged = True
            
            # Keep base64 for WebSocket (backward compatibility, but MJPEG is primary)
            frame_base64 = None
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            elbow_angle = float("nan")
            shoulder_stability = 0.0
            torso_sway = 0.0
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=TEXT_COLOR, thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=TEXT_COLOR, thickness=2, circle_radius=2),
                )
                draw_landmark_labels(frame, results.pose_landmarks)
                raw_landmarks = extract_relevant_landmarks(results.pose_landmarks.landmark)
                
                landmarks_visible = are_landmarks_visible_for_biceps(raw_landmarks)
                rep_completed = False
                normalized_landmarks = None
                torso_angle = float("nan")
                
                if landmarks_visible:
                    normalized_landmarks = normalize_landmarks(raw_landmarks)
                    elbow_angle = calculate_elbow_angle(normalized_landmarks)
                    torso_angle = calculate_torso_angle(normalized_landmarks)
                    
                    # Send landmarks to bridge if available (every frame with landmarks)
                    if bridge is not None:
                        bridge.on_frame_processed(normalized_landmarks, raw_landmarks, None, None, total_reps_completed, frame_base64)
                    
                    if recording and shoulder_baseline_y is None:
                        shoulder_baseline_y = normalized_landmarks["LEFT_SHOULDER"]["y"]
                    if recording and torso_baseline_angle is None and not np.isnan(torso_angle):
                        torso_baseline_angle = torso_angle

                    if shoulder_baseline_y is not None:
                        shoulder_stability = normalized_landmarks["LEFT_SHOULDER"]["y"] - shoulder_baseline_y
                    if torso_baseline_angle is not None and not np.isnan(torso_angle):
                        torso_sway = torso_angle - torso_baseline_angle

                    rep_state, rep_completed = detect_rep_state(elbow_angle, rep_state)
                else:
                    elbow_angle = float("nan")

                if normalized_landmarks is not None:
                    frame_record = {
                        "timestamp": time.time(),
                        "raw_landmarks": raw_landmarks,
                        "normalized_landmarks": normalized_landmarks,
                        "elbow_angle": elbow_angle,
                        "shoulder_stability": shoulder_stability,
                        "torso_sway": torso_sway,
                    }
                else:
                    frame_record = {
                        "timestamp": time.time(),
                        "raw_landmarks": raw_landmarks,
                        "normalized_landmarks": {},
                        "elbow_angle": elbow_angle,
                        "shoulder_stability": shoulder_stability,
                        "torso_sway": torso_sway,
                    }

                if landmarks_visible and normalized_landmarks is not None:

                    if not in_rep and rep_state == "going_up":
                        in_rep = True
                        rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}

                    if in_rep:
                        current_timestamp = time.time()
                        rep_buffer["angles"].append(elbow_angle)
                        rep_buffer["timestamps"].append(current_timestamp)
                        rep_buffer["shoulder_y"].append(normalized_landmarks["LEFT_SHOULDER"]["y"])
                        rep_buffer["wrist_x"].append(normalized_landmarks["LEFT_WRIST"]["x"])
            else:
                # No landmarks detected, but still send frame to bridge
                if bridge is not None and frame_base64:
                    bridge.on_frame_processed({}, {}, None, None, total_reps_completed, frame_base64)
                
                rep_state = STATE_IDLE
                if in_rep:
                    in_rep = False
                    rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}

                # Rep tamamlandığında form score hesapla (sadece uzuvlar görünürse)
                if rep_completed and in_rep and landmarks_visible and normalized_landmarks is not None:
                    if model is not None and scaler is not None:
                        features = extract_rep_features_runtime(rep_buffer)
                        if features:
                            feature_vector = np.array([[features[key] for key in FEATURE_ORDER]], dtype=np.float32)
                            feature_vector_scaled = scaler.transform(feature_vector)
                            score = float(model.predict(feature_vector_scaled)[0])
                            label = score_to_label(score)

                            last_score = score
                            last_label = label
                            last_features = features
                            total_reps_completed += 1

                            log_rep_record(current_rep_id, features, score, label)
                            print(f"Rep {current_rep_id + 1}: Score={score:.1f}, Label={label} (Visible: ✓)")
                            
                            # Send to bridge if available (with updated score)
                            if bridge is not None:
                                # Re-encode frame (might have updated overlay)
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                                bridge.on_frame_processed(normalized_landmarks, raw_landmarks, score, label, total_reps_completed, frame_base64)

                    current_rep_id += 1
                    in_rep = False
                    rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}
                elif rep_completed and in_rep and not landmarks_visible:
                    # Rep tamamlandı ama uzuvlar görünmüyor - rep'i iptal et
                    print(f"Rep {current_rep_id + 1}: CANCELLED (Landmarks not visible during rep completion)")
                    in_rep = False
                    rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}

                if recording:
                    current_rep_frames.append(frame_record)
                    if rep_completed and current_rep_frames:
                        rep_path = save_rep_json(current_rep_id, current_rep_frames)
                        print(f"Rep {current_rep_id} saved to {rep_path}")
                        current_rep_frames = []

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Reps: {total_reps_completed}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            
            if results.pose_landmarks:
                raw_landmarks_check = extract_relevant_landmarks(results.pose_landmarks.landmark)
                landmarks_visible_check = are_landmarks_visible_for_biceps(raw_landmarks_check)
                visibility_status = "Arm: Visible" if landmarks_visible_check else "Arm: Not Visible"
                visibility_color = (0, 255, 0) if landmarks_visible_check else (0, 0, 255)
                cv2.putText(
                    frame,
                    visibility_status,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    visibility_color,
                    2,
                    cv2.LINE_AA,
                )

            if last_score is not None:
                score_color = COLOR_CORRECT
                if last_label == "Partial":
                    score_color = COLOR_PARTIAL
                elif last_label == "Leg_Drive":
                    score_color = COLOR_LEG_DRIVE

                frame_width = frame.shape[1]
                cv2.putText(
                    frame,
                    f"Score: {last_score:.1f}",
                    (frame_width - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    score_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Label: {last_label}",
                    (frame_width - 220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    score_color,
                    2,
                    cv2.LINE_AA,
                )

            if last_features is not None:
                draw_debug_panel(frame, last_features)

            cv2.putText(
                frame,
                "Press Q=Quit | R=Record",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

            # Show window only if not in bridge mode
            if show_window:
                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
                    state_text = "ON" if recording else "OFF"
                    print(f"Recording toggled {state_text}")
                    current_rep_frames = []
                    rep_state = STATE_IDLE
                    shoulder_baseline_y = None
                    torso_baseline_angle = None
                    in_rep = False
                    rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}
            else:
                # Bridge mode: process frames, send to WebSocket, but no OpenCV window
                # (OpenCV can't create windows in threads on macOS)
                pass

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def main():
    ensure_storage_paths()
    
    print("Loading model and scaler...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"Model loaded: {MODEL_PATH}")
        print(f"Scaler loaded: {SCALER_PATH}")
    except FileNotFoundError as e:
        print(f"Warning: Model files not found: {e}")
        print("Real-time scoring disabled. Please train the model first.")
        model = None
        scaler = None
    
    should_start = show_startup_screen(model, scaler)
    
    if should_start:
        start_camera_session(model, scaler)


if __name__ == "__main__":
    main()
