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
import mediapipe as mp
import numpy as np


SNAPSHOT_DIR = Path(__file__).with_name("pose_snapshots")
WINDOW_TITLE = "GymBud - Real-Time Pose Detection"
TEXT_COLOR = (0, 255, 0)

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
ANGLE_THRESHOLDS = {"top_angle": 150.0, "bottom_angle": 70.0}
ANGLE_HYSTERESIS = 5.0


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
    top_angle = ANGLE_THRESHOLDS["top_angle"]
    bottom_angle = ANGLE_THRESHOLDS["bottom_angle"]

    if state == STATE_IDLE:
        if elbow_angle >= top_angle:
            state = "going_up"
    elif state == "going_up":
        if elbow_angle <= bottom_angle:
            state = "top"
    elif state == "top":
        if elbow_angle > bottom_angle + ANGLE_HYSTERESIS:
            state = "going_down"
    elif state == "going_down":
        if elbow_angle >= top_angle - ANGLE_HYSTERESIS:
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


def draw_overlay(frame, lines, origin=(10, 30), line_height=25):
    for idx, text in enumerate(lines):
        y = origin[1] + idx * line_height
        cv2.putText(
            frame,
            text,
            (origin[0], y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )


def find_available_camera():
    """Try different camera indices to find an available one."""
    for idx in range(5):  # Try indices 0-4
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"Camera found at index {idx}")
                return cap, idx
            cap.release()
    return None, None


def main():
    ensure_storage_paths()

    cap, camera_idx = find_available_camera()
    if cap is None:
        raise RuntimeError("Unable to access any camera. Ensure it is connected and permissions are granted.")

    prev_time = time.time()
    recording = False
    rep_count = 0
    rep_state = STATE_IDLE
    current_rep_frames = []
    shoulder_baseline_y = None
    torso_baseline_angle = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
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
                raw_landmarks = extract_relevant_landmarks(results.pose_landmarks.landmark)
                normalized_landmarks = normalize_landmarks(raw_landmarks)
                elbow_angle = calculate_elbow_angle(normalized_landmarks)
                torso_angle = calculate_torso_angle(normalized_landmarks)

                if recording and shoulder_baseline_y is None:
                    shoulder_baseline_y = normalized_landmarks["LEFT_SHOULDER"]["y"]
                if recording and torso_baseline_angle is None and not np.isnan(torso_angle):
                    torso_baseline_angle = torso_angle

                if shoulder_baseline_y is not None:
                    shoulder_stability = normalized_landmarks["LEFT_SHOULDER"]["y"] - shoulder_baseline_y
                if torso_baseline_angle is not None and not np.isnan(torso_angle):
                    torso_sway = torso_angle - torso_baseline_angle

                frame_record = {
                    "timestamp": time.time(),
                    "raw_landmarks": raw_landmarks,
                    "normalized_landmarks": normalized_landmarks,
                    "elbow_angle": elbow_angle,
                    "shoulder_stability": shoulder_stability,
                    "torso_sway": torso_sway,
                }

                if recording:
                    current_rep_frames.append(frame_record)
                    rep_state, rep_completed = detect_rep_state(elbow_angle, rep_state)
                    if rep_completed and current_rep_frames:
                        rep_count += 1
                        rep_path = save_rep_json(rep_count, current_rep_frames)
                        print(f"Rep {rep_count} saved to {rep_path}")
                        current_rep_frames = []
                else:
                    rep_state = STATE_IDLE
            else:
                rep_state = STATE_IDLE

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time

            overlay_lines = [
                f"Elbow Angle: {elbow_angle:.1f}°" if not np.isnan(elbow_angle) else "Elbow Angle: --",
                f"Rep Count: {rep_count}",
                f"Recording: {'ON' if recording else 'OFF'}",
                f"FPS: {fps:.1f}",
            ]
            draw_overlay(frame, overlay_lines)
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

            cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
