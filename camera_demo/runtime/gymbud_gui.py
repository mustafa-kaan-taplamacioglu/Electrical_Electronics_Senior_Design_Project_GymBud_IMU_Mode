#!/usr/bin/env python3
"""
Lightweight desktop GUI to showcase the GymBud camera pipeline and rep scoring.

This script intentionally keeps the existing Streamlit experience untouched.
It wraps the real-time pose detection utilities from `gymbud_pose_detection.py`
inside a small Tkinter interface so the camera feed, rep counter, and last
predicted form score can be demoed without launching Streamlit.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import gymbud_pose_detection as gpd


MODEL_PATH = Path(__file__).with_name("biceps_form_model.joblib")
SCALER_PATH = Path(__file__).with_name("scaler.joblib")


class CameraWorker(threading.Thread):
    """Background thread that mirrors the real-time pose loop for the GUI."""

    def __init__(
        self,
        frame_queue: queue.Queue,
        metrics_queue: queue.Queue,
        stop_event: threading.Event,
        model,
        scaler,
        camera_index: Optional[int] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event
        self.model = model
        self.scaler = scaler
        self.camera_index = camera_index

    def _safe_put(self, q: queue.Queue, payload) -> None:
        try:
            q.put_nowait(payload)
        except queue.Full:
            pass

    def run(self) -> None:
        cap, cam_idx = gpd.find_available_camera(self.camera_index)
        if cap is None:
            self._safe_put(
                self.metrics_queue,
                {"status": "error", "message": "Kamera bulunamadı. Lütfen bağlantıyı kontrol et."},
            )
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        rep_state = gpd.STATE_IDLE
        rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}
        in_rep = False
        total_reps = 0
        last_score = None
        last_label = None
        last_features = None

        prev_time = time.time()

        gpd.ensure_storage_paths()

        with gpd.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                elbow_angle = float("nan")
                rep_completed = False
                landmarks_visible = False
                normalized_landmarks = None

                if results.pose_landmarks:
                    gpd.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        gpd.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=gpd.mp_drawing.DrawingSpec(color=gpd.TEXT_COLOR, thickness=2, circle_radius=2),
                        connection_drawing_spec=gpd.mp_drawing.DrawingSpec(color=gpd.TEXT_COLOR, thickness=2, circle_radius=2),
                    )

                    raw_landmarks = gpd.extract_relevant_landmarks(results.pose_landmarks.landmark)
                    landmarks_visible = gpd.are_landmarks_visible_for_biceps(raw_landmarks)

                    if landmarks_visible:
                        normalized_landmarks = gpd.normalize_landmarks(raw_landmarks)
                        elbow_angle = gpd.calculate_elbow_angle(normalized_landmarks)
                        rep_state, rep_completed = gpd.detect_rep_state(elbow_angle, rep_state)

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
                        rep_state = gpd.STATE_IDLE
                        in_rep = False
                        rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}

                if rep_completed and in_rep and landmarks_visible and normalized_landmarks is not None:
                    features = gpd.extract_rep_features_runtime(rep_buffer)
                    if features and self.model is not None and self.scaler is not None:
                        feature_vector = np.array([[features[key] for key in gpd.FEATURE_ORDER]], dtype=np.float32)
                        feature_vector_scaled = self.scaler.transform(feature_vector)
                        score = float(self.model.predict(feature_vector_scaled)[0])
                        label = gpd.score_to_label(score)

                        last_score = score
                        last_label = label
                        last_features = features
                        total_reps += 1

                        gpd.log_rep_record(total_reps, features, score, label)

                    in_rep = False
                    rep_buffer = {"angles": [], "timestamps": [], "shoulder_y": [], "wrist_x": []}

                current_time = time.time()
                fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
                prev_time = current_time

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gpd.TEXT_COLOR, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Reps: {total_reps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gpd.TEXT_COLOR, 2, cv2.LINE_AA)

                visibility_status = "Arm: Visible" if landmarks_visible else "Arm: Not Visible"
                visibility_color = (0, 255, 0) if landmarks_visible else (0, 0, 255)
                cv2.putText(frame, visibility_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, visibility_color, 2, cv2.LINE_AA)

                if last_score is not None:
                    score_color = gpd.COLOR_CORRECT
                    if last_label == "Partial":
                        score_color = gpd.COLOR_PARTIAL
                    elif last_label == "Leg_Drive":
                        score_color = gpd.COLOR_LEG_DRIVE

                    frame_width = frame.shape[1]
                    cv2.putText(frame, f"Score: {last_score:.1f}", (frame_width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Label: {last_label}", (frame_width - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2, cv2.LINE_AA)

                if last_features is not None:
                    gpd.draw_debug_panel(frame, last_features)

                self._safe_put(self.metrics_queue, {"status": "ok", "reps": total_reps, "score": last_score, "label": last_label, "camera_index": cam_idx})

                if not self.frame_queue.full():
                    self._safe_put(self.frame_queue, frame.copy())

        cap.release()
        self._safe_put(self.metrics_queue, {"status": "stopped"})


class GymBudGUI:
    """Tkinter front-end that talks to the camera worker."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("GymBud Camera Demo")
        self.root.geometry("1100x800")
        self.root.configure(bg="#111111")

        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.worker: Optional[CameraWorker] = None

        self.model = self._load_artifact(MODEL_PATH)
        self.scaler = self._load_artifact(SCALER_PATH)

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._poll_queues)

    def _load_artifact(self, path: Path):
        try:
            artifact = joblib.load(path)
            return artifact
        except FileNotFoundError:
            messagebox.showwarning("Eksik model", f"{path.name} bulunamadı. Skor gösterimi devre dışı.")
            return None

    def _build_layout(self) -> None:
        header = ttk.Label(
            self.root,
            text="GymBud - Kamera Gösterimi",
            font=("Helvetica", 20, "bold"),
            foreground="#00FF7F",
            background="#111111",
        )
        header.pack(pady=10)

        controls = ttk.Frame(self.root)
        controls.pack(pady=10)

        ttk.Label(controls, text="Kamera Index:", font=("Helvetica", 12)).grid(row=0, column=0, padx=5)
        self.camera_var = tk.StringVar()
        camera_entry = ttk.Entry(controls, textvariable=self.camera_var, width=5)
        camera_entry.grid(row=0, column=1, padx=5)

        start_btn = ttk.Button(controls, text="Başlat", command=self.start_camera)
        start_btn.grid(row=0, column=2, padx=10)

        stop_btn = ttk.Button(controls, text="Durdur", command=self.stop_camera)
        stop_btn.grid(row=0, column=3, padx=10)

        self.status_var = tk.StringVar(value="Hazır")
        status_label = ttk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 12), background="#111111", foreground="#DDDDDD")
        status_label.pack(pady=5)

        self.video_panel = ttk.Label(self.root)
        self.video_panel.pack(pady=10)

        metrics_frame = ttk.Frame(self.root)
        metrics_frame.pack(pady=10)

        self.reps_var = tk.StringVar(value="Toplam Rep: 0")
        self.score_var = tk.StringVar(value="Skor: -")
        self.label_var = tk.StringVar(value="Etiket: -")

        ttk.Label(metrics_frame, textvariable=self.reps_var, font=("Helvetica", 14)).grid(row=0, column=0, padx=15)
        ttk.Label(metrics_frame, textvariable=self.score_var, font=("Helvetica", 14)).grid(row=0, column=1, padx=15)
        ttk.Label(metrics_frame, textvariable=self.label_var, font=("Helvetica", 14)).grid(row=0, column=2, padx=15)

    def start_camera(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Bilgi", "Kamera zaten çalışıyor.")
            return

        camera_index = None
        user_value = self.camera_var.get().strip()
        if user_value:
            try:
                camera_index = int(user_value)
            except ValueError:
                messagebox.showerror("Hata", "Kamera index'i sayı olmalı.")
                return

        self.stop_event.clear()
        self.worker = CameraWorker(
            frame_queue=self.frame_queue,
            metrics_queue=self.metrics_queue,
            stop_event=self.stop_event,
            model=self.model,
            scaler=self.scaler,
            camera_index=camera_index,
        )
        self.worker.start()
        self.status_var.set("Kamera başlatılıyor...")

    def stop_camera(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self.worker.join(timeout=2)
        self.worker = None
        self.status_var.set("Kamera durdu.")

    def _poll_queues(self) -> None:
        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.resize((960, 540))
            photo = ImageTk.PhotoImage(image=image)
            self.video_panel.configure(image=photo)
            self.video_panel.image = photo
        except queue.Empty:
            pass

        try:
            metrics = self.metrics_queue.get_nowait()
            if metrics.get("status") == "error":
                messagebox.showerror("Kamera Hatası", metrics.get("message", "Bilinmeyen hata"))
                self.stop_camera()
            elif metrics.get("status") == "ok":
                self.reps_var.set(f"Toplam Rep: {metrics.get('reps', 0)}")
                score_val = metrics.get("score")
                score_text = f"{score_val:.1f}" if score_val is not None else "-"
                self.score_var.set(f"Skor: {score_text}")
                self.label_var.set(f"Etiket: {metrics.get('label') or '-'}")
                self.status_var.set(f"Kamera #{metrics.get('camera_index', '?')} çalışıyor")
            elif metrics.get("status") == "stopped":
                self.status_var.set("Kamera kapandı.")
        except queue.Empty:
            pass

        self.root.after(50, self._poll_queues)

    def _on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    gui = GymBudGUI()
    gui.run()


if __name__ == "__main__":
    main()

