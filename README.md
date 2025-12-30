# GymBud Real-Time Pose Detection

Python script for capturing biceps curl reps with MediaPipe Pose and OpenCV.  
Focuses on eight landmarks, normalizes them to pelvis-centered coordinates, and saves per-rep JSON snapshots.

## Requirements

- macOS with a working webcam (built-in, Continuity Camera, or USB)  
- Python 3.12 (Homebrew `python@3.12` works)  
- Virtual environment recommended

Install Python dependencies:

```bash
cd /Users/derinegeevren/camera_491
/opt/homebrew/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

## Running the Script

```bash
source .venv312/bin/activate
python gymbud_pose_detection.py
```

### Controls

- `R`: Toggle recording on/off. When ON the script buffers frames, detects reps, and writes `pose_snapshots/rep_<count>_<timestamp>.json`.  
- `Q`: Quit application.  
- Overlay shows elbow angle, rep count, recording state, and FPS.

### Camera Detection

The script automatically tries camera indices 0–4, so Continuity Camera or USB cameras are picked up without code changes. If no camera can be opened it raises an error prompting you to check permissions/connections.

## Output Format

Each `rep_XXX.json` file contains:

- `rep_number` — sequential rep counter  
- `frames` — list of frame dictionaries including:  
  - `timestamp`  
  - `raw_landmarks` (8 essential landmarks)  
  - `normalized_landmarks` (pelvis-centered, height-normalized)  
  - `elbow_angle`, `shoulder_stability`, `torso_sway`

Recording is CPU-only; no GPU acceleration is enabled to stay compliant with environments where GPU use is restricted.

