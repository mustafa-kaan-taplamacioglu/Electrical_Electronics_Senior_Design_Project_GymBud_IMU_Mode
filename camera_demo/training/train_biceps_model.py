#!/usr/bin/env python3
"""
GymBud – Biceps Curl ML Training Pipeline

This script:
1. Loads frame-based CSV dataset
2. Detects reps using elbow angle state machine
3. Extracts biomechanically meaningful features per rep
4. Trains a RandomForestRegressor model for form scoring
5. Saves model, scaler, and rep features

CPU-only, no GPU required.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Configuration
# ============================================================================

# Input dataset path (adjust if needed)
# Use augmented dataset if available, otherwise use original
INPUT_CSV = "/mnt/data/training_data_bicep_curls.csv"
AUGMENTED_CSV = "training_data_bicep_curls_augmented.csv"

# Rep detection parameters
TOP_ANGLE_DEG = 150.0
BOTTOM_ANGLE_DEG = 70.0
ANGLE_HYSTERESIS = 5.0

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Output files
MODEL_FILE = "biceps_form_model.joblib"
SCALER_FILE = "scaler.joblib"
FEATURES_FILE = "rep_features.csv"

# Rep state machine states
STATE_IDLE = "idle"
STATE_GOING_UP = "going_up"
STATE_TOP = "top"
STATE_GOING_DOWN = "going_down"


# ============================================================================
# Data Loading
# ============================================================================


def load_data(path: str) -> pd.DataFrame:
    """
    Load the training CSV dataset.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with frame-based data
    """
    print(f"Loading dataset from: {path}")
    
    # Try the specified path first, then fallback to local paths
    if not os.path.exists(path):
        # Try augmented dataset first
        workspace_augmented = Path(__file__).parent / AUGMENTED_CSV
        if workspace_augmented.exists():
            path = str(workspace_augmented)
            print(f"Using augmented dataset: {path}")
        else:
            # Try original dataset as fallback
            workspace_path = Path(__file__).parent / "training_data_bicep_curls.csv"
            if workspace_path.exists():
                path = str(workspace_path)
                print(f"Using fallback path: {path}")
            else:
                raise FileNotFoundError(
                    f"CSV file not found at {path}, {workspace_augmented}, or {workspace_path}. "
                    "Please update INPUT_CSV path."
                )
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} frames")
    print(f"Columns: {len(df.columns)}")
    return df


# ============================================================================
# Rep Detection
# ============================================================================


def get_elbow_angle(df: pd.DataFrame, row_idx: int) -> float:
    """
    Extract elbow angle from a row.

    Tries multiple methods:
    1. Direct 'left elbow' column (if exists)
    2. Calculate from LShoulder-LElbow-LWrist coordinates
    3. Fallback to 'right elbow' if left not available

    Args:
        df: DataFrame with frame data
        row_idx: Row index

    Returns:
        Elbow angle in degrees (or NaN if unavailable)
    """
    row = df.iloc[row_idx]
    
    # Method 1: Check for direct angle column
    angle_columns = ["left elbow", "right elbow", "LElbow", "left_elbow"]
    for col in angle_columns:
        if col in df.columns:
            angle = row[col]
            if pd.notna(angle) and not np.isnan(angle):
                return float(angle)
    
    # Method 2: Calculate from coordinates
    try:
        # Extract 3D coordinates
        shoulder = np.array([row["LShoulder_X20"], row["LShoulder_Y20"], row["LShoulder_Z20"]])
        elbow = np.array([row["LElbow_X21"], row["LElbow_Y21"], row["LElbow_Z21"]])
        wrist = np.array([row["LWrist_X22"], row["LWrist_Y22"], row["LWrist_Z22"]])
        
        # Calculate angle using vectors
        ba = shoulder - elbow
        bc = wrist - elbow
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba > 1e-6 and norm_bc > 1e-6:
            cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle_rad = np.arccos(cosine_angle)
            return float(np.degrees(angle_rad))
    except (KeyError, ValueError, TypeError):
        pass
    
    return float("nan")


def detect_rep_state(elbow_angle: float, current_state: str) -> tuple[str, bool]:
    """
    State machine for rep detection based on elbow angle.

    States: idle -> going_up -> top -> going_down -> idle (rep complete)

    Args:
        elbow_angle: Current elbow angle in degrees
        current_state: Current state

    Returns:
        Tuple of (new_state, rep_completed)
    """
    if np.isnan(elbow_angle):
        return current_state, False
    
    state = current_state
    rep_completed = False
    
    if state == STATE_IDLE:
        if elbow_angle >= TOP_ANGLE_DEG:
            state = STATE_GOING_UP
    elif state == STATE_GOING_UP:
        if elbow_angle <= BOTTOM_ANGLE_DEG:
            state = STATE_TOP
    elif state == STATE_TOP:
        if elbow_angle > BOTTOM_ANGLE_DEG + ANGLE_HYSTERESIS:
            state = STATE_GOING_DOWN
    elif state == STATE_GOING_DOWN:
        if elbow_angle >= TOP_ANGLE_DEG - ANGLE_HYSTERESIS:
            state = STATE_IDLE
            rep_completed = True
    else:
        state = STATE_IDLE
    
    return state, rep_completed


def detect_reps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect rep segments from frame-based data using elbow angle state machine.

    Adds 'rep_id' column to mark which frames belong to which rep.
    Frames not part of a complete rep are marked with rep_id = -1.

    Args:
        df: DataFrame with frame data

    Returns:
        DataFrame with added 'rep_id' column
    """
    print("Detecting reps using elbow angle state machine...")
    
    df = df.copy()
    df["rep_id"] = -1
    df["elbow_angle"] = np.nan
    df["state"] = STATE_IDLE
    
    # Calculate elbow angles for all rows
    for idx in range(len(df)):
        angle = get_elbow_angle(df, idx)
        df.at[idx, "elbow_angle"] = angle
    
    # Detect reps
    current_state = STATE_IDLE
    current_rep_id = 0
    rep_start_idx = None
    in_rep = False
    
    for idx in range(len(df)):
        angle = df.at[idx, "elbow_angle"]
        new_state, rep_completed = detect_rep_state(angle, current_state)
        df.at[idx, "state"] = new_state
        current_state = new_state
        
        # Start tracking a new rep when going_up begins
        if new_state == STATE_GOING_UP and not in_rep:
            in_rep = True
            rep_start_idx = idx
        
        # Mark frames in the rep
        if in_rep:
            df.at[idx, "rep_id"] = current_rep_id
        
        # Rep completed
        if rep_completed:
            in_rep = False
            current_rep_id += 1
            rep_start_idx = None
    
    # Count detected reps
    num_reps = df[df["rep_id"] >= 0]["rep_id"].nunique()
    print(f"Detected {num_reps} complete reps")
    print(f"Frames in reps: {len(df[df['rep_id'] >= 0])}")
    print(f"Frames outside reps: {len(df[df['rep_id'] < 0])}")
    
    return df


# ============================================================================
# Feature Extraction
# ============================================================================


def normalize_landmarks_csv(row):
    """Normalize CSV landmark coordinates using the SAME logic as gymbud_pose_detection runtime."""
    # Hip center
    pelvis = np.array([
        (row['LHip_X8'] + row['RHip_X2']) / 2.0,
        (row['LHip_Y8'] + row['RHip_Y2']) / 2.0,
        (row['LHip_Z8'] + row['RHip_Z2']) / 2.0,
    ])

    # Landmarks used for stability
    lm = {
        'shoulder': np.array([row['LShoulder_X20'], row['LShoulder_Y20'], row['LShoulder_Z20']]),
        'elbow':    np.array([row['LElbow_X21'], row['LElbow_Y21'], row['LElbow_Z21']]),
        'wrist':    np.array([row['LWrist_X22'], row['LWrist_Y22'], row['LWrist_Z22']])
    }

    # Body height
    body_height = max(np.linalg.norm(v - pelvis) for v in lm.values())
    body_height = max(body_height, 1e-6)

    # Normalize all
    norm = {}
    for k, v in lm.items():
        nv = (v - pelvis) / body_height
        norm[k] = nv

    return norm


def extract_rep_features(rep_frames: pd.DataFrame) -> dict:
    """Extract biomechanically meaningful features from normalized CSV rep."""
    if len(rep_frames) == 0:
        return {}

    # Normalize ALL frames
    normalized_frames = [normalize_landmarks_csv(row) for _, row in rep_frames.iterrows()]

    # === ANGLE FEATURES ===
    angles = []
    for nf in normalized_frames:
        shoulder = nf['shoulder']
        elbow = nf['elbow']
        wrist = nf['wrist']

        ba = shoulder - elbow
        bc = wrist - elbow
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = float(np.degrees(np.arccos(cos_angle)))
        angles.append(angle)

    angles = np.array(angles)
    min_elbow = float(np.min(angles))
    max_elbow = float(np.max(angles))
    rom = max_elbow - min_elbow

    # === BODY INVARIANT SHOULDER + WRIST SWAY ===
    shoulder_y = [nf['shoulder'][1] for nf in normalized_frames]
    wrist_x = [nf['wrist'][0] for nf in normalized_frames]

    # Baseline = first 5 frames
    baseline_sh_y = np.mean(shoulder_y[:5])
    baseline_wr_x = np.mean(wrist_x[:5])

    # Relative change → body-size invariant
    shoulder_stability = float(np.std([(y - baseline_sh_y) for y in shoulder_y]))
    wrist_sway = float(np.std([(x - baseline_wr_x) for x in wrist_x]))

    # === TEMPO ===
    min_idx = int(np.argmin(angles))
    up_frames = angles[:min_idx]
    down_frames = angles[min_idx:]
    up_duration = len(up_frames) / 30.0
    down_duration = len(down_frames) / 30.0
    tempo_ratio = up_duration / max(down_duration, 1e-6)

    # === OUTPUT ===
    return {
        'ROM': rom,
        'min_elbow_angle': min_elbow,
        'max_elbow_angle': max_elbow,
        'rep_duration_seconds': len(angles) / 30.0,
        'avg_speed': rom / max(len(angles)/30.0, 1e-6),
        'shoulder_stability': shoulder_stability,
        'wrist_sway': wrist_sway,
        'mean_elbow_angle': float(np.mean(angles)),
        'angle_range': rom,
        'peak_contraction_angle': min_elbow,
        'tempo_up_down_ratio': tempo_ratio
    }


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rep-level feature dataset from frame-based data with rep labels.

    Args:
        df: DataFrame with rep_id column (from detect_reps)

    Returns:
        DataFrame with one row per rep and extracted features
    """
    print("Extracting features for each rep...")
    
    rep_features = []
    rep_ids = sorted(df[df["rep_id"] >= 0]["rep_id"].unique())
    
    for rep_id in rep_ids:
        rep_frames = df[df["rep_id"] == rep_id].copy()
        features = extract_rep_features(rep_frames)
        
        if features:
            # Add rep_id and label if available
            features["rep_id"] = int(rep_id)
            
            # Try to get label from frames (assume all frames in rep have same label)
            if "label" in rep_frames.columns:
                label = rep_frames["label"].iloc[0]
                features["label"] = label
            
            # Try to get filename
            if "filename" in rep_frames.columns:
                filename = rep_frames["filename"].iloc[0]
                features["filename"] = filename
            
            rep_features.append(features)
    
    if not rep_features:
        raise ValueError("No rep features extracted! Check rep detection logic.")
    
    feature_df = pd.DataFrame(rep_features)
    print(f"Built feature dataset with {len(feature_df)} reps")
    print(f"Feature columns: {list(feature_df.columns)}")
    
    return feature_df


# ============================================================================
# Model Training
# ============================================================================


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Train RandomForestRegressor model.

    Args:
        X: Feature matrix
        y: Target scores

    Returns:
        Tuple of (model, scaler)
    """
    print("Training RandomForestRegressor...")
    print(f"Training samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\n" + "="*60)
    print("Model Evaluation:")
    print("="*60)
    print(f"Training MAE: {mae_train:.4f}")
    print(f"Test MAE:     {mae_test:.4f}")
    print(f"Training R²:  {r2_train:.4f}")
    print(f"Test R²:      {r2_test:.4f}")
    print("="*60)
    
    # Feature importance
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))
    
    return model, scaler


def save_model(model, scaler, feature_names: list = None):
    """
    Save trained model and scaler to disk.

    Args:
        model: Trained RandomForestRegressor
        scaler: Fitted StandardScaler
        feature_names: List of feature names (optional)
    """
    print(f"\nSaving model to: {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)
    
    print(f"Saving scaler to: {SCALER_FILE}")
    joblib.dump(scaler, SCALER_FILE)
    
    if feature_names:
        feature_names_file = "feature_names.txt"
        with open(feature_names_file, "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"Saved feature names to: {feature_names_file}")


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Main training pipeline."""
    print("="*60)
    print("GymBud – Biceps Curl ML Training Pipeline")
    print("="*60)
    print()
    
    try:
        # Step 1: Load data
        df = load_data(INPUT_CSV)
        print(f"Dataset shape: {df.shape}\n")
        
        # Step 2: Detect reps
        df_with_reps = detect_reps(df)
        print()
        
        # Step 3: Extract features
        feature_df = build_feature_dataset(df_with_reps)
        print(f"\nFeature dataset head:")
        print(feature_df.head())
        print()
        
        # Step 4: Prepare features and labels
        # Select feature columns (exclude metadata)
        feature_columns = [
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
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in feature_df.columns]
        X = feature_df[available_features].values
        
        # Create target scores
        # If label column exists and has meaningful values, use them
        # Otherwise assign baseline score of 100
        if "label" in feature_df.columns:
            # Map labels to scores
            # Correct = 100 (perfect form)
            # Leg_Drive = 75 (minor issue - using leg drive)
            # Partial = 60 (partial range of motion)
            # Incorrect = 50 (poor form)
            label_mapping = {
                "Correct": 100.0,
                "Leg_Drive": 75.0,
                "Partial": 60.0,
                "Incorrect": 50.0
            }
            y = feature_df["label"].map(label_mapping).fillna(100.0).values
            print(f"Using labels: {feature_df['label'].value_counts().to_dict()}")
            print(f"Score mapping: {label_mapping}")
        else:
            # No labels - assign baseline score
            y = np.full(len(feature_df), 100.0)
            print("No labels found - assigning baseline score of 100 to all reps")
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Features used: {available_features}\n")
        
        # Step 5: Train model
        model, scaler = train_model(X, y)
        
        # Step 6: Save outputs
        save_model(model, scaler, available_features)
        
        # Save rep features
        print(f"\nSaving rep features to: {FEATURES_FILE}")
        feature_df.to_csv(FEATURES_FILE, index=False)
        
        print("\n" + "="*60)
        print("Training pipeline completed successfully!")
        print("="*60)
        print(f"\nOutput files:")
        print(f"  - {MODEL_FILE}")
        print(f"  - {SCALER_FILE}")
        print(f"  - {FEATURES_FILE}")
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

