#!/usr/bin/env python3
"""
Train One-Class Classifier for Bicep Curl Rep Detection
Uses "One-class classification with reject option" approach:
- Accept: Bicep curl reps (from training data)
- Reject: All other movements (lateral arm raise, etc.)

This classifier validates that detected reps are actually bicep curls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from imu_feature_extractor import extract_imu_features

# Configuration
BASE_DIR = Path(__file__).parent
MLTRAIN_DIR = BASE_DIR / "MLTRAINIMU" / "bicep_curls"

# Training data folders (3 CSV files from user)
TRAINING_FOLDERS = [
    "bicep_curls_20260103_230109",  # Medium speed
    "bicep_curls_20260103_230629",  # Fast speed
    "bicep_curls_20260103_230946",  # Slow speed
]

# Output files
MODEL_FILE = BASE_DIR / "bicep_curl_one_class_svm.joblib"
SCALER_FILE = BASE_DIR / "bicep_curl_one_class_scaler.joblib"
FEATURES_FILE = BASE_DIR / "bicep_curl_one_class_features.json"

# Model parameters
USE_ISOLATION_FOREST = False  # Set True to use Isolation Forest instead of One-Class SVM
OCSVM_NU = 0.35  # Expected fraction of outliers in training data (35% - very lenient to accept more variations)
OCSVM_GAMMA = 'scale'  # Kernel coefficient
ISOLATION_CONTAMINATION = 0.35  # Expected fraction of outliers (35% - very lenient)


def csv_to_imu_sequence(df_rep):
    """
    Convert CSV DataFrame (for one rep) to IMU sequence format.
    
    Args:
        df_rep: DataFrame with rows for one rep_number, sorted by timestamp
    
    Returns:
        List of IMU samples in format expected by extract_imu_features
    """
    imu_sequence = []
    
    # Group by timestamp (each timestamp has left_wrist and right_wrist rows)
    for timestamp in df_rep['timestamp'].unique():
        ts_data = df_rep[df_rep['timestamp'] == timestamp].copy()
        
        sample = {'timestamp': timestamp}
        
        # Extract left_wrist and right_wrist data
        for _, row in ts_data.iterrows():
            node_name = row['node_name']
            
            sample[node_name] = {
                'ax': float(row['ax']),
                'ay': float(row['ay']),
                'az': float(row['az']),
                'gx': float(row['gx']),
                'gy': float(row['gy']),
                'gz': float(row['gz']),
                'qw': float(row['qw']),
                'qx': float(row['qx']),
                'qy': float(row['qy']),
                'qz': float(row['qz']),
                'roll': float(row['roll']),
                'pitch': float(row['pitch']),
                'yaw': float(row['yaw']),
            }
        
        imu_sequence.append(sample)
    
    # Sort by timestamp
    imu_sequence.sort(key=lambda x: x['timestamp'])
    
    return imu_sequence


def extract_rep_sequences_from_csv(csv_file):
    """
    Extract rep sequences from CSV file.
    
    Args:
        csv_file: Path to imu_samples.csv file
    
    Returns:
        List of DataFrames, one per valid rep
    """
    print(f"  Loading CSV: {csv_file.name}")
    df = pd.read_csv(csv_file)
    
    # Extract valid reps (rep_number > 0)
    valid_rep_numbers = df[df['rep_number'] > 0]['rep_number'].unique()
    
    reps = []
    for rep_num in sorted(valid_rep_numbers):
        rep_data = df[df['rep_number'] == rep_num].copy()
        
        # Filter: require at least 5 samples (minimum for feature extraction)
        if len(rep_data) >= 5:
            # Sort by timestamp
            rep_data = rep_data.sort_values('timestamp')
            reps.append(rep_data)
    
    print(f"    Found {len(reps)} valid reps (rep_number > 0, >= 5 samples)")
    return reps


def extract_features_from_reps(reps_list):
    """
    Extract IMU features from rep sequences.
    
    Args:
        reps_list: List of DataFrames (one per rep)
    
    Returns:
        List of feature dictionaries
    """
    rep_features = []
    
    for idx, rep_df in enumerate(reps_list):
        # Convert CSV format to IMU sequence format
        imu_sequence = csv_to_imu_sequence(rep_df)
        
        # Extract features
        features = extract_imu_features(imu_sequence)
        
        if features and len(features) > 0:
            rep_features.append(features)
        else:
            print(f"    Warning: Rep {idx+1} produced no features, skipping")
    
    return rep_features


def train_one_class_classifier(X_features):
    """
    Train one-class classifier.
    
    Args:
        X_features: List of feature dictionaries
    
    Returns:
        (model, scaler, feature_names)
    """
    # Convert to DataFrame
    X_df = pd.DataFrame(X_features)
    
    # Get feature names (in consistent order)
    feature_names = sorted(X_df.columns.tolist())
    X = X_df[feature_names].values
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    if USE_ISOLATION_FOREST:
        print(f"\nTraining Isolation Forest (contamination={ISOLATION_CONTAMINATION})...")
        model = IsolationForest(
            contamination=ISOLATION_CONTAMINATION,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        print("✅ Isolation Forest trained")
    else:
        print(f"\nTraining One-Class SVM (nu={OCSVM_NU}, gamma={OCSVM_GAMMA})...")
        model = OneClassSVM(
            nu=OCSVM_NU,
            gamma=OCSVM_GAMMA,
            kernel='rbf'
        )
        model.fit(X_scaled)
        print("✅ One-Class SVM trained")
    
    # Validate on training data (should accept most samples)
    predictions = model.predict(X_scaled)
    n_inliers = np.sum(predictions == 1)
    n_outliers = np.sum(predictions == -1)
    
    print(f"\nTraining data validation:")
    print(f"  Inliers (accepted): {n_inliers} ({100*n_inliers/len(X):.1f}%)")
    print(f"  Outliers (rejected): {n_outliers} ({100*n_outliers/len(X):.1f}%)")
    
    return model, scaler, feature_names


def save_model(model, scaler, feature_names):
    """Save model, scaler, and feature names."""
    print(f"\nSaving model and scaler...")
    
    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"  ✅ Model saved: {MODEL_FILE}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_FILE)
    print(f"  ✅ Scaler saved: {SCALER_FILE}")
    
    # Save feature names (for inference)
    import json
    with open(FEATURES_FILE, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"  ✅ Feature names saved: {FEATURES_FILE}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("Bicep Curl One-Class Classifier Training")
    print("=" * 70)
    
    # Step 1: Load training data
    print("\n[1/4] Loading training data...")
    all_reps = []
    
    for folder_name in TRAINING_FOLDERS:
        folder_path = MLTRAIN_DIR / folder_name
        csv_file = folder_path / "imu_samples.csv"
        
        if not csv_file.exists():
            print(f"  ⚠️  Warning: {csv_file} not found, skipping")
            continue
        
        reps = extract_rep_sequences_from_csv(csv_file)
        all_reps.extend(reps)
        print(f"  ✅ {folder_name}: {len(reps)} reps")
    
    if len(all_reps) == 0:
        print("\n❌ ERROR: No valid reps found in training data!")
        sys.exit(1)
    
    print(f"\nTotal reps collected: {len(all_reps)}")
    
    # Step 2: Extract features
    print("\n[2/4] Extracting features from reps...")
    rep_features = extract_features_from_reps(all_reps)
    
    if len(rep_features) == 0:
        print("\n❌ ERROR: No features extracted!")
        sys.exit(1)
    
    print(f"✅ Extracted features from {len(rep_features)} reps")
    
    # Step 3: Train model
    print("\n[3/4] Training one-class classifier...")
    model, scaler, feature_names = train_one_class_classifier(rep_features)
    
    # Step 4: Save model
    print("\n[4/4] Saving model...")
    save_model(model, scaler, feature_names)
    
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {MODEL_FILE}")
    print(f"  - {SCALER_FILE}")
    print(f"  - {FEATURES_FILE}")
    print(f"\nNext step: Integrate into HybridIMURepDetector")


if __name__ == "__main__":
    main()

