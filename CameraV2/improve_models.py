"""
Model Performansını İyileştirme Script'i
========================================
Bu script mevcut modelleri daha iyi hyperparameter'larla yeniden eğitir.
"""

import argparse
from pathlib import Path
from train_ml_models import train_camera_model, train_imu_model, train_fusion_model
from ml_trainer import FormScorePredictor
from dataset_collector import DatasetCollector
from imu_dataset_collector import IMUDatasetCollector


def train_camera_model_improved(exercise: str):
    """Train camera model with improved hyperparameters (reduced overfitting)."""
    print(f"\n📹 Training IMPROVED Camera Model for {exercise}...")
    
    collector = DatasetCollector("MLTRAINCAMERA")
    samples = collector.load_dataset(exercise=exercise)
    
    if len(samples) == 0:
        print(f"❌ No samples found")
        return False
    
    print(f"   Found {len(samples)} samples")
    
    # Extract features
    print("   Extracting features...")
    for sample in samples:
        if sample.features is None:
            collector.extract_features(sample)
    
    # Z-score perfect form selection
    from zscore_perfect_form_selector import ZScorePerfectFormSelector
    selector = ZScorePerfectFormSelector(z_threshold=1.0, min_features_acceptable=0.9)
    perfect_samples, non_perfect_samples, _ = selector.select_perfect_form_samples(
        samples, use_imu_features=False, verbose=True
    )
    
    # Auto-label
    labeled_samples = [s for s in samples if s.expert_score is not None or s.is_perfect_form is not None]
    if len(labeled_samples) == 0:
        for sample in samples:
            if sample.regional_scores:
                avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                sample.expert_score = avg_score
        labeled_samples = samples
    
    if len(labeled_samples) < 10:
        print(f"❌ Not enough labeled samples")
        return False
    
    # IMPROVED: Custom predictor with better hyperparameters
    class ImprovedCameraPredictor(FormScorePredictor):
        def __init__(self):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.multioutput import MultiOutputRegressor
            
            # More conservative hyperparameters (reduce overfitting)
            base_model = RandomForestRegressor(
                n_estimators=100,      # Keep same
                max_depth=6,           # REDUCED from 10 to 6 (less complexity)
                min_samples_split=10,  # INCREASED from 5 to 10 (more samples needed to split)
                min_samples_leaf=5,    # NEW: minimum samples in leaf
                max_features='sqrt',   # NEW: feature subsampling
                random_state=42,
                n_jobs=-1
            )
            self.model = MultiOutputRegressor(base_model, n_jobs=-1)
            self.model_type = "random_forest"
            self.multi_output = True
            self.scaler = None
            self.feature_names = None
            self.is_trained = False
    
    predictor = ImprovedCameraPredictor()
    
    # Use parent class methods but with our improved model
    predictor.scaler = FormScorePredictor.scaler.__init__(predictor)  # Initialize scaler
    predictor.scaler = __import__('sklearn.preprocessing').preprocessing.StandardScaler()
    
    # Train using parent's prepare_features and train logic
    from ml_trainer import FormScorePredictor as BasePredictor
    predictor.prepare_features = BasePredictor.prepare_features.__get__(predictor, ImprovedCameraPredictor)
    
    # Manual training (simplified)
    X, y = predictor.prepare_features(labeled_samples, use_imu_features=False)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    predictor.model.fit(X_train_scaled, y_train)
    predictor.scaler = scaler
    predictor.feature_names = predictor.feature_names
    predictor.is_trained = True
    
    # Evaluate
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_train_pred = predictor.model.predict(X_train_scaled)
    y_test_pred = predictor.model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
    test_r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
    
    print(f"\n📈 IMPROVED Results:")
    print(f"   Train R²: {train_r2:.3f}")
    print(f"   Test R²:  {test_r2:.3f}")
    print(f"   Gap: {train_r2 - test_r2:.3f} (smaller is better)")
    
    # Save
    model_dir = Path("models") / exercise / "form_score_camera_random_forest_improved"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(predictor.model, f)
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(predictor.scaler, f)
    
    import json
    metadata = {
        'model_type': 'random_forest_improved',
        'multi_output': True,
        'feature_names': predictor.feature_names.tolist() if hasattr(predictor.feature_names, 'tolist') else predictor.feature_names,
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt'
        },
        'performance': {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        }
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Improved model saved to {model_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Improve model performance")
    parser.add_argument('--exercise', type=str, required=True,
                       choices=['bicep_curls', 'squats', 'lateral_shoulder_raises', 
                               'triceps_pushdown', 'dumbbell_rows', 'dumbbell_shoulder_press'],
                       help='Exercise type')
    parser.add_argument('--type', type=str, default='camera',
                       choices=['camera', 'imu', 'fusion', 'all'],
                       help='Model type to improve')
    
    args = parser.parse_args()
    
    print(f"\n🔧 Model Improvement Script")
    print(f"   Exercise: {args.exercise}")
    print(f"   Type: {args.type}")
    
    if args.type in ['camera', 'all']:
        train_camera_model_improved(args.exercise)
    
    # TODO: Add improved IMU and fusion models
    print(f"\n✅ Improvement completed!")


if __name__ == "__main__":
    main()

