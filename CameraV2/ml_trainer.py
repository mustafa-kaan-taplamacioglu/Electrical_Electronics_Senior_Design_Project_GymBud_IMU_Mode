"""
ML Model Training System for Exercise Form Score Prediction
Trains models to predict form scores from pose features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import randint, uniform
import pickle
import json


class FormScorePredictor:
    """ML model for predicting exercise form scores (supports single-output and multi-output)."""
    
    # Regional output order (consistent across all models)
    REGIONAL_OUTPUTS = ['arms', 'legs', 'core', 'head']
    
    def __init__(self, model_type: str = "random_forest", multi_output: bool = True):
        """
        Initialize predictor.
        
        Args:
            model_type: "random_forest", "gradient_boosting", or "ridge"
            multi_output: If True, predict 4 regional scores (arms, legs, core, head). If False, predict single overall score.
        """
        self.model_type = model_type
        self.multi_output = multi_output
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
        # Base model selection
        if model_type == "random_forest":
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            base_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == "ridge":
            base_model = Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Wrap in MultiOutputRegressor if multi_output is True
        if multi_output:
            self.model = MultiOutputRegressor(base_model, n_jobs=-1)
        else:
            self.model = base_model
    
    def prepare_features(self, samples: List, use_imu_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix and labels from samples.
        
        Args:
            samples: List of RepSample objects
            use_imu_features: If True, use imu_features instead of features
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) for single-output or (n_samples, 4) for multi-output
        """
        # Extract features
        feature_vectors = []
        labels = []
        
        for sample in samples:
            # Choose which features to use
            if use_imu_features:
                if sample.imu_features is None:
                    continue
                features_dict = sample.imu_features
            else:
                if sample.features is None:
                    continue
                features_dict = sample.features
            
            # Use features as vector
            if self.feature_names is None:
                self.feature_names = sorted(features_dict.keys())
            
            # Create feature vector
            feature_vec = [features_dict.get(name, 0.0) for name in self.feature_names]
            feature_vectors.append(feature_vec)
            
            # Prepare labels based on multi_output mode
            if self.multi_output:
                # Multi-output: Use regional scores (4 outputs: arms, legs, core, head)
                if sample.regional_scores:
                    # Extract regional scores in consistent order
                    label_vec = [
                        sample.regional_scores.get(region, 0.0)
                        for region in self.REGIONAL_OUTPUTS
                    ]
                    labels.append(label_vec)
                elif sample.expert_score is not None:
                    # Fallback: Use expert_score for all regions (if no regional scores)
                    labels.append([sample.expert_score] * 4)
                elif sample.is_perfect_form is not None:
                    score = 100.0 if sample.is_perfect_form else 50.0
                    labels.append([score] * 4)
                else:
                    continue  # Skip samples without labels
            else:
                # Single-output: Use expert_score or regional score average
                if sample.expert_score is not None:
                    labels.append(sample.expert_score)
                elif sample.regional_scores:
                    avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
                    labels.append(avg_score)
                elif sample.is_perfect_form is not None:
                    labels.append(100.0 if sample.is_perfect_form else 50.0)
                else:
                    continue  # Skip samples without labels
        
        X = np.array(feature_vectors)
        y = np.array(labels)
        
        return X, y
    
    def train(self, samples: List, test_size: float = 0.2, verbose: bool = True, use_imu_features: bool = False):
        """
        Train the model on samples.
        
        Args:
            samples: List of RepSample objects with features and labels
            test_size: Fraction of data to use for testing
            verbose: Print training progress
            use_imu_features: If True, use imu_features instead of features
        """
        if verbose:
            print("=" * 60)
            print(f"Training {self.model_type} model ({'IMU' if use_imu_features else 'Camera'} features)")
            print("=" * 60)
        
        # Prepare data
        X, y = self.prepare_features(samples, use_imu_features=use_imu_features)
        
        if len(X) == 0:
            raise ValueError("No valid samples with features and labels")
        
        if verbose:
            print(f"\n📊 Dataset:")
            print(f"   Total samples: {len(X)}")
            print(f"   Features: {len(self.feature_names)}")
            if self.multi_output:
                print(f"   Outputs: {len(self.REGIONAL_OUTPUTS)} regional scores ({', '.join(self.REGIONAL_OUTPUTS)})")
                print(f"   Label range: {y.min():.1f} - {y.max():.1f} (per region)")
            else:
                print(f"   Output: Single overall score")
                print(f"   Label range: {y.min():.1f} - {y.max():.1f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if verbose:
            print(f"\n📦 Split:")
            print(f"   Train: {len(X_train)} samples")
            print(f"   Test: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if verbose:
            print(f"\n🚀 Training...")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        if self.multi_output:
            # Multi-output evaluation: Calculate metrics per region and average
            train_mse = mean_squared_error(y_train, y_train_pred, multioutput='uniform_average')
            test_mse = mean_squared_error(y_test, y_test_pred, multioutput='uniform_average')
            train_mae = mean_absolute_error(y_train, y_train_pred, multioutput='uniform_average')
            test_mae = mean_absolute_error(y_test, y_test_pred, multioutput='uniform_average')
            train_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
            test_r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')
            
            if verbose:
                print(f"\n📈 Results (Average across regions):")
                print(f"   Train MSE: {train_mse:.2f}")
                print(f"   Test MSE:  {test_mse:.2f}")
                print(f"   Train MAE: {train_mae:.2f}")
                print(f"   Test MAE:  {test_mae:.2f}")
                print(f"   Train R²:  {train_r2:.3f}")
                print(f"   Test R²:   {test_r2:.3f}")
                
                # Per-region breakdown
                print(f"\n📊 Per-Region Test MAE:")
                for i, region in enumerate(self.REGIONAL_OUTPUTS):
                    region_mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
                    region_r2 = r2_score(y_test[:, i], y_test_pred[:, i])
                    print(f"   {region:8s}: MAE={region_mae:.2f}, R²={region_r2:.3f}")
        else:
            # Single-output evaluation
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            if verbose:
                print(f"\n📈 Results:")
                print(f"   Train MSE: {train_mse:.2f}")
                print(f"   Test MSE:  {test_mse:.2f}")
                print(f"   Train MAE: {train_mae:.2f}")
                print(f"   Test MAE:  {test_mae:.2f}")
                print(f"   Train R²:  {train_r2:.3f}")
                print(f"   Test R²:   {test_r2:.3f}")
        
        self.is_trained = True
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def tune_hyperparameters(self, samples: List, cv: int = 5, method: str = "random", n_iter: int = 50, verbose: bool = True, use_imu_features: bool = False):
        """
        Tune hyperparameters using GridSearch or RandomizedSearch.
        
        Args:
            samples: List of RepSample objects
            cv: Number of cross-validation folds
            method: "grid" or "random" (default: "random" - faster)
            n_iter: Number of iterations for RandomizedSearch
            verbose: Print tuning progress
            use_imu_features: If True, use imu_features instead of features
            
        Returns:
            Best hyperparameters dict
        """
        if verbose:
            print("=" * 60)
            print(f"Hyperparameter Tuning ({method}) - {'IMU' if use_imu_features else 'Camera'} features")
            print("=" * 60)
        
        # Prepare data
        X, y = self.prepare_features(samples, use_imu_features=use_imu_features)
        
        if len(X) < cv:
            print(f"⚠️  Not enough samples ({len(X)}) for {cv}-fold CV. Skipping tuning.")
            return None
        
        # Define parameter grids
        if self.model_type == "random_forest":
            if method == "grid":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                search = GridSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=cv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=1 if verbose else 0
                )
            else:  # random
                param_dist = {
                    'n_estimators': randint(50, 300),
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10)
                }
                search = RandomizedSearchCV(
                    RandomForestRegressor(random_state=42, n_jobs=-1),
                    param_dist,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1 if verbose else 0
                )
        elif self.model_type == "gradient_boosting":
            if method == "grid":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'min_samples_split': [2, 5, 10]
                }
                search = GridSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    param_grid,
                    cv=cv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=1 if verbose else 0
                )
            else:  # random
                param_dist = {
                    'n_estimators': randint(50, 200),
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': uniform(0.01, 0.2),
                    'min_samples_split': randint(2, 20)
                }
                search = RandomizedSearchCV(
                    GradientBoostingRegressor(random_state=42),
                    param_dist,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1 if verbose else 0
                )
        elif self.model_type == "ridge":
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
            }
            search = GridSearchCV(
                Ridge(),
                param_grid,
                cv=cv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1 if verbose else 0
            )
        else:
            raise ValueError(f"Hyperparameter tuning not supported for {self.model_type}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Run search
        if verbose:
            print(f"\n🔍 Searching best hyperparameters...")
        search.fit(X_scaled, y)
        
        # Update model with best parameters
        # search.best_estimator_ is already wrapped in MultiOutputRegressor if multi_output=True
        self.model = search.best_estimator_
        
        best_params = search.best_params_
        
        if verbose:
            print(f"\n✅ Best hyperparameters:")
            for param, value in best_params.items():
                print(f"   {param}: {value}")
            print(f"   Best CV MAE: {-search.best_score_:.2f}")
        
        return best_params
    
    def train_with_cv(self, samples: List, cv: int = 5, test_size: float = 0.2, verbose: bool = True, use_imu_features: bool = False):
        """
        Train model with cross-validation evaluation.
        
        Args:
            samples: List of RepSample objects
            cv: Number of cross-validation folds
            test_size: Fraction of data to use for final test
            verbose: Print training progress
            use_imu_features: If True, use imu_features instead of features
            
        Returns:
            Dict with training results including CV scores
        """
        if verbose:
            print("=" * 60)
            print(f"Training {self.model_type} model with {cv}-fold CV ({'IMU' if use_imu_features else 'Camera'} features)")
            print("=" * 60)
        
        # Prepare data
        X, y = self.prepare_features(samples, use_imu_features=use_imu_features)
        
        if len(X) < cv:
            print(f"⚠️  Not enough samples ({len(X)}) for {cv}-fold CV. Using train/test split only.")
            return self.train(samples, test_size=test_size, verbose=verbose, use_imu_features=use_imu_features)
        
        # Split data (final test set)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Cross-validation on training set
        if verbose:
            print(f"\n📊 Cross-Validation ({cv}-fold)...")
        
        cv_scores_mae = -cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        cv_scores_r2 = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=cv, scoring='r2', n_jobs=-1
        )
        
        if verbose:
            print(f"   CV MAE: {cv_scores_mae.mean():.2f} (±{cv_scores_mae.std():.2f})")
            print(f"   CV R²:  {cv_scores_r2.mean():.3f} (±{cv_scores_r2.std():.3f})")
        
        # Train on full training set
        if verbose:
            print(f"\n🚀 Training on full training set...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on test set
        y_test_pred = self.model.predict(X_test_scaled)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        if verbose:
            print(f"\n📈 Test Set Results:")
            print(f"   Test MSE: {test_mse:.2f}")
            print(f"   Test MAE: {test_mae:.2f}")
            print(f"   Test R²:  {test_r2:.3f}")
        
        return {
            'cv_mae_mean': float(cv_scores_mae.mean()),
            'cv_mae_std': float(cv_scores_mae.std()),
            'cv_r2_mean': float(cv_scores_r2.mean()),
            'cv_r2_std': float(cv_scores_r2.std()),
            'test_mse': float(test_mse),
            'test_mae': float(test_mae),
            'test_r2': float(test_r2)
        }
    
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict form scores from features.
        
        Args:
            features: Dictionary of feature_name -> value
            
        Returns:
            If multi_output: Dict with regional scores {"arms": float, "legs": float, "core": float, "head": float}
            If single_output: Dict with overall score {"score": float}
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create feature vector
        feature_vec = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ]).reshape(1, -1)
        
        # Scale
        feature_vec_scaled = self.scaler.transform(feature_vec)
        
        # Predict
        prediction = self.model.predict(feature_vec_scaled)[0]
        
        if self.multi_output:
            # Return regional scores as dictionary
            result = {
                region: float(np.clip(prediction[i], 0, 100))
                for i, region in enumerate(self.REGIONAL_OUTPUTS)
            }
            return result
        else:
            # Return overall score
            return {"score": float(np.clip(prediction, 0, 100))}
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # For Ridge, use coefficient magnitudes
            importances = np.abs(self.model.coef_)
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        return {
            self.feature_names[i]: float(importances[i])
            for i in indices
        }
    
    def save(self, path: str, exercise: str = None, training_samples: int = None, performance_metrics: dict = None):
        """Save model to disk with extended metadata.
        
        Args:
            path: Directory path to save model
            exercise: Exercise name (e.g., "bicep_curls")
            training_samples: Number of samples used for training
            performance_metrics: Dict with train/test metrics (MSE, MAE, R²)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata with extended information
        from datetime import datetime
        metadata = {
            'model_type': self.model_type,
            'multi_output': self.multi_output,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'is_trained': self.is_trained,
            'training_date': datetime.now().isoformat(),
            'exercise': exercise,
            'training_samples': training_samples,
            'performance': performance_metrics or {}
        }
        if self.multi_output:
            metadata['regional_outputs'] = self.REGIONAL_OUTPUTS
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "FormScorePredictor":
        """Load model from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create instance (multi_output defaults to True if not in metadata for backward compatibility)
        multi_output = metadata.get('multi_output', True)
        predictor = cls(model_type=metadata['model_type'], multi_output=multi_output)
        predictor.feature_names = metadata['feature_names']
        predictor.is_trained = metadata['is_trained']
        
        # Load model
        with open(path / "model.pkl", "rb") as f:
            predictor.model = pickle.load(f)
        
        # Load scaler
        with open(path / "scaler.pkl", "rb") as f:
            predictor.scaler = pickle.load(f)
        
        print(f"📂 Model loaded from {path}")
        return predictor


class BaselineCalculator:
    """Calculate baseline values from perfect form samples."""
    
    # Regional output order (must match FormScorePredictor.REGIONAL_OUTPUTS)
    REGIONAL_OUTPUTS = ['arms', 'legs', 'core', 'head']
    
    @staticmethod
    def calculate_baselines(
        perfect_samples: List,
        tolerance_percentile: float = 95.0,
        regional: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate baseline thresholds from perfect form samples.
        
        Args:
            perfect_samples: List of RepSample objects marked as perfect
            tolerance_percentile: Percentile to use for tolerance (e.g., 95 = use 95th percentile as max)
            regional: If True, also calculate regional baselines (separate for each region)
        
        Returns:
            Dict with baseline values for each feature/angle (and regional scores if regional=True)
        """
        if len(perfect_samples) == 0:
            return {}
        
        baselines = {}
        
        # Calculate regional score baselines if requested
        if regional:
            regional_scores = {region: [] for region in BaselineCalculator.REGIONAL_OUTPUTS}
            for sample in perfect_samples:
                if sample.regional_scores:
                    for region in BaselineCalculator.REGIONAL_OUTPUTS:
                        if region in sample.regional_scores:
                            regional_scores[region].append(sample.regional_scores[region])
            
            # Add regional score baselines
            for region, scores in regional_scores.items():
                if len(scores) > 0:
                    scores_array = np.array(scores)
                    baselines[f'regional_score_{region}'] = {
                        'mean': float(np.mean(scores_array)),
                        'std': float(np.std(scores_array)),
                        'min': float(np.percentile(scores_array, 100 - tolerance_percentile)),
                        'max': float(np.percentile(scores_array, tolerance_percentile)),
                        'median': float(np.median(scores_array))
                    }
        
        # Extract all features
        all_features = {}
        for sample in perfect_samples:
            if sample.features:
                for feature_name, value in sample.features.items():
                    if feature_name not in all_features:
                        all_features[feature_name] = []
                    all_features[feature_name].append(value)
        
        # Calculate statistics for each feature
        for feature_name, values in all_features.items():
            values_array = np.array(values)
            
            baselines[feature_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.percentile(values_array, 100 - tolerance_percentile)),
                'max': float(np.percentile(values_array, tolerance_percentile)),
                'median': float(np.median(values_array))
            }
        
        # Calculate angle baselines
        angles = {}
        for sample in perfect_samples:
            if sample.min_angle is not None:
                if 'min_angle' not in angles:
                    angles['min_angle'] = []
                angles['min_angle'].append(sample.min_angle)
            
            if sample.max_angle is not None:
                if 'max_angle' not in angles:
                    angles['max_angle'] = []
                angles['max_angle'].append(sample.max_angle)
            
            if sample.range_of_motion is not None:
                if 'range_of_motion' not in angles:
                    angles['range_of_motion'] = []
                angles['range_of_motion'].append(sample.range_of_motion)
        
        for angle_name, values in angles.items():
            values_array = np.array(values)
            baselines[angle_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.percentile(values_array, 100 - tolerance_percentile)),
                'max': float(np.percentile(values_array, tolerance_percentile))
            }
        
        return baselines


if __name__ == "__main__":
    from dataset_collector import DatasetCollector
    
    # Example usage
    collector = DatasetCollector("dataset")
    
    # Load dataset
    samples = collector.load_dataset()
    
    if len(samples) > 0:
        # Train model
        predictor = FormScorePredictor(model_type="random_forest")
        results = predictor.train(samples, verbose=True)
        
        # Save model
        predictor.save("models/form_score_predictor")
        
        # Calculate baselines
        perfect_samples = [s for s in samples if s.is_perfect_form == True]
        if perfect_samples:
            baselines = BaselineCalculator.calculate_baselines(perfect_samples)
            print("\n📊 Baselines calculated:")
            for key, value in list(baselines.items())[:5]:
                print(f"   {key}: {value}")
    else:
        print("⚠️  No samples found. Collect data first!")

