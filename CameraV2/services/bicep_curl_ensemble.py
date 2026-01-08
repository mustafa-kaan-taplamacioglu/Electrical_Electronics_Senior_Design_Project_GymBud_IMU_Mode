"""
Bicep Curl Ensemble Model
=========================
Creates an ensemble average model from training data for:
1. Form score prediction based on IMU patterns
2. Speed classification (5 levels: very_fast, fast, medium, slow, very_slow)
3. Motion pattern validation

Based on 3 training datasets:
- bicep_curls_20260103_230109 (medium-fast)
- bicep_curls_20260103_230629 (fast)
- bicep_curls_20260103_230946 (slow)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Training data paths
TRAINING_DATA_PATH = Path(__file__).parent.parent / "MLTRAINIMU" / "bicep_curls"
TRAINING_FOLDERS = [
    "bicep_curls_20260103_230109",  # medium-fast (~2.5s per rep)
    "bicep_curls_20260103_230629",  # fast (~1.0s per rep)
    "bicep_curls_20260103_230946",  # slow (~2.0s per rep)
]

# Speed classification thresholds (in seconds per rep)
SPEED_THRESHOLDS = {
    'very_fast': (0.0, 1.2),      # < 1.2s
    'fast': (1.2, 1.8),            # 1.2-1.8s
    'medium': (1.8, 2.5),          # 1.8-2.5s
    'slow': (2.5, 3.5),            # 2.5-3.5s
    'very_slow': (3.5, float('inf'))  # > 3.5s
}

# Speed feedback in Turkish
SPEED_FEEDBACK_TR = {
    'very_fast': {
        'label': 'Çok Hızlı',
        'emoji': '🚀',
        'feedback': 'Çok hızlı yapıyorsun! Biraz yavaşlatarak kasları daha iyi hissedebilirsin.'
    },
    'fast': {
        'label': 'Hızlı',
        'emoji': '⚡',
        'feedback': 'Hızlı tempo, ama dikkat et - formu koruyarak devam et.'
    },
    'medium': {
        'label': 'Orta Hız',
        'emoji': '✅',
        'feedback': 'İdeal tempo! Bu hızda devam et.'
    },
    'slow': {
        'label': 'Yavaş',
        'emoji': '🐢',
        'feedback': 'Yavaş ve kontrollü - kasları iyi hissediyorsun.'
    },
    'very_slow': {
        'label': 'Çok Yavaş',
        'emoji': '🦥',
        'feedback': 'Çok yavaş yapıyorsun. Biraz hızlandırabilirsin.'
    }
}


class BicepCurlEnsembleModel:
    """Ensemble model for bicep curl analysis."""
    
    def __init__(self):
        """Initialize the ensemble model."""
        self.is_loaded = False
        self.reference_patterns: Dict[str, Dict] = {}
        self.speed_stats: Dict[str, float] = {}
        self.pitch_stats: Dict[str, Dict] = {}
        self.gyro_stats: Dict[str, Dict] = {}
        
        # Ensemble weights (based on data quality)
        self.ensemble_weights = {
            "bicep_curls_20260103_230109": 0.35,  # Medium-fast, good quality
            "bicep_curls_20260103_230629": 0.30,  # Fast, good for speed reference
            "bicep_curls_20260103_230946": 0.35,  # Slow, detailed motion
        }
        
        self._load_training_data()
    
    def _load_training_data(self):
        """Load and analyze training data from CSV files."""
        all_rep_durations = []
        all_pitch_ranges = []
        all_gyro_magnitudes = []
        all_samples_per_rep = []
        
        for folder_name in TRAINING_FOLDERS:
            csv_path = TRAINING_DATA_PATH / folder_name / "imu_samples.csv"
            summary_path = TRAINING_DATA_PATH / folder_name / "summary.csv"
            
            if not csv_path.exists():
                print(f"⚠️  Training data not found: {csv_path}")
                continue
            
            try:
                # Load IMU samples
                df = pd.read_csv(csv_path)
                
                # Load summary for rep counts
                if summary_path.exists():
                    summary_df = pd.read_csv(summary_path)
                    # Calculate rep durations from summary
                    rep_timestamps = summary_df[summary_df['rep_number'] > 0]['timestamp'].values
                    if len(rep_timestamps) > 1:
                        rep_durations = np.diff(rep_timestamps)
                        all_rep_durations.extend(rep_durations)
                
                # Analyze each rep in the data
                for rep_num in df['rep_number'].unique():
                    if rep_num <= 0:
                        continue
                    
                    rep_data = df[df['rep_number'] == rep_num]
                    
                    if len(rep_data) < 5:
                        continue
                    
                    # Calculate pitch range for this rep
                    left_wrist_data = rep_data[rep_data['node_name'] == 'left_wrist']
                    right_wrist_data = rep_data[rep_data['node_name'] == 'right_wrist']
                    
                    if len(left_wrist_data) > 0:
                        pitch_range = left_wrist_data['pitch'].max() - left_wrist_data['pitch'].min()
                        all_pitch_ranges.append(pitch_range)
                        
                        # Calculate gyro magnitude
                        gyro_mag = np.sqrt(
                            left_wrist_data['gx']**2 + 
                            left_wrist_data['gy']**2 + 
                            left_wrist_data['gz']**2
                        ).mean()
                        all_gyro_magnitudes.append(gyro_mag)
                    
                    # Samples per rep
                    all_samples_per_rep.append(len(rep_data) // 2)  # Divide by 2 for 2 nodes
                
                print(f"✅ Loaded training data: {folder_name}")
                
            except Exception as e:
                print(f"⚠️  Error loading {folder_name}: {e}")
        
        # Calculate ensemble statistics
        if all_rep_durations:
            self.speed_stats = {
                'mean_duration': float(np.mean(all_rep_durations)),
                'std_duration': float(np.std(all_rep_durations)),
                'min_duration': float(np.min(all_rep_durations)),
                'max_duration': float(np.max(all_rep_durations)),
                'median_duration': float(np.median(all_rep_durations))
            }
        
        if all_pitch_ranges:
            self.pitch_stats = {
                'mean_range': float(np.mean(all_pitch_ranges)),
                'std_range': float(np.std(all_pitch_ranges)),
                'min_range': float(np.min(all_pitch_ranges)),
                'max_range': float(np.max(all_pitch_ranges)),
                'ideal_range': float(np.percentile(all_pitch_ranges, 75))  # Top 25% as ideal
            }
        
        if all_gyro_magnitudes:
            self.gyro_stats = {
                'mean_magnitude': float(np.mean(all_gyro_magnitudes)),
                'std_magnitude': float(np.std(all_gyro_magnitudes)),
                'max_magnitude': float(np.max(all_gyro_magnitudes))
            }
        
        if all_samples_per_rep:
            self.reference_patterns = {
                'mean_samples': float(np.mean(all_samples_per_rep)),
                'std_samples': float(np.std(all_samples_per_rep))
            }
        
        self.is_loaded = bool(all_rep_durations)
        
        if self.is_loaded:
            print(f"📊 Ensemble model loaded with {len(all_rep_durations)} rep durations")
            print(f"   Speed stats: mean={self.speed_stats.get('mean_duration', 0):.2f}s, "
                  f"median={self.speed_stats.get('median_duration', 0):.2f}s")
            print(f"   Pitch range: mean={self.pitch_stats.get('mean_range', 0):.1f}°, "
                  f"ideal={self.pitch_stats.get('ideal_range', 0):.1f}°")
    
    def classify_speed(self, rep_duration: float) -> Dict:
        """
        Classify rep speed into 5 categories.
        
        Args:
            rep_duration: Duration of the rep in seconds
            
        Returns:
            Dict with speed classification and feedback
        """
        for speed_class, (min_dur, max_dur) in SPEED_THRESHOLDS.items():
            if min_dur <= rep_duration < max_dur:
                return {
                    'class': speed_class,
                    'duration': rep_duration,
                    **SPEED_FEEDBACK_TR[speed_class]
                }
        
        # Fallback
        return {
            'class': 'medium',
            'duration': rep_duration,
            **SPEED_FEEDBACK_TR['medium']
        }
    
    def calculate_form_score(
        self,
        pitch_range: float,
        gyro_magnitude: float,
        rep_duration: float,
        samples_count: int
    ) -> Tuple[float, List[str]]:
        """
        Calculate form score using ensemble model.
        
        Args:
            pitch_range: Pitch angle range during rep (degrees)
            gyro_magnitude: Average gyroscope magnitude during rep
            rep_duration: Duration of rep in seconds
            samples_count: Number of IMU samples in rep
            
        Returns:
            Tuple of (form_score 0-100, list of issues)
        """
        issues = []
        score_components = []
        
        # 1. Pitch Range Score (40% weight)
        # Ideal range is around 120-150 degrees for full bicep curl
        if self.pitch_stats:
            ideal_range = self.pitch_stats.get('ideal_range', 130)
            mean_range = self.pitch_stats.get('mean_range', 110)
            
            if pitch_range >= ideal_range:
                pitch_score = 100.0
            elif pitch_range >= mean_range:
                # Scale from mean to ideal
                pitch_score = 70 + 30 * (pitch_range - mean_range) / (ideal_range - mean_range)
            elif pitch_range >= mean_range * 0.6:
                # Below mean but acceptable - daha düşük minimum
                pitch_score = 40 + 20 * (pitch_range - mean_range * 0.6) / (mean_range * 0.4)
                issues.append("Hareket açısı biraz dar")
            else:
                # Çok daha düşük minimum: 30 → 15-25
                pitch_score = max(15, 20 + 10 * pitch_range / (mean_range * 0.6))
                issues.append("Hareket açısı çok dar - kolu tam kaldır")
            
            score_components.append(('pitch', pitch_score, 0.40))
        else:
            # Fallback without training data
            if pitch_range >= 120:
                pitch_score = 100.0
            elif pitch_range >= 90:
                pitch_score = 70 + 30 * (pitch_range - 90) / 30
            else:
                # Daha düşük minimum: 30 → 15-25
                pitch_score = max(15, pitch_range / 90 * 60)
                issues.append("Hareket açısı dar")
            score_components.append(('pitch', pitch_score, 0.40))
        
        # 2. Speed/Tempo Score (25% weight) - Daha sert penalty'ler
        speed_class = self.classify_speed(rep_duration)['class']
        if speed_class == 'medium':
            speed_score = 100.0
        elif speed_class in ['fast', 'slow']:
            speed_score = 80.0  # 85 → 80
        elif speed_class == 'very_fast':
            speed_score = 45.0  # 65 → 45 (çok hızlı = kötü form)
            issues.append("Çok hızlı - yavaşla")
        else:  # very_slow
            speed_score = 55.0  # 70 → 55 (çok yavaş = kötü form)
            issues.append("Çok yavaş")
        score_components.append(('speed', speed_score, 0.25))
        
        # 3. Movement Quality Score (35% weight)
        # Based on gyro smoothness and sample consistency
        if self.gyro_stats and gyro_magnitude > 0:
            mean_gyro = self.gyro_stats.get('mean_magnitude', 150)
            max_gyro = self.gyro_stats.get('max_magnitude', 400)
            
            # Ideal is around mean, too high or too low is bad - Daha sert penalty'ler
            if abs(gyro_magnitude - mean_gyro) < mean_gyro * 0.3:
                quality_score = 100.0
            elif gyro_magnitude > max_gyro * 0.9:
                quality_score = 40.0  # 60 → 40 (çok sert = kötü form)
                issues.append("Hareket çok sert")
            elif gyro_magnitude < mean_gyro * 0.3:
                quality_score = 50.0  # 70 → 50 (çok yavaş = kötü form)
                issues.append("Hareket çok yavaş/durgun")
            else:
                # Scale based on distance from ideal - Daha düşük minimum
                deviation = abs(gyro_magnitude - mean_gyro) / mean_gyro
                quality_score = max(45, 100 - deviation * 50)  # Minimum 60 → 45
        else:
            quality_score = 75.0  # Default without reference
        score_components.append(('quality', quality_score, 0.35))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        # Critical issues için ekstra penalty
        if any('çok' in issue.lower() or 'fazla' in issue.lower() for issue in issues):
            total_score *= 0.85  # %15 penalty
        
        # Çok kötü form için cap
        if pitch_range < mean_range * 0.5 if self.pitch_stats else pitch_range < 60:  # Çok dar ROM
            total_score = min(total_score, 50)  # Max 50'e düşür
        
        if rep_duration < 0.8 or rep_duration > 5.0:  # Çok anormal tempo
            total_score = min(total_score, 55)  # Max 55'e düşür
        
        return max(0, min(100, total_score)), issues
    
    def analyze_wrist_form(
        self,
        lw_pitch_range: float,
        rw_pitch_range: float,
        lw_gyro_mag: float,
        rw_gyro_mag: float,
        sync_diff: float
    ) -> Dict:
        """
        Analyze individual wrist form for bicep curls.
        
        Args:
            lw_pitch_range: Left wrist pitch range (degrees)
            rw_pitch_range: Right wrist pitch range (degrees)
            lw_gyro_mag: Left wrist gyro magnitude
            rw_gyro_mag: Right wrist gyro magnitude
            sync_diff: Synchronization difference between wrists (degrees)
            
        Returns:
            Dict with left_wrist, right_wrist scores and feedback
        """
        ideal_range = self.pitch_stats.get('ideal_range', 130) if self.pitch_stats else 130
        mean_range = self.pitch_stats.get('mean_range', 110) if self.pitch_stats else 110
        
        def calculate_wrist_score(pitch_range: float, gyro_mag: float, wrist_name: str) -> Dict:
            """Calculate form score for a single wrist."""
            issues = []
            
            # Pitch range score (60%)
            if pitch_range >= ideal_range:
                pitch_score = 100.0
            elif pitch_range >= mean_range:
                pitch_score = 70 + 30 * (pitch_range - mean_range) / max(1, ideal_range - mean_range)
            elif pitch_range >= mean_range * 0.6:
                pitch_score = 50 + 20 * (pitch_range - mean_range * 0.6) / max(1, mean_range * 0.4)
                issues.append(f"{wrist_name} hareket açısı dar")
            else:
                pitch_score = max(20, 30 + 20 * pitch_range / max(1, mean_range * 0.6))
                issues.append(f"{wrist_name} hareket açısı çok dar")
            
            # Movement quality score (40%)
            mean_gyro = self.gyro_stats.get('mean_magnitude', 150) if self.gyro_stats else 150
            if gyro_mag > 0:
                if abs(gyro_mag - mean_gyro) < mean_gyro * 0.3:
                    quality_score = 100.0
                elif gyro_mag > mean_gyro * 1.5:
                    quality_score = 60.0
                    issues.append(f"{wrist_name} hareket çok sert")
                elif gyro_mag < mean_gyro * 0.3:
                    quality_score = 70.0
                    issues.append(f"{wrist_name} hareket yavaş")
                else:
                    deviation = abs(gyro_mag - mean_gyro) / max(1, mean_gyro)
                    quality_score = max(50, 100 - deviation * 50)
            else:
                quality_score = 75.0
            
            total_score = 0.6 * pitch_score + 0.4 * quality_score
            
            # Generate feedback
            if total_score >= 85:
                feedback = f"✅ {wrist_name}: Mükemmel form!"
            elif total_score >= 70:
                feedback = f"👍 {wrist_name}: İyi, devam et."
            elif total_score >= 50:
                feedback = f"⚠️ {wrist_name}: Geliştir - {issues[0] if issues else 'form düşük'}"
            else:
                feedback = f"❌ {wrist_name}: Dikkat! {issues[0] if issues else 'form çok düşük'}"
            
            return {
                'score': round(total_score, 1),
                'pitch_range': round(pitch_range, 1),
                'gyro_magnitude': round(gyro_mag, 1),
                'issues': issues,
                'feedback': feedback
            }
        
        lw_result = calculate_wrist_score(lw_pitch_range, lw_gyro_mag, "Sol Bilek")
        rw_result = calculate_wrist_score(rw_pitch_range, rw_gyro_mag, "Sağ Bilek")
        
        # Synchronization analysis
        sync_issues = []
        if sync_diff > 50:
            sync_score = max(20, 100 - (sync_diff - 50) * 2)
            sync_issues.append(f"Bilekler senkron değil (fark: {sync_diff:.0f}°)")
        elif sync_diff > 30:
            sync_score = 70 + (50 - sync_diff) * 1.5
        else:
            sync_score = 100.0
        
        # Overall arms score (average of both wrists + sync bonus)
        arms_score = (lw_result['score'] + rw_result['score'] + sync_score) / 3
        
        return {
            'left_wrist': lw_result,
            'right_wrist': rw_result,
            'sync_score': round(sync_score, 1),
            'sync_diff': round(sync_diff, 1),
            'sync_issues': sync_issues,
            'arms_score': round(arms_score, 1),
            'regional_scores': {
                'arms': round(arms_score, 1),
                'legs': 100.0,  # Not relevant for bicep curls
                'core': 85.0,   # Default for bicep curls (assume good if arms are good)
                'head': 90.0    # Default for bicep curls
            }
        }
    
    def get_rep_analysis(
        self,
        pitch_range: float,
        gyro_magnitude: float,
        rep_duration: float,
        samples_count: int,
        lw_pitch_range: float = None,
        rw_pitch_range: float = None,
        lw_gyro_mag: float = None,
        rw_gyro_mag: float = None,
        sync_diff: float = 0.0
    ) -> Dict:
        """
        Get comprehensive analysis of a rep including LW/RW specific feedback.
        
        Returns dict with form_score, speed_class, feedback, regional scores, and issues.
        """
        form_score, issues = self.calculate_form_score(
            pitch_range, gyro_magnitude, rep_duration, samples_count
        )
        
        speed_info = self.classify_speed(rep_duration)
        
        # Analyze individual wrists if data is available
        wrist_analysis = None
        regional_scores = {'arms': form_score, 'legs': 100.0, 'core': 85.0, 'head': 90.0}
        
        if lw_pitch_range is not None and rw_pitch_range is not None:
            wrist_analysis = self.analyze_wrist_form(
                lw_pitch_range or pitch_range / 2,
                rw_pitch_range or pitch_range / 2,
                lw_gyro_mag or gyro_magnitude / 2,
                rw_gyro_mag or gyro_magnitude / 2,
                sync_diff
            )
            regional_scores = wrist_analysis['regional_scores']
            
            # Add wrist-specific issues
            if wrist_analysis['left_wrist']['issues']:
                issues.extend(wrist_analysis['left_wrist']['issues'])
            if wrist_analysis['right_wrist']['issues']:
                issues.extend(wrist_analysis['right_wrist']['issues'])
            if wrist_analysis['sync_issues']:
                issues.extend(wrist_analysis['sync_issues'])
        
        # Generate combined feedback
        if form_score >= 85:
            form_feedback = "Mükemmel form! 💪"
        elif form_score >= 70:
            form_feedback = "İyi form, ufak düzeltmeler yapılabilir."
        elif form_score >= 50:
            form_feedback = "Form orta düzeyde, iyileştirmeye çalış."
        else:
            form_feedback = "Form düşük, tekniğe odaklan."
        
        result = {
            'form_score': round(form_score, 1),
            'speed_class': speed_info['class'],
            'speed_label': speed_info['label'],
            'speed_emoji': speed_info['emoji'],
            'speed_feedback': speed_info['feedback'],
            'duration': round(rep_duration, 2),
            'form_feedback': form_feedback,
            'issues': issues,
            'pitch_range': round(pitch_range, 1),
            'gyro_magnitude': round(gyro_magnitude, 1),
            'regional_scores': regional_scores
        }
        
        # Add LW/RW pitch ranges to result (important for ROM feedback)
        if lw_pitch_range is not None:
            result['lw_pitch_range'] = round(lw_pitch_range, 1)
        if rw_pitch_range is not None:
            result['rw_pitch_range'] = round(rw_pitch_range, 1)
        
        if wrist_analysis:
            result['left_wrist'] = wrist_analysis['left_wrist']
            result['right_wrist'] = wrist_analysis['right_wrist']
            result['sync_score'] = wrist_analysis['sync_score']
            result['sync_diff'] = wrist_analysis['sync_diff']
        
        return result


# Global instance
_ensemble_model: Optional[BicepCurlEnsembleModel] = None


def get_ensemble_model() -> BicepCurlEnsembleModel:
    """Get or create the singleton ensemble model instance."""
    global _ensemble_model
    if _ensemble_model is None:
        _ensemble_model = BicepCurlEnsembleModel()
    return _ensemble_model


def analyze_bicep_curl_rep(
    pitch_range: float,
    gyro_magnitude: float,
    rep_duration: float,
    samples_count: int = 0,
    lw_pitch_range: float = None,
    rw_pitch_range: float = None,
    lw_gyro_mag: float = None,
    rw_gyro_mag: float = None,
    sync_diff: float = 0.0
) -> Dict:
    """
    Convenience function to analyze a bicep curl rep with LW/RW details.
    
    Args:
        pitch_range: Pitch angle range during rep (degrees)
        gyro_magnitude: Average gyroscope magnitude during rep
        rep_duration: Duration of rep in seconds
        samples_count: Number of IMU samples in rep
        lw_pitch_range: Left wrist pitch range (optional)
        rw_pitch_range: Right wrist pitch range (optional)
        lw_gyro_mag: Left wrist gyro magnitude (optional)
        rw_gyro_mag: Right wrist gyro magnitude (optional)
        sync_diff: Wrist synchronization difference (optional)
        
    Returns:
        Analysis dict with form_score, speed_class, feedback, regional scores, etc.
    """
    model = get_ensemble_model()
    return model.get_rep_analysis(
        pitch_range, gyro_magnitude, rep_duration, samples_count,
        lw_pitch_range, rw_pitch_range, lw_gyro_mag, rw_gyro_mag, sync_diff
    )


def classify_rep_speed(rep_duration: float) -> Dict:
    """Classify rep speed into 5 categories."""
    model = get_ensemble_model()
    return model.classify_speed(rep_duration)


def calculate_wrist_scores(
    lw_pitch_range: float,
    rw_pitch_range: float,
    lw_avg_pitch: float = 0,
    rw_avg_pitch: float = 0,
    lw_roll_range: float = 0,
    rw_roll_range: float = 0,
    sync_diff: float = 0
) -> Dict:
    """
    Calculate individual wrist form scores based on orientation data.
    
    Args:
        lw_pitch_range: Left wrist pitch range (degrees)
        rw_pitch_range: Right wrist pitch range (degrees)
        lw_avg_pitch: Left wrist average pitch
        rw_avg_pitch: Right wrist average pitch
        lw_roll_range: Left wrist roll range
        rw_roll_range: Right wrist roll range
        sync_diff: Synchronization difference between wrists
        
    Returns:
        Dict with LW and RW scores and feedback
    """
    # Ideal pitch range for bicep curl: 120-150 degrees
    ideal_pitch_range = 130.0
    min_acceptable_range = 80.0
    
    # Calculate LW score
    lw_issues = []
    if lw_pitch_range >= ideal_pitch_range:
        lw_pitch_score = 100.0
    elif lw_pitch_range >= min_acceptable_range:
        lw_pitch_score = 60 + 40 * (lw_pitch_range - min_acceptable_range) / (ideal_pitch_range - min_acceptable_range)
    elif lw_pitch_range >= 50:
        lw_pitch_score = 30 + 30 * (lw_pitch_range - 50) / 30
        lw_issues.append("Sol kol hareket açısı dar")
    else:
        lw_pitch_score = max(10, lw_pitch_range / 50 * 30)
        lw_issues.append("Sol kol hareketi çok kısıtlı")
    
    # Roll range penalty (should be minimal for proper bicep curl)
    if lw_roll_range > 45:
        lw_pitch_score -= 10
        lw_issues.append("Sol bilek fazla döndü")
    
    # Calculate RW score
    rw_issues = []
    if rw_pitch_range >= ideal_pitch_range:
        rw_pitch_score = 100.0
    elif rw_pitch_range >= min_acceptable_range:
        rw_pitch_score = 60 + 40 * (rw_pitch_range - min_acceptable_range) / (ideal_pitch_range - min_acceptable_range)
    elif rw_pitch_range >= 50:
        rw_pitch_score = 30 + 30 * (rw_pitch_range - 50) / 30
        rw_issues.append("Sağ kol hareket açısı dar")
    else:
        rw_pitch_score = max(10, rw_pitch_range / 50 * 30)
        rw_issues.append("Sağ kol hareketi çok kısıtlı")
    
    # Roll range penalty
    if rw_roll_range > 45:
        rw_pitch_score -= 10
        rw_issues.append("Sağ bilek fazla döndü")
    
    # Synchronization score
    sync_score = 100.0
    sync_issues = []
    if sync_diff > 50:
        sync_score = max(30, 100 - (sync_diff - 50) * 1.5)
        sync_issues.append("Kollar senkronize değil")
    elif sync_diff > 30:
        sync_score = 80 - (sync_diff - 30) * 0.5
        sync_issues.append("Kollar arası hafif fark var")
    
    # Final scores
    lw_final = max(0, min(100, lw_pitch_score))
    rw_final = max(0, min(100, rw_pitch_score))
    
    # Generate feedback based on comparison
    comparison_feedback = ""
    arms_feedback = ""
    
    # Generate arms feedback with left vs right comparison
    if abs(lw_pitch_range - rw_pitch_range) > 30:
        if lw_pitch_range > rw_pitch_range:
            arms_feedback = f"Sol kol daha geniş hareket ediyor ({lw_pitch_range:.0f}° vs {rw_pitch_range:.0f}°)"
        else:
            arms_feedback = f"Sağ kol daha geniş hareket ediyor ({rw_pitch_range:.0f}° vs {lw_pitch_range:.0f}°)"
    elif abs(lw_pitch_range - rw_pitch_range) > 15:
        if lw_pitch_range > rw_pitch_range:
            arms_feedback = f"Sol kol biraz daha geniş ({lw_pitch_range:.0f}° vs {rw_pitch_range:.0f}°)"
        else:
            arms_feedback = f"Sağ kol biraz daha geniş ({rw_pitch_range:.0f}° vs {lw_pitch_range:.0f}°)"
    else:
        arms_feedback = f"Her iki kol dengeli ({lw_pitch_range:.0f}° vs {rw_pitch_range:.0f}°)"
    
    if abs(lw_pitch_range - rw_pitch_range) > 30:
        if lw_pitch_range > rw_pitch_range:
            comparison_feedback = "Sağ kolun hareket açısı sol koldan daha az, dengelemeye çalış."
        else:
            comparison_feedback = "Sol kolun hareket açısı sağ koldan daha az, dengelemeye çalış."
    elif lw_final >= 80 and rw_final >= 80:
        comparison_feedback = "Her iki kol da iyi çalışıyor! 💪"
    elif lw_final < 60 and rw_final < 60:
        comparison_feedback = "Her iki kolun hareket açısını artırmaya çalış."
    
    return {
        'regional_scores': {
            'left_wrist': round(lw_final, 1),
            'right_wrist': round(rw_final, 1),
            'sync': round(sync_score, 1),
            # Map to traditional regions for frontend compatibility
            'arms': round((lw_final + rw_final) / 2, 1),
            'legs': 100.0,  # Not applicable for bicep curls
            'core': round(sync_score, 1),  # Use sync as core (stability)
            'head': 100.0  # Not applicable for bicep curls
        },
        'regional_issues': {
            'left_wrist': lw_issues,
            'right_wrist': rw_issues,
            'sync': sync_issues,
            'arms': lw_issues + rw_issues,
            'legs': [],
            'core': sync_issues,
            'head': []
        },
        'regional_feedback': {
            'left_wrist': f"Sol Bilek: %{lw_final:.0f} (açı: {lw_pitch_range:.0f}°)",
            'right_wrist': f"Sağ Bilek: %{rw_final:.0f} (açı: {rw_pitch_range:.0f}°)",
            'sync': f"Senkronizasyon: %{sync_score:.0f}",
            'arms': arms_feedback if arms_feedback else f"Kollar: %{(lw_final + rw_final) / 2:.0f}",
            'legs': "Bacaklar: N/A",
            'core': f"Denge: %{sync_score:.0f}",
            'head': "Kafa: N/A"
        },
        'comparison_feedback': comparison_feedback,
        'lw_pitch_range': lw_pitch_range,
        'rw_pitch_range': rw_pitch_range,
        'sync_diff': sync_diff
    }

