"""
Multi-Exercise Ensemble Model
=============================
Creates ensemble average models for multiple exercises:
1. lateral_shoulder_raises
2. squats
3. tricep_extensions

Each exercise has its own:
- Form score prediction based on IMU patterns
- Speed classification (5 levels: very_fast, fast, medium, slow, very_slow)
- Motion pattern validation
- Exercise-specific thresholds and feedback
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Training data paths
MLTRAINIMU_PATH = Path(__file__).parent.parent / "MLTRAINIMU"

# Exercise configurations
EXERCISE_CONFIGS = {
    'lateral_shoulder_raises': {
        'display_name': 'Lateral Shoulder Raises',
        'display_name_tr': 'Yan Omuz Kaldırma',
        'training_folders': [
            'lateral_shoulder_raises_20260106_005526',  # Session 1
            'lateral_shoulder_raises_20260106_005749',  # Session 2
            'lateral_shoulder_raises_20260106_005921',  # Session 3
        ],
        # Speed thresholds (seconds per rep)
        'speed_thresholds': {
            'very_fast': (0.0, 1.0),
            'fast': (1.0, 1.5),
            'medium': (1.5, 2.2),
            'slow': (2.2, 3.0),
            'very_slow': (3.0, float('inf'))
        },
        # Pitch range thresholds (degrees)
        'min_pitch_range': 70.0,      # Minimum for valid rep
        'ideal_pitch_range': 100.0,   # Ideal for good form
        'max_roll_range': 60.0,       # Max acceptable roll (lateral raise should have high roll)
        # Form score weights
        'weights': {
            'pitch': 0.35,
            'roll': 0.35,  # Roll is important for lateral raises
            'speed': 0.15,
            'quality': 0.15
        },
        # Regional score mapping
        'primary_region': 'arms',
        'use_roll_for_form': True,  # Lateral raises use roll movement
        # Detailed feedback in Turkish for different scenarios
        'feedback_templates': {
            'excellent': '🏆 Mükemmel lateral raise! Omuz deltoidleri tam aktivasyon.',
            'good': '💪 İyi form! Kollarını omuz hizasında tutuyorsun.',
            'moderate': '👍 İyi gidiyorsun. Biraz daha yükseğe kaldırmayı dene.',
            'poor': '⚠️ Kolları daha yükseğe ve kontrollü kaldır.',
            'too_fast': '🚀 Çok hızlı! Lateral raise yavaş ve kontrollü yapılmalı.',
            'too_slow': '🐢 Biraz daha hızlandırabilirsin. Ritmi koru.',
            'asymmetric': '⚖️ Sol ve sağ kolun arasında fark var. Simetri önemli!',
            'range_low': '📏 Hareket açısı dar. Kolları en az omuz hizasına kadar kaldır.',
            'range_good': '✅ Hareket açısı ideal. Bu şekilde devam et!',
            'sync_good': '🔄 Her iki kol da senkron çalışıyor. Harika!',
            'sync_poor': '⚠️ Kolların eş zamanlı hareket etmiyor. Simetri önemli!',
        },
        # Scientific feedback for LLM
        'scientific_feedback': {
            'muscle_focus': 'Lateral deltoid (medial deltoid) kasını hedefliyor',
            'key_points': [
                'Dirsekler hafif bükük tutulmalı',
                'Kollar omuz hizasına kadar (90°) kaldırılmalı',
                'Hareket kontrollü ve yavaş olmalı (2-3 saniye yukarı, 2-3 saniye aşağı)',
                'Omuzlar kulağa doğru kalkmamalı (trap dominansı önlemek için)',
                'Gövde sabit kalmalı (sallanma yok)'
            ],
            'common_mistakes': [
                'Momentum kullanmak (sallanarak kaldırmak)',
                'Dirsekleri tam düz tutmak',
                'Kolları çok yükseğe kaldırmak (trap aktivasyonu)',
                'Asimetrik hareket (bir kol diğerinden farklı)',
                'Çok hızlı yapmak (kas aktivasyonu düşer)'
            ]
        },
        'feedback_good': 'Kolları yana güzel kaldırıyorsun! 💪',
        'feedback_moderate': 'İyi hareket, biraz daha yükseğe kaldırabilirsin.',
        'feedback_poor': 'Kolları daha yükseğe ve düzgün kaldırmaya çalış.'
    },
    
    'squats': {
        'display_name': 'Squats',
        'display_name_tr': 'Squat',
        'training_folders': [
            'squats_20260106_010950',  # Session 1 (yavaş ~2.9s)
            'squats_20260106_011923',  # Session 2 (hızlı ~1.5s)
            'squats_20260106_012059',  # Session 3 (yavaş ~3.2s)
        ],
        # Speed thresholds (seconds per rep) - based on training data analysis
        # Session 1: 2.7-3.0s, Session 2: 1.3-2.0s, Session 3: 2.8-3.6s
        'speed_thresholds': {
            'very_fast': (0.0, 1.2),    # <1.2s too fast for proper squat
            'fast': (1.2, 2.0),          # 1.2-2.0s fast (Session 2)
            'medium': (2.0, 3.0),        # 2.0-3.0s ideal tempo
            'slow': (3.0, 4.0),          # 3.0-4.0s slow and controlled
            'very_slow': (4.0, float('inf'))
        },
        # ROM thresholds based on training data
        # Min angles: 48° - 75° (knee bend), Max angles: 169° - 180° (standing)
        # ROM: 102° - 127° (average ~116°)
        'min_pitch_range': 90.0,       # Minimum ROM for valid squat (must go deep)
        'ideal_pitch_range': 115.0,    # Ideal ROM (training data average ~116°)
        'max_roll_range': 20.0,        # Max acceptable roll (lateral lean - should be minimal)
        # Knee angle thresholds
        'min_knee_angle': 90.0,        # Parallel squat threshold
        'ideal_knee_angle': 70.0,      # Below parallel (deep squat)
        'max_standing_angle': 175.0,   # Standing position
        # Form score weights
        'weights': {
            'pitch': 0.40,    # Depth/ROM is critical
            'roll': 0.20,     # Lateral balance important
            'speed': 0.20,
            'quality': 0.20
        },
        # Regional score mapping
        'primary_region': 'legs',
        'use_roll_for_form': True,  # Roll indicates lateral balance
        # Detailed feedback templates in Turkish
        'feedback_templates': {
            'excellent': '🏆 Mükemmel squat! Derin indin ve form harika.',
            'good': '💪 İyi squat! Derinlik ve form iyi.',
            'moderate': '👍 İyi gidiyorsun. Biraz daha derin inmeyi dene.',
            'poor': '⚠️ Daha derin in - dizler en az 90° olmalı.',
            'too_fast': '🚀 Çok hızlı! Squat yavaş ve kontrollü yapılmalı.',
            'too_slow': '🐢 Biraz hızlandırabilirsin. 2-3 sn tempo ideal.',
            'depth_good': '✅ Derinlik mükemmel! Paralel veya altına iniyorsun.',
            'depth_poor': '📏 Daha derin in. Kalça diz hizasına gelene kadar in.',
            'balance_good': '⚖️ Denge harika! Simetrik hareket ediyorsun.',
            'balance_poor': '⚠️ Bir tarafa eğiliyorsun. Simetrik hareket et.',
            'knee_tip': '💡 Dizler ayak uçlarını geçmesin.',
            'core_tip': '💡 Core\'u sık tut, sırtın düz kalsın.'
        },
        # Scientific feedback for LLM
        'scientific_feedback': {
            'muscle_focus': 'Quadriceps, Gluteus Maximus, Hamstrings, Core stabilizers',
            'key_points': [
                'Diz 90° veya altına inilmeli (paralel veya derin squat)',
                'Sırt düz tutulmalı - bel çukuru korunmalı',
                'Dizler ayak uçlarıyla aynı yönde olmalı',
                'Topuklar yerden kalkmamalı',
                'Core kasları sıkı tutulmalı (karın içe çekilmeli)',
                'Nefes: İnerken nefes al, çıkarken ver',
                'Hareket kontrollü olmalı (2-3 sn aşağı, 2-3 sn yukarı)'
            ],
            'common_mistakes': [
                'Yeterince derin inmemek (quarter squat)',
                'Dizlerin içe çökmesi (valgus)',
                'Topukların kalkması',
                'Sırtın yuvarlanması (butt wink)',
                'Çok hızlı yapmak (momentum kullanmak)',
                'Bir tarafa eğilmek (asimetri)',
                'Dizlerin ayak uçlarını aşması'
            ],
            'depth_info': {
                'quarter_squat': '45° - yetersiz, tam kas aktivasyonu yok',
                'half_squat': '90° - minimum kabul edilebilir',
                'parallel': '90-100° - iyi derinlik',
                'below_parallel': '100-120° - ideal, glute aktivasyonu maksimum',
                'deep_squat': '120°+ - en iyi mobilite ve kas aktivasyonu'
            }
        },
        # Legacy feedback (backward compatibility)
        'feedback_good': 'Mükemmel squat! Derin iniyorsun. 🏋️',
        'feedback_moderate': 'İyi squat, biraz daha derinine inmeyi dene.',
        'feedback_poor': 'Diz açısını kontrol et ve daha derin squat yap.'
    },
    
    'tricep_extensions': {
        'display_name': 'Tricep Extensions',
        'display_name_tr': 'Triceps Açma',
        'training_folders': [
            'tricep_extensions_20260106_004736',  # Session 1 (orta hız ~1.6s)
            'tricep_extensions_20260106_005043',  # Session 2 (hızlı ~1.0s)
            'tricep_extensions_20260106_005219',  # Session 3 (yavaş ~2.5s)
        ],
        # Speed thresholds (seconds per rep) - based on training data analysis
        # Session 1: 1.5-1.7s, Session 2: 0.9-1.0s, Session 3: 2.4-2.8s
        'speed_thresholds': {
            'very_fast': (0.0, 0.8),    # <0.8s too fast
            'fast': (0.8, 1.3),          # 0.8-1.3s fast (Session 2)
            'medium': (1.3, 2.0),        # 1.3-2.0s ideal (Session 1)
            'slow': (2.0, 3.0),          # 2.0-3.0s slow (Session 3)
            'very_slow': (3.0, float('inf'))
        },
        # Pitch range thresholds (degrees) - based on training data
        # ROM from data: 150° - 173° (very high extension)
        'min_pitch_range': 140.0,     # Minimum for valid rep (kol tam açılmalı)
        'ideal_pitch_range': 165.0,   # Ideal (training data average ~165°)
        'max_roll_range': 45.0,       # Max acceptable roll (minimal rotation)
        # Min/max angles from training data
        'min_angle_threshold': 25.0,  # Arm should bend at least this much (start position)
        'max_angle_threshold': 170.0, # Arm should extend at least this much (end position)
        # Form score weights
        'weights': {
            'pitch': 0.45,    # Full extension is critical for triceps
            'roll': 0.15,     # Some roll is okay
            'speed': 0.20,
            'quality': 0.20
        },
        # Regional score mapping
        'primary_region': 'arms',
        'use_roll_for_form': False,
        # Detailed feedback templates in Turkish
        'feedback_templates': {
            'excellent': '🏆 Mükemmel triceps extension! Tam açılım, triceps kası tam aktivasyonda.',
            'good': '💪 İyi form! Kol güzel açılıyor, devam et.',
            'moderate': '👍 İyi gidiyorsun. Kolu biraz daha tam açmayı dene.',
            'poor': '⚠️ Kolu daha fazla aç - triceps tam aktive olmuyor.',
            'too_fast': '🚀 Çok hızlı! Yavaşla - kontrollü hareket tricepsi daha iyi çalıştırır.',
            'too_slow': '🐢 Biraz hızlandırabilirsin, 1.5-2 sn tempo ideal.',
            'asymmetric': '⚖️ Sol ve sağ kol arasında fark var. Simetri önemli!',
            'range_low': '📏 Hareket açısı dar. Kolu baştan sona tam aç ve kapat.',
            'range_good': '✅ Hareket açısı mükemmel! Bu şekilde devam et.',
            'elbow_tip': '💡 Dirseği sabit tut, sadece ön kol hareket etmeli.',
            'lock_warning': '⚠️ Dirseği tam kilitleme - hafif bükük tut.'
        },
        # Scientific feedback for LLM
        'scientific_feedback': {
            'muscle_focus': 'Triceps brachii (lateral head, long head, medial head) kasını hedefliyor',
            'key_points': [
                'Dirsek sabit tutulmalı - sadece ön kol hareket etmeli',
                'Kol 150-170° açıya kadar tam açılmalı (tam extension)',
                'Başlangıç pozisyonu: dirsek yaklaşık 30° bükük',
                'Hareket kontrollü olmalı (1.5-2.0 sn yukarı, 1.5-2.0 sn aşağı)',
                'En üst noktada kısa bir duraklama tricepsi daha iyi aktive eder',
                'Omuz ve üst kol hareketsiz kalmalı'
            ],
            'common_mistakes': [
                'Dirseği hareket ettirmek (omuzdan yardım almak)',
                'Momentum kullanmak (sallanarak yapmak)',
                'Kolu tam açmamak (eksik extension)',
                'Dirseği tam kilitlemek (eklem yükü)',
                'Çok hızlı yapmak (triceps yeterince kasılmaz)',
                'Asimetrik hareket (bir kol diğerinden farklı)'
            ],
            'rom_info': {
                'start_angle': '20-30° (dirsek bükük, triceps gerilmiş)',
                'end_angle': '165-180° (kol tam açık, triceps kasılmış)',
                'ideal_rom': '150-170° hareket açısı'
            }
        },
        # Legacy feedback (backward compatibility)
        'feedback_good': 'Harika triceps açması! 💪',
        'feedback_moderate': 'İyi form, kolu tam açmaya çalış.',
        'feedback_poor': 'Kolu daha fazla açarak tricepsi aktive et.'
    },
    
    'dumbbell_shoulder_press': {
        'display_name': 'Dumbbell Shoulder Press',
        'display_name_tr': 'Dumbbell Omuz Presi',
        'training_folders': [],  # No training data yet
        # Speed thresholds (seconds per rep)
        'speed_thresholds': {
            'very_fast': (0.0, 1.2),
            'fast': (1.2, 1.8),
            'medium': (1.8, 2.5),
            'slow': (2.5, 3.5),
            'very_slow': (3.5, float('inf'))
        },
        # Pitch range thresholds (degrees) - arms go up and down
        'min_pitch_range': 50.0,      # Minimum for valid rep (more lenient)
        'ideal_pitch_range': 80.0,    # Ideal for full press
        'max_roll_range': 40.0,       # Max acceptable roll
        # Form score weights
        'weights': {
            'pitch': 0.45,
            'roll': 0.15,
            'speed': 0.20,
            'quality': 0.20
        },
        # Regional score mapping
        'primary_region': 'arms',
        'use_roll_for_form': False,
        # Feedback in Turkish
        'feedback_good': 'Mükemmel omuz presi! 💪',
        'feedback_moderate': 'İyi form, kolları tam yukarı kaldırmaya çalış.',
        'feedback_poor': 'Kolları daha yükseğe kaldır ve kontrollü indir.'
    },
    
    'lateral_raises': {
        'display_name': 'Lateral Raises',
        'display_name_tr': 'Yan Kol Kaldırma',
        'training_folders': [],  # Alias for lateral_shoulder_raises
        # Speed thresholds (seconds per rep)
        'speed_thresholds': {
            'very_fast': (0.0, 1.0),
            'fast': (1.0, 1.5),
            'medium': (1.5, 2.2),
            'slow': (2.2, 3.0),
            'very_slow': (3.0, float('inf'))
        },
        # Pitch range thresholds (degrees)
        'min_pitch_range': 50.0,      # More lenient minimum
        'ideal_pitch_range': 90.0,    # Ideal for good form
        'max_roll_range': 60.0,       # Roll is important for lateral raises
        # Form score weights
        'weights': {
            'pitch': 0.35,
            'roll': 0.35,  # Roll is important
            'speed': 0.15,
            'quality': 0.15
        },
        # Regional score mapping
        'primary_region': 'arms',
        'use_roll_for_form': True,  # Lateral raises use roll movement
        # Feedback in Turkish
        'feedback_good': 'Kolları yana güzel kaldırıyorsun! 💪',
        'feedback_moderate': 'İyi hareket, biraz daha yükseğe kaldırabilirsin.',
        'feedback_poor': 'Kolları daha yükseğe ve düzgün kaldırmaya çalış.'
    }
}

# Speed feedback in Turkish (shared across exercises)
SPEED_FEEDBACK_TR = {
    'very_fast': {
        'label': 'Çok Hızlı',
        'emoji': '🚀',
        'feedback': 'Çok hızlı yapıyorsun! Yavaşlatarak kasları daha iyi hisset.'
    },
    'fast': {
        'label': 'Hızlı',
        'emoji': '⚡',
        'feedback': 'Hızlı tempo, formunu koruyarak devam et.'
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


class ExerciseEnsembleModel:
    """Ensemble model for a specific exercise."""
    
    def __init__(self, exercise: str):
        """Initialize the ensemble model for given exercise."""
        self.exercise = exercise
        self.config = EXERCISE_CONFIGS.get(exercise, {})
        
        if not self.config:
            print(f"⚠️  Unknown exercise: {exercise}")
            self.is_loaded = False
            return
        
        self.is_loaded = False
        self.reference_patterns: Dict[str, Dict] = {}
        self.speed_stats: Dict[str, float] = {}
        self.pitch_stats: Dict[str, Dict] = {}
        self.roll_stats: Dict[str, Dict] = {}
        self.gyro_stats: Dict[str, Dict] = {}
        
        self._load_training_data()
    
    def _load_training_data(self):
        """Load and analyze training data from summary.csv files."""
        all_rep_durations = []
        all_pitch_ranges = []
        all_roll_ranges = []
        all_range_of_motion = []
        
        exercise_path = MLTRAINIMU_PATH / self.exercise
        training_folders = self.config.get('training_folders', [])
        
        for folder_name in training_folders:
            summary_path = exercise_path / folder_name / "summary.csv"
            
            if not summary_path.exists():
                print(f"⚠️  Training data not found: {summary_path}")
                continue
            
            try:
                # Load summary data
                df = pd.read_csv(summary_path)
                
                # Filter valid reps (rep_number > 0)
                valid_reps = df[df['rep_number'] > 0]
                
                if len(valid_reps) < 2:
                    continue
                
                # Calculate rep durations from timestamps
                timestamps = valid_reps['timestamp'].values
                if len(timestamps) > 1:
                    rep_durations = np.diff(timestamps)
                    # Filter reasonable durations (0.5s to 10s)
                    valid_durations = rep_durations[(rep_durations > 0.5) & (rep_durations < 10)]
                    all_rep_durations.extend(valid_durations)
                
                # Get range of motion data
                if 'range_of_motion' in valid_reps.columns:
                    rom_values = valid_reps['range_of_motion'].dropna().values
                    all_range_of_motion.extend(rom_values)
                
                # Get min/max angles for pitch estimation
                if 'min_angle' in valid_reps.columns and 'max_angle' in valid_reps.columns:
                    min_angles = valid_reps['min_angle'].dropna().values
                    max_angles = valid_reps['max_angle'].dropna().values
                    if len(min_angles) > 0 and len(max_angles) > 0:
                        pitch_ranges = max_angles - min_angles
                        all_pitch_ranges.extend(pitch_ranges)
                
                print(f"✅ Loaded training data: {folder_name} ({len(valid_reps)} reps)")
                
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
                'ideal_range': float(np.percentile(all_pitch_ranges, 75))
            }
        
        if all_range_of_motion:
            self.roll_stats = {
                'mean_rom': float(np.mean(all_range_of_motion)),
                'std_rom': float(np.std(all_range_of_motion)),
                'ideal_rom': float(np.percentile(all_range_of_motion, 75))
            }
        
        self.is_loaded = bool(all_rep_durations) or bool(all_pitch_ranges)
        
        if self.is_loaded:
            print(f"📊 Ensemble model loaded for {self.exercise}")
            if self.speed_stats:
                print(f"   Speed stats: mean={self.speed_stats.get('mean_duration', 0):.2f}s, "
                      f"median={self.speed_stats.get('median_duration', 0):.2f}s")
            if self.pitch_stats:
                print(f"   Range stats: mean={self.pitch_stats.get('mean_range', 0):.1f}°, "
                      f"ideal={self.pitch_stats.get('ideal_range', 0):.1f}°")
    
    def classify_speed(self, rep_duration: float) -> Dict:
        """
        Classify rep speed into 5 categories.
        
        Args:
            rep_duration: Duration of the rep in seconds
            
        Returns:
            Dict with speed classification and feedback
        """
        thresholds = self.config.get('speed_thresholds', {
            'very_fast': (0.0, 1.2),
            'fast': (1.2, 1.8),
            'medium': (1.8, 2.5),
            'slow': (2.5, 3.5),
            'very_slow': (3.5, float('inf'))
        })
        
        for speed_class, (min_dur, max_dur) in thresholds.items():
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
        roll_range: float = 0.0,
        gyro_magnitude: float = 0.0,
        rep_duration: float = 2.0,
        samples_count: int = 0
    ) -> Tuple[float, List[str]]:
        """
        Calculate form score using exercise-specific ensemble model.
        
        Args:
            pitch_range: Pitch angle range during rep (degrees)
            roll_range: Roll angle range during rep (degrees)
            gyro_magnitude: Average gyroscope magnitude during rep
            rep_duration: Duration of rep in seconds
            samples_count: Number of IMU samples in rep
            
        Returns:
            Tuple of (form_score 0-100, list of issues)
        """
        issues = []
        score_components = []
        weights = self.config.get('weights', {'pitch': 0.40, 'roll': 0.20, 'speed': 0.20, 'quality': 0.20})
        
        # Get exercise-specific thresholds
        min_pitch_range = self.config.get('min_pitch_range', 60.0)
        ideal_pitch_range = self.config.get('ideal_pitch_range', 100.0)
        
        # Use training data if available
        if self.pitch_stats:
            ideal_pitch_range = self.pitch_stats.get('ideal_range', ideal_pitch_range)
            mean_range = self.pitch_stats.get('mean_range', min_pitch_range)
        else:
            mean_range = min_pitch_range
        
        # 1. Pitch Range Score
        if pitch_range >= ideal_pitch_range:
            pitch_score = 100.0
        elif pitch_range >= mean_range:
            pitch_score = 70 + 30 * (pitch_range - mean_range) / max(1, ideal_pitch_range - mean_range)
        elif pitch_range >= min_pitch_range:
            pitch_score = 50 + 20 * (pitch_range - min_pitch_range) / max(1, mean_range - min_pitch_range)
            issues.append("Hareket açısı biraz dar")
        else:
            pitch_score = max(20, 30 + 20 * pitch_range / max(1, min_pitch_range))
            issues.append("Hareket açısı çok dar - tam hareket yap")
        
        score_components.append(('pitch', pitch_score, weights.get('pitch', 0.40)))
        
        # 2. Roll Score (for exercises that use roll, like lateral raises)
        if self.config.get('use_roll_for_form', False) and roll_range > 0:
            max_roll_range = self.config.get('max_roll_range', 60.0)
            if roll_range >= max_roll_range * 0.8:
                roll_score = 100.0
            elif roll_range >= max_roll_range * 0.5:
                roll_score = 70 + 30 * (roll_range - max_roll_range * 0.5) / (max_roll_range * 0.3)
            else:
                roll_score = max(30, roll_range / (max_roll_range * 0.5) * 70)
                issues.append("Kolları daha yana kaldır")
        else:
            # For exercises where roll should be minimal
            max_roll_range = self.config.get('max_roll_range', 45.0)
            if roll_range <= max_roll_range * 0.3:
                roll_score = 100.0
            elif roll_range <= max_roll_range:
                roll_score = 70 + 30 * (1 - (roll_range - max_roll_range * 0.3) / (max_roll_range * 0.7))
            else:
                roll_score = max(40, 70 - (roll_range - max_roll_range) * 0.5)
                issues.append("Bilek fazla dönüyor")
        
        score_components.append(('roll', roll_score, weights.get('roll', 0.20)))
        
        # 3. Speed/Tempo Score
        speed_class = self.classify_speed(rep_duration)['class']
        if speed_class == 'medium':
            speed_score = 100.0
        elif speed_class in ['fast', 'slow']:
            speed_score = 85.0
        elif speed_class == 'very_fast':
            speed_score = 65.0
            issues.append("Çok hızlı - yavaşla")
        else:  # very_slow
            speed_score = 70.0
            issues.append("Çok yavaş")
        score_components.append(('speed', speed_score, weights.get('speed', 0.20)))
        
        # 4. Movement Quality Score (based on consistency)
        if gyro_magnitude > 0:
            # Use gyro magnitude as quality indicator
            quality_score = min(100, max(50, 80 + (gyro_magnitude - 100) * 0.1))
        else:
            quality_score = 75.0  # Default
        score_components.append(('quality', quality_score, weights.get('quality', 0.20)))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        return max(0, min(100, total_score)), issues
    
    def analyze_wrist_form(
        self,
        lw_pitch_range: float,
        rw_pitch_range: float,
        lw_roll_range: float = 0.0,
        rw_roll_range: float = 0.0,
        lw_gyro_mag: float = 0.0,
        rw_gyro_mag: float = 0.0,
        sync_diff: float = 0.0
    ) -> Dict:
        """
        Analyze individual wrist form for the exercise.
        
        Returns:
            Dict with left_wrist, right_wrist scores and feedback
        """
        min_pitch_range = self.config.get('min_pitch_range', 60.0)
        ideal_pitch_range = self.config.get('ideal_pitch_range', 100.0)
        
        if self.pitch_stats:
            ideal_pitch_range = self.pitch_stats.get('ideal_range', ideal_pitch_range)
            mean_range = self.pitch_stats.get('mean_range', min_pitch_range)
        else:
            mean_range = min_pitch_range
        
        def calculate_wrist_score(pitch_range: float, roll_range: float, wrist_name: str) -> Dict:
            """Calculate form score for a single wrist."""
            issues = []
            
            # Pitch range score (60%)
            if pitch_range >= ideal_pitch_range:
                pitch_score = 100.0
            elif pitch_range >= mean_range:
                pitch_score = 70 + 30 * (pitch_range - mean_range) / max(1, ideal_pitch_range - mean_range)
            elif pitch_range >= min_pitch_range:
                pitch_score = 50 + 20 * (pitch_range - min_pitch_range) / max(1, mean_range - min_pitch_range)
                issues.append(f"{wrist_name} hareket açısı dar")
            else:
                pitch_score = max(20, 30 + 20 * pitch_range / max(1, min_pitch_range))
                issues.append(f"{wrist_name} hareket açısı çok dar")
            
            # Quality score (40%)
            quality_score = 80.0  # Default
            
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
                'roll_range': round(roll_range, 1),
                'issues': issues,
                'feedback': feedback
            }
        
        lw_result = calculate_wrist_score(lw_pitch_range, lw_roll_range, "Sol Bilek")
        rw_result = calculate_wrist_score(rw_pitch_range, rw_roll_range, "Sağ Bilek")
        
        # Synchronization analysis
        sync_issues = []
        if sync_diff > 50:
            sync_score = max(20, 100 - (sync_diff - 50) * 2)
            sync_issues.append(f"Bilekler senkron değil (fark: {sync_diff:.0f}°)")
        elif sync_diff > 30:
            sync_score = 70 + (50 - sync_diff) * 1.5
        else:
            sync_score = 100.0
        
        # Overall score
        primary_region = self.config.get('primary_region', 'arms')
        arms_score = (lw_result['score'] + rw_result['score'] + sync_score) / 3
        
        return {
            'left_wrist': lw_result,
            'right_wrist': rw_result,
            'sync_score': round(sync_score, 1),
            'sync_diff': round(sync_diff, 1),
            'sync_issues': sync_issues,
            'arms_score': round(arms_score, 1),
            'regional_scores': {
                primary_region: round(arms_score, 1),
                'arms': round(arms_score, 1) if primary_region != 'arms' else round(arms_score, 1),
                'legs': 100.0 if primary_region != 'legs' else round(arms_score, 1),
                'core': 85.0,
                'head': 90.0
            }
        }
    
    def get_rep_analysis(
        self,
        pitch_range: float,
        roll_range: float = 0.0,
        gyro_magnitude: float = 0.0,
        rep_duration: float = 2.0,
        samples_count: int = 0,
        lw_pitch_range: float = None,
        rw_pitch_range: float = None,
        lw_roll_range: float = 0.0,
        rw_roll_range: float = 0.0,
        lw_gyro_mag: float = None,
        rw_gyro_mag: float = None,
        sync_diff: float = 0.0
    ) -> Dict:
        """
        Get comprehensive analysis of a rep including LW/RW specific feedback.
        
        Returns dict with form_score, speed_class, feedback, regional scores, and issues.
        """
        form_score, issues = self.calculate_form_score(
            pitch_range, roll_range, gyro_magnitude, rep_duration, samples_count
        )
        
        speed_info = self.classify_speed(rep_duration)
        
        # Analyze individual wrists if data is available
        wrist_analysis = None
        primary_region = self.config.get('primary_region', 'arms')
        regional_scores = {primary_region: form_score, 'arms': form_score, 'legs': 100.0, 'core': 85.0, 'head': 90.0}
        
        if lw_pitch_range is not None and rw_pitch_range is not None:
            wrist_analysis = self.analyze_wrist_form(
                lw_pitch_range or pitch_range / 2,
                rw_pitch_range or pitch_range / 2,
                lw_roll_range,
                rw_roll_range,
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
            form_feedback = self.config.get('feedback_good', 'Mükemmel form! 💪')
        elif form_score >= 70:
            form_feedback = self.config.get('feedback_moderate', 'İyi form, ufak düzeltmeler yapılabilir.')
        else:
            form_feedback = self.config.get('feedback_poor', 'Form düşük, tekniğe odaklan.')
        
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
            'roll_range': round(roll_range, 1),
            'gyro_magnitude': round(gyro_magnitude, 1),
            'regional_scores': regional_scores,
            'exercise': self.exercise,
            'exercise_name_tr': self.config.get('display_name_tr', self.exercise)
        }
        
        if wrist_analysis:
            result['left_wrist'] = wrist_analysis['left_wrist']
            result['right_wrist'] = wrist_analysis['right_wrist']
            result['sync_score'] = wrist_analysis['sync_score']
            result['sync_diff'] = wrist_analysis['sync_diff']
        
        return result


# Cache for loaded models
_exercise_models: Dict[str, ExerciseEnsembleModel] = {}


def get_exercise_ensemble_model(exercise: str) -> ExerciseEnsembleModel:
    """Get or create the ensemble model for given exercise."""
    global _exercise_models
    
    # Normalize exercise name
    exercise_normalized = exercise.lower().replace(' ', '_').replace('-', '_')
    
    if exercise_normalized not in _exercise_models:
        _exercise_models[exercise_normalized] = ExerciseEnsembleModel(exercise_normalized)
    
    return _exercise_models[exercise_normalized]


def analyze_exercise_rep(
    exercise: str,
    pitch_range: float,
    roll_range: float = 0.0,
    gyro_magnitude: float = 0.0,
    rep_duration: float = 2.0,
    samples_count: int = 0,
    lw_pitch_range: float = None,
    rw_pitch_range: float = None,
    lw_roll_range: float = 0.0,
    rw_roll_range: float = 0.0,
    lw_gyro_mag: float = None,
    rw_gyro_mag: float = None,
    sync_diff: float = 0.0
) -> Dict:
    """
    Convenience function to analyze a rep for any exercise.
    
    Args:
        exercise: Exercise name
        pitch_range: Pitch angle range during rep (degrees)
        roll_range: Roll angle range during rep (degrees)
        gyro_magnitude: Average gyroscope magnitude during rep
        rep_duration: Duration of rep in seconds
        samples_count: Number of IMU samples in rep
        lw_pitch_range: Left wrist pitch range (optional)
        rw_pitch_range: Right wrist pitch range (optional)
        lw_roll_range: Left wrist roll range (optional)
        rw_roll_range: Right wrist roll range (optional)
        lw_gyro_mag: Left wrist gyro magnitude (optional)
        rw_gyro_mag: Right wrist gyro magnitude (optional)
        sync_diff: Wrist synchronization difference (optional)
        
    Returns:
        Analysis dict with form_score, speed_class, feedback, regional scores, etc.
    """
    model = get_exercise_ensemble_model(exercise)
    return model.get_rep_analysis(
        pitch_range, roll_range, gyro_magnitude, rep_duration, samples_count,
        lw_pitch_range, rw_pitch_range, lw_roll_range, rw_roll_range,
        lw_gyro_mag, rw_gyro_mag, sync_diff
    )


def classify_exercise_rep_speed(exercise: str, rep_duration: float) -> Dict:
    """Classify rep speed for given exercise."""
    model = get_exercise_ensemble_model(exercise)
    return model.classify_speed(rep_duration)


def calculate_exercise_wrist_scores(
    exercise: str,
    lw_pitch_range: float,
    rw_pitch_range: float,
    lw_roll_range: float = 0.0,
    rw_roll_range: float = 0.0,
    sync_diff: float = 0.0
) -> Dict:
    """
    Calculate individual wrist form scores for given exercise.
    
    Args:
        exercise: Exercise name
        lw_pitch_range: Left wrist pitch range (degrees)
        rw_pitch_range: Right wrist pitch range (degrees)
        lw_roll_range: Left wrist roll range
        rw_roll_range: Right wrist roll range
        sync_diff: Synchronization difference between wrists
        
    Returns:
        Dict with LW and RW scores and feedback
    """
    model = get_exercise_ensemble_model(exercise)
    return model.analyze_wrist_form(
        lw_pitch_range, rw_pitch_range, lw_roll_range, rw_roll_range, 0.0, 0.0, sync_diff
    )


def get_supported_exercises() -> List[str]:
    """Get list of exercises with ensemble models."""
    return list(EXERCISE_CONFIGS.keys())


def get_exercise_config(exercise: str) -> Dict:
    """Get configuration for given exercise."""
    return EXERCISE_CONFIGS.get(exercise, {})

