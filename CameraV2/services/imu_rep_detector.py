"""IMU periodic rep detector for IMU-only rep counting."""

import numpy as np
import time
from typing import Optional

class IMUPeriodicRepDetector:
    """
    IMU-only modunda periyodik sinyal analizi ile rep tespiti.
    Gyroscope magnitude kullanarak periyodik hareketleri tespit eder.
    Her periyod = 1 rep olarak sayılır.
    """
    
    def __init__(self, exercise: str, window_size: int = 90, min_period_samples: int = 29):
        """
        Args:
            exercise: Exercise name
            window_size: Sliding window size (samples) - periyod analizi için (~5.8s at 15.5Hz)
            min_period_samples: Minimum periyod uzunluğu (~1.9s at 15.5Hz, adjusted for bicep curls)
        """
        self.exercise = exercise
        # Real data analysis from bicep_curls_20260103_230109:
        # - Rep durations: Min: 8 sample, Max: 23 sample, Mean: 19.8 sample, Median: 21 sample
        # - Sample rate: ~66.44 Hz
        # - Rep'ler çok kısa: 8-23 sample = ~0.12-0.35 saniye
        # - Peak gyro: Min: 288.6, Max: 437.1, Mean: 349.6, Median: 350.1 deg/s
        self.window_size = 100  # ~1.5 saniye window (rep'ler 0.12-0.35s, window yeterli)
        self.min_period_samples = 5  # Minimum rep length (rep'ler 8-23 sample, 5 yeterli)
        
        # Combined magnitude buffer (sliding window) - gyro + accel
        # Rep detection: Peak-based detection (104 peaks for 100 reps = 1.04 ratio, excellent!)
        self.magnitude_buffer = []  # Combined magnitude (gyro + accel)
        self.timestamps = []
        
        # Rep tracking
        self.rep_count = 0
        self.last_rep_time = None
        self.current_period_start = None
        self.period_complete = False
        self.rep_durations = []  # Track rep durations for adaptive thresholding
        
        # Cooldown period to prevent double counting (in seconds)
        # Rep'ler çok kısa (0.12-0.35s), cooldown sadece çift saymayı önlemek için (çok kısa)
        # Kullanıcı istediği kadar bekleyebilir, uzun bekleme sonrası yeni rep algılanır
        self.rep_cooldown = 0.2  # Minimum 0.2s between reps (sadece çift saymayı önlemek için, çok kısa)
        self.last_rep_detection_time = None
        self.max_idle_time = 5.0  # 5 saniye bekleme sonrası yeni rep cycle başlat (kullanıcı istediği kadar bekleyebilir)
        
        # Adaptive peak detection parameters (adjusted based on REAL bicep curl data analysis)
        # Real data analysis from bicep_curls_20260103_230109:
        #   Rep durations: 8-23 sample (mean: 19.8, median: 21)
        #   Peak gyro: Min: 288.6, Max: 437.1, Mean: 349.6, Median: 350.1 deg/s
        #   Sample rate: ~66.44 Hz
        self.min_peak_height_base = 200.0  # Base minimum gyro magnitude (deg/s) - peaks are 288-437, 200 is conservative
        self.min_peak_distance = self.min_period_samples  # Minimum distance between peaks (samples)
        self.adaptive_threshold_factor = 1.0  # Adaptive threshold multiplier
        
        # State tracking
        self.last_peak_index = -1
        self.is_tracking = False
        self.current_period_start_time = None
        self.sample_rate = 66.44  # Real sample rate from data analysis
    
    def add_imu_sample(self, imu_sample: dict) -> Optional[dict]:
        """
        IMU sample ekle ve rep tespiti yap.
        
        Args:
            imu_sample: IMU sample dict (left_wrist, right_wrist, chest içeren)
            
        Returns:
            Rep completion dict veya None (rep tamamlanmadıysa)
        """
        timestamp = imu_sample.get('timestamp', time.time())
        
        # Calculate combined magnitude from wrist nodes (gyro + accel)
        # Rep detection: Peak-based detection (each peak = one rep)
        combined_mag = self._calculate_combined_magnitude(imu_sample)
        
        if combined_mag is None:
            return None
        
        # Add to buffer
        self.magnitude_buffer.append(combined_mag)
        self.timestamps.append(timestamp)
        
        # Keep buffer size limited
        if len(self.magnitude_buffer) > self.window_size * 2:
            self.magnitude_buffer.pop(0)
            self.timestamps.pop(0)
        
        # Need minimum samples for analysis
        # Rep'ler kısa (8-23 sample), buffer'da en az 2 rep olmalı
        if len(self.magnitude_buffer) < self.min_period_samples * 3:
            return None
        
        # Cooldown check: prevent double counting (sadece çok kısa süreler için)
        # Kullanıcı uzun süre bekleyip tekrar başladığında, sistem bunu algılamalı
        if self.last_rep_detection_time is not None:
            time_since_last_rep = timestamp - self.last_rep_detection_time
            # Rep detection durmasını önlemek için cooldown'u daha kısa yap
            # Sadece çok kısa süreler için cooldown uygula (çift saymayı önlemek için)
            # Uzun bekleme sonrası yeni rep algılanabilir
            effective_cooldown = min(self.rep_cooldown, 0.15)  # Max 0.15s cooldown (daha agresif)
            if time_since_last_rep < effective_cooldown:
                return None  # Too soon after last rep, ignore (sadece çift saymayı önlemek için)
        
        # Detect period (rep completion)
        rep_detected = self._detect_period()
        
        if rep_detected:
            # Additional cooldown check after detection
            if self.last_rep_detection_time is not None:
                time_since_last_rep = timestamp - self.last_rep_detection_time
                effective_cooldown = min(self.rep_cooldown, 0.15)  # Max 0.15s cooldown (daha agresif)
                if time_since_last_rep < effective_cooldown:
                    return None  # Too soon, ignore this detection
            
            self.rep_count += 1
            self.last_rep_time = timestamp
            self.last_rep_detection_time = timestamp  # Update cooldown timer
            
            # Calculate rep duration (for speed detection)
            rep_duration = None
            if self.current_period_start_time is not None:
                rep_duration = timestamp - self.current_period_start_time
                if rep_duration > 0:
                    self.rep_durations.append(rep_duration)
                    # Keep only last 10 durations for adaptive thresholding
                    if len(self.rep_durations) > 10:
                        self.rep_durations.pop(0)
            
            self.period_complete = True
            period_start_time_backup = self.current_period_start_time
            self.current_period_start_time = None  # Reset for next rep
            
            return {
                'rep': self.rep_count,
                'timestamp': timestamp,
                'detection_method': 'periodic_imu',
                'rep_duration': rep_duration  # Duration in seconds (for speed detection)
            }
        
        return None
    
    def _calculate_combined_magnitude(self, imu_sample: dict) -> Optional[float]:
        """
        Wrist nodes'larından combined magnitude hesapla (gyro + accel).
        Exercise'e göre left veya right wrist kullan.
        Rep detection için: Hareketin başında/sonunda magnitude azalır (valley detection).
        """
        # Exercise'e göre hangi node'ları kullanacağımızı belirle
        if self.exercise in ['bicep_curls', 'triceps_pushdown', 'dumbbell_rows']:
            # Her iki wrist de kullanılabilir - average al
            left_wrist = imu_sample.get('left_wrist', {})
            right_wrist = imu_sample.get('right_wrist', {})
            
            combined_mags = []
            for wrist_data in [left_wrist, right_wrist]:
                if wrist_data and isinstance(wrist_data, dict):
                    # Gyroscope magnitude
                    gx = wrist_data.get('gx', 0) or 0
                    gy = wrist_data.get('gy', 0) or 0
                    gz = wrist_data.get('gz', 0) or 0
                    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
                    
                    # Accelerometer magnitude (scale for compatibility)
                    ax = wrist_data.get('ax', 0) or 0
                    ay = wrist_data.get('ay', 0) or 0
                    az = wrist_data.get('az', 0) or 0
                    accel_mag = np.sqrt(ax**2 + ay**2 + az**2) * 100  # Scale accel
                    
                    # Combined magnitude
                    combined = (gyro_mag + accel_mag) / 2
                    combined_mags.append(combined)
            
            if len(combined_mags) > 0:
                return np.mean(combined_mags)
        elif self.exercise in ['dumbbell_shoulder_press', 'lateral_shoulder_raises']:
            # Shoulder exercises - her iki wrist
            left_wrist = imu_sample.get('left_wrist', {})
            right_wrist = imu_sample.get('right_wrist', {})
            
            gyro_mags = []
            for wrist_data in [left_wrist, right_wrist]:
                if wrist_data and isinstance(wrist_data, dict):
                    gx = wrist_data.get('gx', 0) or 0
                    gy = wrist_data.get('gy', 0) or 0
                    gz = wrist_data.get('gz', 0) or 0
                    mag = np.sqrt(gx**2 + gy**2 + gz**2)
                    gyro_mags.append(mag)
            
            if len(gyro_mags) > 0:
                return np.mean(gyro_mags)
        elif self.exercise == 'squats':
            # Squats - chest node kullan (gövde hareketi)
            chest = imu_sample.get('chest', {})
            if chest and isinstance(chest, dict):
                gx = chest.get('gx', 0) or 0
                gy = chest.get('gy', 0) or 0
                gz = chest.get('gz', 0) or 0
                return np.sqrt(gx**2 + gy**2 + gz**2)
        
        return None
    
    def _detect_period(self) -> bool:
        """
        Peak-based rep detection optimized for bicep curls (100 rep dataset analysis).
        Real data analysis from bicep_curls_20260103_230109:
        - 104 peaks detected for 100 reps = 1.04 ratio (excellent!)
        - Peak distances: min=4, max=64, mean=20.7, median=21.0 samples
        - Peak values: Min: 58.0, Max: 312.6, Mean: 224.2, Median: 236.4
        - Signal median: 99.6, 75th percentile: 162.8
        - Optimal peak threshold: 130.2 (80% of 75th percentile)
        Each peak represents a rep's maximum movement intensity.
        Two peaks = one complete rep cycle (up and down).
        Kısa, orta ve uzun rep'leri detect edebilir (adaptive thresholds).
        """
        if len(self.magnitude_buffer) < self.window_size:
            return False
        
        # Son window_size sample'ı al
        signal = np.array(self.magnitude_buffer[-self.window_size:])
        
        signal_median = np.median(signal)
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_75th = np.percentile(signal, 75)
        signal_90th = np.percentile(signal, 90)
        
        # Peak threshold: 75th percentile bazlı (real data analysis)
        # Optimal threshold: 80% of 75th percentile = 130.2 (from data analysis)
        # Rep detection durmasını önlemek için threshold'u daha agresif yapıyoruz
        if len(self.rep_durations) > 0:
            avg_duration = np.median(self.rep_durations)
            # Kısa rep'ler için threshold daha düşük (daha agresif detection)
            duration_factor = 1.0 / max(0.1, min(3.0, avg_duration))
            peak_threshold_factor = 0.70 + (0.10 * duration_factor)  # 0.70-0.80 range (daha agresif)
        else:
            peak_threshold_factor = 0.75  # Default: 75% of 75th percentile (daha agresif, rep detection durmasını önlemek için)
        
        peak_threshold = signal_75th * peak_threshold_factor
        # Minimum peak height'i daha düşük yapıyoruz (rep detection durmasını önlemek için)
        min_peak_height = max(30.0, signal_median * 0.9)  # Peak must be at least 0.9x median (daha agresif)
        
        # Son recent_window sample içinde peak (maximum) ara
        # Rep'ler kısa (8-23 sample, median ~21), recent_window yeterli
        # Rep detection durmasını önlemek için window'u daha büyük yapıyoruz
        recent_window = min(100, len(signal))  # Son 100 sample (4-5 rep için yeterli, rep detection durmasını önlemek için)
        recent_signal = signal[-recent_window:]
        
        if len(recent_signal) < 7:  # Minimum 7 sample gerekli (3 sample neighborhood check için)
            return False
        
        # Peak detection: local maximum bul (wider neighborhood for better detection)
        peaks_found = []
        for i in range(3, len(recent_signal) - 3):
            # Local maximum: komşularından daha büyük (wider neighborhood: ±3 samples)
            is_local_max = (recent_signal[i] > recent_signal[i-1] and 
                           recent_signal[i] > recent_signal[i+1] and
                           recent_signal[i] > recent_signal[i-2] and
                           recent_signal[i] > recent_signal[i+2] and
                           recent_signal[i] > recent_signal[i-3] and
                           recent_signal[i] > recent_signal[i+3])
            
            if is_local_max:
                peak_value = recent_signal[i]
                # Peak threshold'un üstünde olmalı ve minimum height'tan büyük olmalı
                if peak_value > peak_threshold and peak_value > min_peak_height:
                    peak_idx = len(self.magnitude_buffer) - recent_window + i
                    peaks_found.append((peak_idx, peak_value))
        
        # Her peak bir rep'i gösterir (peak detection: 104 peaks for 100 reps)
        # Eğer tracking başlamışsa ve yeni bir peak bulunduysa rep tamamlandı
        if len(peaks_found) > 0:
            latest_peak_idx, latest_peak_value = peaks_found[-1]
            latest_peak_time = self.timestamps[latest_peak_idx] if latest_peak_idx < len(self.timestamps) else None
            
            # Adaptive min distance: rep sürelerine göre
            # Real data: peak distances min=4, max=64, mean=20.7, median=21.0
            # Rep detection durmasını önlemek için min_distance'i daha küçük yapıyoruz
            if len(self.rep_durations) > 0:
                avg_duration_samples = int(np.median(self.rep_durations) * self.sample_rate)
                min_peak_distance = max(int(avg_duration_samples * 0.5), 3)  # 50% of avg duration, min 3 samples (daha agresif)
            else:
                min_peak_distance = 3  # Default: minimum 3 samples (daha agresif, rep detection durmasını önlemek için)
            
            # İlk peak'ten sonraki peak = rep tamamlandı
            # CRITICAL: latest_peak_idx must be > last_peak_index (prevent double counting of same peak)
            # But we also need to handle the case where buffer wraps around or peak indices get out of sync
            if self.last_peak_index >= 0:
                # Check if this is a new peak (different index)
                if latest_peak_idx <= self.last_peak_index:
                    # Check if buffer wrapped around (new peak is at end but old index is from earlier)
                    buffer_len = len(self.magnitude_buffer)
                    # If old index is far from end and new index is near end, might be a new rep
                    if (self.last_peak_index < buffer_len * 0.5 and latest_peak_idx > buffer_len * 0.8):
                        # Buffer likely wrapped, allow this as new peak
                        pass  # Continue to process as new peak
                    else:
                        # This peak was already counted, ignore
                        return False
            
            # Eğer son peak'ten çok uzun zaman geçtiyse (kullanıcı uzun süre bekledi), yeni cycle başlat
            if self.last_peak_index >= 0 and latest_peak_time:
                last_peak_time = self.timestamps[self.last_peak_index] if self.last_peak_index < len(self.timestamps) else None
                if last_peak_time:
                    time_since_last_peak = latest_peak_time - last_peak_time
                    # Reduce max_idle_time to detect reps faster after pauses
                    if time_since_last_peak > min(self.max_idle_time, 3.0):  # Max 3s instead of 5s
                        # Kullanıcı uzun süre bekledi, yeni rep cycle başlat (cooldown'u bypass et)
                        print(f"🔄 Long idle detected ({time_since_last_peak:.1f}s), starting new rep cycle")
                        self.last_peak_index = -1  # Reset to allow new cycle
                        self.is_tracking = False
                        # Continue to normal detection logic below
            
            if self.is_tracking:
                # Son peak'ten yeterince zaman geçtiyse rep say
                if self.last_peak_index >= 0:
                    distance = latest_peak_idx - self.last_peak_index
                    # Rep detection durmasını önlemek için: hem distance hem de time-based check daha agresif
                    time_check_passed = False
                    time_since_last_peak = None
                    if latest_peak_time:
                        last_peak_time = self.timestamps[self.last_peak_index] if self.last_peak_index < len(self.timestamps) else None
                        if last_peak_time:
                            time_since_last_peak = latest_peak_time - last_peak_time
                            # Time check'i daha kısa yap (0.2s instead of 0.3s) - rep detection durmasını önlemek için
                            time_check_passed = time_since_last_peak > 0.2
                    
                    # Distance check'i de daha küçük yap (50% of min_peak_distance) - rep detection durmasını önlemek için
                    distance_threshold = max(int(min_peak_distance * 0.5), 2)  # At least 2 samples
                    
                    if distance >= distance_threshold or time_check_passed:
                        self.last_peak_index = latest_peak_idx
                        self.is_tracking = False
                        if latest_peak_time and self.current_period_start_time:
                            self.current_period_start_time = None
                        return True
                else:
                    # İlk peak bulundu, tracking başlat
                    self.last_peak_index = latest_peak_idx
                    self.is_tracking = False
                    return False
            else:
                # İlk peak: tracking başlat
                # Rep detection durmasını önlemek için: eğer yeterince zaman geçtiyse veya ilk peak ise tracking başlat
                if self.last_peak_index < 0 or (latest_peak_idx - self.last_peak_index) >= min_peak_distance:
                    self.last_peak_index = latest_peak_idx
                    self.is_tracking = True
                    self.current_period_start_time = latest_peak_time
                    return False
                # Eğer yeterince zaman geçtiyse ama min_peak_distance'den küçükse, yine de tracking başlat (rep detection durmasını önlemek için)
                elif self.last_peak_index >= 0 and latest_peak_time:
                    last_peak_time = self.timestamps[self.last_peak_index] if self.last_peak_index < len(self.timestamps) else None
                    if last_peak_time:
                        time_since_last_peak = latest_peak_time - last_peak_time
                        if time_since_last_peak > 0.5:  # 0.5 saniye geçtiyse, yeni peak olarak kabul et
                            self.last_peak_index = latest_peak_idx
                            self.is_tracking = True
                            self.current_period_start_time = latest_peak_time
                            return False
        
        return False
    
    def reset(self):
        """Reset detector state."""
        self.magnitude_buffer = []
        self.timestamps = []
        self.rep_count = 0
        self.last_rep_time = None
        self.current_period_start = None
        self.period_complete = False
        self.last_peak_index = -1
        self.is_tracking = False
        self.rep_durations = []
        self.current_period_start_time = None
        self.adaptive_threshold_factor = 1.0
        self.last_rep_detection_time = None  # Reset cooldown timer


# AI Feedback with variety
FEEDBACK_TEMPLATES = [
    "Great job! {detail}",
    "Looking good! {detail}",
    "Nice work! {detail}",
    "Keep it up! {detail}",
    "Excellent! {detail}",
    "{detail} Keep going!",
    "Amazing energy! {detail}",
]

CORRECTION_TEMPLATES = [
    "{issue} - pay attention.",
    "Small fix needed: {issue}",
    "{issue} - stay controlled.",
    "Watch out: {issue}",
]

import random

# EXERCISE FEEDBACK LIBRARY - 72 feedback options (6 exercises x 12 categories)
EXERCISE_FEEDBACK_LIBRARY = {
    'bicep_curls': {
        1: "🎉 Mükemmel biceps curl! Form, hız ve kontrol harika. Devam et!",
        2: "💪 Çok iyi! Dirsekler sabit, hareket kontrollü. İyi gidiyorsun!",
        3: "👍 İyi form, dirseklerin biraz daha sabit kalmalı. Küçük bir iyileştirme yap.",
        4: "✅ İyi gidiyorsun, omuzların daha düşük kalmalı. Gövdeni sabitle.",
        5: "⚠️ Orta seviye, dirsekleri gövdene sabitle. Daha kontrollü hareket et.",
        6: "🔴 Kollarına odaklan: dirsekleri sabit tut, sallama. Gövdeni sabitle.",
        7: "🔴 Gövdeni sabitle, öne eğilme. Dikey dur ve dirsekleri sabit tut.",
        8: "🔴 Kafanı nötr tut, aşağı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: dirsekleri sabitle ve gövdeni düz tut. Yavaşla.",
        10: "🟡 Hareketi tamamla, kolları tam uzat. Tam hareket menzili kullan.",
        11: "🟡 Kontrolü artır, daha yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Dirseklerin omuzun üstüne çıkmasın, daha düşük tut. Yanlış açıda hareket ediyorsun."
    },
    'squats': {
        1: "🎉 Mükemmel squat! Derinlik ve form harika. Mükemmel çalışma!",
        2: "💪 Çok iyi! Dizler ayak parmaklarının üzerinde, gövde düz. İyi gidiyorsun!",
        3: "👍 İyi form, biraz daha derine inebilirsin. Derinliği artır.",
        4: "✅ İyi gidiyorsun, gövdeni daha dik tut. Omurganı düzleştir.",
        5: "⚠️ Orta seviye, dizlerin içe düşmesin. Dizlerini dışarı doğru it.",
        6: "🔴 Bacaklarına odaklan: dizleri dışarı doğru it. İçe çökmesin.",
        7: "🔴 Gövdeni düz tut, öne çok eğilme. Dikey dur, göğsünü kaldır.",
        8: "🔴 İleri bak, kafanı öne eğme. Gözlerin öne baksın.",
        9: "🔴 Birkaç sorun var: diz pozisyonu ve gövde düzgünlüğüne dikkat. Yavaşla.",
        10: "🟡 Daha derine in, kalçalar diz seviyesinin altına gelsin. Derinlik artır.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Dizlerin içe çökmesin! Ayak parmaklarınla hizalı tut, dışarı doğru it."
    },
    'lateral_shoulder_raises': {
        1: "🎉 Mükemmel lateral raise! Omuz kontrolü harika. Devam et!",
        2: "💪 Çok iyi! Kollar omuz hizasında, simetrik. İyi gidiyorsun!",
        3: "👍 İyi form, kolları biraz daha simetrik kaldır. Eşit yüksekliğe getir.",
        4: "✅ İyi gidiyorsun, omuzların yukarı kalkmasın. Omuzları düşük tut.",
        5: "⚠️ Orta seviye, kolları omuz hizasına kadar kaldır. Yeterince yükseğe çık.",
        6: "🔴 Kollarına odaklan: simetrik kaldır, eşit yüksekliğe getir. Asimetri var.",
        7: "🔴 Gövdeni sabitle, sallanma. Dikey dur, core'unu sık.",
        8: "🔴 Kafanı nötr tut, yukarı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: simetrik kaldır ve gövdeni sabitle. Yavaşla.",
        10: "🟡 Kolları omuz hizasına kadar kaldır, daha yukarı çıkar. Tam menzil kullan.",
        11: "🟡 Kontrolü artır, omuzları silkmeyi bırak. Yavaş ve kontrollü hareket et.",
        12: "🔴 Omuzlarını yukarı kaldırma! Sadece kolları kaldır, omuzlar düşük kalsın."
    },
    'triceps_pushdown': {
        1: "🎉 Mükemmel triceps pushdown! Üst kol sabit, form harika. Devam et!",
        2: "💪 Çok iyi! Üst kol sabit, sadece dirsek hareket ediyor. İyi gidiyorsun!",
        3: "👍 İyi form, üst kolunu biraz daha sabit tut. Sallanmayı azalt.",
        4: "✅ İyi gidiyorsun, dirseği tam aç. Tam hareket menzili kullan.",
        5: "⚠️ Orta seviye, üst kolunu sabit tut, sallama. Kontrolü artır.",
        6: "🔴 Kollarına odaklan: üst kol sabit, sadece dirsek hareket etsin. Sallama.",
        7: "🔴 Gövdeni sabitle, öne eğilme. Dikey dur, core'unu sık.",
        8: "🔴 Kafanı nötr tut, aşağı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: üst kol sabitliği ve gövde pozisyonuna dikkat. Yavaşla.",
        10: "🟡 Dirseği tam aç, kolları tam uzat. Tam hareket menzili kullan.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Üst kolunu sabit tut! Sadece ön kol hareket etmeli, üst kol sabit kalmalı."
    },
    'dumbbell_rows': {
        1: "🎉 Mükemmel row! Sırt kasların aktif, form harika. Devam et!",
        2: "💪 Çok iyi! Gövde sabit, kürek kemikleri sıkılıyor. İyi gidiyorsun!",
        3: "👍 İyi form, gövdeni biraz daha sabit tut. Sallanmayı azalt.",
        4: "✅ İyi gidiyorsun, dirseği vücuda daha yakın çek. Daha yakın tut.",
        5: "⚠️ Orta seviye, sırtını düz tut, eğilme. Gövdeni sabitle.",
        6: "🔴 Gövdeni sabitle, sırtını düz tut. Öne çok eğilme, düz kal.",
        7: "🔴 Kollarına odaklan: dirseği vücuda yakın çek. Daha yakın tut.",
        8: "🔴 Kafanı nötr tut, boynunu eğme. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: sırt düzgünlüğü ve dirsek pozisyonuna dikkat. Yavaşla.",
        10: "🟡 Daha geriye çek, kürek kemiklerini sıkıştır. Tam menzil kullan.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Sırtını düz tut, fazla kavisli olmasın! Omurganı nötr tut."
    },
    'dumbbell_shoulder_press': {
        1: "🎉 Mükemmel shoulder press! Core aktif, form harika. Devam et!",
        2: "💪 Çok iyi! Kollar tam yukarı, gövde sabit. İyi gidiyorsun!",
        3: "👍 İyi form, kolları biraz daha tam yukarı it. Tam aç.",
        4: "✅ İyi gidiyorsun, gövdeni daha sabit tut. Core'unu sık.",
        5: "⚠️ Orta seviye, core'unu sık, sırtına yaslanma. Dikey dur.",
        6: "🔴 Kollarına odaklan: tam yukarı it, tam aç. Yeterince yukarı çıkmıyor.",
        7: "🔴 Gövdeni sabitle, core'unu sık. Sallanmayı azalt.",
        8: "🔴 Kafanı nötr tut, yukarı bakma. İleri bak, boynunu rahatlat.",
        9: "🔴 Birkaç sorun var: core stabilitesi ve kol hareketi düzgünlüğüne dikkat. Yavaşla.",
        10: "🟡 Kolları tam yukarı it, tam aç. Tam hareket menzili kullan.",
        11: "🟡 Kontrolü artır, yavaş ve kontrollü hareket et. Acele etme.",
        12: "🔴 Arkaya yaslanma! Gövdeni dik tut, core'unu sık. Öne eğilme."
    }
}


