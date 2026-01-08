"""IMU-only mode için form analyzer - sadece IMU verilerini kullanır."""

import numpy as np
from typing import Dict, List, Optional

class IMUFormAnalyzer:
    """
    IMU-only mode için form analyzer.
    Sadece IMU verilerini (left_wrist, right_wrist) kullanarak form skoru hesaplar.
    Camera landmark'larına ihtiyaç duymaz.
    """
    
    def __init__(self, exercise: str = 'bicep_curls'):
        self.exercise = exercise
        
        # Biceps curl bilimsel gerçekleri
        self.OPTIMAL_PITCH_RANGE = (120, 150)  # Derece - ideal ROM
        self.MIN_ACCEPTABLE_PITCH = 90  # Minimum kabul edilebilir ROM
        self.MAX_ROLL_TOLERANCE = 45  # Bilek rotasyonu toleransı
        self.MAX_YAW_TOLERANCE = 30  # Öne/geriye sapma toleransı
        self.OPTIMAL_TEMPO = (1.8, 2.5)  # Saniye - ideal tempo
        self.MAX_GYRO_MAGNITUDE = 500  # deg/s - çok hızlı hareket eşiği
    
    def analyze_bicep_curl_imu_only(
        self,
        imu_sequence: List[Dict],
        rep_duration: float = 0.0
    ) -> Dict:
        """
        IMU-only mode için biceps curl form analizi.
        
        Args:
            imu_sequence: IMU sample'ları listesi (left_wrist, right_wrist içerir)
            rep_duration: Rep süresi (saniye)
        
        Returns:
            Form skoru, bölgesel skorlar, sorunlar ve detaylı IMU analizi
        """
        if not imu_sequence or len(imu_sequence) < 5:
            return self._default_response()
        
        # IMU verilerini extract et
        left_wrist_data = []
        right_wrist_data = []
        
        for sample in imu_sequence:
            if sample.get('left_wrist'):
                left_wrist_data.append(sample['left_wrist'])
            if sample.get('right_wrist'):
                right_wrist_data.append(sample['right_wrist'])
        
        if not left_wrist_data and not right_wrist_data:
            return self._default_response()
        
        # Analiz sonuçları
        issues = []
        scores = []
        arms_issues = []
        arms_scores = []
        
        imu_analysis = {
            'left_wrist': {},
            'right_wrist': {},
            'bilateral_symmetry': {},
            'movement_quality': {}
        }
        
        # === SOL BİLEK (LW) ANALİZİ ===
        if left_wrist_data:
            lw_analysis = self._analyze_wrist(
                wrist_data=left_wrist_data,
                wrist_name='Sol Bilek',
                wrist_side='left'
            )
            imu_analysis['left_wrist'] = lw_analysis['analysis']
            arms_issues.extend(lw_analysis['issues'])
            arms_scores.extend(lw_analysis['scores'])
        
        # === SAĞ BİLEK (RW) ANALİZİ ===
        if right_wrist_data:
            rw_analysis = self._analyze_wrist(
                wrist_data=right_wrist_data,
                wrist_name='Sağ Bilek',
                wrist_side='right'
            )
            imu_analysis['right_wrist'] = rw_analysis['analysis']
            arms_issues.extend(rw_analysis['issues'])
            arms_scores.extend(rw_analysis['scores'])
        
        # === BİLATERAL SİMETRİ ANALİZİ ===
        if left_wrist_data and right_wrist_data:
            symmetry_analysis = self._analyze_bilateral_symmetry(
                left_wrist_data=left_wrist_data,
                right_wrist_data=right_wrist_data
            )
            imu_analysis['bilateral_symmetry'] = symmetry_analysis['analysis']
            if symmetry_analysis['issues']:
                arms_issues.extend(symmetry_analysis['issues'])
                arms_scores.append(symmetry_analysis['score'])
        
        # === TEMPO ANALİZİ ===
        if rep_duration > 0:
            tempo_analysis = self._analyze_tempo(rep_duration)
            imu_analysis['movement_quality']['tempo'] = tempo_analysis
            if tempo_analysis['status'] == 'too_fast':
                arms_issues.append('Çok hızlı - yavaşla! 2-3 saniye tempo ideal.')
                arms_scores.append(50)  # 70 → 50 (daha sert penalty)
            elif tempo_analysis['status'] == 'too_slow':
                arms_issues.append('Biraz hızlandırabilirsin. 2-3 saniye tempo optimal.')
                arms_scores.append(55)  # 75 → 55 (daha sert penalty)
            else:
                arms_scores.append(95)  # Ideal tempo bonus
        
        # === FINAL SKOR HESAPLAMA ===
        # Weighted average yerine daha agresif hesaplama
        if arms_scores:
            # Pitch range skorları daha ağırlıklı (ROM en önemli)
            pitch_scores = [s for s in arms_scores if s > 0]  # Sadece pozitif skorlar
            if pitch_scores:
                # Minimum skorları düşür
                min_score = min(pitch_scores)
                if min_score < 60:  # Kötü pitch range varsa
                    arms_score = sum(arms_scores) / len(arms_scores) * 0.8  # %20 penalty
                else:
                    arms_score = sum(arms_scores) / len(arms_scores)
            else:
                arms_score = 50  # Default daha düşük (70 → 50)
        else:
            arms_score = 50  # Default daha düşük (70 → 50)
        
        # IMU-only mode için sadece arms score kullanılır (diğer bölgeler N/A)
        final_score = arms_score
        
        # Daha agresif penalty'ler
        critical_issues = [i for i in arms_issues if any(word in i.lower() for word in ['çok', 'fazla', 'yetersiz', 'kısıtlı'])]
        if critical_issues:
            # Her kritik issue için %10 penalty
            penalty = len(critical_issues) * 0.10
            final_score = final_score * (1 - penalty)
            final_score = max(20, final_score)  # Minimum 20'ye düşebilir
        
        # Çok kötü form için cap
        if any('çok' in issue.lower() and ('dar' in issue.lower() or 'kısıtlı' in issue.lower()) for issue in arms_issues):
            final_score = min(final_score, 45)  # Max 45'e düşür
        
        if any('fazla' in issue.lower() and ('dönüş' in issue.lower() or 'sapma' in issue.lower()) for issue in arms_issues):
            final_score = min(final_score, 50)  # Max 50'ye düşür
        
        return {
            'score': round(final_score, 1),
            'issues': list(set(arms_issues)),  # Duplicate'leri kaldır
            'regional_scores': {
                'arms': round(arms_score, 1),
                'legs': 100.0,  # IMU-only mode'da legs N/A
                'core': 85.0,   # IMU-only mode'da core tahmini
                'head': 90.0   # IMU-only mode'da head N/A
            },
            'regional_issues': {
                'arms': arms_issues,
                'legs': [],
                'core': [],
                'head': []
            },
            'imu_analysis': imu_analysis  # Detaylı IMU analizi
        }
    
    def _analyze_wrist(
        self,
        wrist_data: List[Dict],
        wrist_name: str,
        wrist_side: str
    ) -> Dict:
        """Tek bir bilek için detaylı analiz."""
        issues = []
        scores = []
        analysis = {}
        
        # Extract pitch, roll, yaw, accelerometer, gyroscope values
        pitches = [w.get('pitch', 0) for w in wrist_data if w.get('pitch') is not None]
        rolls = [w.get('roll', 0) for w in wrist_data if w.get('roll') is not None]
        yaws = [w.get('yaw', 0) for w in wrist_data if w.get('yaw') is not None]
        ax_values = [w.get('ax', 0) for w in wrist_data if w.get('ax') is not None]
        ay_values = [w.get('ay', 0) for w in wrist_data if w.get('ay') is not None]
        az_values = [w.get('az', 0) for w in wrist_data if w.get('az') is not None]
        gx_values = [w.get('gx', 0) for w in wrist_data if w.get('gx') is not None]
        gy_values = [w.get('gy', 0) for w in wrist_data if w.get('gy') is not None]
        gz_values = [w.get('gz', 0) for w in wrist_data if w.get('gz') is not None]
        
        if not pitches:
            return {'analysis': {}, 'issues': [], 'scores': []}
        
        # 1. PITCH RANGE ANALİZİ (Yukarı/Aşağı Hareket) - ANA METRİK
        pitch_min = min(pitches)
        pitch_max = max(pitches)
        pitch_range = pitch_max - pitch_min
        
        # Normalize pitch: Biceps curl için 0° (aşağı) → 180° (yukarı)
        # Gerçekte: pitch değerleri -90° ile +90° arasında olabilir
        # Biceps curl için: pitch artışı = yukarı hareket
        pitch_range_normalized = abs(pitch_range)
        
        if pitch_range_normalized >= self.OPTIMAL_PITCH_RANGE[0]:
            if pitch_range_normalized >= self.OPTIMAL_PITCH_RANGE[1]:
                analysis['pitch_status'] = 'excellent'
                analysis['pitch_feedback'] = f'{wrist_name}: {pitch_range_normalized:.0f}° ROM (Mükemmel! Tam ROM)'
                scores.append(100)
            else:
                analysis['pitch_status'] = 'good'
                analysis['pitch_feedback'] = f'{wrist_name}: {pitch_range_normalized:.0f}° ROM (İyi - Optimal aralıkta)'
                scores.append(90)
        elif pitch_range_normalized >= self.MIN_ACCEPTABLE_PITCH:
            analysis['pitch_status'] = 'moderate'
            analysis['pitch_feedback'] = f'{wrist_name}: {pitch_range_normalized:.0f}° ROM (Orta - Daha geniş açı hedefle)'
            issues.append(f'{wrist_name} hareket açısı dar - daha geniş açı kullan')
            scores.append(65)  # 75 → 65 (daha sert penalty)
        else:
            analysis['pitch_status'] = 'insufficient'
            analysis['pitch_feedback'] = f'{wrist_name}: {pitch_range_normalized:.0f}° ROM (Yetersiz - Tam açı kullan)'
            issues.append(f'{wrist_name} tam açılmamış - daha aşağı indir')
            scores.append(35)  # 55 → 35 (çok daha sert penalty)
        
        analysis['pitch_range'] = round(pitch_range_normalized, 1)
        analysis['pitch_min'] = round(pitch_min, 1)
        analysis['pitch_max'] = round(pitch_max, 1)
        
        # 2. ROLL RANGE ANALİZİ (Sağa/Sola Rotasyon) - Bilek Stabilitesi
        if rolls:
            roll_min = min(rolls)
            roll_max = max(rolls)
            roll_range = abs(roll_max - roll_min)
            roll_abs = max(abs(roll_min), abs(roll_max))
            
            if roll_range > self.MAX_ROLL_TOLERANCE:
                analysis['roll_status'] = 'excessive'
                analysis['roll_feedback'] = f'{wrist_name} bilek: {roll_range:.0f}° dönüş (Fazla - Nötr tut)'
                issues.append(f'{wrist_name} bilek fazla dönüyor - sabit tut')
                scores.append(max(30, 100 - roll_range))  # 50 → 30 (daha sert penalty)
            elif roll_range > 25:
                analysis['roll_status'] = 'moderate'
                analysis['roll_feedback'] = f'{wrist_name} bilek: {roll_range:.0f}° dönüş (Orta)'
                scores.append(65)  # 75 → 65 (daha sert penalty)
            else:
                analysis['roll_status'] = 'good'
                analysis['roll_feedback'] = f'{wrist_name} bilek: {roll_range:.0f}° dönüş (İyi - Stabil)'
                scores.append(90)
            
            analysis['roll_range'] = round(roll_range, 1)
        
        # 3. YAW RANGE ANALİZİ (Öne/Geriye Sapma)
        if yaws:
            yaw_min = min(yaws)
            yaw_max = max(yaws)
            yaw_range = abs(yaw_max - yaw_min)
            yaw_abs = max(abs(yaw_min), abs(yaw_max))
            
            if yaw_range > self.MAX_YAW_TOLERANCE:
                analysis['yaw_status'] = 'excessive'
                analysis['yaw_feedback'] = f'{wrist_name}: {yaw_range:.0f}° öne/geriye sapma (Fazla)'
                issues.append(f'{wrist_name} öne/geriye sapıyor - düz tut')
                scores.append(max(40, 100 - yaw_range))  # 60 → 40 (daha sert penalty)
            else:
                analysis['yaw_status'] = 'good'
                analysis['yaw_feedback'] = f'{wrist_name}: {yaw_range:.0f}° sapma (İyi)'
                scores.append(90)
            
            analysis['yaw_range'] = round(yaw_range, 1)
        
        # 4. ACCELEROMETER ANALİZİ (Hareket Kalitesi)
        if ay_values:
            ay_abs = [abs(a) for a in ay_values]
            ay_max = max(ay_abs) if ay_abs else 0
            ay_avg = np.mean(ay_abs) if ay_abs else 0
            
            if ay_max > 15:  # m/s²
                analysis['accel_status'] = 'fast'
                analysis['accel_feedback'] = f'{wrist_name}: {ay_max:.1f}m/s² max ivme (Hızlı)'
            elif ay_max > 8:
                analysis['accel_status'] = 'moderate'
                analysis['accel_feedback'] = f'{wrist_name}: {ay_max:.1f}m/s² max ivme (Orta)'
            else:
                analysis['accel_status'] = 'slow'
                analysis['accel_feedback'] = f'{wrist_name}: {ay_max:.1f}m/s² max ivme (Yavaş)'
            
            analysis['accel_max'] = round(ay_max, 1)
            analysis['accel_avg'] = round(ay_avg, 1)
        
        # 5. GYROSCOPE ANALİZİ (Açısal Hız - Hareket Kontrolü)
        if gx_values and gy_values and gz_values:
            gyro_magnitudes = []
            for i in range(min(len(gx_values), len(gy_values), len(gz_values))):
                mag = (gx_values[i]**2 + gy_values[i]**2 + gz_values[i]**2)**0.5
                gyro_magnitudes.append(mag)
            
            if gyro_magnitudes:
                gyro_max = max(gyro_magnitudes)
                gyro_avg = np.mean(gyro_magnitudes)
                
                if gyro_max > self.MAX_GYRO_MAGNITUDE:
                    analysis['gyro_status'] = 'too_fast'
                    analysis['gyro_feedback'] = f'{wrist_name}: {gyro_max:.0f}°/s max (Çok hızlı - Yavaşla!)'
                    issues.append(f'{wrist_name} çok hızlı hareket ediyor')
                    scores.append(45)  # 65 → 45 (daha sert penalty)
                elif gyro_max > 300:
                    analysis['gyro_status'] = 'fast'
                    analysis['gyro_feedback'] = f'{wrist_name}: {gyro_max:.0f}°/s max (Hızlı)'
                    scores.append(70)  # 80 → 70 (daha sert penalty)
                else:
                    analysis['gyro_status'] = 'controlled'
                    analysis['gyro_feedback'] = f'{wrist_name}: {gyro_max:.0f}°/s max (Kontrollü)'
                    scores.append(95)
                
                analysis['gyro_max'] = round(gyro_max, 1)
                analysis['gyro_avg'] = round(gyro_avg, 1)
        
        return {
            'analysis': analysis,
            'issues': issues,
            'scores': scores
        }
    
    def _analyze_bilateral_symmetry(
        self,
        left_wrist_data: List[Dict],
        right_wrist_data: List[Dict]
    ) -> Dict:
        """Bilateral simetri analizi."""
        issues = []
        analysis = {}
        score = 70  # Default (85 → 70)
        
        # Pitch range karşılaştırması
        lw_pitches = [w.get('pitch', 0) for w in left_wrist_data if w.get('pitch') is not None]
        rw_pitches = [w.get('pitch', 0) for w in right_wrist_data if w.get('pitch') is not None]
        
        if lw_pitches and rw_pitches:
            lw_pitch_range = max(lw_pitches) - min(lw_pitches)
            rw_pitch_range = max(rw_pitches) - min(rw_pitches)
            
            pitch_diff = abs(lw_pitch_range - rw_pitch_range)
            pitch_diff_pct = (pitch_diff / max(lw_pitch_range, rw_pitch_range, 1)) * 100
            
            if pitch_diff_pct <= 5:
                analysis['status'] = 'excellent'
                analysis['feedback'] = f'Simetri: {pitch_diff_pct:.1f}% fark (Mükemmel! Dengeli kas gelişimi)'
                score = 100
            elif pitch_diff_pct <= 10:
                analysis['status'] = 'good'
                analysis['feedback'] = f'Simetri: {pitch_diff_pct:.1f}% fark (İyi - Küçük fark)'
                score = 90
            elif pitch_diff_pct <= 20:
                analysis['status'] = 'moderate'
                analysis['feedback'] = f'Simetri: {pitch_diff_pct:.1f}% fark (Orta - Eşitlemeye çalış)'
                issues.append('Kollar asimetrik - eşit hareket et')
                score = 65  # 75 → 65 (daha sert penalty)
            else:
                weaker = 'sol' if lw_pitch_range < rw_pitch_range else 'sağ'
                analysis['status'] = 'poor'
                analysis['feedback'] = f'Simetri: {pitch_diff_pct:.1f}% fark ({weaker.capitalize()} kol zayıf - Odaklan!)'
                issues.append(f'{weaker.capitalize()} kol daha az hareket ediyor - eşitle')
                score = 45  # 60 → 45 (daha sert penalty)
            
            analysis['pitch_diff'] = round(pitch_diff, 1)
            analysis['pitch_diff_pct'] = round(pitch_diff_pct, 1)
            analysis['lw_pitch_range'] = round(lw_pitch_range, 1)
            analysis['rw_pitch_range'] = round(rw_pitch_range, 1)
        
        return {
            'analysis': analysis,
            'issues': issues,
            'score': score
        }
    
    def _analyze_tempo(self, rep_duration: float) -> Dict:
        """Tempo analizi."""
        if self.OPTIMAL_TEMPO[0] <= rep_duration <= self.OPTIMAL_TEMPO[1]:
            return {
                'status': 'optimal',
                'feedback': f'Tempo: {rep_duration:.1f}s (İdeal! TUT optimal)',
                'score': 100
            }
        elif rep_duration < 1.2:
            return {
                'status': 'too_fast',
                'feedback': f'Tempo: {rep_duration:.1f}s (Çok hızlı! 2-3s hedefle)',
                'score': 65
            }
        elif rep_duration > 3.5:
            return {
                'status': 'too_slow',
                'feedback': f'Tempo: {rep_duration:.1f}s (Yavaş - 2-3s optimal)',
                'score': 75
            }
        else:
            return {
                'status': 'acceptable',
                'feedback': f'Tempo: {rep_duration:.1f}s (Kabul edilebilir)',
                'score': 85
            }
    
    def _default_response(self) -> Dict:
        """Varsayılan yanıt (veri yoksa)."""
        return {
            'score': 70.0,
            'issues': [],
            'regional_scores': {'arms': 70.0, 'legs': 100.0, 'core': 85.0, 'head': 90.0},
            'regional_issues': {'arms': [], 'legs': [], 'core': [], 'head': []},
            'imu_analysis': {}
        }

