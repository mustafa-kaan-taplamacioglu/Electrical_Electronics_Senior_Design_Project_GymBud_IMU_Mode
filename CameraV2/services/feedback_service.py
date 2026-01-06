"""Feedback service for exercise feedback generation."""

import random

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
    'tricep_extensions': {
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


def select_feedback_category(
    exercise: str,
    score: float,
    regional_scores: dict,
    regional_issues: dict,
    min_angle: float = None,
    max_angle: float = None,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary'  # 'camera_only', 'imu_only', 'camera_primary'
) -> int:
    """
    Select appropriate feedback category (1-12) based on ML predictions, scores, IMU data, and landmarks.
    Supports Camera-only, IMU-only, and Sensor Fusion modes.
    
    Args:
        exercise: Exercise name
        score: Overall form score (0-100)
        regional_scores: Dict with regional scores {'arms': float, 'legs': float, ...}
        regional_issues: Dict with regional issues {'arms': [str, ...], ...}
        min_angle: Minimum angle during rep
        max_angle: Maximum angle during rep
        ml_prediction: ML model prediction dict (regional scores) - from camera or fusion
        imu_data: IMU data dict (left_wrist, right_wrist, chest)
        landmarks: Raw landmark data (list of 33 landmarks)
        initial_positions: Calibration initial positions
        fusion_mode: 'camera_only', 'imu_only', or 'camera_primary' (sensor fusion)
    
    Returns:
        Feedback category ID (1-12)
    """
    # Use ML prediction if available (preferred - works for all modes)
    if ml_prediction and isinstance(ml_prediction, dict) and 'arms' in ml_prediction:
        score = sum(ml_prediction.values()) / len(ml_prediction)
        regional_scores = ml_prediction
    
    # Calculate range of motion
    rom = None
    if min_angle is not None and max_angle is not None:
        rom = max_angle - min_angle
    
    # Expected ROM ranges per exercise
    expected_rom = {
        'bicep_curls': (90, 120),
        'squats': (70, 100),
        'lateral_shoulder_raises': (50, 80),
        'tricep_extensions': (80, 140),
        'dumbbell_rows': (80, 110),
        'dumbbell_shoulder_press': (60, 90)
    }
    
    # Category 1: Perfect Form (Score >=95, no issues)
    if score >= 95 and not any(regional_issues.values() if regional_issues else []):
        return 1
    
    # Category 2: Excellent Form (Score 90-94)
    if score >= 90:
        return 2
    
    # Category 3: Good - Minor Issues (Score 85-89)
    if score >= 85:
        return 3
    
    # Category 4: Good - Needs Improvement (Score 80-84)
    if score >= 80:
        return 4
    
    # Category 5: Moderate Form (Score 70-79)
    if score >= 70:
        return 5
    
    # Category 11: Range Too Limited (works for all modes - uses angle data)
    if rom is not None and expected_rom.get(exercise):
        exp_min, exp_max = expected_rom[exercise]
        if rom < exp_min * 0.8:
            return 11
    
    # Category 12: Range Too Wide or specific landmark-based issues
    if rom is not None and expected_rom.get(exercise):
        exp_min, exp_max = expected_rom[exercise]
        if rom > exp_max * 1.2:
            return 12
    
    # ✅ Landmark-based checks (Camera mode or Fusion mode)
    if (fusion_mode in ['camera_only', 'camera_primary']) and landmarks and initial_positions:
        lm = {i: {'x': landmarks[i]['x'], 'y': landmarks[i]['y']} for i in range(min(len(landmarks), 33))}
        
        # Bicep curls - elbow above shoulder check
        if exercise == 'bicep_curls' and 13 < len(landmarks) and 'left_elbow' in initial_positions:
            left_elbow_current = lm.get(13, {})
            left_elbow_init = initial_positions.get('left_elbow', {})
            if left_elbow_current.get('x') and left_elbow_init.get('x'):
                elbow_drift = abs(left_elbow_current['x'] - left_elbow_init['x'])
                if elbow_drift > 0.1:
                    return 6  # Arms issue
        
        # Squats - knee valgus check
        if exercise == 'squats' and len(landmarks) >= 28:
            left_knee_x = lm.get(25, {}).get('x', 0)
            left_ankle_x = lm.get(27, {}).get('x', 0)
            right_knee_x = lm.get(26, {}).get('x', 0)
            right_ankle_x = lm.get(28, {}).get('x', 0)
            
            knee_width = abs(right_knee_x - left_knee_x)
            ankle_width = abs(right_ankle_x - left_ankle_x)
            
            if ankle_width > 0 and knee_width < ankle_width * 0.8:
                return 12  # Knee valgus
    
    # ✅ IMU-based checks (IMU mode or Fusion mode)
    if (fusion_mode in ['imu_only', 'camera_primary']) and imu_data:
        # Check for excessive wrist movement (bicep curls, triceps pushdown)
        if exercise in ['bicep_curls', 'tricep_extensions']:
            left_wrist = imu_data.get('left_wrist', {})
            right_wrist = imu_data.get('right_wrist', {})
            
            # Check gyroscope magnitude (indicates movement)
            if left_wrist and right_wrist:
                left_gyro = left_wrist.get('gyro', {})
                right_gyro = right_wrist.get('gyro', {})
                
                if left_gyro and right_gyro:
                    left_mag = (left_gyro.get('x', 0)**2 + left_gyro.get('y', 0)**2 + left_gyro.get('z', 0)**2)**0.5
                    right_mag = (right_gyro.get('x', 0)**2 + right_gyro.get('y', 0)**2 + right_gyro.get('z', 0)**2)**0.5
                    
                    # High gyro magnitude indicates excessive movement
                    if left_mag > 500 or right_mag > 500:  # Threshold in deg/s
                        return 6  # Arms issue - too much movement
    
    # Category 10: Multiple Issues (3+ issues across regions)
    total_issues = sum(len(issues) for issues in (regional_issues.values() if regional_issues else []))
    if total_issues >= 3:
        return 10
    
    # Category 6-9: Poor Form - Region-specific (Score <70, find lowest region)
    if score < 70 and regional_scores:
        min_region = min(regional_scores.items(), key=lambda x: x[1])
        region_name = min_region[0]
        
        region_to_category = {
            'arms': 6,
            'legs': 7,
            'core': 8,
            'head': 9
        }
        return region_to_category.get(region_name, 6)
    
    # Default: Category 5 (Moderate)
    return 5


def get_smart_feedback(
    exercise: str,
    score: float,
    regional_scores: dict,
    regional_issues: dict,
    min_angle: float = None,
    max_angle: float = None,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary',
    rep_num: int = 0
) -> str:
    """
    Get smart feedback using ML predictions, IMU data, and landmark analysis.
    Supports Camera-only, IMU-only, and Sensor Fusion modes.
    
    Args:
        exercise: Exercise name
        score: Overall form score
        regional_scores: Regional scores dict
        regional_issues: Regional issues dict
        min_angle: Min angle
        max_angle: Max angle
        ml_prediction: ML model prediction (regional scores)
        imu_data: IMU data (left_wrist, right_wrist, chest)
        landmarks: Raw landmark data
        initial_positions: Calibration initial positions
        fusion_mode: 'camera_only', 'imu_only', or 'camera_primary'
        rep_num: Rep number
    
    Returns:
        Feedback message string
    """
    # Select feedback category based on mode
    category = select_feedback_category(
        exercise, score, regional_scores, regional_issues,
        min_angle, max_angle, ml_prediction, imu_data,
        landmarks=landmarks,
        initial_positions=initial_positions,
        fusion_mode=fusion_mode
    )
    
    # Get feedback from library
    feedback_lib = EXERCISE_FEEDBACK_LIBRARY.get(exercise, {})
    feedback = feedback_lib.get(category, "Formunu iyileştirmeye devam et.")
    
    if rep_num > 0:
        return f"Rep #{rep_num}: {feedback}"
    
    return feedback


def get_rule_based_regional_feedback(
    exercise: str,
    region: str,
    region_score: float,
    region_issues: list,
    rep_num: int,
    min_angle: float = None,
    max_angle: float = None
) -> str:
    """Get rule-based feedback for a specific body region using MediaPipe data."""
    region_names = {
        'arms': 'Kollar',
        'legs': 'Bacaklar',
        'core': 'Gövde',
        'head': 'Kafa'
    }
    
    region_name_tr = region_names.get(region, region)
    
    # If score is high, give positive feedback
    if region_score >= 85:
        if region == 'arms':
            return f"💪 Kollar mükemmel! Form çok iyi."
        elif region == 'legs':
            return f"🦵 Bacaklar mükemmel! Form çok iyi."
        elif region == 'core':
            return f"✅ Gövde mükemmel! Duruş çok iyi."
        elif region == 'head':
            return f"👍 Kafa pozisyonu mükemmel!"
        else:
            return f"{region_name_tr} mükemmel! Skor: %{region_score:.0f}"
    
    # If there are specific issues, provide targeted feedback
    if region_issues:
        # Exercise-specific feedback based on issues
        issue_lower = region_issues[0].lower()
        
        # Arms feedback
        if region == 'arms':
            if 'dirsek' in issue_lower or 'elbow' in issue_lower or 'oynuyor' in issue_lower:
                if 'sol' in issue_lower or 'left' in issue_lower:
                    return "Sol dirseğini gövdene sabitle, daha az oynatmalısın."
                elif 'sağ' in issue_lower or 'right' in issue_lower:
                    return "Sağ dirseğini gövdene sabitle, daha az oynatmalısın."
                else:
                    return "Dirseklerini sabit tutmalısın, gövdene yakın tut."
            elif 'kol' in issue_lower and 'esit' in issue_lower:
                return "Kollarını eşit yüksekliğe getirmelisin, simetrik hareket et."
            elif 'uzat' in issue_lower or 'extend' in issue_lower:
                return "Kollarını daha fazla uzatmalısın, tam hareket menzili kullan."
            elif 'bük' in issue_lower or 'curl' in issue_lower:
                return "Kollarını daha fazla bük, hareket menzilini artır."
            else:
                return f"Kollar: {region_issues[0]}"
        
        # Legs feedback
        elif region == 'legs':
            if 'diz' in issue_lower or 'knee' in issue_lower:
                if 'içe' in issue_lower or 'valgus' in issue_lower:
                    return "Dizlerini ayak parmaklarınla hizalı tut, içe düşmesin."
                elif 'öne' in issue_lower or 'forward' in issue_lower:
                    return "Dizlerini ayak bileklerinin üzerinde tut, çok öne çıkmasın."
                else:
                    return "Diz pozisyonuna dikkat et, doğru açıda tut."
            elif 'duruş' in issue_lower or 'genişlik' in issue_lower:
                return "Bacaklarını omuz genişliğinde tut, daha dengeli dur."
            elif 'derinlik' in issue_lower or 'depth' in issue_lower:
                return "Daha derin inmelisin, tam hareket menzili kullan."
            else:
                return f"Bacaklar: {region_issues[0]}"
        
        # Core feedback
        elif region == 'core':
            if 'gövde' in issue_lower or 'sırt' in issue_lower or 'omurga' in issue_lower:
                if 'düz' in issue_lower or 'straight' in issue_lower:
                    return "Gövdeni düz tut, omurganı nötr pozisyonda tut."
                elif 'kavis' in issue_lower or 'arch' in issue_lower:
                    return "Sırtını düzleştir, fazla kavisli olmasın."
                elif 'eğil' in issue_lower or 'lean' in issue_lower:
                    return "Gövdeni dikey tut, öne veya arkaya eğilme."
                else:
                    return "Gövdeni stabilize et, düz ve dengeli tut."
            elif 'pelvis' in issue_lower or 'kalça' in issue_lower:
                return "Kalça pozisyonunu kontrol et, pelvis nötr olsun."
            else:
                return f"Gövde: {region_issues[0]}"
        
        # Head feedback
        elif region == 'head':
            if 'öne' in issue_lower or 'forward' in issue_lower:
                return "Başını öne eğme, ileri bak."
            elif 'yukarı' in issue_lower or 'up' in issue_lower:
                return "Başını çok yukarı kaldırma, nötr pozisyonda tut."
            elif 'aşağı' in issue_lower or 'down' in issue_lower:
                return "Başını aşağı bakma, öne doğru bak."
            else:
                return f"Kafa: {region_issues[0]}"
    
    # Default feedback based on score range
    if region_score >= 70:
        return f"{region_name_tr} iyi (Skor: %{region_score:.0f}), küçük iyileştirmeler yapabilirsin."
    elif region_score >= 50:
        return f"{region_name_tr} orta (Skor: %{region_score:.0f}), formunu iyileştirmeye odaklan."
    else:
        return f"{region_name_tr} düşük (Skor: %{region_score:.0f}), formunu düzeltmeye öncelik ver."


async def get_regional_ai_feedback(
    exercise: str,
    region: str,
    region_score: float,
    region_issues: list,
    rep_num: int,
    min_angle: float = None,
    max_angle: float = None
) -> str:
    """Get AI feedback for a specific body region. Falls back to rule-based if OpenAI unavailable."""
    # Always use rule-based feedback (faster and more reliable)
    return get_rule_based_regional_feedback(
        exercise, region, region_score, region_issues, rep_num, min_angle, max_angle
    )
    


def get_rule_based_overall_feedback(
    exercise: str,
    rep_num: int,
    score: float,
    issues: list,
    regional_scores: dict = None,
    regional_issues: dict = None,
    min_angle: float = None,
    max_angle: float = None,
    is_valid: bool = True,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary'
) -> str:
    """Get rule-based overall feedback using MediaPipe data, ML predictions, IMU data, and landmarks."""
    if not is_valid:
        if issues:
            return f"Rep #{rep_num}: Geçersiz rep. {issues[0] if issues else 'Form hatası'}."
        return f"Rep #{rep_num}: Geçersiz rep, formunu düzelt."
    
    # Use smart feedback system (includes ML, IMU, and landmark data)
    return get_smart_feedback(
        exercise=exercise,
        score=score,
        regional_scores=regional_scores or {},
        regional_issues=regional_issues or {},
        min_angle=min_angle,
        max_angle=max_angle,
        ml_prediction=ml_prediction,
        imu_data=imu_data,
        landmarks=landmarks,
        initial_positions=initial_positions,
        fusion_mode=fusion_mode,
        rep_num=rep_num
    )


