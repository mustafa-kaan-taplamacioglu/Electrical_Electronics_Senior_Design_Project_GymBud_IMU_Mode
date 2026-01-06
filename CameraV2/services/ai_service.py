"""AI service for OpenAI integration and feedback generation."""

from typing import Optional
from fastapi import WebSocket
from openai import OpenAI

# OpenAI client (will be set from api_server)
openai_client: Optional[OpenAI] = None

def init_openai_client(client: OpenAI):
    """Initialize OpenAI client from api_server."""
    global openai_client
    openai_client = client

from services.feedback_service import get_rule_based_regional_feedback

async def get_ai_feedback(
    exercise: str,
    rep_data: dict,
    issues: list,
    regional_scores: dict = None,
    regional_issues: dict = None,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary'
) -> dict:
    """Get technical and specific AI feedback based on rep quality data with regional breakdown.
    Uses OpenAI if available, otherwise falls back to rule-based feedback.
    Supports Camera-only, IMU-only, and Sensor Fusion modes.
    """
    rep_num = rep_data.get('rep', 0)
    score = rep_data.get('form_score', 0)
    min_angle = rep_data.get('min_angle', 0)
    max_angle = rep_data.get('max_angle', 0)
    is_valid = rep_data.get('is_valid', True)
    
    # Speed classification data from ensemble model
    speed_class = rep_data.get('speed_class', 'medium')
    speed_label = rep_data.get('speed_label', 'Orta Hız')
    rep_duration = rep_data.get('duration', 0)
    form_feedback = rep_data.get('form_feedback', '')
    rep_issues_from_detector = rep_data.get('issues', [])
    
    # Try OpenAI first (if available)
    if openai_client:
        try:
            # Build comprehensive prompt with all available data
            exercise_names = {
                'bicep_curls': 'Biceps Curl',
                'squats': 'Squat',
                'lunges': 'Lunge',
                'pushups': 'Push-up',
                'lateral_shoulder_raises': 'Lateral Shoulder Raise',
                'tricep_extensions': 'Triceps Extension',
                'dumbbell_rows': 'Dumbbell Row',
                'dumbbell_shoulder_press': 'Shoulder Press'
            }
            ex_name = exercise_names.get(exercise, exercise)
            
            # Combine issues from both sources
            all_issues = list(issues) + list(rep_issues_from_detector) if issues else list(rep_issues_from_detector)
            issues_text = ', '.join(all_issues) if all_issues else 'Yok'
            
            regional_info = ""
            if regional_scores:
                regional_info = f"\nRegional Scores:\n"
                for region, reg_score in regional_scores.items():
                    region_name = {'arms': 'Arms', 'legs': 'Legs', 'core': 'Core/Torso', 'head': 'Head/Neck'}.get(region, region)
                    region_issues_str = ', '.join(regional_issues.get(region, [])) if regional_issues else 'Yok'
                    regional_info += f"- {region_name}: {reg_score:.1f}% (Sorunlar: {region_issues_str})\n"
            
            angle_info = ""
            if min_angle and max_angle:
                angle_info = f"\nHareket Açısı: {min_angle:.1f}° - {max_angle:.1f}° (Aralık: {max_angle - min_angle:.1f}°)"
            
            speed_info = f"\nHız: {speed_label} ({rep_duration:.1f} saniye)" if rep_duration else ""
            
            # LW/RW pitch range info for wrist-based exercises (bicep curls, tricep extensions, etc.)
            lw_pitch = rep_data.get('lw_pitch_range', 0)
            rw_pitch = rep_data.get('rw_pitch_range', 0)
            lw_rw_info = ""
            scientific_context = ""
            
            if lw_pitch > 0 or rw_pitch > 0:
                lw_rw_info = f"\nKol Hareket Aralıkları (IMU): Sol: {lw_pitch:.0f}°, Sağ: {rw_pitch:.0f}°"
                if lw_pitch > 0 and rw_pitch > 0:
                    diff = abs(lw_pitch - rw_pitch)
                    avg_rom = (lw_pitch + rw_pitch) / 2
                    if diff > 20:
                        lw_rw_info += f" (⚠️ Fark: {diff:.0f}° - senkronizasyon gerekli!)"
                    else:
                        lw_rw_info += f" (✅ Senkron - fark: {diff:.0f}°)"
                    
                    # Tricep extensions specific scientific context
                    if exercise == 'tricep_extensions':
                        if avg_rom >= 160:
                            scientific_context = "\n🔬 Bilimsel Not: ROM mükemmel! Triceps'in 3 başı (lateral, long, medial) tam aktive oluyor."
                        elif avg_rom >= 140:
                            scientific_context = f"\n🔬 Bilimsel Not: ROM iyi ama tam extension için {160 - avg_rom:.0f}° daha aç. Triceps maksimum kasılma için 150-170° ideal."
                        elif avg_rom >= 100:
                            scientific_context = f"\n🔬 Bilimsel Not: ROM orta. Triceps tam aktivasyonu için {160 - avg_rom:.0f}° daha açılmalı (hedef: 150-170°)."
                        else:
                            scientific_context = f"\n🔬 Bilimsel Not: ROM dar! Triceps extension için kol neredeyse tam açılmalı. {160 - avg_rom:.0f}° daha açılmalı."
                        
                        # Tempo analysis
                        if rep_duration > 0:
                            if rep_duration < 0.8:
                                scientific_context += " ⚡ Çok hızlı! 1.5-2.0s TUT (Time Under Tension) kas hipertrofisi için optimal."
                            elif rep_duration > 3.0:
                                scientific_context += " 🐢 Çok yavaş. 1.5-2.0s tempo triceps için daha etkili."
                elif lw_pitch > 0:
                    lw_rw_info += f" (Sadece sol kol verisi mevcut)"
                elif rw_pitch > 0:
                    lw_rw_info += f" (Sadece sağ kol verisi mevcut)"
            
            # Exercise-specific system prompt
            if exercise == 'tricep_extensions':
                system_prompt = """You are a professional fitness coach and exercise physiologist specializing in triceps training. 
You provide scientifically-accurate, concise feedback in Turkish based on IMU sensor data.
You understand:
- Triceps brachii anatomy (lateral, long, medial heads)
- Optimal ROM for triceps activation (150-170°)
- Time Under Tension (TUT) principles (1.5-2.0s ideal)
- Bilateral symmetry importance
- Elbow stability biomechanics

Provide 1-2 sentences of actionable, evidence-based feedback."""
            else:
                system_prompt = 'You are a professional fitness coach. Provide concise, actionable feedback in Turkish.'
            
            prompt = f"""Sen uzman bir fitness koçusun ve {ex_name} hareketini analiz ediyorsun.

Rep #{rep_num} Analizi:
- Form Skoru: {score:.1f}%
- Geçerli Rep: {'Evet' if is_valid else 'Hayır'}
- Tespit Edilen Sorunlar: {issues_text}{speed_info}{lw_rw_info}{scientific_context}
{regional_info}{angle_info}

KISA, BİLİMSEL ve AKSİYON ALINACAK feedback ver (Türkçe):
1. Pozitif bir notla başla (skor düşük olsa bile)
2. IMU verilerini yorumla (ROM, tempo, bilateral symmetry)
3. Varsa en kritik sorunu belirt ve bilimsel düzeltme önerisi ver
4. Teşvik edici bir cümleyle bitir

2 cümleyi geçme. Samimi, destekleyici ve bilimsel ol."""

            response = openai_client.chat.completions.create(
                model='gpt-4o-mini',  # Faster and cheaper than gpt-4
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=200,
                temperature=0.7,
            )
            
            overall_feedback = response.choices[0].message.content.strip()
            
            # Get regional feedbacks using rule-based (faster for regions, OpenAI for overall)
            regional_feedbacks = {}
            if regional_scores and regional_issues:
                for region in ['arms', 'legs', 'core', 'head']:
                    region_score = regional_scores.get(region, 100)
                    region_issues_list = regional_issues.get(region, [])
                    regional_feedbacks[region] = get_rule_based_regional_feedback(
                        exercise, region, region_score, region_issues_list,
                        rep_num, min_angle, max_angle
                    )
            
            return {
                'overall': overall_feedback,
                'regional': regional_feedbacks
            }
        except Exception as e:
            print(f"⚠️  OpenAI feedback error: {e}, falling back to rule-based")
            # Fall through to rule-based feedback
    
    # Fallback: Rule-based feedback for bicep curls and other exercises
    score = rep_data.get('form_score', 0) or 0
    rep_num = rep_data.get('rep', 0)
    
    # Speed classification data
    speed_class = rep_data.get('speed_class', 'medium')
    speed_label = rep_data.get('speed_label', '')
    speed_emoji = rep_data.get('speed_emoji', '')
    rep_duration = rep_data.get('duration', 0) or 0
    form_feedback = rep_data.get('form_feedback', '')
    rep_issues_from_detector = rep_data.get('issues', [])
    
    # Generate comprehensive feedback based on score and speed
    speed_str = f" | {speed_emoji} {speed_label}" if speed_emoji and speed_label else ""
    duration_str = f" ({rep_duration:.1f}s)" if rep_duration > 0 else ""
    
    if score >= 85:
        overall = f"🎉 Mükemmel Rep #{rep_num}! Form: %{score:.0f}{speed_str}{duration_str}"
    elif score >= 70:
        overall = f"👍 İyi Rep #{rep_num}! Form: %{score:.0f}{speed_str}{duration_str}"
    elif score >= 50:
        overall = f"💪 Rep #{rep_num} tamamlandı. Form: %{score:.0f}{speed_str}{duration_str}"
    else:
        overall = f"⚠️ Rep #{rep_num} algılandı. Form: %{score:.0f}{speed_str}{duration_str}"
    
    # Add speed-based tips (exercise-specific)
    if exercise == 'lateral_shoulder_raises':
        if speed_class == 'very_fast':
            overall += " Çok hızlı! Lateral raise yavaş ve kontrollü yapılmalı."
        elif speed_class == 'very_slow':
            overall += " Biraz hızlandırabilirsin, ritmi koru."
    elif exercise == 'tricep_extensions':
        if speed_class == 'very_fast':
            overall += " Çok hızlı! Triceps extension kontrollü yapılmalı - kasları hisset."
        elif speed_class == 'very_slow':
            overall += " Biraz hızlandırabilirsin. 1.5-2 sn tempo ideal."
        elif speed_class == 'fast':
            overall += " Biraz yavaşla, tam extension için zaman tanı."
    elif exercise == 'squats':
        if speed_class == 'very_fast':
            overall += " Çok hızlı! Squat kontrollü yapılmalı - derin in ve yavaş çık."
        elif speed_class == 'very_slow':
            overall += " Biraz hızlandırabilirsin. 2-3 sn tempo ideal."
        elif speed_class == 'fast':
            overall += " Yavaşla! Derin squat için zaman tanı."
    else:
        if speed_class == 'very_fast':
            overall += " Çok hızlı yapıyorsun, biraz yavaşla!"
        elif speed_class == 'very_slow':
            overall += " Biraz hızlandırabilirsin."
    
    # Add issue-based tips from detector (exercise-specific)
    if rep_issues_from_detector:
        overall += " " + " ".join(rep_issues_from_detector[:2])  # First 2 issues
    elif issues:
        if exercise == 'lateral_shoulder_raises':
            if 'asymmetric' in str(issues).lower() or 'asimetrik' in str(issues).lower():
                overall += " Kolları eş zamanlı kaldır."
            elif 'range' in str(issues).lower() or 'açı' in str(issues).lower():
                overall += " Kolları omuz hizasına kadar kaldır."
            elif 'momentum' in str(issues).lower() or 'sallanma' in str(issues).lower():
                overall += " Gövdeyi sabit tut, sallanma."
            else:
                overall += " Kontrollü hareket et."
        elif exercise == 'tricep_extensions':
            if 'asymmetric' in str(issues).lower() or 'asimetrik' in str(issues).lower():
                overall += " Her iki kolu eşit aç."
            elif 'range' in str(issues).lower() or 'açı' in str(issues).lower():
                overall += " Kolu tam aç - triceps kasılsın!"
            elif 'elbow' in str(issues).lower() or 'dirsek' in str(issues).lower():
                overall += " Dirseği sabit tut, sadece ön kol hareket etsin."
            elif 'fast' in str(issues).lower() or 'hızlı' in str(issues).lower():
                overall += " Yavaşla! Kontrollü hareket tricepsi daha iyi çalıştırır."
            else:
                overall += " Kontrollü ve tam açılımla devam et."
        elif exercise == 'squats':
            if 'depth' in str(issues).lower() or 'derinlik' in str(issues).lower() or 'shallow' in str(issues).lower():
                overall += " Daha derin in! Kalça diz hizasına gelsin."
            elif 'balance' in str(issues).lower() or 'denge' in str(issues).lower():
                overall += " Dengeni koru, bir tarafa eğilme."
            elif 'knee' in str(issues).lower() or 'diz' in str(issues).lower():
                overall += " Dizleri ayak uçlarıyla aynı hizada tut."
            elif 'back' in str(issues).lower() or 'sırt' in str(issues).lower():
                overall += " Sırtını düz tut, öne eğilme."
            elif 'fast' in str(issues).lower() or 'hızlı' in str(issues).lower():
                overall += " Yavaşla! Kontrollü in, kontrollü çık."
            else:
                overall += " Derin ve kontrollü squat yap."
        elif 'elbow_moving' in str(issues).lower() or 'dirsek' in str(issues).lower():
            overall += " Dirseklerini vücuduna yakın tut."
        elif 'incomplete' in str(issues).lower() or 'eksik' in str(issues).lower():
            overall += " Hareketi tam kapsamda yap."
        elif 'fast' in str(issues).lower() or 'hızlı' in str(issues).lower():
            overall += " Daha yavaş ve kontrollü hareket et."
    
    # Add form feedback from ensemble model
    if form_feedback and score < 85:
        overall += f" {form_feedback}"
    
    # Regional feedback
    regional_feedbacks = {}
    if regional_scores:
        for region, reg_score in regional_scores.items():
            if reg_score >= 85:
                regional_feedbacks[region] = f"✅ {region.capitalize()}: Mükemmel form!"
            elif reg_score >= 70:
                regional_feedbacks[region] = f"👍 {region.capitalize()}: İyi, biraz iyileştir."
            else:
                regional_feedbacks[region] = f"⚠️ {region.capitalize()}: Dikkat, form düşük."
    
    # LW/RW specific feedback for bicep curls
    lw_pitch_range = rep_data.get('lw_pitch_range', 0)
    rw_pitch_range = rep_data.get('rw_pitch_range', 0)
    if lw_pitch_range > 0 or rw_pitch_range > 0:
        lw_rw_feedback = ""
        if lw_pitch_range > 0 and rw_pitch_range > 0:
            pitch_diff = abs(lw_pitch_range - rw_pitch_range)
            if pitch_diff > 20:
                if lw_pitch_range > rw_pitch_range:
                    lw_rw_feedback = f"⚠️ Sol kol daha aktif ({lw_pitch_range:.0f}° vs {rw_pitch_range:.0f}°). Sağ kolunu da eşit hareket ettir."
                else:
                    lw_rw_feedback = f"⚠️ Sağ kol daha aktif ({rw_pitch_range:.0f}° vs {lw_pitch_range:.0f}°). Sol kolunu da eşit hareket ettir."
            else:
                lw_rw_feedback = f"✅ Kollar senkron! Sol: {lw_pitch_range:.0f}°, Sağ: {rw_pitch_range:.0f}°"
        elif lw_pitch_range > 0:
            lw_rw_feedback = f"Sol kol hareket aralığı: {lw_pitch_range:.0f}°"
        elif rw_pitch_range > 0:
            lw_rw_feedback = f"Sağ kol hareket aralığı: {rw_pitch_range:.0f}°"
        
        regional_feedbacks['lw_rw'] = lw_rw_feedback
    
    return {
        'overall': overall,
        'regional': regional_feedbacks,
        'speed_class': speed_class,
        'speed_label': speed_label,
        'duration': rep_duration
    }



async def send_ai_feedback_async(
    websocket: WebSocket,
    exercise: str,
    rep_result: dict,
    issues: list,
    regional_scores: dict = None,
    regional_issues: dict = None,
    ml_prediction: dict = None,
    imu_data: dict = None,
    landmarks: list = None,
    initial_positions: dict = None,
    fusion_mode: str = 'camera_primary'
):
    """Send AI feedback asynchronously without blocking rep detection.
    Supports Camera-only, IMU-only, and Sensor Fusion modes.
    """
    try:
        feedback_data = await get_ai_feedback(
            exercise,
            rep_result,
            issues,
            regional_scores,
            regional_issues,
            ml_prediction=ml_prediction,
            imu_data=imu_data,
            landmarks=landmarks,
            initial_positions=initial_positions,
            fusion_mode=fusion_mode
        )
        
        # Send feedback as separate message
        if websocket.client_state.name == 'CONNECTED':
            if isinstance(feedback_data, dict):
                await websocket.send_json({
                    'type': 'rep_feedback',
                    'rep': rep_result.get('rep', 0),
                    'feedback': feedback_data.get('overall', ''),
                    'regional_feedback': feedback_data.get('regional', {})
                })
            else:
                await websocket.send_json({
                    'type': 'rep_feedback',
                    'rep': rep_result.get('rep', 0),
                    'feedback': feedback_data,
                    'regional_feedback': {}
                })
    except Exception as e:
        print(f"⚠️  Error sending async AI feedback: {e}")
        # Silently fail - feedback is optional




async def get_session_feedback(exercise: str, reps_data: list, all_issues: list) -> str:
    """Get comprehensive feedback at session end. Uses OpenAI if available, otherwise rule-based."""
    
    if not reps_data:
        return "Henüz rep tamamlanmadı. Devam et, daha uzun süre yapmaya çalış!"
    
    total_reps = len(reps_data)
    avg_score = sum(r['form_score'] for r in reps_data) / total_reps
    best_score = max(r['form_score'] for r in reps_data)
    worst_score = min(r['form_score'] for r in reps_data)
    
    # Find most common issues
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    top_issues = sorted(issue_counts.items(), key=lambda x: -x[1])[:3]
    
    # Exercise names
    exercise_names = {
        'bicep_curls': 'Biceps Curl',
        'squats': 'Squat',
        'lateral_shoulder_raises': 'Lateral Raise',
        'tricep_extensions': 'Triceps Extension',
        'dumbbell_rows': 'Dumbbell Row',
        'dumbbell_shoulder_press': 'Shoulder Press'
    }
    ex_name = exercise_names.get(exercise, exercise)
    
    # Try OpenAI first (if available)
    if openai_client:
        try:
            top_issues_text = ', '.join([f"{issue} ({count}x)" for issue, count in top_issues]) if top_issues else 'None'
            
            # Extract IMU data for tricep extensions
            imu_context = ""
            if exercise == 'tricep_extensions':
                lw_ranges = [r.get('lw_pitch_range', 0) for r in reps_data if r.get('lw_pitch_range', 0) > 0]
                rw_ranges = [r.get('rw_pitch_range', 0) for r in reps_data if r.get('rw_pitch_range', 0) > 0]
                durations = [r.get('duration', 0) for r in reps_data if r.get('duration', 0) > 0]
                
                avg_lw = sum(lw_ranges) / len(lw_ranges) if lw_ranges else 0
                avg_rw = sum(rw_ranges) / len(rw_ranges) if rw_ranges else 0
                avg_duration = sum(durations) / len(durations) if durations else 0
                avg_rom = (avg_lw + avg_rw) / 2 if avg_lw > 0 and avg_rw > 0 else max(avg_lw, avg_rw)
                
                symmetry_diff = abs(avg_lw - avg_rw) / max(avg_lw, avg_rw) * 100 if avg_lw > 0 and avg_rw > 0 else 0
                
                imu_context = f"""
📊 IMU SENSOR DATA ANALYSIS (Triceps Extension):
- Left Wrist ROM: {avg_lw:.1f}° (average pitch range)
- Right Wrist ROM: {avg_rw:.1f}° (average pitch range)
- Combined ROM: {avg_rom:.1f}° (ideal: 150-170° for full triceps activation)
- Bilateral Symmetry: {symmetry_diff:.1f}% difference (ideal: <10%)
- Average Rep Duration: {avg_duration:.2f}s (ideal: 1.5-2.0s for optimal TUT)
- Speed Classification: {'Too fast' if avg_duration < 0.8 else 'Fast' if avg_duration < 1.3 else 'Ideal' if avg_duration <= 2.0 else 'Slow' if avg_duration <= 3.0 else 'Too slow'}

🔬 SCIENTIFIC CONTEXT:
- Triceps brachii has 3 heads: lateral, long, and medial head
- Full extension (150-170° ROM) activates all 3 heads maximally
- Time Under Tension (TUT) of 1.5-2.0s optimizes muscle hypertrophy
- Bilateral symmetry prevents muscle imbalances
- Elbow stability is critical - only forearm should move, not shoulder
- Locking elbow fully can cause joint stress - slight bend (5-10°) is safer
"""
            
            prompt = f"""You are an expert fitness coach and exercise physiologist providing scientifically-based workout session feedback.

📊 WORKOUT SUMMARY ({ex_name}):
- Total Reps Completed: {total_reps}
- Average Form Score: {avg_score:.1f}%
- Best Rep Score: {best_score:.1f}%
- Worst Rep Score: {worst_score:.1f}%
- Most Common Issues: {top_issues_text}
{imu_context}

Provide comprehensive, scientifically-accurate feedback in Turkish:
1. Congratulate them for completing the workout
2. Analyze their performance using the IMU sensor data (ROM, tempo, bilateral symmetry)
3. Provide 2-3 specific, actionable improvement recommendations based on:
   - Range of Motion (ROM) analysis - triceps activation
   - Tempo/TUT (Time Under Tension) - muscle hypertrophy optimization
   - Bilateral symmetry - muscle balance
   - Form quality - elbow stability, full extension
4. Include scientific rationale (e.g., "Triceps'in 3 başı da tam aktive olması için...")
5. Motivating closing message

Keep it friendly, professional, scientifically accurate, and under 6-8 sentences. Focus on actionable, evidence-based advice."""

            system_prompt = f"""You are a professional fitness coach and exercise physiologist specializing in triceps training. 
You provide scientifically-accurate, evidence-based feedback in Turkish. 
You understand:
- Triceps brachii anatomy (lateral, long, medial heads)
- Optimal ROM for triceps activation (150-170°)
- Time Under Tension (TUT) principles for muscle hypertrophy
- Bilateral symmetry importance for muscle balance
- Biomechanics of triceps extension (elbow stability, forearm movement)

Always provide specific, actionable advice based on IMU sensor data analysis."""

            response = openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=400,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  OpenAI session feedback error: {e}, falling back to rule-based")
            # Fall through to rule-based feedback
    
    # Fallback: Build feedback based on performance (rule-based)
    feedback_parts = []
    
    # Opening
    if avg_score >= 85:
        feedback_parts.append(f"🎉 Harika iş! {total_reps} rep {ex_name} tamamladın!")
    elif avg_score >= 70:
        feedback_parts.append(f"👍 İyi gidiyorsun! {total_reps} rep {ex_name} tamamladın!")
    else:
        feedback_parts.append(f"💪 Tebrikler! {total_reps} rep {ex_name} tamamladın!")
    
    # Performance summary
    feedback_parts.append(f"Ortalama form skoru: %{avg_score:.0f}")
    if best_score >= 85:
        feedback_parts.append(f"En iyi rep: %{best_score:.0f} (Mükemmel!)")
    
    # BICEP CURL SPECIFIC SCIENTIFIC FEEDBACK
    if exercise == 'bicep_curls':
        # Analyze LW/RW pitch ranges from reps_data
        lw_ranges = [r.get('lw_pitch_range', 0) for r in reps_data if r.get('lw_pitch_range', 0) > 0]
        rw_ranges = [r.get('rw_pitch_range', 0) for r in reps_data if r.get('rw_pitch_range', 0) > 0]
        
        avg_lw = sum(lw_ranges) / len(lw_ranges) if lw_ranges else 0
        avg_rw = sum(rw_ranges) / len(rw_ranges) if rw_ranges else 0
        
        # Speed analysis
        durations = [r.get('duration', 0) for r in reps_data if r.get('duration', 0) > 0]
        avg_duration = sum(durations) / len(durations) if durations else 2.0
        
        # Scientific bicep curl feedback
        bicep_tips = []
        
        # 1. Range of Motion (ROM) Analysis - Scientific tip
        if avg_lw > 0 or avg_rw > 0:
            avg_rom = (avg_lw + avg_rw) / 2 if avg_lw > 0 and avg_rw > 0 else max(avg_lw, avg_rw)
            if avg_rom >= 120:
                bicep_tips.append("🎯 Hareket açısı mükemmel! Tam ROM (120°+) biceps kası için ideal.")
            elif avg_rom >= 90:
                bicep_tips.append("📐 Hareket açısı iyi. Daha geniş açı (120°+) için dirsekleri sabit tutarak tam aşağı indir.")
            else:
                bicep_tips.append("📏 Hareket açısı dar. Bilimsel olarak biceps curl için 120-150° açı optimal kas aktivasyonu sağlar.")
        
        # 2. Tempo/TUT (Time Under Tension) Analysis
        if avg_duration > 0:
            if 1.8 <= avg_duration <= 2.5:
                bicep_tips.append("⏱️ Tempo ideal! 2-2.5 saniye süre (TUT) kas hipertrofisi için optimal.")
            elif avg_duration < 1.2:
                bicep_tips.append("⚡ Tempo çok hızlı. Yavaşla! Araştırmalar 2-4 saniyelik konsentrik fazın kas gelişimi için daha etkili olduğunu gösteriyor.")
            elif avg_duration > 3.5:
                bicep_tips.append("🐢 Tempo yavaş. İyi kontrol, ama 2-3 sn hedefle - kas yorgunluğunu optimize eder.")
        
        # 3. Bilateral Symmetry (LW vs RW)
        if avg_lw > 0 and avg_rw > 0:
            diff_pct = abs(avg_lw - avg_rw) / max(avg_lw, avg_rw) * 100
            if diff_pct <= 10:
                bicep_tips.append("✅ Kollar simetrik çalışıyor! Bu dengesiz kas gelişimini önler.")
            elif diff_pct <= 20:
                weaker = "sol" if avg_lw < avg_rw else "sağ"
                bicep_tips.append(f"⚖️ {weaker.capitalize()} kol biraz daha az hareket ediyor (%{diff_pct:.0f} fark). Bilateral eşitlik için dikkat et.")
            else:
                weaker = "sol" if avg_lw < avg_rw else "sağ"
                bicep_tips.append(f"⚠️ {weaker.capitalize()} kol önemli ölçüde daha az hareket ediyor! Kas dengesizliğini önlemek için {weaker} koluna odaklan.")
        
        # Add bicep tips
        if bicep_tips:
            feedback_parts.append("\n\n🔬 Bilimsel Biceps Curl Analizi:")
            feedback_parts.extend(bicep_tips)
    
    # LATERAL SHOULDER RAISES SPECIFIC SCIENTIFIC FEEDBACK
    elif exercise == 'lateral_shoulder_raises':
        # Analyze LW/RW pitch ranges from reps_data
        lw_ranges = [r.get('lw_pitch_range', 0) for r in reps_data if r.get('lw_pitch_range', 0) > 0]
        rw_ranges = [r.get('rw_pitch_range', 0) for r in reps_data if r.get('rw_pitch_range', 0) > 0]
        
        avg_lw = sum(lw_ranges) / len(lw_ranges) if lw_ranges else 0
        avg_rw = sum(rw_ranges) / len(rw_ranges) if rw_ranges else 0
        
        # Speed analysis
        durations = [r.get('duration', 0) for r in reps_data if r.get('duration', 0) > 0]
        avg_duration = sum(durations) / len(durations) if durations else 1.7
        
        # Scientific lateral raise feedback
        lateral_tips = []
        
        # 1. Range of Motion (ROM) Analysis - Scientific tip
        avg_rom = 0
        if avg_lw > 0 or avg_rw > 0:
            avg_rom = (avg_lw + avg_rw) / 2 if avg_lw > 0 and avg_rw > 0 else max(avg_lw, avg_rw)
            if avg_rom >= 90:
                lateral_tips.append("🎯 Hareket açısı mükemmel! Omuz hizasına (90°+) ulaşıyorsun - lateral deltoid tam aktivasyonu!")
            elif avg_rom >= 70:
                lateral_tips.append("📐 Hareket açısı iyi. Omuz hizasına (90°) kadar kaldırmaya çalış, deltoid aktivasyonunu artırır.")
            else:
                lateral_tips.append("📏 Hareket açısı dar. Lateral raise için kolları en az omuz hizasına (90°) kaldır.")
        
        # 2. Tempo/TUT (Time Under Tension) Analysis
        if avg_duration > 0:
            if 1.5 <= avg_duration <= 2.5:
                lateral_tips.append("⏱️ Tempo ideal! 1.5-2.5 saniye lateral raise için deltoidleri optimal çalıştırıyor.")
            elif avg_duration < 1.0:
                lateral_tips.append("⚡ Çok hızlı! Yavaşla - momentum değil kas gücü kullan. 2-3 sn yukarı, 2-3 sn aşağı hedefle.")
            elif avg_duration > 3.5:
                lateral_tips.append("🐢 İyi kontrol ama biraz hızlandırabilirsin. 2-2.5 sn tempo deltoid hipertrofisi için optimal.")
        
        # 3. Bilateral Symmetry (LW vs RW)
        if avg_lw > 0 and avg_rw > 0:
            diff_pct = abs(avg_lw - avg_rw) / max(avg_lw, avg_rw) * 100
            if diff_pct <= 10:
                lateral_tips.append("✅ Her iki omuz da simetrik çalışıyor! Dengeli deltoid gelişimi için mükemmel.")
            elif diff_pct <= 20:
                weaker = "sol" if avg_lw < avg_rw else "sağ"
                lateral_tips.append(f"⚖️ {weaker.capitalize()} omuz biraz daha az hareket ediyor. Ayna karşısında simetri kontrolü yap.")
            else:
                weaker = "sol" if avg_lw < avg_rw else "sağ"
                lateral_tips.append(f"⚠️ {weaker.capitalize()} omuz belirgin şekilde daha az kalkıyor! Tek kol lateral raise ile {weaker} omuzu güçlendir.")
        
        # 4. Form Tips
        if avg_score < 75:
            lateral_tips.append("💡 Form İpucu: Dirsekleri hafif bükük tut, omuzları kulağa doğru kaldırma (trap yerine deltoid çalışsın).")
        elif avg_score < 85:
            lateral_tips.append("💡 Form İpucu: Gövdeyi sabit tut, sallanma momentum kullandığını gösterir.")
        else:
            lateral_tips.append("💡 Mükemmel teknik! Lateral deltoidler tam aktivasyonda.")
        
        # 5. Pitch range specific feedback
        if avg_rom > 0:
            if avg_rom >= 100:
                lateral_tips.append("🏆 Omuz hizasını aştın - dikkat: çok yüksekte trap kasları devreye girer, 90° civarı ideal.")
            elif avg_rom < 60:
                lateral_tips.append("📊 Hareket kısıtlı. Omuz mobilitesi sorun olabilir - ısınma ve stretching önerilir.")
        
        # Add lateral raise tips
        if lateral_tips:
            feedback_parts.append("\n\n🔬 Bilimsel Lateral Raise Analizi:")
            feedback_parts.extend(lateral_tips)
    
    # TRICEP EXTENSIONS SPECIFIC SCIENTIFIC FEEDBACK
    elif exercise == 'tricep_extensions':
        # Analyze LW/RW pitch ranges from reps_data
        lw_ranges = [r.get('lw_pitch_range', 0) for r in reps_data if r.get('lw_pitch_range', 0) > 0]
        rw_ranges = [r.get('rw_pitch_range', 0) for r in reps_data if r.get('rw_pitch_range', 0) > 0]
        
        avg_lw = sum(lw_ranges) / len(lw_ranges) if lw_ranges else 0
        avg_rw = sum(rw_ranges) / len(rw_ranges) if rw_ranges else 0
        
        # Speed analysis
        durations = [r.get('duration', 0) for r in reps_data if r.get('duration', 0) > 0]
        avg_duration = sum(durations) / len(durations) if durations else 1.7
        
        # Scientific tricep extension feedback
        tricep_tips = []
        
        # 1. Range of Motion (ROM) Analysis
        # Training data shows ideal ROM is 150-170° (arm nearly full extension)
        avg_rom = 0
        if avg_lw > 0 or avg_rw > 0:
            avg_rom = (avg_lw + avg_rw) / 2 if avg_lw > 0 and avg_rw > 0 else max(avg_lw, avg_rw)
            if avg_rom >= 160:
                tricep_tips.append("🎯 Hareket açısı mükemmel! Kol tam açılıyor - triceps maksimum kasılıyor.")
            elif avg_rom >= 140:
                tricep_tips.append("📐 İyi hareket açısı. Tam extension (160°+) için kolu biraz daha aç.")
            elif avg_rom >= 100:
                tricep_tips.append("📏 Hareket açısı orta. Triceps için kolu 150-170° açıya kadar tam aç.")
            else:
                tricep_tips.append("⚠️ Hareket açısı dar! Triceps extension için kol neredeyse tam açılmalı (150-170°).")
        
        # 2. Tempo/TUT (Time Under Tension) Analysis
        # Training data: Session 1 ~1.6s (medium), Session 2 ~1.0s (fast), Session 3 ~2.5s (slow)
        if avg_duration > 0:
            if 1.3 <= avg_duration <= 2.0:
                tricep_tips.append("⏱️ Tempo ideal! 1.5-2.0 saniye triceps extension için optimal TUT (Time Under Tension).")
            elif avg_duration < 0.8:
                tricep_tips.append("⚡ Çok hızlı! Triceps extension yavaş ve kontrollü yapılmalı. 1.5-2 sn hedefle.")
            elif avg_duration < 1.3:
                tricep_tips.append("⚡ Biraz hızlı. Yavaşlatarak tricepsi daha iyi kasabilirsin.")
            elif avg_duration > 3.0:
                tricep_tips.append("🐢 Çok yavaş. 1.5-2.0 sn tempo kas hipertrofisi için daha etkili.")
            else:
                tricep_tips.append("🐢 İyi kontrol. Tempo uygun ama biraz daha dinamik olabilir.")
        
        # 3. Bilateral Symmetry (LW vs RW)
        if avg_lw > 0 and avg_rw > 0:
            diff_pct = abs(avg_lw - avg_rw) / max(avg_lw, avg_rw) * 100
            if diff_pct <= 10:
                tricep_tips.append("✅ Her iki kol da simetrik çalışıyor! Dengeli triceps gelişimi için mükemmel.")
            elif diff_pct <= 20:
                weaker = "sol" if avg_lw < avg_rw else "sağ"
                tricep_tips.append(f"⚖️ {weaker.capitalize()} kol biraz daha az açılıyor. Tek kol triceps extension ile {weaker} kolu güçlendir.")
            else:
                weaker = "sol" if avg_lw < avg_rw else "sağ"
                tricep_tips.append(f"⚠️ {weaker.capitalize()} kol belirgin şekilde daha az açılıyor! Kas dengesizliği oluşabilir - {weaker} kola odaklan.")
        
        # 4. Form Tips based on score
        if avg_score < 70:
            tricep_tips.append("💡 Form İpucu: Dirseği sabit tut! Sadece ön kol hareket etmeli - omuzdan yardım alma.")
        elif avg_score < 80:
            tricep_tips.append("💡 Form İpucu: Kolu tam aç, en üst noktada 1 sn bekle - triceps maksimum kasılır.")
        elif avg_score < 90:
            tricep_tips.append("💡 İyi form! Dirseği tam kilitleme, hafif bükük tut - eklem sağlığı için.")
        else:
            tricep_tips.append("💡 Mükemmel teknik! Triceps tam aktivasyonda, bu formu koru.")
        
        # 5. Extension quality feedback
        if avg_rom > 0:
            if avg_rom >= 165:
                tricep_tips.append("🏆 Tam extension başarılı! Triceps'in lateral ve long head'i tam çalışıyor.")
            elif avg_rom < 120:
                tricep_tips.append("📊 Extension eksik. Triceps kasının tam kasılması için kol 160°+ açılmalı.")
        
        # Add tricep extension tips
        if tricep_tips:
            feedback_parts.append("\n\n🔬 Bilimsel Triceps Extension Analizi:")
            feedback_parts.extend(tricep_tips)
    
    # SQUATS SPECIFIC SCIENTIFIC FEEDBACK
    elif exercise == 'squats':
        # For squats, we use chest IMU or combined body sensors
        # ROM is measured differently - based on knee angle or body pitch
        
        # Speed analysis
        durations = [r.get('duration', 0) for r in reps_data if r.get('duration', 0) > 0]
        avg_duration = sum(durations) / len(durations) if durations else 2.5
        
        # ROM analysis (using pitch_range as depth indicator)
        rom_values = [r.get('pitch_range', 0) for r in reps_data if r.get('pitch_range', 0) > 0]
        avg_rom = sum(rom_values) / len(rom_values) if rom_values else 0
        
        # If no pitch_range, try lw/rw (though squats typically use chest sensor)
        if avg_rom == 0:
            lw_ranges = [r.get('lw_pitch_range', 0) for r in reps_data if r.get('lw_pitch_range', 0) > 0]
            rw_ranges = [r.get('rw_pitch_range', 0) for r in reps_data if r.get('rw_pitch_range', 0) > 0]
            if lw_ranges or rw_ranges:
                avg_rom = (sum(lw_ranges) / len(lw_ranges)) if lw_ranges else 0
                avg_rom = max(avg_rom, (sum(rw_ranges) / len(rw_ranges)) if rw_ranges else 0)
        
        # Scientific squat feedback
        squat_tips = []
        
        # 1. Depth Analysis (ROM)
        # Training data: ROM 102° - 127° (average ~116°)
        if avg_rom >= 115:
            squat_tips.append("🎯 Derinlik mükemmel! Paralel altına iniyorsun - glute ve quad tam aktivasyonda.")
        elif avg_rom >= 100:
            squat_tips.append("📐 İyi derinlik. Paralele (90°) ulaşıyorsun. Biraz daha derin inmeyi dene.")
        elif avg_rom >= 80:
            squat_tips.append("📏 Derinlik orta. En az kalça diz hizasına gelene kadar in (paralel squat).")
        elif avg_rom > 0:
            squat_tips.append("⚠️ Quarter squat - çok sığ! Derin squat için kalça diz hizasının altına inmeli.")
        
        # 2. Tempo Analysis
        # Training data: Session 1 ~2.9s, Session 2 ~1.5s, Session 3 ~3.2s
        if avg_duration > 0:
            if 2.0 <= avg_duration <= 3.0:
                squat_tips.append("⏱️ Tempo ideal! 2-3 saniye squat için kas aktivasyonu ve güvenlik açısından optimal.")
            elif avg_duration < 1.2:
                squat_tips.append("⚡ Çok hızlı! Squat kontrollü yapılmalı. 2-3 sn aşağı, 2-3 sn yukarı hedefle.")
            elif avg_duration < 2.0:
                squat_tips.append("⚡ Biraz hızlı. Yavaşlatarak kas aktivasyonunu artırabilirsin.")
            elif avg_duration > 4.0:
                squat_tips.append("🐢 Çok yavaş. 2-3 sn tempo kas yorgunluğunu optimize eder.")
            else:
                squat_tips.append("🐢 İyi kontrol. Yavaş ve kontrollü - eklem sağlığı için iyi.")
        
        # 3. Form Tips based on score
        if avg_score < 70:
            squat_tips.append("💡 Form İpucu: Sırtını düz tut! Bel çukurunu koru, öne eğilme.")
            squat_tips.append("💡 Dizlerin ayak uçlarıyla aynı yönde olmalı - içe çökmesin.")
        elif avg_score < 80:
            squat_tips.append("💡 Form İpucu: Core'u sık tut. Karını içe çek, sırt stabil kalsın.")
            squat_tips.append("💡 Topuklar yerden kalkmamalı - ayak tabanı tam yere basmalı.")
        elif avg_score < 90:
            squat_tips.append("💡 İyi form! Denge ve derinlik tutarlı. Bu şekilde devam et.")
        else:
            squat_tips.append("💡 Mükemmel teknik! Quad, glute ve core tam sinerji içinde çalışıyor.")
        
        # 4. Depth classification
        if avg_rom > 0:
            if avg_rom >= 120:
                squat_tips.append("🏆 Deep squat! Maksimum glute aktivasyonu ve mobilite. ATG (Ass To Grass) seviyesi!")
            elif avg_rom >= 100:
                squat_tips.append("✅ Below parallel! Glute ve quadriceps tam aktivasyonda.")
            elif avg_rom >= 80:
                squat_tips.append("📊 Parallel squat. İyi başlangıç, hedef daha derin.")
            else:
                squat_tips.append("⚠️ Yarım squat. Tam kas aktivasyonu için daha derin in.")
        
        # 5. Safety tips
        squat_tips.append("🛡️ Güvenlik: Dizler ayak uçlarını aşmamalı, sırt düz kalmalı.")
        
        # Add squat tips
        if squat_tips:
            feedback_parts.append("\n\n🔬 Bilimsel Squat Analizi:")
            feedback_parts.extend(squat_tips)
    
    # General improvement areas
    if top_issues:
        if len(top_issues) == 1:
            feedback_parts.append(f"\n📋 İyileştirme alanı: {top_issues[0][0]} ({top_issues[0][1]} kez tespit edildi).")
        else:
            issues_str = ", ".join([f"{issue} ({count}x)" for issue, count in top_issues[:2]])
            feedback_parts.append(f"\n📋 İyileştirme alanları: {issues_str}.")
    elif avg_score >= 80:
        feedback_parts.append("\nFormun çok iyi, devam et!")
    elif exercise != 'bicep_curls':  # Skip if bicep curl tips already given
        feedback_parts.append("\nFormunu iyileştirmeye devam et, yavaş ve kontrollü hareket et.")
    
    # Closing motivation
    if avg_score >= 85:
        feedback_parts.append("\n🏆 Harika çalışma, bu şekilde devam et! 💪")
    elif avg_score >= 70:
        feedback_parts.append("\n🎯 İyi performans, bir sonraki antrenmanda daha da iyileşeceksin!")
    else:
        feedback_parts.append("\n💪 İlk adımlar zor, ama devam ettiğin sürece ilerleyeceksin!")
    
    return " ".join(feedback_parts)
