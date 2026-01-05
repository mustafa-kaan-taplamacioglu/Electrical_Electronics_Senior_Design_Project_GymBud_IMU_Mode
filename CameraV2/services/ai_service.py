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
                'triceps_pushdown': 'Triceps Extension',
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
            
            # LW/RW pitch range info for bicep curls
            lw_pitch = rep_data.get('lw_pitch_range', 0)
            rw_pitch = rep_data.get('rw_pitch_range', 0)
            lw_rw_info = ""
            if lw_pitch > 0 or rw_pitch > 0:
                lw_rw_info = f"\nKol Hareket Aralıkları: Sol: {lw_pitch:.0f}°, Sağ: {rw_pitch:.0f}°"
                if lw_pitch > 0 and rw_pitch > 0:
                    diff = abs(lw_pitch - rw_pitch)
                    if diff > 20:
                        lw_rw_info += f" (⚠️ Fark: {diff:.0f}° - senkronizasyon gerekli!)"
                    else:
                        lw_rw_info += f" (✅ Senkron - fark: {diff:.0f}°)"
            
            prompt = f"""Sen uzman bir fitness koçusun ve {ex_name} hareketini analiz ediyorsun.

Rep #{rep_num} Analizi:
- Form Skoru: {score:.1f}%
- Geçerli Rep: {'Evet' if is_valid else 'Hayır'}
- Tespit Edilen Sorunlar: {issues_text}{speed_info}{lw_rw_info}
{regional_info}{angle_info}

KISA, MOTİVE EDİCİ ve AKSİYON ALINACAK feedback ver (Türkçe):
1. Pozitif bir notla başla (skor düşük olsa bile)
2. Özellikle sol ve sağ kol senkronizasyonuna dikkat et
3. Varsa en kritik sorunu belirt ve düzeltme önerisi ver
4. Teşvik edici bir cümleyle bitir

2 cümleyi geçme. Samimi ve destekleyici ol."""

            response = openai_client.chat.completions.create(
                model='gpt-4o-mini',  # Faster and cheaper than gpt-4
                messages=[
                    {'role': 'system', 'content': 'You are a professional fitness coach. Provide concise, actionable feedback in Turkish.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=150,
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
    
    # Add speed-based tips
    if speed_class == 'very_fast':
        overall += " Çok hızlı yapıyorsun, biraz yavaşla!"
    elif speed_class == 'very_slow':
        overall += " Biraz hızlandırabilirsin."
    
    # Add issue-based tips from detector
    if rep_issues_from_detector:
        overall += " " + " ".join(rep_issues_from_detector[:2])  # First 2 issues
    elif issues:
        if 'elbow_moving' in str(issues).lower() or 'dirsek' in str(issues).lower():
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
        'triceps_pushdown': 'Triceps Extension',
        'dumbbell_rows': 'Dumbbell Row',
        'dumbbell_shoulder_press': 'Shoulder Press'
    }
    ex_name = exercise_names.get(exercise, exercise)
    
    # Try OpenAI first (if available)
    if openai_client:
        try:
            top_issues_text = ', '.join([f"{issue} ({count}x)" for issue, count in top_issues]) if top_issues else 'None'
            
            prompt = f"""You are an expert fitness coach providing workout session feedback.

📊 WORKOUT SUMMARY ({ex_name}):
- Total Reps Completed: {total_reps}
- Average Form Score: {avg_score:.1f}%
- Best Rep Score: {best_score:.1f}%
- Worst Rep Score: {worst_score:.1f}%
- Most Common Issues: {top_issues_text}

Provide comprehensive feedback in Turkish:
1. Congratulate them for completing the workout
2. Overall performance assessment (be encouraging but honest)
3. 2-3 specific improvement areas based on the most common issues
4. Motivating closing message

Keep it friendly, professional, and under 4-5 sentences. Focus on actionable advice."""

            response = openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': 'You are a professional fitness coach. Provide detailed, encouraging workout feedback in Turkish.'},
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=300,
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
