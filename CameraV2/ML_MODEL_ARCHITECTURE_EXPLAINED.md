# 🤖 ML Model Mimarisi ve Pattern Öğrenme Açıklaması

Bu doküman, ML modelimizin MLTRAINIMU ve MLTRAINCAMERA verilerinden pattern öğrenme, ensemble average ile doğru form belirleme, benzerlik tabanlı correction scoring ve LLM feedback süreçlerini detaylı olarak açıklar.

---

## 📊 1. VERİ TOPLAMA VE PATTERN ÖĞRENME

### 1.1 Veri Kaynakları

**MLTRAINCAMERA/** klasöründe:
- `samples.json`: Her rep için 33 landmark point'lerin zaman serisi (20Hz, ~30-60 frame/rep)
- `summary.csv`: Her rep için metadata (rep_number, timestamp, expert_score, regional_scores, min_angle, max_angle, range_of_motion)
- Her session için: `{exercise}/{session_id}/` altında saklanır

**MLTRAINIMU/** klasöründe:
- `imu_samples.json`: Her rep için IMU sensor verileri (left_wrist, right_wrist, chest - 20Hz)
- `imu_samples.csv`: Detaylı CSV formatında IMU verileri (timestamp, node_id, ax, ay, az, gx, gy, gz, quaternions, euler angles)
- `summary.csv`: Her rep için metadata (rep_number, timestamp, num_samples)

### 1.2 Feature Extraction (Özellik Çıkarımı)

**Camera Features:**
- `DatasetCollector.extract_features()` → `exercise_embeddings.feature_extractor.extract_all_features()`
- Her rep için:
  - **Angle Features**: Her eklem için (elbow, shoulder, hip, knee, etc.)
    - `min_angle`, `max_angle`, `range_of_motion`
    - `mean`, `std`, `median`
    - `velocity_mean`, `velocity_max` (açısal hız)
    - `acceleration_mean` (açısal ivme)
    - `smoothness` (hareket düzgünlüğü)
    - `peaks`, `valleys`, `oscillation_rate`
  - **Position Features**: Normalize edilmiş landmark pozisyonları
  - **Joint-Specific Features**: Her egzersize özel (örn: bicep curls için elbow ROM)

**IMU Features:**
- `imu_feature_extractor.extract_imu_features()`
- Her node (left_wrist, right_wrist, chest) için:
  - **Accelerometer**: `ax`, `ay`, `az` → mean, std, min, max, range
  - **Gyroscope**: `gx`, `gy`, `gz` → mean, std, min, max, range
  - **Quaternions**: `qw`, `qx`, `qy`, `qz` → mean, std
  - **Euler Angles**: `roll`, `pitch`, `yaw` → mean, std, min, max, range
  - **Cross-sensor features**: Wrist-wrist correlation, chest-wrist alignment

**Toplam Feature Sayısı:**
- Camera: ~120-150 features (egzersize göre değişir)
- IMU: ~60-90 features (3 node × ~20-30 features/node)

---

## 🎯 2. PATTERN ÖĞRENME VE MODEL EĞİTİMİ

### 2.1 Training Pipeline (`train_ml_models.py`)

```python
# 1. Veri Yükleme
collector = DatasetCollector("MLTRAINCAMERA")
samples = collector.load_dataset(exercise="bicep_curls")  # Sadece bu egzersiz

# 2. Feature Extraction (eğer yapılmamışsa)
for sample in samples:
    if sample.features is None:
        collector.extract_features(sample)

# 3. Labeling (Auto-labeling if needed)
for sample in samples:
    if sample.expert_score is None:
        # Regional scores ortalamasını kullan
        avg_score = sum(sample.regional_scores.values()) / len(sample.regional_scores)
        sample.expert_score = avg_score
        sample.is_perfect_form = (avg_score >= 90)

# 4. Model Training
predictor = FormScorePredictor(model_type="random_forest")
results = predictor.train(samples, verbose=True)

# 5. Model Saving
predictor.save("models/bicep_curls/form_score_camera_random_forest/")
```

### 2.2 Model Training Detayları (`ml_trainer.py`)

**FormScorePredictor.train():**
1. **Feature Matrix Oluşturma**: Tüm samples'ları feature vector'larına çevirir
2. **Label Extraction**: `expert_score` veya `regional_scores` average'ını label olarak kullanır
3. **Train/Test Split**: 80/20 split
4. **Feature Scaling**: StandardScaler ile normalize eder
5. **Model Training**: Random Forest (100 trees), Gradient Boosting, veya Ridge Regression
6. **Evaluation**: MSE, MAE, R² scores

**Model Output:**
- **Trained model**: `model.pkl`
- **Scaler**: `scaler.pkl`
- **Metadata**: `metadata.json` (feature_names, performance_metrics, etc.)

---

## ✅ 3. PERFECT FORM BASELINE: ENSEMBLE AVERAGE YAKLAŞIMI

### 3.1 Perfect Form Samples Seçimi

Training sırasında `is_perfect_form == True` olan samples'lar seçilir:
- `expert_score >= 90` olan samples
- Veya kullanıcı tarafından "perfect" olarak işaretlenen samples

### 3.2 Baseline Calculation (`BaselineCalculator.calculate_baselines()`)

**Ensemble Average Yaklaşımı:**

```python
perfect_samples = [s for s in samples if s.is_perfect_form == True]

# Her feature için perfect samples'ların ortalamasını al
baselines = {}
for feature_name in feature_names:
    values = [s.features[feature_name] for s in perfect_samples]
    baselines[feature_name] = {
        'mean': np.mean(values),      # Ensemble average
        'std': np.std(values),         # Standard deviation
        'min': np.percentile(values, 5),   # 5th percentile (lower bound)
        'max': np.percentile(values, 95),  # 95th percentile (upper bound)
        'median': np.median(values)
    }
```

**Örnek Baseline:**
```json
{
  "left_elbow_range": {
    "mean": 120.5,     // Perfect form'da ortalama ROM
    "std": 8.3,        // Standart sapma
    "min": 105.2,      // Minimum kabul edilebilir ROM (5th percentile)
    "max": 135.8,      // Maximum kabul edilebilir ROM (95th percentile)
    "median": 120.0
  },
  "left_elbow_vel_mean": {
    "mean": 45.2,
    "std": 5.1,
    "min": 35.0,
    "max": 55.0,
    "median": 45.0
  }
}
```

**Baseline Dosyası:**
- `models/{exercise}/form_score_camera_random_forest/baselines.json`
- Her training sonrası otomatik oluşturulur

---

## 📏 4. BENZERLİK TABANLI CORRECTION SCORING

### 4.1 Real-Time Prediction (Model Inference)

**Mevcut Sistem (`model_inference.py`):**

```python
# 1. Yeni bir rep için feature extraction
sample = RepSample(
    landmarks_sequence=current_rep_landmarks,
    imu_sequence=current_rep_imu_data
)
features = collector.extract_features(sample)

# 2. ML Model ile score prediction
predictor = FormScorePredictor.load("models/bicep_curls/form_score_camera_random_forest/")
predicted_score = predictor.predict(features)  # 0-100 arası score
```

**Predicted Score Anlamı:**
- **90-100**: Mükemmel form (perfect form'a çok benzer)
- **80-89**: İyi form
- **70-79**: Orta form
- **60-69**: Kötü form
- **<60**: Çok kötü form (düzeltme gerekli)

### 4.2 Baseline Benzerliği Hesaplama (ÖNERİLEN YAKLAŞIM)

**Eksik Kısım:** Şu anda sistem sadece ML model score'u kullanıyor. Baseline benzerliği eklenmeli:

```python
def calculate_baseline_similarity(current_features, baselines):
    """
    Current rep'in perfect form baseline'larına ne kadar benzediğini hesapla.
    
    Returns:
        similarity_score: 0-100 arası (100 = perfect match)
    """
    similarity_scores = []
    
    for feature_name, baseline in baselines.items():
        current_value = current_features.get(feature_name, 0)
        
        # Baseline mean'e ne kadar yakın?
        baseline_mean = baseline['mean']
        baseline_std = baseline['std']
        
        # Z-score hesapla (normalize edilmiş uzaklık)
        if baseline_std > 0:
            z_score = abs(current_value - baseline_mean) / baseline_std
            # Z-score'u similarity score'a çevir (0-100)
            # Z-score < 1: Çok yakın (score > 84)
            # Z-score < 2: Yakın (score > 68)
            # Z-score > 2: Uzak (score < 68)
            similarity = max(0, 100 - (z_score * 16))  # 16 = scaling factor
        else:
            # Std = 0 ise, sadece mean'e eşit olup olmadığına bak
            similarity = 100.0 if abs(current_value - baseline_mean) < 0.001 else 50.0
        
        similarity_scores.append(similarity)
    
    # Weighted average (önemli feature'lar daha fazla ağırlık alabilir)
    return np.mean(similarity_scores)
```

### 4.3 Hybrid Scoring (ML Model + Baseline Similarity)

**Önerilen Yaklaşım:**

```python
def calculate_correction_score(ml_prediction, baseline_similarity, ml_weight=0.6, baseline_weight=0.4):
    """
    ML model prediction ve baseline similarity'yi birleştir.
    
    Args:
        ml_prediction: ML model'den gelen score (0-100)
        baseline_similarity: Baseline'a benzerlik score'u (0-100)
        ml_weight: ML model ağırlığı (default: 0.6)
        baseline_weight: Baseline similarity ağırlığı (default: 0.4)
    
    Returns:
        final_correction_score: 0-100 arası final score
    """
    final_score = (ml_prediction * ml_weight) + (baseline_similarity * baseline_weight)
    return np.clip(final_score, 0, 100)
```

**Kullanım Senaryosu:**
```python
# Real-time workout sırasında
current_features = extract_features(current_rep)

# 1. ML Model Prediction
ml_score = predictor.predict(current_features)  # Örn: 75

# 2. Baseline Similarity
baseline_similarity = calculate_baseline_similarity(current_features, baselines)  # Örn: 82

# 3. Hybrid Score
correction_score = calculate_correction_score(
    ml_prediction=75,
    baseline_similarity=82,
    ml_weight=0.6,
    baseline_weight=0.4
)  # = 75 * 0.6 + 82 * 0.4 = 77.8

# 4. Feedback için kullan
if correction_score < 70:
    feedback = "Form düzeltme gerekli!"
elif correction_score < 85:
    feedback = "İyi, ama daha iyi olabilir."
else:
    feedback = "Mükemmel form!"
```

---

## 🤖 5. LLM FEEDBACK SİSTEMİ

### 5.1 Mevcut LLM Entegrasyonu (`api_server.py`)

**OpenAI API Integration:**
```python
def get_llm_feedback(exercise, rep_data, regional_scores, regional_issues):
    """
    OpenAI GPT-4 ile detaylı feedback üret.
    
    Args:
        exercise: Egzersiz adı
        rep_data: Rep detayları (form_score, issues, etc.)
        regional_scores: Bölgesel skorlar (arms, legs, core, head)
        regional_issues: Bölgesel sorunlar
    
    Returns:
        feedback_text: LLM tarafından üretilen feedback
    """
    if not openai_client:
        return get_rule_based_feedback(...)  # Fallback
    
    prompt = f"""
    Sen bir fitness antrenörüsün. Aşağıdaki egzersiz form analizi için 
    detaylı, motivasyonel ve yapıcı feedback ver.
    
    Egzersiz: {exercise}
    Form Skoru: {rep_data['form_score']}/100
    Bölgesel Skorlar:
    - Kollar: {regional_scores['arms']}/100
    - Bacaklar: {regional_scores['legs']}/100
    - Gövde: {regional_scores['core']}/100
    - Kafa: {regional_scores['head']}/100
    
    Sorunlar: {regional_issues}
    
    Feedback (Türkçe, kısa ve öz, 2-3 cümle):
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    return response.choices[0].message.content
```

### 5.2 LLM Feedback Kullanım Senaryosu

**Real-Time Feedback (Her Rep Sonrası):**
```python
# Rep tamamlandığında
rep_data = {
    'form_score': correction_score,  # Hybrid score
    'issues': detected_issues,
    'regional_scores': regional_scores,
    'regional_issues': regional_issues
}

# LLM Feedback
if openai_api_available:
    llm_feedback = get_llm_feedback(
        exercise=exercise,
        rep_data=rep_data,
        regional_scores=regional_scores,
        regional_issues=regional_issues
    )
else:
    # Fallback: Rule-based feedback
    llm_feedback = get_rule_based_regional_feedback(...)

# WebSocket'e gönder
await websocket.send_json({
    'type': 'rep_complete',
    'rep_number': rep_counter.count,
    'form_score': correction_score,
    'feedback': llm_feedback,
    'regional_scores': regional_scores,
    'regional_issues': regional_issues
})
```

**Session Summary Feedback (Workout Sonrası):**
```python
# Tüm workout bittiğinde
session_summary = {
    'total_reps': len(session['reps_data']),
    'avg_form_score': np.mean([r['form_score'] for r in session['reps_data']]),
    'regional_scores': {
        'arms': np.mean([r['regional_scores']['arms'] for r in session['reps_data']]),
        'legs': np.mean([r['regional_scores']['legs'] for r in session['reps_data']]),
        'core': np.mean([r['regional_scores']['core'] for r in session['reps_data']]),
        'head': np.mean([r['regional_scores']['head'] for r in session['reps_data']])
    },
    'all_issues': collect_all_issues(session['reps_data'])
}

# LLM Session Feedback
if openai_api_available:
    session_feedback = get_llm_session_feedback(exercise, session_summary)
else:
    session_feedback = get_rule_based_session_feedback(session_summary)
```

---

## 🔄 6. TAM İŞ AKIŞI (END-TO-END)

### 6.1 Training Phase (Model Eğitimi)

```
1. Kullanıcı workout yapar (Train Mode)
   ↓
2. MLTRAINCAMERA ve MLTRAINIMU'ya veri kaydedilir
   ↓
3. train_ml_models.py çalıştırılır
   ↓
4. Feature extraction → Model training → Baseline calculation
   ↓
5. Model ve baseline'lar models/{exercise}/ altına kaydedilir
```

### 6.2 Inference Phase (Real-Time Kullanım)

```
1. Kullanıcı workout yapar (Usage Mode)
   ↓
2. Her rep tamamlandığında:
   a) Feature extraction (camera + IMU)
   b) ML model prediction (FormScorePredictor.predict())
   c) Baseline similarity calculation (YENİ - eklenecek)
   d) Hybrid correction score (ML + Baseline)
   e) Regional score calculation (mevcut FormAnalyzer)
   f) LLM feedback generation (OpenAI API veya rule-based)
   ↓
3. WebSocket ile frontend'e gönderilir:
   {
     'type': 'rep_complete',
     'form_score': 77.8,  // Hybrid correction score
     'ml_prediction': 75,  // ML model score
     'baseline_similarity': 82,  // Baseline similarity
     'regional_scores': {...},
     'feedback': "İyi form! Dirseklerini biraz daha sabit tutabilirsin."
   }
   ↓
4. Session sonunda:
   a) Average scores hesaplanır
   b) Session summary LLM feedback üretilir
   c) Kullanıcıya gösterilir
```

---

## 📝 7. EKSİK KISIMLAR VE ÖNERİLER

### 7.1 Eksik Kısımlar

1. **Baseline Similarity Calculation**: Şu anda sadece ML model prediction var, baseline similarity yok
2. **Hybrid Scoring**: ML + Baseline birleştirmesi yok
3. **IMU Model Training**: IMU model training tam implement edilmemiş
4. **Sensor Fusion**: Camera + IMU fusion için ağırlıklandırma yok

### 7.2 Önerilen İyileştirmeler

1. **`ml_trainer.py`'a Baseline Similarity Function Ekle:**
   ```python
   def calculate_baseline_similarity(features, baselines):
       # Implementation
   ```

2. **`model_inference.py`'a Hybrid Scoring Ekle:**
   ```python
   def predict_with_baseline(self, features, baselines, ml_weight=0.6):
       ml_score = self.predict(features)
       baseline_sim = calculate_baseline_similarity(features, baselines)
       return calculate_correction_score(ml_score, baseline_sim, ml_weight)
   ```

3. **`api_server.py`'da Real-Time Inference Entegrasyonu:**
   ```python
   # Rep complete olduğunda
   features = extract_features(current_rep)
   ml_score = model_inference.predict(features)
   baselines = load_baselines(exercise)
   baseline_sim = calculate_baseline_similarity(features, baselines)
   correction_score = calculate_correction_score(ml_score, baseline_sim)
   ```

---

## 🎯 SONUÇ

Mevcut sistem:
- ✅ Veri toplama (MLTRAINCAMERA, MLTRAINIMU)
- ✅ Feature extraction
- ✅ ML model training
- ✅ Baseline calculation (ensemble average)
- ✅ Model inference
- ✅ LLM feedback (OpenAI API)

Eksik kısımlar:
- ❌ Baseline similarity calculation (eklenecek)
- ❌ Hybrid scoring (ML + Baseline) (eklenecek)
- ❌ Real-time inference entegrasyonu (api_server.py'da) (eklenecek)

Bu eksik kısımlar implement edildikten sonra, sistem tam bir pattern learning ve similarity-based correction scoring sistemi olacak.

