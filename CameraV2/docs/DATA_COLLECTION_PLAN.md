# 📊 Veri Toplama ve ML Model Eğitimi Planı

## 🎯 ÖNEMLİ: Video Çekmeye GEREK YOK!

**Mevcut sistem zaten real-time çalışıyor ve otomatik olarak veri topluyor!**

### Nasıl Çalışıyor?

1. **Real-time Landmarks Toplama:**
   - Kullanıcı antrenman yaparken, sistem her frame'de MediaPipe landmarks'ları otomatik topluyor
   - Her rep tamamlandığında, o rep'in tüm frame'leri (`landmarks_sequence`) otomatik kaydediliyor
   - Video dosyası oluşturmaya veya işlemeye **GEREK YOK**

2. **Veri Formatı:**
   - Her rep = `landmarks_sequence` (frame'lerin listesi)
   - Her frame = 33 MediaPipe landmark noktası (x, y, z, visibility)
   - Ek olarak: form skorları, açılar, bölgesel skorlar otomatik kaydediliyor

3. **Dataset Collection:**
   - `api_server.py` içinde `DATASET_COLLECTION_ENABLED = True` yapıldığında
   - Her rep otomatik olarak `dataset_collector.py` ile kaydediliyor
   - Veriler `dataset/` klasörüne JSON ve CSV formatında kaydediliyor

---

## 📈 Optimize Edilmiş Veri Toplama Planı

### ❌ ÖNERİLMEYEN: 4 × 10 × 10 × 12 = 4800 Rep
**Neden çok fazla?**
- 4800 rep = ~160 saat antrenman (her rep 2 dakika)
- Çoğu rep gereksiz tekrar olur
- ML modeli için 200-400 rep yeterli

### ✅ ÖNERİLEN: Optimize Edilmiş Plan

#### **Aşama 1: Baseline İçin (Minimum)**
- **4 hareket × 20 perfect rep = 80 rep**
- **Süre:** 1-2 hafta
- **Amaç:** Baseline hesaplama için perfect form örnekleri

#### **Aşama 2: Model Eğitimi İçin (İdeal)**
- **4 hareket × 100 rep = 400 rep**
- **Dağılım:**
  - Perfect form: 30% (120 rep)
  - Good form: 50% (200 rep)
  - Bad form: 20% (80 rep)
- **Kişi sayısı:** 3-5 kişi (her kişi 20-30 rep/hareket)
- **Süre:** 3-4 hafta

#### **Aşama 3: Production Ready (Opsiyonel)**
- **4 hareket × 200 rep = 800 rep**
- **Kişi sayısı:** 5-10 kişi
- **Süre:** 6-8 hafta

---

## 🔄 Veri Toplama Süreci

### 1. **Sistem Hazırlığı**

```python
# api_server.py içinde
DATASET_COLLECTION_ENABLED = True  # Aktif et
```

### 2. **Antrenman Yapma**

1. Kullanıcı normal antrenman yapar (web UI üzerinden)
2. Her rep otomatik olarak kaydedilir
3. Sistem her rep için:
   - Landmarks sequence (tüm frame'ler)
   - Form skorları (genel + bölgesel)
   - Açılar (min, max, range)
   - Issues (hatalar)
   - Otomatik olarak kaydeder

### 3. **Labeling (Etiketleme)**

Rep'leri kategorize et:
- **Perfect:** Form skoru ≥ 90%, tüm açılar doğru
- **Good:** Form skoru 70-89%
- **Bad:** Form skoru < 70%

**Nasıl label edilir?**
```python
# dataset_collector.py kullanarak
collector.label_sample(
    sample_index=0,
    expert_score=95,
    is_perfect_form=True,
    user_feedback="perfect"
)
```

### 4. **Feature Extraction**

Her rep için kinematic özellikler çıkarılır:
- Açılar (elbow, knee, shoulder, etc.)
- Range of motion
- Velocity (hız)
- Smoothness (akıcılık)
- Stability (kararlılık)
- Symmetry (simetri)

**Otomatik olarak yapılır:** `dataset_collector.extract_features()`

---

## 🤖 ML Model Eğitimi Süreci

### 1. **Veri Hazırlama**

```bash
# Tüm rep'leri yükle
python -c "from dataset_collector import DatasetCollector; \
    collector = DatasetCollector(); \
    samples = collector.load_dataset()"
```

### 2. **Feature Extraction**

```python
# Her rep için özellikler çıkarılır
for sample in samples:
    features = collector.extract_features(sample)
    # features = {
    #     'elbow_angle_mean': 45.2,
    #     'knee_angle_range': 85.3,
    #     'shoulder_stability': 0.05,
    #     ...
    # }
```

### 3. **Train/Validation/Test Split**

```python
# 70% train, 15% validation, 15% test
train_samples = samples[:280]  # 70%
val_samples = samples[280:340]  # 15%
test_samples = samples[340:]    # 15%
```

### 4. **K-Fold Cross Validation**

```python
# 5-fold CV ile model performansını test et
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(X_train):
    # Model eğit ve değerlendir
    ...
```

### 5. **Model Eğitimi**

```bash
# train_form_model.py ile model eğit
python train_form_model.py \
    --exercise bicep_curls \
    --model_type random_forest \
    --dataset_dir dataset
```

**Model Tipleri:**
- `RandomForestRegressor` (önerilen)
- `GradientBoostingRegressor`
- `Ridge` (baseline)

### 6. **Baseline Hesaplama**

```bash
# Perfect form örneklerinden baseline hesapla
python calculate_baselines.py \
    --exercise bicep_curls \
    --dataset_dir dataset
```

**Çıktı:**
```json
{
  "elbow_angle_mean": {"mean": 45.2, "std": 2.1},
  "shoulder_stability": {"mean": 0.03, "std": 0.01},
  ...
}
```

---

## ⚡ Real-Time Çalışma

### ✅ EVET, Sistem Zaten Real-Time Çalışıyor!

**Nasıl?**

1. **Frontend (React):**
   - Kameradan frame'leri alır
   - MediaPipe ile landmarks çıkarır
   - WebSocket ile backend'e gönderir

2. **Backend (Python):**
   - Landmarks'ları alır
   - **Eğitilmiş ML modeli** ile form skorunu tahmin eder
   - Real-time feedback gönderir

3. **ML Model Entegrasyonu:**

```python
# api_server.py içinde
from ml_trainer import FormScorePredictor

# Model yükle
predictor = FormScorePredictor.load(Path("models/bicep_curls"))

# Real-time tahmin
features = extract_features_from_landmarks(landmarks)
form_score = predictor.predict(features)
```

### Model Performansı

- **Inference Time:** < 10ms (real-time için yeterli)
- **Accuracy:** %85-95 (yeterli veri ile)
- **Latency:** < 50ms (kullanıcı fark etmez)

---

## 📋 Özet: Veri Toplama Checklist

### Minimum (Baseline için)
- [ ] 4 hareket × 20 perfect rep = **80 rep**
- [ ] 1-2 kişi
- [ ] 1-2 hafta

### İdeal (Model için)
- [ ] 4 hareket × 100 rep = **400 rep**
- [ ] 3-5 kişi
- [ ] Perfect: 120, Good: 200, Bad: 80
- [ ] 3-4 hafta

### Production (Opsiyonel)
- [ ] 4 hareket × 200 rep = **800 rep**
- [ ] 5-10 kişi
- [ ] 6-8 hafta

---

## 🚀 Hızlı Başlangıç

### 1. Dataset Collection'ı Aktif Et

```python
# api_server.py
DATASET_COLLECTION_ENABLED = True
```

### 2. Antrenman Yap ve Veri Topla

- Normal antrenman yap
- Her rep otomatik kaydedilir
- `dataset/` klasörüne bak

### 3. Rep'leri Label Et

```python
python -c "
from dataset_collector import DatasetCollector
collector = DatasetCollector()
samples = collector.load_dataset()

# Perfect rep'leri işaretle
for i, sample in enumerate(samples):
    if sample.regional_scores and sum(sample.regional_scores.values())/4 >= 90:
        collector.label_sample(i, expert_score=95, is_perfect_form=True)
"
```

### 4. Model Eğit

```bash
python train_form_model.py --exercise bicep_curls
```

### 5. Baseline Hesapla

```bash
python calculate_baselines.py --exercise bicep_curls
```

### 6. Real-Time Kullan

- Model otomatik yüklenir
- Real-time form skorları gösterilir
- LLM feedback'leri ML model skorlarına göre verilir

---

## ❓ Sık Sorulan Sorular

### Q: Video çekmem gerekiyor mu?
**A: HAYIR!** Sistem zaten real-time landmarks topluyor.

### Q: 4800 rep çok fazla değil mi?
**A: EVET!** 400 rep yeterli, 800 rep ideal.

### Q: Model real-time çalışacak mı?
**A: EVET!** Sistem zaten real-time, ML model sadece form skorunu iyileştirir.

### Q: K-fold CV gerekli mi?
**A: EVET!** Model performansını doğru değerlendirmek için önemli.

### Q: Her hareket için ayrı model mi?
**A: EVET!** Her hareket için ayrı model eğitilir (bicep_curls, squats, etc.)

---

## 📊 Beklenen Sonuçlar

### Model Performansı (400 rep ile)
- **R² Score:** 0.85-0.95
- **MAE:** 5-10 puan
- **Inference Time:** < 10ms

### Baseline Accuracy
- **Perfect form detection:** %90+
- **Form score prediction:** %85-95

---

**Sonuç:** Video çekmeye gerek yok, sistem zaten real-time çalışıyor. 400 rep yeterli, model real-time çalışacak! 🎉

