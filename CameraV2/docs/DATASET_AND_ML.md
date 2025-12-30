# Dataset Collection and ML Model Training Guide

## 📊 Dataset Collection Sistemi

### Nasıl Çalışır?

1. **Otomatik Kayıt**: Her rep tamamlandığında otomatik olarak kaydedilir
2. **Landmarks Sequence**: Her rep için tüm frame'lerdeki landmark'lar saklanır
3. **Feature Extraction**: Otomatik olarak kinematic feature'lar çıkarılır
4. **Labeling**: Expert score veya user feedback ile label'lenebilir

### Dataset Yapısı

```
dataset/
├── bicep_curls_20251223_143000/
│   ├── samples.json          # Tüm rep samples (landmarks + features)
│   └── summary.csv           # Özet CSV
├── squats_20251223_150000/
│   ├── samples.json
│   └── summary.csv
└── ...
```

### Manuel Kullanım

```python
from dataset_collector import DatasetCollector

# Başlat
collector = DatasetCollector("dataset")

# Session başlat
session_id = collector.start_session("bicep_curls", user_id="user1")

# Rep ekle (gerçek kullanımda API server otomatik yapar)
sample = collector.add_rep_sample(
    exercise="bicep_curls",
    rep_number=1,
    landmarks_sequence=[...],  # List of frames with landmarks
    regional_scores={'arms': 85, 'legs': 90, 'core': 80, 'head': 95},
    min_angle=45,
    max_angle=155
)

# Label'le
collector.label_sample(0, expert_score=90, is_perfect_form=True)

# Kaydet
collector.save_session(auto_label_perfect=True)
```

## 🤖 ML Model Training

### 1. Dataset Toplama

Önce yeterli veri toplamalısınız:
- **Minimum**: 20-30 rep (farklı form kalitelerinde)
- **İdeal**: 100+ rep (perfect, good, bad form karışımı)

### 2. Model Eğitimi

```bash
# Belirli bir hareket için
python train_form_model.py bicep_curls random_forest

# Tüm hareketler için
python train_form_model.py all random_forest

# Model tipleri:
# - random_forest (önerilen)
# - gradient_boosting
# - ridge
```

### 3. Baseline Hesaplama

```bash
# Belirli hareket için
python calculate_baselines.py bicep_curls

# Tüm hareketler için
python calculate_baselines.py
```

### 4. Model Kullanımı

```python
from ml_trainer import FormScorePredictor

# Model yükle
predictor = FormScorePredictor.load("models/form_score_bicep_curls_random_forest")

# Feature'ları çıkar (dataset_collector kullanarak)
features = collector.extract_features(sample)

# Form skoru tahmin et
predicted_score = predictor.predict(features)
print(f"Predicted form score: {predicted_score:.1f}%")
```

## 📈 Baseline Kullanımı

Baseline'lar perfect form için referans değerlerdir:

```python
import json

# Baseline'ları yükle
with open("baselines/bicep_curls_baselines.json") as f:
    baselines = json.load(f)

# Örnek: Left elbow angle baseline
left_elbow_baseline = baselines['left_elbow_min']
# {
#   'mean': 42.5,
#   'std': 3.2,
#   'min': 38.0,
#   'max': 48.0
# }

# Kullanıcının değerini kontrol et
user_elbow_angle = 50.0
if user_elbow_angle > left_elbow_baseline['max']:
    print("Elbow angle too high!")
```

## 🔄 Workflow

### İlk Kurulum

1. **Veri Toplama** (1-2 hafta)
   - Normal kullanım sırasında otomatik kayıt
   - Her rep otomatik kaydedilir

2. **Labeling** (Manuel veya otomatik)
   - Perfect form rep'leri işaretle
   - Expert score ver (0-100)

3. **Model Eğitimi**
   ```bash
   python train_form_model.py bicep_curls random_forest
   ```

4. **Baseline Hesaplama**
   ```bash
   python calculate_baselines.py bicep_curls
   ```

5. **API Server'a Entegre Et**
   - Model'i yükle
   - Baseline'ları kullan
   - Prediction'ları form analysis'e ekle

### Sürekli İyileştirme

- Daha fazla veri topla → Model'i yeniden eğit
- Perfect form örnekleri ekle → Baseline'ları güncelle
- Farklı kullanıcılardan veri topla → Daha genel model

## 📊 Feature'lar

Her rep için şu feature'lar otomatik çıkarılır:

### Angle Features
- `left_elbow_min`, `left_elbow_max`, `left_elbow_range`
- `right_elbow_min`, `right_elbow_max`, `right_elbow_range`
- `left_knee_min`, `left_knee_max`, `left_knee_range`
- ... (tüm joint'ler için)

### Dynamics Features
- `left_elbow_vel_mean`, `left_elbow_vel_max`
- `left_elbow_acc_mean`
- `left_elbow_smoothness`
- ... (tüm joint'ler için)

### Temporal Features
- `left_elbow_skew`, `left_elbow_kurtosis`
- `left_elbow_zero_crossings`
- `left_elbow_peak_count`
- ... (tüm joint'ler için)

**Toplam**: ~100-150 feature per rep

## 🎯 Best Practices

1. **Veri Kalitesi**
   - Farklı form kalitelerinden örnekler topla
   - Perfect form örneklerini özellikle işaretle
   - Farklı açılardan/kamera pozisyonlarından veri topla

2. **Labeling**
   - Expert score: 0-100 arası detaylı skor
   - Perfect form: Boolean (sadece mükemmel rep'ler)
   - User feedback: "perfect", "good", "bad"

3. **Model Seçimi**
   - **Random Forest**: Genel kullanım için önerilen
   - **Gradient Boosting**: Daha iyi accuracy, daha yavaş
   - **Ridge**: Hızlı, basit, az veri için uygun

4. **Baseline Kullanımı**
   - Perfect form örneklerinden hesapla
   - Tolerance percentile: 95% (varsayılan)
   - Her hareket için ayrı baseline

## 📝 Örnek Kullanım Senaryosu

```python
# 1. Dataset topla (otomatik - API server yapar)
# 2. Label'le
collector = DatasetCollector("dataset")
samples = collector.load_dataset()
collector.label_sample(0, expert_score=95, is_perfect_form=True)

# 3. Model eğit
# python train_form_model.py bicep_curls

# 4. Baseline hesapla
# python calculate_baselines.py bicep_curls

# 5. Kullan
from ml_trainer import FormScorePredictor
predictor = FormScorePredictor.load("models/form_score_bicep_curls_random_forest")
score = predictor.predict(features)
```

## ⚠️ Notlar

- **Minimum Veri**: En az 10-20 labeled sample gerekli
- **Perfect Form**: Baseline hesaplamak için en az 5-10 perfect sample
- **Memory**: Landmarks sequence'ları büyük olabilir (her frame 33 landmark)
- **Performance**: Feature extraction biraz zaman alabilir

