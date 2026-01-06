# Single-Output Models Eğitim Rehberi

## 🎯 Amaç

6 hareket × 3 mod = **18 adet single-output model** eğitmek.

Her model **overall form score** (0-100) tahmin eder (regional scores değil).

---

## 📊 Model Yapısı

### Her Hareket İçin 3 Model:

1. **Camera Model** (`form_score_camera_random_forest_single_output`)
   - Sadece kamera (landmark) verilerini kullanır
   - Overall score tahmin eder

2. **IMU Model** (`form_score_imu_random_forest_single_output`)
   - Sadece IMU verilerini kullanır
   - Overall score tahmin eder

3. **Fusion Model** (`form_score_fusion_random_forest_single_output`)
   - Hem kamera hem IMU verilerini kullanır
   - Overall score tahmin eder

### Toplam: 6 × 3 = 18 Model

```
bicep_curls/
  ├── form_score_camera_random_forest_single_output/
  ├── form_score_imu_random_forest_single_output/
  └── form_score_fusion_random_forest_single_output/
squats/
  ├── form_score_camera_random_forest_single_output/
  ├── form_score_imu_random_forest_single_output/
  └── form_score_fusion_random_forest_single_output/
... (diğer 4 hareket)
```

---

## 🚀 Eğitim Komutları

### Seçenek 1: Tek Tek Eğit

```bash
# Bicep Curls için 3 model
python3 train_ml_models.py --exercise bicep_curls --camera-only --single-output
python3 train_ml_models.py --exercise bicep_curls --imu-only --single-output
python3 train_ml_models.py --exercise bicep_curls --fusion --single-output

# Squats için 3 model
python3 train_ml_models.py --exercise squats --camera-only --single-output
python3 train_ml_models.py --exercise squats --imu-only --single-output
python3 train_ml_models.py --exercise squats --fusion --single-output

# ... (diğer hareketler için)
```

### Seçenek 2: Tüm Modelleri Tek Komutla Eğit

```bash
# Tüm 18 modeli otomatik eğit
./train_all_single_output_models.sh
```

### Seçenek 3: Bir Hareket İçin Tüm Modelleri Eğit

```bash
# Bicep curls için 3 modeli birden eğit
python3 train_ml_models.py --exercise bicep_curls --single-output
```

---

## 📈 Beklenen Performans

**Eski Single-Output Sonuçları (Referans):**
- Test R²: 0.735 ✅
- Test MAE: 0.25 puan ✅
- Test MSE: 0.10 ✅

**Beklenen Performans (Her Model İçin):**
- Camera: Test R² ≈ 0.70-0.75 (en iyi)
- IMU: Test R² ≈ 0.40-0.60 (daha düşük)
- Fusion: Test R² ≈ 0.70-0.80 (en iyi potansiyel)

---

## 📁 Model Lokasyonları

Tüm modeller `models/{exercise}/form_score_{mode}_random_forest_single_output/` klasöründe saklanır.

**Örnek:**
```
models/
  bicep_curls/
    form_score_camera_random_forest_single_output/
      ├── model.pkl
      ├── scaler.pkl
      ├── metadata.json
      └── baselines.json
    form_score_imu_random_forest_single_output/
      └── ...
    form_score_fusion_random_forest_single_output/
      └── ...
```

---

## ✅ Kontrol

Tüm modellerin eğitildiğini kontrol etmek için:

```bash
# Python script ile kontrol
python3 << 'PYEOF'
from pathlib import Path

exercises = ['bicep_curls', 'squats', 'lateral_shoulder_raises', 
             'tricep_extensions', 'dumbbell_rows', 'dumbbell_shoulder_press']
modes = ['camera', 'imu', 'fusion']

total_expected = len(exercises) * len(modes)
total_found = 0

for exercise in exercises:
    for mode in modes:
        model_dir = Path(f"models/{exercise}/form_score_{mode}_random_forest_single_output")
        if model_dir.exists() and (model_dir / "model.pkl").exists():
            total_found += 1
            print(f"✅ {exercise} - {mode}")
        else:
            print(f"❌ {exercise} - {mode} (MISSING)")

print(f"\n📊 {total_found}/{total_expected} models found")
PYEOF
```

---

## 🔄 Multi-Output vs Single-Output

### Single-Output Model (Bu Rehber)
- **Çıktı:** Tek bir overall score (0-100)
- **Kullanım:** Genel form değerlendirmesi
- **Performans:** Daha iyi (Test R² = 0.735)
- **Avantaj:** Basit, hızlı, yüksek performans

### Multi-Output Model (Önceki)
- **Çıktı:** 4 regional score (arms, legs, core, head)
- **Kullanım:** Detaylı regional feedback
- **Performans:** Daha düşük (Test R² = 0.37-0.41)
- **Avantaj:** Detaylı analiz

**Öneri:** Her ikisini de kullanın!
- Overall score → Single-output model
- Regional scores → Multi-output model veya rule-based

