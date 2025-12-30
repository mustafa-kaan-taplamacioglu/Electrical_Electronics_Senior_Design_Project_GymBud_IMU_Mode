# 🤖 ML Model Kullanım Kılavuzu

## 📊 Model Özeti

### **Model Tipi: REGRESSION ✅**

- **Amaç:** Form skorunu 0-100 arası sürekli değer olarak tahmin etmek
- **Algoritmalar:** Random Forest, Gradient Boosting, Ridge
- **Performance Metrics:** MSE, MAE, R² Score

---

## 🚀 Kullanım

### **1. Temel Model Eğitimi**

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

---

### **2. Cross-Validation ile Eğitim**

```bash
# 5-fold cross-validation ile eğit
python train_form_model.py bicep_curls random_forest --cv
# veya kısa: -c
python train_form_model.py bicep_curls random_forest -c
```

**Avantajlar:**
- ✅ Daha robust performance estimation
- ✅ Overfitting tespiti
- ✅ Tüm veriyi kullanır

**Çıktı:**
```
📊 Cross-Validation (5-fold)...
   CV MAE: 6.25 (±1.2)
   CV R²:  0.87 (±0.05)

📈 Test Set Results:
   Test MAE: 6.50
   Test R²:  0.85
```

---

### **3. Hyperparameter Tuning ile Eğitim**

```bash
# RandomizedSearch ile hyperparameter tuning
python train_form_model.py bicep_curls random_forest --tune
# veya kısa: -t
python train_form_model.py bicep_curls random_forest -t
```

**Avantajlar:**
- ✅ Optimal hyperparameter'ları bulur
- ✅ Cross-validation ile robust
- ✅ Overfitting riskini azaltır

**Çıktı:**
```
🔍 Hyperparameter tuning enabled (random search)...
🔍 Searching best hyperparameters...
✅ Best hyperparameters:
   n_estimators: 200
   max_depth: 15
   min_samples_split: 5
   min_samples_leaf: 2
   Best CV MAE: 5.8
```

---

### **4. Kombine: CV + Tuning**

```bash
# Hem CV hem tuning
python train_form_model.py bicep_curls random_forest --cv --tune
# veya kısa
python train_form_model.py bicep_curls random_forest -c -t
```

---

## 📈 Performance Metrics

### **Mean Absolute Error (MAE)**
- **Birim:** Score (puan)
- **Yorumlama:** Ortalama mutlak hata
- **Hedef:** < 7 puan

**Örnek:**
```
MAE = 6.5
→ Model tahminleri ortalama 6.5 puan hata yapıyor
```

---

### **Mean Squared Error (MSE)**
- **Birim:** Score²
- **Yorumlama:** Büyük hatalara daha fazla ağırlık verir
- **Hedef:** < 50

**Örnek:**
```
MSE = 45.2
→ RMSE = √45.2 = 6.7 puan
```

---

### **R² Score (Coefficient of Determination)**
- **Aralık:** 0-1 (veya negatif)
- **Yorumlama:** Variance'ın ne kadarını açıklıyor
- **Hedef:** > 0.85

**Örnek:**
```
R² = 0.87
→ Model variance'ın %87'sini açıklıyor
```

---

## 🎯 Hyperparameter Tuning Detayları

### **Random Forest:**

**Tuned Parameters:**
- `n_estimators`: 50-300 (ağaç sayısı)
- `max_depth`: 5, 10, 15, 20, None
- `min_samples_split`: 2-20
- `min_samples_leaf`: 1-10

**Default (Tuning olmadan):**
```python
n_estimators=100
max_depth=10
min_samples_split=5
min_samples_leaf=1
```

---

### **Gradient Boosting:**

**Tuned Parameters:**
- `n_estimators`: 50-200
- `max_depth`: 3, 5, 7, 10
- `learning_rate`: 0.01-0.2
- `min_samples_split`: 2-20

**Default:**
```python
n_estimators=100
max_depth=5
learning_rate=0.1
min_samples_split=2
```

---

### **Ridge:**

**Tuned Parameters:**
- `alpha`: 0.1, 0.5, 1.0, 5.0, 10.0, 50.0

**Default:**
```python
alpha=1.0
```

---

## 📊 Beklenen Performans

### **Minimum (Baseline - Ridge):**
- **MAE:** < 10 puan
- **R²:** > 0.70
- **MSE:** < 100

### **İyi (Random Forest - Default):**
- **MAE:** < 7 puan
- **R²:** > 0.85
- **MSE:** < 50

### **Mükemmel (Random Forest - Tuned):**
- **MAE:** < 5 puan
- **R²:** > 0.90
- **MSE:** < 25

---

## 🔧 Python API Kullanımı

### **Temel Eğitim:**

```python
from ml_trainer import FormScorePredictor
from dataset_collector import DatasetCollector

# Dataset yükle
collector = DatasetCollector("dataset")
samples = collector.load_dataset()

# Model eğit
predictor = FormScorePredictor(model_type="random_forest")
results = predictor.train(samples, verbose=True)

# Model kaydet
predictor.save("models/form_score_predictor")
```

---

### **Hyperparameter Tuning:**

```python
# Tuning yap
best_params = predictor.tune_hyperparameters(
    samples,
    cv=5,
    method="random",  # "grid" or "random"
    n_iter=50,
    verbose=True
)

# Eğit (tuned hyperparameter'lar ile)
results = predictor.train(samples, verbose=True)
```

---

### **Cross-Validation ile Eğitim:**

```python
# CV ile eğit
results = predictor.train_with_cv(
    samples,
    cv=5,
    test_size=0.2,
    verbose=True
)

# CV scores:
# - cv_mae_mean: Ortalama CV MAE
# - cv_mae_std: CV MAE standart sapması
# - cv_r2_mean: Ortalama CV R²
# - cv_r2_std: CV R² standart sapması
```

---

### **Model Kullanımı:**

```python
# Model yükle
predictor = FormScorePredictor.load("models/form_score_predictor")

# Feature'ları çıkar
features = collector.extract_features(sample)

# Form skoru tahmin et
predicted_score = predictor.predict(features)
print(f"Predicted form score: {predicted_score:.1f}%")

# Feature importance
importances = predictor.get_feature_importance(top_n=10)
for feature, importance in importances.items():
    print(f"{feature}: {importance:.4f}")
```

---

## ⚙️ Komut Satırı Seçenekleri

```bash
# Temel kullanım
python train_form_model.py [exercise] [model_type]

# Seçenekler:
# --cv, -c          : Cross-validation kullan
# --tune, -t        : Hyperparameter tuning yap

# Örnekler:
python train_form_model.py bicep_curls random_forest
python train_form_model.py bicep_curls random_forest --cv
python train_form_model.py bicep_curls random_forest --tune
python train_form_model.py bicep_curls random_forest --cv --tune
python train_form_model.py all gradient_boosting -c -t
```

---

## 📝 Örnek Çıktı

```
============================================================
Training random_forest model
============================================================

📊 Dataset:
   Total samples: 150
   Features: 120
   Label range: 45.0 - 98.0

📦 Split:
   Train: 120 samples
   Test: 30 samples

🔍 Hyperparameter tuning enabled (random search)...
🔍 Searching best hyperparameters...
Fitting 5 folds for each of 50 candidates, totalling 250 fits
✅ Best hyperparameters:
   n_estimators: 200
   max_depth: 15
   min_samples_split: 5
   min_samples_leaf: 2
   Best CV MAE: 5.8

🚀 Training...

📊 Cross-Validation (5-fold)...
   CV MAE: 5.85 (±1.1)
   CV R²:  0.89 (±0.04)

📈 Test Set Results:
   Test MSE: 38.5
   Test MAE: 6.2
   Test R²:  0.87

📈 Top 10 Most Important Features:
   left_elbow_range: 0.1523
   left_elbow_min: 0.1234
   right_elbow_vel_mean: 0.0987
   ...

✅ Training complete!
```

---

## 🎯 Öneriler

1. **İlk eğitim:** Temel train (tuning olmadan)
2. **Performans yetersizse:** `--tune` ile hyperparameter tuning
3. **Robust evaluation:** `--cv` ile cross-validation
4. **Production:** Tuned model + CV evaluation

---

## ⚠️ Notlar

- **Minimum veri:** En az 20-30 sample (CV için daha fazla önerilir)
- **Tuning süresi:** RandomizedSearch daha hızlı (GridSearch daha uzun)
- **CV folds:** 5-fold önerilir (3-fold daha az veri için)

