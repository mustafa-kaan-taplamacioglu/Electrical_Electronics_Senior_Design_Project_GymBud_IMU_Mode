# 🤖 ML Model Specifications & Hyperparameters

## 📊 Model Özeti

### **Model Tipi: REGRESSION ✅**

**Amaç:** Form skorunu 0-100 arası sürekli değer olarak tahmin etmek

**Neden Regression?**
- Form skoru sürekli bir değer (0-100)
- Classification değil (binary veya multi-class değil)
- Skorun kendisini tahmin etmek istiyoruz

---

## 🎯 Mevcut Model Tipleri

### **1. Random Forest Regressor (Önerilen) ✅**

**Hyperparameters (Şu anki):**
```python
RandomForestRegressor(
    n_estimators=100,      # Ağaç sayısı
    max_depth=10,          # Maksimum derinlik
    min_samples_split=5,   # Split için minimum sample
    random_state=42,       # Reproducibility
    n_jobs=-1             # Paralel işlem
)
```

**Avantajlar:**
- ✅ Robust (outlier'lara karşı dayanıklı)
- ✅ Feature importance sağlar
- ✅ Overfitting riski düşük
- ✅ Hızlı eğitilir

**Dezavantajlar:**
- ❌ Interpretability düşük (black box)

---

### **2. Gradient Boosting Regressor**

**Hyperparameters (Şu anki):**
```python
GradientBoostingRegressor(
    n_estimators=100,      # Iteration sayısı
    max_depth=5,           # Maksimum derinlik
    learning_rate=0.1,     # Öğrenme hızı
    random_state=42        # Reproducibility
)
```

**Avantajlar:**
- ✅ Yüksek accuracy
- ✅ Feature importance sağlar

**Dezavantajlar:**
- ❌ Yavaş eğitilir
- ❌ Overfitting riski (tuning gerekir)

---

### **3. Ridge Regression (Baseline)**

**Hyperparameters (Şu anki):**
```python
Ridge(
    alpha=1.0              # Regularization strength
)
```

**Avantajlar:**
- ✅ Hızlı (linear model)
- ✅ Interpretable (coefficient'lar)
- ✅ Baseline için uygun

**Dezavantajlar:**
- ❌ Non-linear pattern'leri yakalayamaz
- ❌ Düşük accuracy (baseline için)

---

## 📈 Performance Evaluation Metrics

### **1. Mean Squared Error (MSE)**

**Formül:**
```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**Yorumlama:**
- Düşük = İyi
- Birim: Score² (örnek: 25.0 = ±5 puan hatası)
- Büyük hatalara daha fazla ağırlık verir

**Hedef:** < 50 (ortalama ±7 puan hatası)

---

### **2. Mean Absolute Error (MAE)**

**Formül:**
```
MAE = (1/n) * Σ|y_true - y_pred|
```

**Yorumlama:**
- Düşük = İyi
- Birim: Score (örnek: 5.0 = ortalama 5 puan hatası)
- Tüm hatalara eşit ağırlık verir

**Hedef:** < 7 (ortalama 7 puan hatası)

---

### **3. R² Score (Coefficient of Determination)**

**Formül:**
```
R² = 1 - (SS_res / SS_tot)
```

**Yorumlama:**
- 0-1 arası değer
- 1.0 = Mükemmel (tam tahmin)
- 0.0 = Baseline (ortalama değer kadar iyi)
- < 0 = Baseline'dan kötü

**Hedef:** > 0.85 (variance'ın %85'ini açıklıyor)

---

## 🔧 Hyperparameter Tuning (Şu anki durum: YOK ❌)

### **Mevcut Durum:**
- ❌ Hyperparameter'lar **sabit** (tuning yok)
- ❌ GridSearch/RandomSearch yok
- ❌ Cross-validation yok (sadece train/test split)

### **Önerilen İyileştirme:**

#### **1. GridSearchCV ile Hyperparameter Tuning**

```python
from sklearn.model_selection import GridSearchCV

# Random Forest için
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**Avantajlar:**
- ✅ Optimal hyperparameter'ları bulur
- ✅ Cross-validation ile robust
- ✅ Overfitting riskini azaltır

---

#### **2. RandomizedSearchCV (Daha Hızlı)**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Random Forest için
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_dist,
    n_iter=50,  # 50 kombinasyon dene
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

**Avantajlar:**
- ✅ Daha hızlı (tüm kombinasyonları denemez)
- ✅ Yeterince iyi sonuçlar verir

---

#### **3. K-Fold Cross-Validation**

**Şu anki:** Sadece train/test split (80/20)

**Önerilen:** 5-Fold Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X, y,
    cv=5,
    scoring='neg_mean_absolute_error'
)

print(f"CV MAE: {-scores.mean():.2f} (±{scores.std():.2f})")
```

**Avantajlar:**
- ✅ Daha robust performance estimation
- ✅ Overfitting tespiti
- ✅ Tüm veriyi kullanır (daha iyi evaluation)

---

## 📊 Mevcut vs Önerilen

| Özellik | Mevcut Durum | Önerilen |
|---------|-------------|----------|
| **Hyperparameter Tuning** | ❌ Yok (sabit değerler) | ✅ GridSearch/RandomSearch |
| **Cross-Validation** | ❌ Yok (sadece train/test) | ✅ 5-Fold CV |
| **Performance Metrics** | ✅ MSE, MAE, R² | ✅ MSE, MAE, R² + CV scores |
| **Model Selection** | ✅ 3 tip var | ✅ + Hyperparameter tuning |

---

## 🎯 Önerilen Hyperparameter Ranges

### **Random Forest:**
```python
{
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

### **Gradient Boosting:**
```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}
```

### **Ridge:**
```python
{
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
}
```

---

## 🚀 Implementation Plan

### **1. Hyperparameter Tuning Ekle**

```python
# ml_trainer.py içine ekle
def tune_hyperparameters(self, X, y, cv=5):
    """Tune hyperparameters using GridSearchCV."""
    if self.model_type == "random_forest":
        param_grid = {...}
        grid_search = GridSearchCV(...)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_
```

### **2. Cross-Validation Ekle**

```python
def train_with_cv(self, samples, cv=5, verbose=True):
    """Train with cross-validation."""
    X, y = self.prepare_features(samples)
    
    # Cross-validation scores
    cv_scores = cross_val_score(
        self.model, X, y,
        cv=cv,
        scoring='neg_mean_absolute_error'
    )
    
    # Train on full dataset
    self.model.fit(X, y)
    
    return cv_scores
```

### **3. Performance Reporting İyileştir**

```python
def evaluate(self, X_test, y_test):
    """Comprehensive evaluation."""
    y_pred = self.model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return metrics
```

---

## 📈 Beklenen Performans

### **Minimum (Baseline):**
- **MAE:** < 10 puan
- **R²:** > 0.70
- **MSE:** < 100

### **İyi:**
- **MAE:** < 7 puan
- **R²:** > 0.85
- **MSE:** < 50

### **Mükemmel:**
- **MAE:** < 5 puan
- **R²:** > 0.90
- **MSE:** < 25

---

## ✅ Checklist

- [x] Model tipi belirlendi (REGRESSION)
- [x] 3 model tipi eklendi (RF, GB, Ridge)
- [x] Performance metrics eklendi (MSE, MAE, R²)
- [ ] Hyperparameter tuning eklenecek
- [ ] Cross-validation eklenecek
- [ ] Performance reporting iyileştirilecek

---

## 🎯 Sonuç

**Mevcut Durum:**
- ✅ Regression modeli var
- ✅ 3 farklı algoritma var
- ✅ Temel performance metrics var
- ❌ Hyperparameter tuning yok
- ❌ Cross-validation yok

**Önerilen İyileştirmeler:**
1. GridSearch/RandomSearch ekle
2. 5-Fold cross-validation ekle
3. Performance reporting iyileştir
4. Hyperparameter ranges optimize et

