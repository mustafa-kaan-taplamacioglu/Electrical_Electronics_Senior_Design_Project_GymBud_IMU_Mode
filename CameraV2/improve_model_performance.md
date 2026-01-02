# Model Performansını İyileştirme Rehberi

## Mevcut Durum Analizi

### Performans Metrikleri:
- **Camera Model**: Test R² = 0.372 (Train R² = 0.883) → OVERFITTING
- **Fusion Model**: Test R² = 0.373 (Train R² = 0.896) → OVERFITTING  
- **IMU Model**: Test R² = 0.078 (Train R² = 0.571) → OVERFITTING + Düşük performans

### Sorunlar:
1. **OVERFITTING**: Train R² çok yüksek, Test R² düşük (train >> test)
2. **Yetersiz Veri**: 101 sample yeterli olmayabilir
3. **IMU Feature Sayısı**: IMU modelde sadece 4 feature var (çok az!)
4. **Hyperparameter Tuning**: Model parametreleri optimize edilmemiş

## İyileştirme Stratejileri

### 1. Hyperparameter Tuning (En Hızlı İyileştirme)

**Mevcut Parametreler:**
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

**Önerilen İyileştirmeler:**
```python
# Option 1: Daha conservative (overfitting azaltma)
RandomForestRegressor(
    n_estimators=50,        # Daha az tree (100 → 50)
    max_depth=5,            # Daha shallow (10 → 5)
    min_samples_split=10,   # Daha fazla sample gerekli (5 → 10)
    min_samples_leaf=5,     # Yeni: leaf'te minimum sample
    max_features='sqrt',    # Yeni: feature subsampling
    random_state=42
)

# Option 2: Gradient Boosting (daha iyi performans potansiyeli)
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,            # Daha shallow
    learning_rate=0.1,
    min_samples_split=10,
    subsample=0.8,          # Row subsampling (overfitting önleme)
    random_state=42
)
```

### 2. Daha Fazla Veri Toplama

**Hedef:**
- Minimum 200-300 sample (şu an 101)
- Daha çeşitli form hataları içeren veriler
- Perfect form samples sayısını artırma (şu an sadece 10 perfect sample var!)

### 3. Feature Engineering İyileştirmeleri

**Camera Features:**
- Mevcut: 104 feature (iyi)
- Eklenebilir: Temporal features (velocity, acceleration of landmarks)

**IMU Features:**
- Mevcut: Sadece 4 feature (ÇOK AZ!)
- Sorun: `extract_imu_features` fonksiyonu tüm IMU datayı extract etmiyor
- Düzeltme gerekli: Accelerometer ve gyroscope feature'ları eksik!

### 4. Z-Score Perfect Form Selection Tuning

**Mevcut:**
```python
ZScorePerfectFormSelector(z_threshold=1.0, min_features_acceptable=0.9)
```

**Önerilen:**
```python
# Daha strict selection (daha az perfect sample, ama daha kaliteli)
ZScorePerfectFormSelector(z_threshold=0.8, min_features_acceptable=0.95)

# Veya daha lenient (daha fazla perfect sample)
ZScorePerfectFormSelector(z_threshold=1.5, min_features_acceptable=0.85)
```

### 5. Cross-Validation ile Model Seçimi

**Şu an:** Train/Test split (80/20)
**Önerilen:** 5-fold Cross-Validation (daha güvenilir metrikler)

### 6. Feature Selection

- Feature importance'ı kontrol et
- Düşük importance'lı feature'ları kaldır
- Correlation analysis (yüksek korelasyonlu feature'ları birleştir)

## Hızlı Uygulama Adımları

### Adım 1: IMU Feature Extraction'ı Düzelt (KRİTİK!)
IMU modelinde sadece 4 feature var - bu çok düşük! `imu_feature_extractor.py`'yi kontrol et.

### Adım 2: Hyperparameter Tuning Ekle
`ml_trainer.py`'ye GridSearchCV veya RandomizedSearchCV ekle.

### Adım 3: Cross-Validation Metrikleri
Train/test split yerine CV kullan.

### Adım 4: Daha Fazla Veri Topla
200-300 sample hedefi.

## Öncelik Sırası

1. **🔴 KRİTİK**: IMU feature extraction düzeltmesi (sadece 4 feature!)
2. **🟠 YÜKSEK**: Hyperparameter tuning (overfitting azaltma)
3. **🟡 ORTA**: Daha fazla veri toplama
4. **🟢 DÜŞÜK**: Feature engineering (mevcut features iyi)

