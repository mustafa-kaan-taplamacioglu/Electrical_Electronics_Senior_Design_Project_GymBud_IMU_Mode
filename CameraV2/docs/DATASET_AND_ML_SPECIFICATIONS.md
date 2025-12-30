# Dataset ve ML Model Detaylı Özellikler
## (Dataset and ML Model Detailed Specifications)

---

## 1. DATASET YAPISI (Dataset Structure)

### 1.1 RepSample Dataclass - Kaydedilen Attribute'lar

Her rep için kaydedilen veriler:

```python
@dataclass
class RepSample:
    # METADATA
    timestamp: float              # Rep zaman damgası
    exercise: str                 # Egzersiz adı (e.g., "bicep_curls")
    rep_number: int               # Rep numarası (1, 2, 3, ...)
    user_id: str                  # Kullanıcı ID (default: "default")
    
    # RAW DATA (Ham Veriler)
    landmarks_sequence: List[List[Dict]]  # MediaPipe landmarks (her frame için 33 landmark)
                                          # Format: [[{x, y, z, visibility}, ...], ...]
                                          # Her frame: 33 landmark (MediaPipe format)
    
    imu_sequence: List[Dict]              # IMU sensor verileri
                                          # Her sample: {
                                          #   left_wrist: {quaternion, euler, accel, gyro},
                                          #   right_wrist: {quaternion, euler, accel, gyro},
                                          #   timestamp: float
                                          # }
    
    # EXTRACTED FEATURES (Çıkarılan Özellikler)
    features: Dict[str, float]            # Camera-based features (MediaPipe'den)
    imu_features: Dict[str, float]        # IMU-based features (sensor'lerden)
    
    # LABELS (Etiketler - Ground Truth)
    expert_score: Optional[float]         # Uzman skoru (0-100)
    user_feedback: Optional[str]          # Kullanıcı geri bildirimi ("perfect", "good", "bad")
    is_perfect_form: Optional[bool]       # Mükemmel form mu? (True/False)
    
    # REGIONAL SCORES (Bölgesel Skorlar)
    regional_scores: Dict[str, float]     # {"arms": 85.0, "legs": 90.0, "core": 80.0, "head": 95.0}
    regional_issues: Dict[str, List[str]] # {"arms": ["elbow flare", ...], ...}
    
    # ANGLE DATA (Açı Verileri)
    min_angle: Optional[float]            # Minimum açı (rep sırasında)
    max_angle: Optional[float]            # Maksimum açı (rep sırasında)
    range_of_motion: Optional[float]      # Hareket aralığı (max_angle - min_angle)
```

### 1.2 Dataset Format (Dosya Yapısı)

```
MLTRAINCAMERA/
├── {exercise}_{timestamp}/
│   ├── samples.json          # Tüm RepSample'lar (JSON format)
│   └── summary.csv           # Özet CSV (rep_number, scores, angles, vb.)

MLTRAINIMU/
├── {exercise}_{timestamp}/
│   ├── imu_samples.json      # IMU rep sequence'leri
│   └── summary.csv           # Özet CSV
```

**samples.json Örneği:**
```json
[
  {
    "timestamp": 1703123456.789,
    "exercise": "bicep_curls",
    "rep_number": 1,
    "landmarks_sequence": [
      [{"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.9}, ...], // Frame 1: 33 landmarks
      [{"x": 0.51, "y": 0.31, "z": 0.11, "visibility": 0.89}, ...], // Frame 2
      ...
    ],
    "imu_sequence": [
      {"left_wrist": {...}, "right_wrist": {...}, "timestamp": 1703123456.789},
      ...
    ],
    "features": {
      "left_elbow_rom": 120.5,
      "left_elbow_vel_mean": 45.2,
      ...
    },
    "expert_score": 85.0,
    "is_perfect_form": false,
    "regional_scores": {"arms": 85.0, "legs": 90.0, "core": 80.0, "head": 95.0},
    "min_angle": 40.0,
    "max_angle": 160.0,
    "range_of_motion": 120.0
  },
  ...
]
```

---

## 2. FEATURE EXTRACTION (Özellik Çıkarımı)

### 2.1 Camera Features (MediaPipe - Kamera Modu)

**Pipeline:**
```
MediaPipe Landmarks (33 joints) 
  → Normalize (Pelvis-center, Shoulder-width)
  → Extract Angles (8 joints: elbow, shoulder, hip, knee)
  → Calculate Features (ROM, Velocity, Acceleration, Smoothness, Temporal)
```

**Extracted Features:**

#### A. Range of Motion (ROM) Features
```python
# Her eklem için:
{angle_name}_rom          # Range of Motion (max - min açı)
{angle_name}_mean         # Ortalama açı
{angle_name}_std          # Standart sapma
{angle_name}_min          # Minimum açı
{angle_name}_max          # Maksimum açı
{angle_name}_median       # Medyan açı

# Örnek:
"left_elbow_rom": 120.5
"left_elbow_mean": 90.2
"right_shoulder_rom": 45.8
```

#### B. Dynamics Features (Hız, İvme, Smoothness)
```python
# Her eklem için:
{angle_name}_vel_mean     # Ortalama hız (deg/s)
{angle_name}_vel_max      # Maksimum hız
{angle_name}_acc_mean     # Ortalama ivme (deg/s²)
{angle_name}_smoothness   # Smoothness skoru (düşük = pürüzsüz)

# Örnek:
"left_elbow_vel_mean": 45.2
"left_elbow_vel_max": 120.5
"left_elbow_smoothness": 0.85
```

#### C. Temporal Features (Zamansal Özellikler)
```python
# Her eklem için:
{angle_name}_peaks        # Peak sayısı
{angle_name}_valleys      # Valley sayısı
{angle_name}_oscillation_rate  # Salınım oranı

# Örnek:
"left_elbow_peaks": 2.0
"left_elbow_oscillation_rate": 0.15
```

**Toplam Camera Features:** ~50-100 feature (8 eklem × ~6-12 feature/eklem)

**Matematiksel Hesaplamalar:**

1. **Angle Calculation (Açı Hesaplama):**
```python
# 3 nokta: p1 (parent), p2 (joint), p3 (child)
v1 = p1 - p2  # Vektör 1
v2 = p3 - p2  # Vektör 2

# Dot product ve magnitudes
dot = v1 · v2
mag1 = ||v1||
mag2 = ||v2||

# Açı (derece)
cos_angle = dot / (mag1 * mag2)
angle = arccos(cos_angle) * (180 / π)
```

2. **Velocity (Hız):**
```python
velocity = diff(angles) * fps  # diff = frame-to-frame fark
# Unit: degrees/second
```

3. **Acceleration (İvme):**
```python
acceleration = diff(velocity) * fps
# Unit: degrees/second²
```

4. **Smoothness (Pürüzsüzlük):**
```python
# Jerk (ivme değişimi) kullanarak
jerk = diff(acceleration) * fps
smoothness = mean(abs(jerk))
# Düşük smoothness = pürüzsüz hareket
```

5. **Normalization (Normalizasyon):**
```python
# Pelvis-center normalization
pelvis_center = (left_hip + right_hip) / 2
normalized_joints = joints - pelvis_center

# Shoulder-width normalization
shoulder_width = ||right_shoulder - left_shoulder||
normalized_joints = normalized_joints / shoulder_width
```

### 2.2 IMU Features (IMU Modu)

**Extracted Features:**

#### A. Euler Angles (Roll, Pitch, Yaw)
```python
# Her sensor için (left_wrist, right_wrist):
{sensor}_roll_mean        # Ortalama roll açısı
{sensor}_roll_std         # Standart sapma
{sensor}_roll_min         # Minimum
{sensor}_roll_max         # Maksimum
{sensor}_roll_range       # Range (max - min)
{sensor}_roll_median      # Medyan

# Aynı şekilde pitch ve yaw için:
{sensor}_pitch_mean, {sensor}_pitch_std, ...
{sensor}_yaw_mean, {sensor}_yaw_std, ...

# Örnek:
"lw_roll_mean": 15.5      # Left wrist roll mean
"rw_pitch_max": 45.2      # Right wrist pitch max
```

#### B. Quaternion Features
```python
# Her sensor için:
{sensor}_qw_mean, {sensor}_qw_std, ...  # Quaternion w component
{sensor}_qx_mean, {sensor}_qx_std, ...  # Quaternion x component
{sensor}_qy_mean, {sensor}_qy_std, ...  # Quaternion y component
{sensor}_qz_mean, {sensor}_qz_std, ...  # Quaternion z component
```

#### C. Cross-Sensor Features (Simetri, Koordinasyon)
```python
wrist_symmetry_mean       # Sol-sağ simetri (ortalaması)
wrist_symmetry_max        # Maksimum simetri farkı
```

#### D. Metadata
```python
sequence_length           # IMU sequence uzunluğu
has_left_wrist           # 1.0 veya 0.0
has_right_wrist          # 1.0 veya 0.0
```

**Toplam IMU Features:** ~50-70 feature (2 sensor × ~25-35 feature/sensor + cross-sensor)

---

## 3. ML MODEL DETAYLARI

### 3.1 Model Türleri

**1. Random Forest Regressor** (Varsayılan)
```python
RandomForestRegressor(
    n_estimators=100,      # Ağaç sayısı
    max_depth=10,          # Maksimum derinlik
    min_samples_split=5,   # Split için minimum sample
    random_state=42,       # Reproducibility
    n_jobs=-1              # Paralel işleme
)
```

**2. Gradient Boosting Regressor**
```python
GradientBoostingRegressor(
    n_estimators=100,      # Boosting iteration sayısı
    max_depth=5,           # Maksimum derinlik
    learning_rate=0.1,     # Öğrenme oranı
    random_state=42
)
```

**3. Ridge Regression** (Linear)
```python
Ridge(
    alpha=1.0              # Regularization strength
)
```

### 3.2 Train/Test/Validation Split

**Varsayılan Split:**
```python
# Train/Test Split
test_size = 0.2  # %20 test, %80 train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Cross-Validation:**
```python
# K-Fold Cross-Validation (varsayılan: K=5)
cv = 5  # 5-fold CV

# Train set'i 5 parçaya böler:
# Fold 1: Train on [2,3,4,5], Validate on [1]
# Fold 2: Train on [1,3,4,5], Validate on [2]
# ...
# Fold 5: Train on [1,2,3,4], Validate on [5]
```

**Validation Strategy:**
- **Training Set:** %80 (model eğitimi için)
- **Test Set:** %20 (final değerlendirme için)
- **Cross-Validation:** Training set üzerinde K-fold CV (hyperparameter tuning için)

### 3.3 Feature Scaling

**StandardScaler (Z-score Normalization):**
```python
# Her feature için:
scaled = (x - mean) / std

# Örnek:
# Feature "left_elbow_rom": mean=100, std=20
# Original value: 120
# Scaled: (120 - 100) / 20 = 1.0
```

**Neden Scaling?**
- Farklı feature'ların scale'leri farklı olabilir (ROM: 0-180°, velocity: 0-200°/s)
- Scaling, tüm feature'ları aynı ölçeğe getirir
- Random Forest için gerekli değil ama Ridge için kritik

### 3.4 Hyperparameter Tuning

**Random Forest Tuning:**
```python
param_dist = {
    'n_estimators': randint(50, 300),      # 50-300 ağaç
    'max_depth': [5, 10, 15, 20, None],    # Derinlik seçenekleri
    'min_samples_split': randint(2, 20),   # Split threshold
    'min_samples_leaf': randint(1, 10)     # Leaf minimum samples
}

# RandomizedSearchCV ile arama (n_iter=50 iteration)
```

**Gradient Boosting Tuning:**
```python
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, 10],
    'learning_rate': uniform(0.01, 0.2),   # 0.01-0.2 arası
    'min_samples_split': randint(2, 20)
}
```

### 3.5 Performance Metrics

**1. Mean Squared Error (MSE):**
```python
MSE = (1/n) * Σ(y_true - y_pred)²
# Birim: score² (e.g., 100² = 10000)
# Düşük = iyi
```

**2. Mean Absolute Error (MAE):**
```python
MAE = (1/n) * Σ|y_true - y_pred|
# Birim: score (e.g., 5.2 puan fark)
# Daha interpretable (ortalama kaç puan fark var?)
```

**3. R² Score (Coefficient of Determination):**
```python
R² = 1 - (SS_res / SS_tot)
# SS_res = Σ(y_true - y_pred)²
# SS_tot = Σ(y_true - y_mean)²
# Range: -∞ to 1.0 (1.0 = perfect, 0 = mean baseline, negatif = kötü)
```

**Örnek Output:**
```
Train MSE: 45.2
Test MSE:  52.8
Train MAE: 5.2
Test MAE:  6.1
Train R²:  0.875
Test R²:   0.842
```

### 3.6 Batch Size ve Epoch

**NOT:** Mevcut model **sklearn** tabanlı (Random Forest, Gradient Boosting, Ridge).
- **Batch Size:** Kullanılmıyor (sklearn batch processing yapmaz)
- **Epoch:** Kullanılmıyor (sklearn iterative değil, tek seferde eğitilir)

**Neden?**
- Random Forest: Ağaçlar bağımsız oluşturulur (paralel)
- Gradient Boosting: Sequential ama sklearn içinde otomatik
- Ridge: Analytical solution (tek adımda çözülür)

**Eğer Deep Learning kullanılsaydı:**
- Batch Size: 32, 64, 128 (genelde)
- Epoch: 50-200 (early stopping ile)

---

## 4. KAMERA MODU vs IMU MODU

### 4.1 Camera Mode (MediaPipe)

**Input:**
- MediaPipe landmarks (33 joints, ~30 FPS)
- Her frame: 33 landmark × (x, y, z, visibility) = 132 values

**Processing:**
1. Normalize (pelvis-center, shoulder-width)
2. Map to common format (33 → 13 joints)
3. Extract angles (8 joints: elbow, shoulder, hip, knee)
4. Calculate features (ROM, velocity, acceleration, smoothness)

**Output Features:** ~50-100 feature

**Avantajlar:**
- Tüm vücut görünürlüğü
- Postür analizi
- Eklem açıları (tam açısal bilgi)

**Dezavantajlar:**
- Kamera görüş açısı gereksinimi
- Işık koşullarına bağımlı
- Giyim/kıyafet etkisi

### 4.2 IMU Mode

**Input:**
- IMU sensor data (left_wrist, right_wrist)
- Her sample: quaternion (w,x,y,z), euler (roll,pitch,yaw), accel (x,y,z), gyro (x,y,z)
- Sample rate: ~20 Hz

**Processing:**
1. Extract euler angles (roll, pitch, yaw)
2. Extract quaternion components
3. Calculate statistics (mean, std, min, max, range, median)
4. Cross-sensor features (symmetry)

**Output Features:** ~50-70 feature

**Avantajlar:**
- Işık koşullarından bağımsız
- Giyim/kıyafet etkisi yok
- Yüksek hassasiyet (doğrudan hareket ölçümü)

**Dezavantajlar:**
- Sadece sensor'lerin olduğu bölgeler (şu an: wrists)
- Kalibrasyon gereksinimi
- Sensor takılma/bağlantı sorunları

### 4.3 Sensor Fusion Mode (Camera + IMU)

**Strateji:**
1. **Camera model** eğit (camera features ile)
2. **IMU model** eğit (IMU features ile)
3. **Fusion:** İki modelin prediction'larını birleştir

**Fusion Yaklaşımları:**
```python
# Yaklaşım 1: Weighted Average
final_score = w_camera * camera_score + w_imu * imu_score
# w_camera + w_imu = 1.0

# Yaklaşım 2: Voting/Ensemble
final_score = (camera_score + imu_score) / 2

# Yaklaşım 3: Confidence-based
if camera_confidence > imu_confidence:
    final_score = camera_score
else:
    final_score = imu_score
```

---

## 5. MODEL EĞİTİM AKIŞI (Training Pipeline)

### 5.1 Data Collection (Veri Toplama)

```
1. User starts workout session (ML Training Mode)
2. For each rep:
   - Collect landmarks_sequence (MediaPipe, ~30 FPS)
   - Collect imu_sequence (IMU sensors, ~20 Hz)
   - Calculate regional_scores (Arms, Legs, Core, Head)
   - Store min_angle, max_angle
3. Save to MLTRAINCAMERA/ and MLTRAINIMU/
```

### 5.2 Feature Extraction (Özellik Çıkarımı)

```
For each RepSample:
  1. Extract camera features:
     - Normalize landmarks
     - Calculate angles
     - Extract ROM, velocity, acceleration, smoothness
  2. Extract IMU features:
     - Calculate statistics (mean, std, min, max, range)
     - Cross-sensor features
```

### 5.3 Model Training (Model Eğitimi)

```
1. Load all RepSamples from MLTRAINCAMERA/ (or MLTRAINIMU/)
2. Filter by exercise (e.g., "bicep_curls")
3. Extract features (camera or IMU)
4. Prepare labels (expert_score or regional_scores average)
5. Split: Train (80%) / Test (20%)
6. Scale features (StandardScaler)
7. Train model:
   - Random Forest: Build 100 trees
   - Gradient Boosting: 100 iterations
   - Ridge: Analytical solution
8. Evaluate: MSE, MAE, R²
9. (Optional) Hyperparameter tuning with CV
10. Save model to models/form_score_{exercise}_{model_type}/
```

### 5.4 Model Inference (Tahmin)

```
For real-time prediction:
1. Get current rep's landmarks_sequence (or imu_sequence)
2. Extract features (same pipeline as training)
3. Scale features (using saved scaler)
4. Predict score: model.predict(features_scaled)
5. Clip to [0, 100] range
```

---

## 6. ÖRNEK: Bicep Curls İçin Feature Listesi

### Camera Features (Örnek):
```
left_elbow_rom: 120.5
left_elbow_mean: 90.2
left_elbow_std: 35.5
left_elbow_min: 40.0
left_elbow_max: 160.0
left_elbow_median: 95.0
left_elbow_vel_mean: 45.2
left_elbow_vel_max: 120.5
left_elbow_acc_mean: 25.3
left_elbow_smoothness: 0.85
left_elbow_peaks: 2.0
left_elbow_valleys: 2.0
left_elbow_oscillation_rate: 0.15

right_elbow_rom: 118.3
right_elbow_mean: 91.5
... (aynı şekilde tüm feature'lar)

left_shoulder_rom: 45.2
right_shoulder_rom: 43.8
left_hip_rom: 25.3
right_hip_rom: 24.1
left_knee_rom: 30.5
right_knee_rom: 29.8
```

### IMU Features (Örnek):
```
lw_roll_mean: 15.5
lw_roll_std: 5.2
lw_roll_min: 10.0
lw_roll_max: 25.0
lw_roll_range: 15.0
lw_roll_median: 15.2
lw_pitch_mean: 20.3
lw_pitch_std: 8.5
... (tüm pitch, yaw, quaternion features)

rw_roll_mean: 16.1
rw_roll_std: 5.5
... (sağ bilek için aynı)

wrist_symmetry_mean: 0.6
wrist_symmetry_max: 2.5
sequence_length: 45.0
has_left_wrist: 1.0
has_right_wrist: 1.0
```

---

## 7. MODEL PERFORMANCE BEKLENTİLERİ

**İyi Model İçin:**
- **Test MAE:** < 8.0 (ortalama 8 puan hatadan az)
- **Test R²:** > 0.80 (model %80+ variance açıklıyor)
- **Train-Test Gap:** Küçük (overfitting yok)

**Minimum Veri Gereksinimi:**
- **Per Exercise:** Minimum 10 rep (test için)
- **Önerilen:** 50-100 rep (robust model için)
- **Perfect Form Samples:** En az 5-10 rep (baseline için)

---

## 8. ÖZET TABLO

| Özellik | Camera Mode | IMU Mode |
|---------|-------------|----------|
| **Input** | MediaPipe landmarks (33 joints, 30 FPS) | IMU sensors (left/right wrist, 20 Hz) |
| **Features** | ~50-100 (angles, ROM, velocity, smoothness) | ~50-70 (euler, quaternion, statistics) |
| **Normalization** | Pelvis-center, Shoulder-width | Gerekli değil (zaten relative) |
| **Model** | Random Forest / Gradient Boosting / Ridge | Random Forest / Gradient Boosting / Ridge |
| **Train/Test Split** | 80/20 | 80/20 |
| **Cross-Validation** | 5-fold CV | 5-fold CV |
| **Scaling** | StandardScaler (Z-score) | StandardScaler (Z-score) |
| **Metrics** | MSE, MAE, R² | MSE, MAE, R² |
| **Batch Size** | N/A (sklearn) | N/A (sklearn) |
| **Epoch** | N/A (sklearn) | N/A (sklearn) |

---

## 9. SONUÇ

- **Dataset Format:** JSON (samples.json) + CSV (summary.csv)
- **Features:** Camera ~50-100, IMU ~50-70
- **Model:** Regression (0-100 score prediction)
- **Training:** 80/20 split, 5-fold CV
- **No Batch/Epoch:** sklearn tabanlı (batch processing yok)
- **Fusion:** İki ayrı model (camera + IMU) → ensemble

