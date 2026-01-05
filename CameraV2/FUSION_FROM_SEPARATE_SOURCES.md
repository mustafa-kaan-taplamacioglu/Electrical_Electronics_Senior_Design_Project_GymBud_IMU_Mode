# Sensor Fusion from Separate Data Sources (Ayrı Kaynaklardan Sensor Fusion)

## ✅ Yapılabilir!

Ayrı yerlerden toplanan IMU ve kamera verileri ile sensor fusion yapılabilir, ancak bazı gereksinimler vardır.

---

## 📋 Gereksinimler (Requirements)

### 1. **Veri Formatı Uyumluluğu**

#### **Kamera Verileri (Camera Data):**
- ✅ MediaPipe formatında olmalı (33 landmark)
- ✅ Her landmark: `{"x": float, "y": float, "z": float, "visibility": float}`
- ✅ `landmarks_sequence`: List[List[Dict]] - Her frame için 33 landmark
- ✅ `rep_number`: Her rep için numara (1, 2, 3, ...)
- ✅ `timestamp`: Unix timestamp (rep başlangıç zamanı)

**Örnek format:**
```json
{
  "rep_number": 1,
  "timestamp": 1703123456.789,
  "landmarks_sequence": [
    [  // Frame 1: 33 landmark
      {"x": 0.5, "y": 0.3, "z": -0.2, "visibility": 0.99},
      {"x": 0.51, "y": 0.31, "z": -0.21, "visibility": 0.98},
      ...  // 33 landmark total
    ],
    [  // Frame 2: 33 landmark
      ...
    ]
  ]
}
```

#### **IMU Verileri (IMU Data):**
- ✅ Format: Bizim sistem formatı (gymbud_imu_bridge formatı)
- ✅ `rep_number`: Her rep için numara (1, 2, 3, ...)
- ✅ `rep_start_time`: Unix timestamp (rep başlangıç zamanı)
- ✅ `samples`: IMU sample listesi

**Örnek format:**
```json
{
  "rep_number": 1,
  "rep_start_time": 1703123456.789,
  "samples": [
    {
      "timestamp": 1703123456.789,
      "left_wrist": {...},
      "right_wrist": {...}
    },
    ...
  ]
}
```

### 2. **Rep Eşleştirme (Rep Matching)**

Mevcut sistem **rep_number** ile eşleştirme yapıyor:
- ✅ Exact timestamp match yoksa bile, `rep_number` ile eşleştiriyor
- ✅ Örnek: Camera rep_number=1 ↔ IMU rep_number=1

**Mevcut kod (train_ml_models.py, line 517-522):**
```python
# Try to find by rep_number only (more flexible matching)
for imu_key, imu_data in imu_data_map.items():
    if imu_key[0] == cam_sample.rep_number:  # Same rep_number
        matching_imu = imu_data
        matched_count += 1
        break
```

### 3. **Timestamp Senkronizasyonu (Timestamp Synchronization)**

⚠️ **Önemli:** Timestamp'ler eşit olmasa bile fusion yapılabilir, ancak:

- ✅ **Rep-based fusion:** Rep başına feature extraction yapılıyor (temporal alignment gerekmiyor)
- ✅ **Feature-level fusion:** Her rep için ayrı ayrı feature çıkarılıyor, sonra birleştiriliyor
- ⚠️ **Frame-level fusion için:** Timestamp interpolation gerekir (şu an yapılmıyor)

**Şu anki yaklaşım:**
- Her rep için:
  1. Camera features extract ediliyor (landmarks_sequence'den)
  2. IMU features extract ediliyor (imu_sequence'den)
  3. İki feature set'i birleştiriliyor

Bu yaklaşım **timestamp eşitsizliğine toleranslı** çünkü:
- Feature extraction rep-level yapılıyor (temporal alignment gerekmiyor)
- Sadece rep_number eşleştirmesi yeterli

---

## 🔧 Nasıl Yapılır? (How to Do It?)

### **Adım 1: Veri Formatını Kontrol Et**

Kamera verilerinin MediaPipe formatında olduğundan emin ol:
- 33 landmark
- Her landmark: `{x, y, z, visibility}`
- `rep_number` mevcut

### **Adım 2: Verileri Doğru Klasörlere Koy**

```
MLTRAINCAMERA/
└── {exercise}/
    └── {session_id}/
        └── samples.json  # Arkadaşının kamera verileri (bizim format)

MLTRAINIMU/
└── {exercise}/
    └── {session_id}/
        └── imu_samples.json  # Senin IMU verilerin (bizim format)
```

**Önemli:** `session_id` farklı olabilir, ama `rep_number` eşleştirmesi yapılacak!

### **Adım 3: Rep Number'ları Senkronize Et**

Eğer rep_number'lar farklıysa, manuel olarak düzenle:
- Örnek: Camera'da rep_number=1, IMU'da rep_number=1 → ✅ Eşleşiyor
- Örnek: Camera'da rep_number=1, IMU'da rep_number=2 → ❌ Düzelt gerekir

### **Adım 4: Fusion Model Eğit**

```bash
python train_ml_models.py --exercise bicep_curls --fusion
```

Mevcut kod otomatik olarak:
1. Camera samples'ları yükler
2. IMU samples'ları yükler
3. `rep_number` ile eşleştirir (timestamp'e bakmaz, sadece rep_number'a bakar)
4. Feature extraction yapar (her rep için ayrı)
5. Fusion model eğitir

---

## ⚠️ Dikkat Edilmesi Gerekenler

### 1. **Rep Number Eşleştirmesi**

- ✅ Rep_number'lar eşleşmeli (1-1, 2-2, 3-3, ...)
- ✅ Farklı session_id'ler kullanılabilir (sorun değil)
- ✅ Timestamp'ler farklı olabilir (sorun değil, rep_number ile eşleştirme yapılıyor)

### 2. **Veri Kalitesi**

- ✅ Aynı exercise olmalı (bicep_curls, squats, vs.)
- ✅ Aynı hareket yapılmış olmalı (aynı exercise, aynı form)
- ⚠️ Farklı kişilerden toplanmış olabilir (sorun değil, ama model performansını etkileyebilir)

### 3. **Feature Extraction**

- ✅ Camera features: MediaPipe landmarks'den çıkarılıyor (rep-level)
- ✅ IMU features: IMU samples'lerden çıkarılıyor (rep-level)
- ✅ Her ikisi de rep-level feature extraction yapıyor (temporal alignment gerekmiyor)

---

## 📝 Örnek Kullanım Senaryosu

### **Senaryo:**
1. Sen IMU verilerini topladın: `MLTRAINIMU/bicep_curls/my_imu_session/imu_samples.json`
2. Arkadaşın kamera verilerini topladı: MediaPipe landmarks (33 landmark format)
3. Timestamp'ler farklı (farklı zamanlarda toplanmış)

### **Çözüm:**

**Adım 1: Arkadaşının verilerini bizim formata çevir**

```python
# Arkadaşının verilerini bizim samples.json formatına çevir
camera_data = [
    {
        "rep_number": 1,
        "timestamp": 1703123456.789,  # Önemli değil, rep_number eşleştirmesi yapılıyor
        "landmarks_sequence": [
            [  # Frame 1: 33 landmark
                {"x": 0.5, "y": 0.3, "z": -0.2, "visibility": 0.99},
                ...
            ],
            ...
        ]
    },
    {
        "rep_number": 2,
        ...
    }
]

# MLTRAINCAMERA/bicep_curls/friend_camera_session/samples.json olarak kaydet
```

**Adım 2: Rep number'ları kontrol et**

- IMU'da: rep_number 1, 2, 3, ...
- Camera'da: rep_number 1, 2, 3, ...
- ✅ Eşleşiyor → Devam et

**Adım 3: Fusion model eğit**

```bash
python train_ml_models.py --exercise bicep_curls --fusion
```

Kod otomatik olarak:
- Camera samples'ları yükler
- IMU samples'ları yükler
- `rep_number` ile eşleştirir (timestamp'e bakmaz)
- Fusion model eğitir

---

## 🔍 Mevcut Kod Detayları

### **train_fusion_model Fonksiyonu (train_ml_models.py)**

**Eşleştirme Mantığı:**
```python
# 1. Exact match (rep_number + timestamp)
key = (cam_sample.rep_number, round(cam_sample.timestamp, 1))
if key in imu_data_map:
    matching_imu = imu_data_map[key]

# 2. Flexible match (sadece rep_number)
else:
    for imu_key, imu_data in imu_data_map.items():
        if imu_key[0] == cam_sample.rep_number:  # Same rep_number
            matching_imu = imu_data
            break
```

**Feature Extraction:**
- Camera features: `extract_camera_features(sample)` → Rep-level features
- IMU features: `extract_imu_features(imu_sequence)` → Rep-level features
- Fusion: İki feature set'i birleştiriliyor

---

## ✅ Sonuç

**Evet, yapılabilir!** Ayrı yerlerden toplanan verilerle sensor fusion yapılabilir:

1. ✅ **Format uyumluluğu:** MediaPipe landmarks (33 landmark)
2. ✅ **Rep eşleştirme:** `rep_number` ile eşleştirme (timestamp'e bakmıyor)
3. ✅ **Feature-level fusion:** Rep-level feature extraction (temporal alignment gerekmiyor)
4. ⚠️ **Timestamp farklılıkları:** Sorun değil (rep_number ile eşleştirme yapılıyor)

**Sadece dikkat edilmesi gereken:**
- Rep_number'lar eşleşmeli
- Aynı exercise olmalı
- MediaPipe formatında olmalı (33 landmark)

