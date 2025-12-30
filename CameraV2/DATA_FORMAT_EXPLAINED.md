# 📊 Veri Kayıt Formatı ve Attributelar

## ✅ Eş Zamanlı Kayıt

**EVET, şu an eş zamanlı olarak IMU ve kamera verilerini kaydediyor!**

### Nasıl Çalışıyor?

1. **Aynı Rep Tamamlandığında:**
   - Camera collector'a rep kaydedilir (line 2696-2709)
   - IMU collector'a rep kaydedilir (line 2717-2734)
   - Her ikisi de **aynı `rep_number`** ile kaydedilir
   - Her ikisi de **aynı `rep_start_time`** kullanır

2. **Aynı Session ID:**
   - Her iki collector da **aynı `shared_session_id`** kullanır
   - Format: `{exercise}_{YYYYMMDD_HHMMSS}`
   - Örnek: `bicep_curls_20251226_140808`

3. **Aynı Anda Başlayıp Biter:**
   - Başlatma: `init` mesajında her ikisi de başlatılır (aynı session_id ile)
   - Bitiş: `end_session` mesajında her ikisi de kaydedilir

---

## 📁 Kayıt Formatları

### 1. **KAMERA VERİLERİ** (MLTRAINCAMERA)

**Klasör:** `MLTRAINCAMERA/{exercise}/{session_id}/`

**Dosyalar:**
- `samples.json` - Tüm rep'lerin detaylı verileri
- `summary.csv` - Rep özet bilgileri

#### **samples.json Formatı:**

```json
[
  {
    "timestamp": 1703123456.789,           // Rep zaman damgası (Unix timestamp)
    "exercise": "bicep_curls",             // Egzersiz adı
    "rep_number": 1,                        // Rep numarası (1, 2, 3, ...)
    "user_id": "default",                   // Kullanıcı ID
    
    // RAW DATA - Ham Veriler
    "landmarks_sequence": [                 // MediaPipe landmarks (her frame için)
      [                                     // Frame 1: 33 landmark
        {"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.9},  // Landmark 0 (nose)
        {"x": 0.51, "y": 0.31, "z": 0.11, "visibility": 0.89}, // Landmark 1
        ...                                 // Toplam 33 landmark (0-32)
      ],
      [                                     // Frame 2
        {"x": 0.52, "y": 0.32, "z": 0.12, "visibility": 0.88},
        ...
      ],
      ...                                   // ~20Hz'de toplanan frame'ler (50ms interval)
    ],
    
    "imu_sequence": null,                   // Training mode'da IMU burada değil (ayrı dosyada)
    
    // EXTRACTED FEATURES - Çıkarılan Özellikler
    "features": {                           // Camera-based features (MediaPipe'den)
      "left_elbow_rom": 120.5,              // Range of Motion
      "left_elbow_vel_mean": 45.2,          // Velocity (mean)
      "left_elbow_vel_std": 12.3,           // Velocity (std)
      "left_elbow_accel_mean": 8.5,         // Acceleration (mean)
      "left_elbow_smoothness": 0.85,        // Smoothness score
      "right_elbow_rom": 118.2,
      ...                                   // ~100+ feature
    },
    
    "imu_features": null,                   // IMU features burada değil (ayrı dosyada)
    
    // LABELS - Etiketler (Ground Truth)
    "expert_score": 85.0,                   // Uzman skoru (0-100)
    "user_feedback": null,                  // Kullanıcı geri bildirimi
    "is_perfect_form": false,               // Mükemmel form mu? (True/False)
    
    // REGIONAL SCORES - Bölgesel Skorlar
    "regional_scores": {                    // Her bölge için skor
      "arms": 85.0,                         // Kollar
      "legs": 90.0,                         // Bacaklar
      "core": 80.0,                         // Gövde
      "head": 95.0                          // Kafa
    },
    
    "regional_issues": {                    // Her bölge için tespit edilen sorunlar
      "arms": ["Sol dirsek oynuyor", "Sağ omuz kalkıyor"],
      "legs": [],
      "core": ["Gövde eğiliyor"],
      "head": []
    },
    
    // ANGLE DATA - Açı Verileri
    "min_angle": 40.0,                      // Minimum açı (rep sırasında)
    "max_angle": 160.0,                     // Maksimum açı (rep sırasında)
    "range_of_motion": 120.0                 // Hareket aralığı (max - min)
  },
  {
    "timestamp": 1703123458.123,
    "rep_number": 2,
    ...
  }
]
```

#### **summary.csv Formatı:**

```csv
rep_number,expert_score,is_perfect,user_feedback,arms_score,legs_score,core_score,head_score,min_angle,max_angle,range_of_motion
1,85.0,False,,85.0,90.0,80.0,95.0,40.0,160.0,120.0
2,88.0,False,,88.0,92.0,82.0,96.0,38.0,162.0,124.0
...
```

---

### 2. **IMU VERİLERİ** (MLTRAINIMU)

**Klasör:** `MLTRAINIMU/{exercise}/{session_id}/`

**Dosyalar:**
- `imu_samples.json` - Tüm rep'lerin IMU sequence'ları
- `summary.csv` - Rep özet bilgileri

#### **imu_samples.json Formatı:**

```json
{
  "session_id": "bicep_curls_20251226_140808",
  "total_reps": 12,
  "samples": [
    {
      "rep_number": 1,
      "rep_start_time": 1703123456.789,     // Rep başlangıç zamanı
      "samples": [                           // IMU samples (her sample ~20Hz'de)
        {
          "timestamp": 1703123456.789,      // Sample zaman damgası
          "imu_data": {                      // IMU sensor verileri
            "left_wrist": {                  // Sol bilek IMU
              "node_id": 1,
              "timestamp": 1703123456.789,
              "accel": {                     // Accelerometer (g)
                "x": 0.0,
                "y": -0.5144,
                "z": 0.8808
              },
              "gyro": {                      // Gyroscope (deg/s)
                "x": -1.26,
                "y": -5.39,
                "z": -0.56
              },
              "quaternion": {                // Quaternion (orientation)
                "w": 0.998,
                "x": 0.012,
                "y": 0.034,
                "z": 0.056
              },
              "euler": {                     // Euler angles (degrees)
                "roll": 5.2,
                "pitch": 12.5,
                "yaw": 178.3
              }
            },
            "right_wrist": {                 // Sağ bilek IMU
              "node_id": 2,
              "timestamp": 1703123456.789,
              "accel": {
                "x": 0.1,
                "y": -0.5234,
                "z": 0.8756
              },
              "gyro": {
                "x": -1.15,
                "y": -5.21,
                "z": -0.48
              },
              "quaternion": {
                "w": 0.997,
                "x": 0.015,
                "y": 0.028,
                "z": 0.052
              },
              "euler": {
                "roll": 4.8,
                "pitch": 13.1,
                "yaw": 179.1
              }
            },
            "chest": {                       // Göğüs IMU (opsiyonel)
              "node_id": 3,
              "timestamp": 1703123456.789,
              "accel": {
                "x": 0.05,
                "y": -0.1023,
                "z": 0.9945
              },
              "gyro": {
                "x": 0.12,
                "y": -0.23,
                "z": 0.15
              },
              "quaternion": {
                "w": 0.999,
                "x": 0.002,
                "y": 0.008,
                "z": 0.012
              },
              "euler": {
                "roll": 1.2,
                "pitch": 2.5,
                "yaw": 180.0
              }
            }
          }
        },
        {
          "timestamp": 1703123456.839,      // 50ms sonra (20Hz)
          "imu_data": {
            "left_wrist": {...},
            "right_wrist": {...},
            "chest": {...}
          }
        },
        ...                                  // ~20Hz'de toplanan samples (50ms interval per node)
      ]
    },
    {
      "rep_number": 2,
      "rep_start_time": 1703123458.123,
      "samples": [...]
    }
  ]
}
```

#### **summary.csv Formatı:**

```csv
rep_number,num_samples,timestamp
1,45,1703123456.789
2,48,1703123458.123
...
```

---

## 🔄 Senkronizasyon Detayları

### **Rep Eşleştirme:**

Aynı rep için camera ve IMU verilerini eşleştirmek için:

```python
# Camera rep'leri
camera_samples = load_camera_samples("bicep_curls_20251226_140808")
camera_rep_1 = camera_samples[0]  # rep_number=1

# IMU rep'leri
imu_data = load_imu_samples("bicep_curls_20251226_140808")
imu_rep_1 = imu_data['samples'][0]  # rep_number=1

# Artık camera_rep_1 ve imu_rep_1 aynı rep'i temsil ediyor!
# camera_rep_1.timestamp ≈ imu_rep_1.rep_start_time
```

### **Veri Toplama Hızı:**

- **Camera:** ~20Hz (50ms interval) - Her landmark için
- **IMU:** ~20Hz per node (50ms interval per node)
  - Left wrist: 20Hz
  - Right wrist: 20Hz
  - Chest: 20Hz (if available)

### **Örnek Timeline:**

```
Rep #1 başlangıcı: 1703123456.789

Camera Frame 1:  1703123456.789  (33 landmarks)
IMU Sample 1:    1703123456.789  (left_wrist, right_wrist, chest)

Camera Frame 2:  1703123456.839  (50ms sonra)
IMU Sample 2:    1703123456.839  (50ms sonra)

Camera Frame 3:  1703123456.889  (50ms sonra)
IMU Sample 3:    1703123456.889  (50ms sonra)

...

Rep #1 bitişi: 1703123458.123
```

---

## 📊 Feature Extraction (Özellik Çıkarımı)

### **Camera Features:**

MediaPipe landmarks'ten çıkarılan özellikler:
- **ROM (Range of Motion):** Her eklem için min-max açı farkı
- **Velocity:** Açısal hız (mean, std, max, min)
- **Acceleration:** Açısal ivme (mean, std, max, min)
- **Smoothness:** Hareket pürüzsüzlüğü (0-1)
- **Temporal Features:** Zaman bazlı özellikler

### **IMU Features:**

IMU sensor verilerinden çıkarılan özellikler:
- **Euler Angles:** Roll, Pitch, Yaw (mean, std, min, max, range)
- **Quaternions:** w, x, y, z (mean, std, min, max)
- **Accelerometer:** x, y, z (mean, std, min, max, range)
- **Gyroscope:** x, y, z (mean, std, min, max, range)

Her node için (left_wrist, right_wrist, chest) ayrı ayrı hesaplanır.

---

## ✅ Özet

1. **Eş Zamanlı Kayıt:** ✅ Evet, aynı rep_number ile kaydediliyor
2. **Session ID:** ✅ Aynı session_id kullanılıyor (senkronize)
3. **Format:** ✅ JSON (detaylı) + CSV (özet)
4. **Hız:** ✅ ~20Hz (camera) + ~20Hz per node (IMU)
5. **Eşleştirme:** ✅ rep_number ile eşleştirilebilir

