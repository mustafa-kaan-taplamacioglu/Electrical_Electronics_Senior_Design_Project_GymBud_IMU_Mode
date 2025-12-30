# 📊 Veri Kayıt Özeti (Data Saving Summary)

## 📁 Klasör Yapısı

```
CameraV2/
├── MLTRAINCAMERA/              # Kamera verileri (Training Mode)
│   └── {exercise}/             # Örnek: bicep_curls
│       └── {session_id}/       # Örnek: bicep_curls_20251230_204351
│           ├── samples.json
│           └── summary.csv
│
└── MLTRAINIMU/                 # IMU verileri (Training Mode)
    └── {exercise}/             # Örnek: bicep_curls
        └── {session_id}/       # Örnek: bicep_curls_20251230_204351
            ├── imu_samples.json
            ├── summary.csv
            └── imu_samples.csv
```

**Session ID Formatı:** `{exercise}_{YYYYMMDD_HHMMSS}`  
**Örnek:** `bicep_curls_20251230_204351`

---

## 📋 Detaylı Dosya Tablosu

| **Veri Tipi** | **Klasör** | **Dosya Adı** | **Format** | **İçerik** | **Notlar** |
|---------------|------------|---------------|------------|------------|------------|
| **Kamera Verileri** | `MLTRAINCAMERA/{exercise}/{session_id}/` | `samples.json` | JSON | Tüm rep'lerin detaylı landmark sequence'ları | Her rep için: landmarks_sequence (33 landmark × N frames), regional_scores, angles, vb. |
| **Kamera Özeti** | `MLTRAINCAMERA/{exercise}/{session_id}/` | `summary.csv` | CSV | Rep-level özet bilgileri | Columns: rep_number, expert_score, is_perfect, user_feedback, arms_score, legs_score, core_score, head_score, min_angle, max_angle, range_of_motion |
| **IMU JSON** | `MLTRAINIMU/{exercise}/{session_id}/` | `imu_samples.json` | JSON | Tüm rep'lerin IMU sequence'ları | Her rep için: samples array (left_wrist, right_wrist, chest node data) |
| **IMU Özeti** | `MLTRAINIMU/{exercise}/{session_id}/` | `summary.csv` | CSV | Rep-level özet bilgileri | Columns: rep_number, num_samples, timestamp |
| **IMU Detaylı CSV** | `MLTRAINIMU/{exercise}/{session_id}/` | `imu_samples.csv` | CSV | Tüm IMU verileri (gymbud_imu_bridge formatı) | Her satır bir node için bir sample. Columns: timestamp, node_id, node_name, ax, ay, az, gx, gy, gz, qw, qx, qy, qz, roll, pitch, yaw, rep_number |

---

## 🔍 Dosya Formatları Detayı

### 1. **samples.json** (Kamera Verileri)

**Konum:** `MLTRAINCAMERA/{exercise}/{session_id}/samples.json`

**Format:** JSON Array

**Yapı:**
```json
[
  {
    "timestamp": 1767114952.459285,
    "exercise": "bicep_curls",
    "rep_number": 1,
    "landmarks_sequence": [
      [  // Frame 1: 33 landmark
        {"x": 0.5, "y": 0.3, "z": -0.2, "visibility": 0.99},
        {"x": 0.51, "y": 0.31, "z": -0.21, "visibility": 0.98},
        ...  // 33 landmark total
      ],
      [  // Frame 2: 33 landmark
        ...
      ],
      ...  // N frames (20Hz, ~50ms per frame)
    ],
    "regional_scores": {"arms": 85.0, "legs": 100, "core": 60.0, "head": 100},
    "regional_issues": {"arms": ["elbow flare"], ...},
    "min_angle": 5.8,
    "max_angle": 174.5,
    "range_of_motion": 168.7,
    "expert_score": null,
    "is_perfect_form": null,
    "features": {...},  // Extracted camera features
    "imu_features": null
  },
  {
    "rep_number": 2,
    ...
  },
  {
    "rep_number": 0,  // Session-level continuous data (tüm session boyunca)
    "landmarks_sequence": [...],  // Tüm frame'ler
    ...
  }
]
```

**Notlar:**
- `rep_number=0`: Session-level continuous data (tüm session boyunca, rep sayılsın ya da sayılmasın)
- `rep_number=1,2,3,...`: Sayılan rep'ler (rep tamamlandığında kaydediliyor)
- Her frame 33 MediaPipe landmark içerir
- 20Hz throttling (50ms per frame)

---

### 2. **summary.csv** (Kamera Özeti)

**Konum:** `MLTRAINCAMERA/{exercise}/{session_id}/summary.csv`

**Format:** CSV

**Columns:**
```
rep_number, expert_score, is_perfect, user_feedback, arms_score, legs_score, core_score, head_score, min_angle, max_angle, range_of_motion
```

**Örnek:**
```csv
rep_number,expert_score,is_perfect,user_feedback,arms_score,legs_score,core_score,head_score,min_angle,max_angle,range_of_motion
1,,,,50.0,100,74.0,100,7.7,179.8,172.1
2,,,,50.0,100,60.0,70.0,11.0,179.4,168.4
0,,,,,,,,,,
```

**Notlar:**
- Her satır bir rep'i temsil eder
- `rep_number=0`: Session-level continuous data özeti
- Boş değerler için boş string veya `null`

---

### 3. **imu_samples.json** (IMU Verileri)

**Konum:** `MLTRAINIMU/{exercise}/{session_id}/imu_samples.json`

**Format:** JSON Object

**Yapı:**
```json
{
  "session_id": "bicep_curls_20251230_204351",
  "total_reps": 25,
  "samples": [
    {
      "rep_number": 1,
      "rep_start_time": 1767116644.560874,
      "samples": [
        {
          "timestamp": 1767116644.560874,
          "left_wrist": {"ax": -0.0224, "ay": 0.1366, "az": -0.977, "gx": 1.61, "gy": -2.73, "gz": -0.07, "qw": 0.9999, "qx": 0.0056, "qy": -0.0003, "qz": -0.00003, "roll": 0.645, "pitch": -0.043, "yaw": -0.0037},
          "right_wrist": {...},
          "chest": {...},
          "rep_number": 1
        },
        {
          "timestamp": 1767116644.610874,
          "left_wrist": {...},
          "right_wrist": {...},
          "chest": {...},
          "rep_number": 1
        },
        ...
      ]
    },
    {
      "rep_number": 2,
      "rep_start_time": 1767116648.560616,
      "samples": [...]
    },
    {
      "rep_number": 0,  // Session-level continuous data
      "rep_start_time": 1767116644.560874,
      "samples": [...]  // Tüm session boyunca tüm IMU samples
    }
  ]
}
```

**Notlar:**
- `rep_number=0`: Session-level continuous data (tüm session boyunca)
- `rep_number=1,2,3,...`: Sayılan rep'ler
- Her sample 3 node içerebilir: left_wrist, right_wrist, chest
- 20Hz throttling per node (50ms per sample per node)

---

### 4. **summary.csv** (IMU Özeti)

**Konum:** `MLTRAINIMU/{exercise}/{session_id}/summary.csv`

**Format:** CSV

**Columns:**
```
rep_number, num_samples, timestamp
```

**Örnek:**
```csv
rep_number,num_samples,timestamp
1,24,1767116644.560874
2,30,1767116648.560616
0,386,1767116644.560874
```

**Notlar:**
- Her satır bir rep'i temsil eder
- `num_samples`: O rep'teki IMU sample sayısı
- `timestamp`: Rep başlangıç zamanı (Unix timestamp)
- `rep_number=0`: Session-level continuous data

---

### 5. **imu_samples.csv** (IMU Detaylı CSV - gymbud_imu_bridge Formatı)

**Konum:** `MLTRAINIMU/{exercise}/{session_id}/imu_samples.csv`

**Format:** CSV (gymbud_imu_bridge formatı + rep_number)

**Columns:**
```
timestamp, node_id, node_name, ax, ay, az, gx, gy, gz, qw, qx, qy, qz, roll, pitch, yaw, rep_number
```

**Örnek:**
```csv
timestamp,node_id,node_name,ax,ay,az,gx,gy,gz,qw,qx,qy,qz,roll,pitch,yaw,rep_number
1767116644.560874,1,left_wrist,-0.0224,0.1366,-0.977,1.61,-2.73,-0.07,0.9999,0.0056,-0.0003,-0.00003,0.645,-0.043,-0.0037,1
1767116644.560874,2,right_wrist,-0.1903,0.0229,-0.9584,0.91,-2.03,-0.28,0.9999,0.0009,0.0040,-0.0001,0.113,0.467,-0.013,1
1767116644.560874,3,chest,0.0123,0.0456,-0.9989,0.42,-1.15,0.12,0.9998,0.0021,0.0015,0.0002,0.234,0.087,0.015,1
1767116644.610874,1,left_wrist,-0.0166,0.1405,-0.9833,2.45,-5.25,-0.42,0.9999,0.0116,-0.0021,-0.0002,1.338,-0.246,-0.029,1
...
```

**Notlar:**
- **gymbud_imu_bridge formatına uygun:** `timestamp, node_id, node_name, ax, ay, az, gx, gy, gz, qw, qx, qy, qz, roll, pitch, yaw`
- **+ rep_number column:** Her sample'ın hangi rep'e ait olduğunu belirtir
- Her satır bir node için bir sample
- `node_id`: 1=left_wrist, 2=right_wrist, 3=chest
- `rep_number=0`: Session-level continuous data (rep sayılsın ya da sayılmasın)
- `rep_number>0`: Belirli bir rep'e ait sample'lar
- Tüm session boyunca tüm IMU verileri bu dosyada (rep sayılsın ya da sayılmasın)

---

## 🔄 Veri Toplama Akışı

### **Train Mode (`ml_mode='train'`)**

1. **Session Başlangıcı:**
   - `init` mesajı gelir
   - Camera collector başlatılır: `MLTRAINCAMERA/{exercise}/{session_id}/`
   - IMU collector başlatılır: `MLTRAINIMU/{exercise}/{session_id}/`
   - Aynı `session_id` kullanılır (senkronizasyon)

2. **Veri Toplama (Tracking State):**
   - Her frame'de (20Hz throttling):
     - Camera: `current_rep_landmarks` array'ine eklenir
     - Camera: `session_landmarks` array'ine eklenir (rep_number ile işaretlenmiş)
     - IMU: `current_rep_imu_samples` array'ine eklenir
     - IMU: `session_imu_samples` array'ine eklenir (rep_number ile işaretlenmiş)

3. **Rep Tamamlandığında:**
   - `current_rep_landmarks` → Camera collector'a `rep_number` ile kaydedilir
   - `current_rep_imu_samples` → IMU collector'a `rep_number` ile kaydedilir
   - Array'ler reset edilir (sıfırlanır)

4. **Session Bitişi:**
   - `end_session` mesajı veya workout completion
   - `session_landmarks` → Camera collector'a `rep_number=0` ile kaydedilir (tüm session)
   - `session_imu_samples` → IMU collector'a `rep_number=0` ile kaydedilir (tüm session)
   - Tüm veriler dosyalara kaydedilir

---

## 📊 Rep Number Mantığı

| **Rep Number** | **Açıklama** | **Ne Zaman Kaydedilir** |
|----------------|--------------|-------------------------|
| `0` | Session-level continuous data | Session bitişinde (tüm session boyunca, rep sayılsın ya da sayılmasın) |
| `1, 2, 3, ...` | Sayılan rep'ler | Rep tamamlandığında (phase 'up' → 'down' geçişi) |
| `null` veya boş | Yok | - |

**Önemli:**
- `rep_number=0`: Tüm session boyunca her frame kaydediliyor (sayılsın ya da sayılmasın)
- `rep_number>0`: Sadece sayılan (tamamlanan) rep'ler için kaydediliyor
- Bir frame hem `rep_number=0` içinde hem de kendi `rep_number`'ı ile kaydediliyor (örneğin rep 1 hem 0 içinde hem de 1 olarak)

---

## ✅ Özet

### **Kamera Verileri:**
- **Klasör:** `MLTRAINCAMERA/{exercise}/{session_id}/`
- **Dosyalar:**
  1. `samples.json` - Tüm rep'lerin detaylı landmark sequence'ları (JSON)
  2. `summary.csv` - Rep-level özet bilgileri (CSV)

### **IMU Verileri:**
- **Klasör:** `MLTRAINIMU/{exercise}/{session_id}/`
- **Dosyalar:**
  1. `imu_samples.json` - Tüm rep'lerin IMU sequence'ları (JSON)
  2. `summary.csv` - Rep-level özet bilgileri (CSV)
  3. `imu_samples.csv` - Detaylı IMU verileri (CSV, gymbud_imu_bridge formatı + rep_number)

### **Formatlar:**
- **JSON:** Structured data (rep-based organization)
- **CSV:** Tabular data (gymbud_imu_bridge formatı IMU için, summary için özet bilgiler)

### **Senkronizasyon:**
- Camera ve IMU verileri **aynı `session_id`** kullanır
- Camera ve IMU verileri **aynı `rep_number`** kullanır
- Camera ve IMU verileri **aynı timestamp** kullanır (frame-level senkronizasyon)

