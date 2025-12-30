# 📊 ML Training Mode Veri Kaydetme Bilgisi

## 🗂️ Folder Yapısı

### **Kamera Verileri (Camera Data)**
- **Klasör:** `MLTRAINCAMERA/{exercise}/{session_id}/`
- **Dosyalar:**
  - `samples.json` - Tüm rep'lerin landmark sequence'ları (MediaPipe 33 landmarks)
  - `summary.csv` - Rep özet bilgileri (rep_number, timestamp, num_samples, form_score)

**Örnek:**
```
MLTRAINCAMERA/
  └── bicep_curls/
      └── bicep_curls_20251226_140808/
          ├── samples.json          # Landmark sequences (her rep için)
          └── summary.csv           # Rep özetleri
```

### **IMU Verileri (IMU Data)**
- **Klasör:** `MLTRAINIMU/{exercise}/{session_id}/`
- **Dosyalar:**
  - `imu_samples.json` - Tüm rep'lerin IMU sequence'ları (left_wrist, right_wrist, chest)
  - `summary.csv` - Rep özet bilgileri (rep_number, num_samples, timestamp)

**Örnek:**
```
MLTRAINIMU/
  └── bicep_curls/
      └── bicep_curls_20251226_140808/
          ├── imu_samples.json      # IMU sequences (her rep için)
          └── summary.csv           # Rep özetleri
```

## 🔄 Senkronizasyon

### **Session ID Formatı**
```
{exercise}_{YYYYMMDD_HHMMSS}
```
Örnek: `bicep_curls_20251226_140808`

**ÖNEMLİ:** Her iki collector (camera ve IMU) **AYNI session_id** kullanır!

### **Başlatma (Start)**
- ✅ **BAŞLATMA:** Camera ve IMU collector'ları **AYNI ANDA** başlatılıyor
- ✅ **Session ID:** Ortak bir `shared_session_id` oluşturulur ve her iki collector'a atanır
- ✅ **Senkronizasyon:** Her iki collector aynı `session_id` kullanır (garantili)

**Kod Akışı:**
1. `api_server.py` içinde ortak `shared_session_id` oluşturulur
2. Camera collector başlatılır → session_id override edilir
3. IMU collector başlatılır → session_id override edilir
4. Her ikisi de aynı session_id ile çalışır

### **Veri Toplama (Data Collection)**
Her rep tamamlandığında (`rep_result` oluşturulduğunda):
1. **Camera collector'a rep kaydedilir:**
   - `camera_collector.add_rep_sample()` çağrılır
   - `landmarks_sequence` kaydedilir (20Hz throttled)
   - Rep number: `rep_number`

2. **IMU collector'a rep kaydedilir:**
   - `imu_collector.add_rep_sequence()` çağrılır
   - `imu_sequence` kaydedilir (20Hz throttled per node)
   - Rep number: `rep_number` (aynı!)

3. Her ikisi de **aynı rep_number** ile kaydedilir

### **Veri Kaydetme (Save)**
`end_session` mesajı geldiğinde:
1. **Camera collector kaydedilir:**
   - `camera_collector.save_session()` çağrılır
   - `MLTRAINCAMERA/{exercise}/{session_id}/` klasörüne kaydedilir
   - `session_id` = `camera_collector.current_session_id`

2. **IMU collector kaydedilir:**
   - `imu_collector.save_session()` çağrılır
   - `MLTRAINIMU/{exercise}/{session_id}/` klasörüne kaydedilir
   - `session_id` = `imu_collector.current_session_id`

3. Her ikisi de **aynı session_id** ile kaydedilir (senkronize!)

## 📝 Kullanım Örnekleri

### **Session ID Eşleştirme**
Aynı egzersiz ve aynı zaman için kaydedilen verileri eşleştirmek için:

```python
from pathlib import Path

exercise = "bicep_curls"
session_id = "bicep_curls_20251226_140808"  # Her iki collector için aynı!

camera_path = Path("MLTRAINCAMERA") / exercise / session_id / "samples.json"
imu_path = Path("MLTRAINIMU") / exercise / session_id / "imu_samples.json"

# Her iki dosya da aynı session_id'ye sahip
assert camera_path.exists() == imu_path.exists()  # Aynı anda kaydedilmiş olmalı
```

### **Veri Yükleme**
```python
from dataset_collector import DatasetCollector
import json
from pathlib import Path

exercise = "bicep_curls"
session_id = "bicep_curls_20251226_140808"

# Camera verilerini yükle
camera_collector = DatasetCollector("MLTRAINCAMERA")
camera_samples = camera_collector.load_dataset(exercise=exercise)
# Filter by session_id if needed

# IMU verilerini yükle (JSON'dan direkt)
imu_path = Path("MLTRAINIMU") / exercise / session_id / "imu_samples.json"
with open(imu_path, 'r') as f:
    imu_data = json.load(f)
    imu_reps = imu_data['samples']  # List of rep data
```

### **Rep Eşleştirme**
Aynı rep_number'a sahip camera ve IMU verilerini eşleştirmek için:

```python
# Camera rep'leri
camera_rep_1 = camera_samples[0]  # rep_number=1
landmarks = camera_rep_1.landmarks_sequence  # Camera data

# IMU rep'leri
imu_rep_1 = imu_reps[0]  # rep_number=1
imu_samples = imu_rep_1['samples']  # IMU data

# Artık camera_rep_1 ve imu_rep_1 eşleşmiş durumda!
```

## ⚠️ Önemli Notlar

1. **Session ID Formatı:** `{exercise}_{timestamp}` formatındadır, saniye bazlıdır
2. **Senkronizasyon:** Her iki collector aynı `shared_session_id` ile başlatıldığı için veriler eşleştirilebilir
3. **Rep Numaraları:** Camera ve IMU verilerinde aynı `rep_number` kullanılır (1, 2, 3, ...)
4. **Timestamp:** Her rep için `rep_start_time` kullanılır (IMU için), `timestamp` kullanılır (Camera için)
5. **Data Rate:** Her iki veri de 20Hz'de toplanır (throttled)
6. **Aynı Anda Başlayıp Biter:** Evet, hem başlatma hem de kaydetme aynı anda yapılır

