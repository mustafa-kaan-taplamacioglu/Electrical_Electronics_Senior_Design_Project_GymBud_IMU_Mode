# ML Training IMU Dataset

Bu klasör ML modeli eğitimi için IMU sensör verilerini içerir.

## Klasör Yapısı

```
MLTRAINIMU/
├── bicep_curls/          # Bicep curl egzersizi IMU verileri
├── squats/               # Squat egzersizi IMU verileri
├── lunges/               # Lunge egzersizi IMU verileri
├── pushups/              # Push-up egzersizi IMU verileri
├── lateral_shoulder_raises/  # Lateral raise egzersizi IMU verileri
├── tricep_extensions/    # Tricep extension egzersizi IMU verileri
├── dumbbell_rows/        # Dumbbell row egzersizi IMU verileri
└── dumbbell_shoulder_press/  # Shoulder press egzersizi IMU verileri
```

## Veri Formatı

Her egzersiz klasörü altında otomatik olarak session klasörleri oluşturulur:

```
{exercise}/
└── {exercise}_{timestamp}/
    ├── samples.json      # Tüm rep samples (IMU sequences + features)
    └── summary.csv       # Özet CSV (rep_number, IMU stats, etc.)
```

## Kullanım

1. **Frontend'de "ML Training Mode" seçin**
2. IMU sensörlerini bağlayın (left_wrist, right_wrist, chest)
3. Antrenman yapın - IMU verileri otomatik olarak kaydedilir
4. Her rep tamamlandığında `samples.json` ve `summary.csv` dosyalarına yazılır

## Veri İçeriği

- **IMU Sequences**: Her rep için tüm IMU sample'ları (20 Hz)
  - Accelerometer: ax, ay, az (g)
  - Gyroscope: gx, gy, gz (deg/s)
  - Orientation: Quaternion (qw, qx, qy, qz)
  - Euler Angles: Roll, Pitch, Yaw (degrees)
- **Features**: Statistical features (mean, std, min, max, range, median)
  - Her node için: left_wrist, right_wrist, chest
  - Euler angles ve quaternion değerleri üzerinden

