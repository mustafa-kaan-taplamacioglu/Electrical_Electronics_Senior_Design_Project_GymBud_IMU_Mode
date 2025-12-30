# ML Training Camera Dataset

Bu klasör ML modeli eğitimi için kamera verilerini (MediaPipe landmarks) içerir.

## Klasör Yapısı

```
MLTRAINCAMERA/
├── bicep_curls/          # Bicep curl egzersizi verileri
├── squats/               # Squat egzersizi verileri
├── lunges/               # Lunge egzersizi verileri
├── pushups/              # Push-up egzersizi verileri
├── lateral_shoulder_raises/  # Lateral raise egzersizi verileri
├── tricep_extensions/    # Tricep extension egzersizi verileri
├── dumbbell_rows/        # Dumbbell row egzersizi verileri
└── dumbbell_shoulder_press/  # Shoulder press egzersizi verileri
```

## Veri Formatı

Her egzersiz klasörü altında otomatik olarak session klasörleri oluşturulur:

```
{exercise}/
└── {exercise}_{timestamp}/
    ├── samples.json      # Tüm rep samples (landmarks + features)
    └── summary.csv       # Özet CSV (rep_number, scores, angles, etc.)
```

## Kullanım

1. **Frontend'de "ML Training Mode" seçin**
2. Antrenman yapın - veriler otomatik olarak kaydedilir
3. Her rep tamamlandığında `samples.json` ve `summary.csv` dosyalarına yazılır

## Veri İçeriği

- **Landmarks Sequence**: Her rep için tüm frame'lerdeki MediaPipe landmarks
- **Regional Scores**: Kollar, Bacaklar, Gövde, Kafa skorları
- **Angles**: Min/max açılar ve hareket menzili
- **Features**: Kinematic features (ROM, velocity, acceleration, smoothness)

