# Modüler Yapı - Çalıştırma Talimatları

## 📁 Yeni Modüler Yapı

`api_server.py` dosyası artık modüler bir yapıya ayrıldı:

```
CameraV2/
├── api_server.py (ANA DOSYA - ~2800 satır, sadece WebSocket ve route'lar)
├── utils/
│   └── pose_utils.py (~200 satır) - Helper functions
└── services/
    ├── form_analyzer.py (~870 satır) - FormAnalyzer class + EXERCISE_CONFIG
    ├── rep_counter.py (~361 satır) - RepCounter class
    ├── imu_rep_detector.py (~463 satır) - IMUPeriodicRepDetector class
    ├── feedback_service.py (~400 satır) - Feedback functions
    ├── ai_service.py (~200 satır) - OpenAI integration
    └── ml_service.py (~140 satır) - ML training functions
```

## 🚀 Nasıl Çalıştırılır?

### Önceki ile AYNI şekilde çalıştırılır:

```bash
cd CameraV2
python3 api_server.py
```

Veya:

```bash
cd CameraV2
uvicorn api_server:app --reload --port 8000
```

## ✅ Avantajlar

1. **Okunabilirlik**: Her dosya artık ~200-900 satır (önceden 4893 satır!)
2. **Bakım**: Her modül bağımsız olarak geliştirilebilir
3. **Test**: Modüller ayrı ayrı test edilebilir
4. **İş birliği**: Farklı geliştiriciler farklı modüllerde çalışabilir

## 📝 Değişiklikler

- **api_server.py**: Sadece FastAPI app, WebSocket endpoints ve route'lar
- **utils/pose_utils.py**: `calculate_angle`, `check_required_landmarks`, `get_bone_*` functions
- **services/form_analyzer.py**: `FormAnalyzer` class ve `EXERCISE_CONFIG`
- **services/rep_counter.py**: `RepCounter` class
- **services/imu_rep_detector.py**: `IMUPeriodicRepDetector` class
- **services/feedback_service.py**: Tüm feedback fonksiyonları
- **services/ai_service.py**: OpenAI entegrasyonu
- **services/ml_service.py**: ML model training

## ⚠️ Notlar

- Tüm import'lar otomatik olarak yapılandırıldı
- Eski kodlar hala `api_server.py` içinde (import'lar öncelikli olduğu için çalışmıyor)
- Hiçbir fonksiyonalite değişmedi, sadece dosya yapısı modülerleştirildi
- Backward compatible - aynı şekilde çalışır!

## 🔍 Test

Modüllerin doğru çalıştığını test etmek için:

```bash
cd CameraV2
python3 -c "import api_server; print('✅ api_server.py OK')"
```

Tüm modüller başarıyla import edildiyse, sistem hazır! 🎉

