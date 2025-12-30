# ML Model Training Rehberi

Bu rehber, toplanan verilerle ML modelini eğitmek için kullanılır.

## Hızlı Başlangıç

### 1. Veri Toplama (Frontend)

1. Frontend'de egzersiz yapın
2. Session sonunda "💾 Eğitim Setini Kaydet" seçeneğini seçin
3. Veriler `MLTRAINCAMERA/{exercise}/` klasörüne kaydedilir

### 2. Model Eğitimi

#### Seçenek A: Lokal Python Script (CPU)

```bash
cd CameraV2
python train_ml_models.py --exercise bicep_curls --camera-only
```

#### Seçenek B: Google Colab (GPU - Önerilen)

1. **Google Colab'ı açın:**
   - [Google Colab](https://colab.research.google.com/)
   - Runtime > Change runtime type > Hardware accelerator > **GPU** seçin

2. **Dosyaları yükleyin:**
   - `train_model_colab.ipynb` notebook'unu açın
   - Gerekli Python dosyalarını yükleyin:
     - `dataset_collector.py`
     - `ml_trainer.py`
     - `imu_feature_extractor.py` (opsiyonel)
   - `MLTRAINCAMERA/{exercise}/` klasörlerini yükleyin

3. **Notebook'u çalıştırın:**
   - Hücreleri sırayla çalıştırın
   - EXERCISE değişkenini değiştirin (örn: 'bicep_curls', 'squats')

4. **Modeli indirin:**
   - Eğitilmiş model `models/{exercise}/form_score_camera_random_forest/` klasöründe
   - Google Drive'a kaydedin veya ZIP olarak indirin

#### Seçenek C: Kendi GPU'nuz (CUDA)

```bash
# GPU kontrolü
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Model eğitimi (GPU otomatik kullanılacak)
python train_ml_models.py --exercise bicep_curls --camera-only
```

## Model Eğitimi Parametreleri

### train_ml_models.py

```bash
# Yeni model eğit
python train_ml_models.py --exercise bicep_curls --camera-only

# Mevcut modeli güncelle (sadece yeni verilerle)
python train_ml_models.py --exercise bicep_curls --mode update --camera-only

# Hem camera hem IMU modeli eğit
python train_ml_models.py --exercise bicep_curls

# Sadece IMU modeli eğit
python train_ml_models.py --exercise bicep_curls --imu-only
```

### Desteklenen Egzersizler

- `bicep_curls`
- `squats`
- `lunges`
- `pushups`
- `lateral_shoulder_raises`
- `tricep_extensions`
- `dumbbell_rows`
- `dumbbell_shoulder_press`

## Veri Gereksinimleri

- **Minimum:** 10 etiketli rep örneği
- **Önerilen:** 50+ rep örneği
- **İdeal:** 100+ rep örneği (farklı form kalitesiyle)

## Model Çıktıları

Eğitilmiş model şu dosyaları içerir:

```
models/{exercise}/form_score_camera_random_forest/
├── model.pkl              # Eğitilmiş model
├── scaler.pkl             # Feature normalizasyon scaler'ı
├── metadata.json          # Model metadata (tarih, performans, vb.)
└── baselines.json         # Perfect form baselines (opsiyonel)
```

## Performans Metrikleri

Model eğitimi sonrası şu metrikler gösterilir:

- **Train R²:** Training set'teki açıklama oranı
- **Test R²:** Test set'teki açıklama oranı
- **Test MAE:** Ortalama mutlak hata (daha düşük = daha iyi)
- **Test MSE:** Ortalama kare hata (daha düşük = daha iyi)

## Sorun Giderme

### "No samples found"

- `MLTRAINCAMERA/{exercise}/` klasörünü kontrol edin
- Veri toplama sırasında "Eğitim Setini Kaydet" seçeneğini seçtiğinizden emin olun

### "Not enough labeled samples"

- Daha fazla veri toplayın (minimum 10 rep)
- Verilerin `expert_score` veya `regional_scores` ile etiketlendiğinden emin olun

### GPU Kullanımı

- Google Colab'da GPU ücretsizdir (sınırlı süre)
- Lokal GPU için CUDA kurulumu gerekir
- CPU ile eğitim yavaş ama çalışır (küçük veri setleri için yeterli)

## İleri Seviye

### Hyperparameter Tuning

`ml_trainer.py` içinde `tune_hyperparameters()` fonksiyonunu kullanarak hyperparameter'ları optimize edebilirsiniz.

### Model Güncelleme

Yeni veri topladıktan sonra mevcut modeli güncellemek için:

```bash
python train_ml_models.py --exercise bicep_curls --mode update
```

Bu, sadece yeni verilerle modeli yeniden eğitir (hızlı).

## Daha Fazla Bilgi

- `ml_trainer.py`: Model eğitimi implementasyonu
- `train_ml_models.py`: Training script
- `dataset_collector.py`: Veri yükleme ve feature extraction
