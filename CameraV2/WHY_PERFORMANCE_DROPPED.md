# Neden Performans Düştü?

## 📊 Performans Karşılaştırması

### Eski Sonuçlar (Single-Output Model):
```
Train R²: 0.827
Test R²:  0.735  ✅ İYİ!
Train MAE: 0.39 puan
Test MAE:  0.25 puan  ✅ ÇOK İYİ!
Train MSE: 2.26
Test MSE:  0.10  ✅ MÜKEMMEL!
```

### Şimdiki Sonuçlar (Multi-Output Model):
```
Camera Model:
  Test R²: 0.372  ❌ Düşük
  Test MAE: 0.312 puan
  Test MSE: 0.348

IMU Model:
  Test R²: 0.406  ❌ Düşük
  Test MAE: 0.319 puan
  Test MSE: 0.269
```

## 🔍 Neden Bu Kadar Fark Var?

### 1. Model Tipi Değişti

**ÖNCE: Single-Output Model**
- Tek bir hedef: Overall form score (0-100)
- Basit regression problemi
- Daha kolay öğrenilebilir

**ŞİMDİ: Multi-Output Model**
- 4 farklı hedef: arms, legs, core, head scores
- Daha karmaşık problem
- 4 farklı target'ı aynı anda tahmin etmeye çalışıyor
- MultiOutputRegressor kullanılıyor (her region için ayrı model)

**Sonuç:** Multi-output genellikle single-output'tan daha zor bir problem!

---

### 2. Labeling Yaklaşımı

**ÖNCE (Muhtemelen):**
- Expert-rated overall score (tek bir skor)
- Veya belirli bir metodoloji ile hesaplanmış overall score

**ŞİMDİ:**
- Regional scores'dan average alınarak overall score hesaplanıyor
- Regional scores: arms, legs, core, head (4 ayrı skor)
- Bu skorların ortalaması kullanılıyor

**Etkisi:** Farklı labeling yaklaşımı = farklı target distribution

---

### 3. Veri Seti Farklı Olabilir

- Farklı sample'lar kullanılmış olabilir
- Farklı workout session'ları
- Farklı form kaliteleri

---

## 💡 Çözüm Önerileri

### Seçenek 1: Eski Single-Output Modeli Geri Getir ✅ ÖNERİLEN

**Avantajlar:**
- Daha basit problem (tek target)
- Daha iyi performans potansiyeli (test edilmiş: Test R² = 0.735)
- Daha hızlı eğitilir
- Daha kolay interpret edilebilir

**Dezavantajlar:**
- Regional scores bilgisi kaybolur
- Sadece overall score tahmin edilir

**Kullanım:**
```python
# train_ml_models.py'de
predictor = FormScorePredictor(model_type="random_forest", multi_output=False)  # False!
```

---

### Seçenek 2: Multi-Output Modeli İyileştir

**Yapılacaklar:**
1. Hyperparameter tuning (overfitting azaltma)
2. Daha fazla veri toplama (200-300 sample)
3. Feature engineering
4. Cross-validation

**Beklenen İyileşme:**
- Test R²: 0.37 → 0.50-0.60 (ama muhtemelen 0.735'e ulaşamaz)

---

### Seçenek 3: Hybrid Yaklaşım (İKİSİ DE!)

**Strateji:**
1. Single-output model → Overall score için (eski performans)
2. Multi-output model → Regional scores için (detaylı analiz)

**Kullanım:**
```python
# Overall score için
single_predictor = FormScorePredictor(multi_output=False)
overall_score = single_predictor.predict(features)['score']

# Regional scores için
multi_predictor = FormScorePredictor(multi_output=True)
regional_scores = multi_predictor.predict(features)  # {'arms': ..., 'legs': ..., ...}
```

---

## 🎯 ÖNERİ

**Kullanıcının eski performansına ulaşmak için:**

1. **Single-output model'i de eğit** (eski sonuçları geri getirmek için)
2. **Multi-output model'i de eğit** (regional feedback için)
3. **İkisini de kullan:**
   - Overall score → Single-output model'den
   - Regional scores → Multi-output model'den (veya rule-based'den)

Bu şekilde hem eski performansı koruyabilir, hem de regional detayları kullanabilirsiniz!

---

## 📝 Uygulama

Eğer single-output model'i de eğitmek isterseniz:

```bash
# Single-output model için
python3 train_ml_models.py --exercise bicep_curls --camera-only --single-output

# Veya train_ml_models.py'ye --single-output flag'i ekleyelim
```

**Öneri:** `train_ml_models.py`'ye `--single-output` ve `--multi-output` flag'leri ekleyip, kullanıcının seçim yapmasına izin verelim.

