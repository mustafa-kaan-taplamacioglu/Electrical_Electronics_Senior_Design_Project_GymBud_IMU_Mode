# Dataset Toplama Gereksinimleri

## 📊 Önerilen Veri Miktarları

### Minimum (Proof of Concept)
**Amaç**: Sistemin çalıştığını doğrulamak

- **Rep sayısı**: 20-30 rep/hareket
- **Set sayısı**: 3-5 set/kişi
- **Rep/set**: 6-10 rep
- **Kişi sayısı**: 1-2 kişi
- **Form dağılımı**: 
  - Perfect: 5-10 rep (%30-40)
  - Good: 10-15 rep (%50-60)
  - Bad: 5-10 rep (%20-30)

**Toplam süre**: ~1-2 hafta

---

### Good (Production Ready) ⭐ **ÖNERİLEN**
**Amaç**: Gerçek kullanım için yeterli doğruluk

- **Rep sayısı**: 100-200 rep/hareket
- **Set sayısı**: 5-8 set/kişi
- **Rep/set**: 8-12 rep
- **Kişi sayısı**: 5-10 kişi
- **Form dağılımı**: 
  - Perfect: 30-60 rep (%30-40)
  - Good: 50-80 rep (%50-60)
  - Bad: 20-40 rep (%15-25)

**Toplam süre**: ~1-2 ay

**Örnek Plan**:
- 5 kişi × 6 set × 10 rep = 300 rep
- Her hareket için 100-200 rep seçilir

---

### Excellent (Research Grade)
**Amaç**: Yayın kalitesinde sonuçlar

- **Rep sayısı**: 500+ rep/hareket
- **Set sayısı**: 10-15 set/kişi
- **Rep/set**: 8-15 rep
- **Kişi sayısı**: 20+ kişi
- **Form dağılımı**: 
  - Perfect: 150-200 rep (%30-40)
  - Good: 250-300 rep (%50-60)
  - Bad: 100-150 rep (%20-30)

**Toplam süre**: ~3-6 ay

---

## 🎯 Her Hareket İçin Detaylı Plan

### Bicep Curls Örneği (Production Ready)

| Kişi | Set | Rep/Set | Toplam Rep | Perfect | Good | Bad |
|------|-----|---------|------------|---------|------|-----|
| K1   | 5   | 10      | 50         | 15      | 25   | 10  |
| K2   | 5   | 10      | 50         | 15      | 25   | 10  |
| K3   | 5   | 10      | 50         | 15      | 25   | 10  |
| K4   | 5   | 10      | 50         | 15      | 25   | 10  |
| K5   | 5   | 10      | 50         | 15      | 25   | 10  |
| **TOPLAM** | **25** | **50** | **250** | **75** | **125** | **50** |

**Seçilen**: 100-200 rep (best quality samples)

---

## 👥 Kişi Çeşitliliği Neden Önemli?

### Farklı Vücut Tipleri
- Kısa/Uzun boy
- İnce/Kaslı
- Farklı kol/bacak uzunlukları

### Farklı Form Seviyeleri
- Yeni başlayanlar (bad form örnekleri için)
- Orta seviye (good form örnekleri için)
- İleri seviye (perfect form örnekleri için)

### Farklı Stil Varyasyonları
- Hızlı/Yavaş yapma
- Farklı açılar
- Farklı range of motion

---

## 📈 Veri Kalitesi Kriterleri

### 1. Perfect Form Örnekleri (Baseline için kritik)
- **Minimum**: 10-15 perfect rep/hareket
- **İdeal**: 30-50 perfect rep/hareket
- **Kriterler**:
  - Regional scores >= 90
  - Expert score >= 95
  - Range of motion tam
  - Hiç kritik hata yok

### 2. Labeling Kalitesi
- Expert score: Detaylı 0-100 skor
- User feedback: "perfect", "good", "bad"
- Boolean: Perfect form (true/false)

### 3. Çeşitlilik
- Farklı kameradan açılar
- Farklı ışık koşulları
- Farklı arka planlar
- Farklı saatler (yorgun/taze)

---

## 🗓️ Toplama Planı Önerisi

### Haftalık Plan (Production Ready)

**Hafta 1-2**: Veri Toplama
- Günlük 3-5 set × 5 hareket
- Her sette 8-12 rep
- = ~150-300 rep/hafta
- 2 hafta = 300-600 rep toplam

**Hafta 3**: Labeling
- Perfect form rep'leri işaretle
- Expert score ver
- Quality check

**Hafta 4**: Model Eğitimi
- Train model
- Calculate baselines
- Test & validate

### Aylık Plan

**Ay 1**: Initial Collection
- 5 kişi × 100 rep = 500 rep
- Tüm hareketler için

**Ay 2**: Expansion
- 5 kişi daha ekle
- 5 kişi × 100 rep = 500 rep daha
- = 1000 rep toplam

**Ay 3**: Refinement
- Perfect form örnekleri artır
- Model retrain
- Baseline update

---

## ⚡ Hızlı Başlangıç (Minimum Viable)

Sadece test için:

1. **1 Kişi × 5 Set × 10 Rep = 50 Rep**
2. **Labeling**: En iyi 10-15 rep'i "perfect" işaretle
3. **Baseline**: Perfect rep'lerden baseline hesapla
4. **Model**: Tüm 50 rep ile train et

**Süre**: 1 gün

---

## 📊 Gerçekçi Öneriler

### Senaryo 1: Tek Kişi (Kendi Kullanımı)
- **30-50 rep/hareket**
- **Perfect**: 10-15 rep
- **Good/Bad**: 20-35 rep
- **Süre**: 1 hafta

### Senaryo 2: Küçük Grup (5 Kişi)
- **100-150 rep/hareket**
- **Perfect**: 30-50 rep
- **Good/Bad**: 70-100 rep
- **Süre**: 2-3 hafta

### Senaryo 3: Büyük Grup (20+ Kişi)
- **500+ rep/hareket**
- **Perfect**: 150-200 rep
- **Good/Bad**: 300+ rep
- **Süre**: 2-3 ay

---

## 🎯 Öncelik Sırası

### 1. Önce Perfect Form Örnekleri (En Kritik!)
**Amaç**: Baseline hesaplama
- **Minimum**: 10 perfect rep/hareket
- **İdeal**: 30-50 perfect rep/hareket
- **Öncelik**: EN YÜKSEK ⭐⭐⭐

### 2. Sonra Genel Dataset
**Amaç**: Model training
- **Minimum**: 20-30 rep/hareket
- **İdeal**: 100-200 rep/hareket
- **Öncelik**: Yüksek ⭐⭐

### 3. En Son Çeşitlilik
**Amaç**: Generalization
- Farklı kişiler
- Farklı koşullar
- Farklı form kaliteleri
- **Öncelik**: Orta ⭐

---

## 💡 Pratik Tavsiyeler

1. **Perfect Form Öncelikli**: İlk 20-30 perfect rep'i mutlaka topla
2. **Kalite > Miktar**: 50 iyi labeled sample > 200 unlabeled sample
3. **Düzenli Labeling**: Her hafta toplanan verileri label'le
4. **Iterative Training**: Her 50 yeni rep'te model'i retrain et
5. **Baseline Update**: Her 10 yeni perfect rep'te baseline'ı güncelle

---

## 📝 Özet Tablo

| Seviye | Rep/Hareket | Kişi | Set/Kişi | Rep/Set | Perfect Rep | Süre |
|--------|-------------|------|----------|---------|-------------|------|
| **Minimum** | 20-30 | 1-2 | 3-5 | 6-10 | 5-10 | 1 hafta |
| **Good** ⭐ | 100-200 | 5-10 | 5-8 | 8-12 | 30-60 | 1-2 ay |
| **Excellent** | 500+ | 20+ | 10-15 | 8-15 | 150-200 | 3-6 ay |

---

## 🚀 Hızlı Başlangıç Komutu

```bash
# 1. Minimum veri topla (30 rep)
# 2. Perfect form örnekleri işaretle (10 rep)
# 3. Baseline hesapla
python calculate_baselines.py bicep_curls

# 4. Model eğit
python train_form_model.py bicep_curls random_forest
```

