# 🎯 Normalization & Robustness: Body Size, Environment, Clothing

## ❓ Sorunlar

### 1. **Farklı Vücut Boyutları**
- Uzun/kısa/orta boylu kişiler
- Farklı eklem mesafeleri
- Farklı vücut oranları

### 2. **Ortam Koşulları**
- Aydınlık/karanlık ortamlar
- Farklı kamera açıları
- Farklı arka planlar

### 3. **Kıyafet Farklılıkları**
- Farklı giysi tipleri
- Koyu/açık renkler
- Vücut hatlarını gizleyen kıyafetler

---

## ✅ Çözüm: Multi-Level Normalization

### **1. Pelvis-Center Normalization (Position Invariance)**

**Amaç:** Kamera mesafesi ve pozisyon farklılıklarını ortadan kaldır.

```python
# Pelvis (hip center) koordinatını orijin yap
pelvis_center = (left_hip + right_hip) / 2
normalized_pose = landmarks - pelvis_center
```

**Avantajlar:**
- ✅ Kamera mesafesi farklılıklarından bağımsız
- ✅ Kişinin ekrandaki pozisyonundan bağımsız
- ✅ Açı farklılıklarına daha toleranslı

**Kullanım:**
- Her frame için pelvis merkezi hesaplanır
- Tüm landmark'lar pelvis'e göre normalize edilir

---

### **2. Body Height Normalization (Scale Invariance)**

**Amaç:** Farklı boy uzunluklarını normalize et.

```python
# Body height hesapla (pelvis'ten en uzak landmark'a kadar)
body_height = max(
    norm(landmark - pelvis_center) 
    for landmark in all_landmarks
)

# Normalize
normalized_pose = (landmarks - pelvis_center) / body_height
```

**Avantajlar:**
- ✅ Uzun/kısa boylu kişiler için aynı özellik uzayı
- ✅ Eklem mesafeleri normalize edilir
- ✅ Tek bir model tüm boy uzunlukları için çalışır

**Not:** 
- Body height her rep'te hesaplanır (dinamik)
- Alternatif: Shoulder width'e göre normalize (daha stabil)

---

### **3. Shoulder Width Normalization (Alternative)**

**Amaç:** Shoulder width'e göre normalize (daha stabil ölçüm).

```python
shoulder_width = norm(left_shoulder - right_shoulder)
normalized_pose = landmarks / shoulder_width
```

**Avantajlar:**
- ✅ Daha stabil (shoulder genişliği sabit kalır)
- ✅ Body height'ten daha güvenilir ölçüm
- ✅ MediaPipe'ın doğru tespit ettiği bir ölçüm

**Kullanım:**
- Her frame için shoulder width hesaplanır
- Tüm landmark'lar shoulder width'e bölünür

---

### **4. Angle-Based Features (Size Invariant)**

**En önemli:** Açılar boy uzunluğundan bağımsızdır!

```python
# Örnek: Elbow angle
elbow_angle = angle(shoulder, elbow, wrist)  # Derece cinsinden
# Bu açı boy uzunluğundan BAĞIMSIZ!
```

**Mevcut Sistem:**
- ✅ `extract_joint_angles()` fonksiyonu zaten var
- ✅ Tüm açılar otomatik olarak hesaplanıyor
- ✅ Bu açılar boy uzunluğundan bağımsız!

**Sonuç:**
- Açı tabanlı özellikler kullanıldığında boy farklılıkları sorun olmaz
- ROM (Range of Motion), velocity, smoothness gibi özellikler normalize

---

## 🎯 Önerilen Strateji

### **Tek Model (Ensemble Gereksiz)**

**Neden?**
- Normalization ile boy uzunluğu farklılıkları ortadan kalkar
- Açı tabanlı özellikler zaten size-invariant
- Ensemble averaging gereksiz komplekslik yaratır

**Nasıl?**
1. **Feature Extraction:**
   - Önce pelvis-center normalization
   - Sonra body height veya shoulder width normalization
   - Açı tabanlı özellikler (zaten normalize)
   - ROM, velocity, smoothness gibi dinamik özellikler

2. **Model Training:**
   - Tek bir RandomForest/GradientBoosting modeli
   - Farklı boy uzunluklarından veri ile eğit
   - Normalize edilmiş özellikler kullan

3. **Inference:**
   - Aynı normalization pipeline'ı kullan
   - Tek model ile tahmin

---

## 🌍 Ortam Koşulları

### **Aydınlık/Karanlık**

**MediaPipe Robust mu?**
- ✅ MediaPipe **çok robust** ortam koşullarına karşı
- ✅ Düşük ışıkta da çalışır (ama kalite düşer)
- ✅ Yüksek ışıkta da çalışır

**Test Edilmesi Gerekenler:**
1. **Çok karanlık ortam** (< 50 lux)
   - MediaPipe kalitesi düşebilir
   - Çözüm: Minimum ışık gereksinimi koy

2. **Çok aydınlık ortam** (güneş ışığı)
   - Overexposure problemi olabilir
   - Çözüm: Kamera ayarları ile düzeltilebilir

3. **Kontrast düşük ortamlar**
   - Arka plan ile vücut ayrımı zor
   - Çözüm: MediaPipe genellikle başa çıkar

**Öneri:**
- Veri toplarken **çeşitli ışık koşulları** kullan
- Model farklı koşullarda test edilmeli
- Minimum kalite threshold koy (visibility > 0.5)

---

### **Kıyafet Farklılıkları**

**MediaPipe Skeleton-Based**

**İyi Haber:**
- ✅ MediaPipe **skeleton tabanlı** çalışır
- ✅ Kıyafet **önemli değil** (vücut hatlarını gizlese bile)
- ✅ Eklem noktalarını tespit eder (kıyafet rengi önemli değil)

**Test Edilmesi Gerekenler:**
1. **Çok gevşek kıyafetler**
   - Hareket sırasında kıyafet sallanabilir
   - Çözüm: MediaPipe eklem noktalarını doğru tespit eder

2. **Koyu/açık renkli kıyafetler**
   - Arka plan ile kontrast problemi
   - Çözüm: MediaPipe genellikle başa çıkar

3. **Vücut hatlarını gizleyen kıyafetler**
   - Örnek: Şalvar, bol etek
   - Çözüm: Skeleton-based olduğu için sorun yok

**Sonuç:**
- Kıyafet farklılıkları **büyük sorun yaratmaz**
- Ama veri toplarken **çeşitli kıyafetler** kullanılmalı (robustness için)

---

## 📊 Normalization Pipeline

### **Feature Extraction'da Normalization**

```python
# 1. Raw landmarks al
landmarks = get_mediapipe_landmarks(frame)

# 2. Pelvis-center normalization
pelvis_center = (landmarks[left_hip] + landmarks[right_hip]) / 2
centered_landmarks = landmarks - pelvis_center

# 3. Body height normalization
body_height = max(norm(lm - pelvis_center) for lm in centered_landmarks)
normalized_landmarks = centered_landmarks / body_height

# 4. Angle-based features (size-invariant)
elbow_angle = angle(shoulder, elbow, wrist)
knee_angle = angle(hip, knee, ankle)

# 5. ROM, velocity, smoothness (normalize edilmiş landmarks'tan)
rom = max_angle - min_angle
velocity = gradient(angle_series)
smoothness = spectral_arc_length(velocity)
```

**Şu anki durum:**
- ❌ `dataset_collector.py`'de normalization yok
- ❌ `feature_extractor.py`'de normalization yok
- ✅ Açı hesaplamaları var (size-invariant)

**Yapılacaklar:**
- ✅ Normalization pipeline ekle
- ✅ Feature extraction'da kullan

---

## 🔧 Implementation Plan

### **1. Dataset Collection'da Normalization**

```python
# dataset_collector.py içinde
def extract_features(self, sample: RepSample, fps: float = 30.0):
    # ... landmarks'ı al ...
    
    # NORMALIZATION EKLE
    from exercise_embeddings.joint_mapping import (
        normalize_pose_to_relative,
        normalize_pose_scale
    )
    
    # Pelvis-center normalization
    normalized_pose = normalize_pose_to_relative(landmarks_np, reference_joint=7)  # pelvis
    
    # Shoulder width normalization
    normalized_pose = normalize_pose_scale(normalized_pose)
    
    # Feature extraction (normalize edilmiş pose ile)
    features = extract_all_features(normalized_pose, fps=fps)
    
    return features
```

### **2. Real-Time Inference'da Normalization**

```python
# api_server.py içinde
def extract_features_from_landmarks(landmarks):
    # Aynı normalization pipeline'ı kullan
    normalized_pose = normalize_pose_to_relative(landmarks)
    normalized_pose = normalize_pose_scale(normalized_pose)
    features = extract_all_features(normalized_pose)
    return features
```

---

## 📈 Veri Toplama Stratejisi

### **Boy Uzunluğu Çeşitliliği**

**Minimum:**
- 1 uzun boylu kişi (180cm+)
- 1 orta boylu kişi (165-180cm)
- 1 kısa boylu kişi (<165cm)

**İdeal:**
- 2-3 kişi her kategoriden
- Toplam 6-9 kişi

**Sonuç:**
- Normalization ile tek model yeterli
- Ensemble averaging gereksiz

---

### **Ortam Çeşitliliği**

**Test Edilecekler:**
1. **Aydınlık ortamlar:**
   - Güneş ışığı (outdoor)
   - Yapay ışık (indoor)
   - Parlak ortam (stüdyo)

2. **Karanlık ortamlar:**
   - Düşük ışık (50-100 lux)
   - Minimum ışık (100-200 lux)
   - Normal iç mekan (200-500 lux)

3. **Arka plan çeşitliliği:**
   - Düz duvar
   - Karmaşık arka plan
   - Hareketli arka plan (dikkatli)

**Sonuç:**
- MediaPipe robust ama test edilmeli
- Minimum kalite threshold koy

---

### **Kıyafet Çeşitliliği**

**Test Edilecekler:**
1. **Dar kıyafetler:**
   - T-shirt + şort
   - Spor kıyafeti
   - Vücut hatlarını gösteren

2. **Gevşek kıyafetler:**
   - Bol tişört
   - Şalvar
   - Vücut hatlarını gizleyen

3. **Renk çeşitliliği:**
   - Açık renkli (beyaz, açık gri)
   - Koyu renkli (siyah, koyu mavi)
   - Renkli (kırmızı, mavi)

**Sonuç:**
- Kıyafet farklılıkları büyük sorun yaratmaz
- Ama çeşitlilik için test edilmeli

---

## 🎯 Sonuç ve Öneriler

### **1. Normalization Stratejisi**

**Önerilen:**
- ✅ **Pelvis-center + Shoulder width normalization**
- ✅ **Angle-based features (size-invariant)**
- ✅ **Tek model (ensemble gereksiz)**

**Yapılacaklar:**
1. Feature extraction'a normalization ekle
2. Dataset collection'da normalization kullan
3. Real-time inference'da aynı pipeline kullan

### **2. Veri Toplama**

**Minimum:**
- 3 farklı boy uzunluğundan kişiler
- 2-3 farklı ortam koşulu
- 2-3 farklı kıyafet tipi

**İdeal:**
- 6-9 farklı kişi (boy çeşitliliği)
- 5-10 farklı ortam (ışık çeşitliliği)
- 5-10 farklı kıyafet kombinasyonu

### **3. Model Training**

**Önerilen:**
- Tek RandomForest/GradientBoosting modeli
- Normalize edilmiş özellikler ile
- Çeşitli koşullardan veri ile eğit

**Sonuç:**
- Normalization ile boy farklılıkları sorun olmaz
- MediaPipe ile ortam/kıyafet farklılıkları minimize edilir
- Tek model yeterli, ensemble gereksiz

---

## 📝 Implementation Checklist

- [ ] `dataset_collector.py`'e normalization ekle
- [ ] `feature_extractor.py`'e normalization wrapper ekle
- [ ] Real-time inference'da normalization kullan
- [ ] Farklı boy uzunluklarından veri topla
- [ ] Farklı ortam koşullarında test et
- [ ] Farklı kıyafetlerle test et
- [ ] Model performansını değerlendir

---

**Normalization ile tek model yeterli! Ensemble averaging gereksiz.** 🎉

