# MediaPipe Landmark Kalibrasyon Dokümantasyonu

Bu doküman, MediaPipe'ın 33 landmark'ının kalibrasyon sürecini, hangi landmarkların zorunlu/opsiyonel olduğunu, verilerin nasıl loglandığını ve etkilerini açıklar.

---

## 📋 İçindekiler

1. [Zorunlu Landmarklar (Her Hareket İçin)](#zorunlu-landmarklar)
2. [Kalibrasyon Süreci](#kalibrasyon-süreci)
3. [Veri Loglama](#veri-loglama)
4. [Etkileri ve Sonuçları](#etkileri-ve-sonuçları)
5. [Örnek Senaryolar](#örnek-senaryolar)

---

## 🔴 Zorunlu Landmarklar (Her Hareket İçin)

### Bicep Curls
**Zorunlu Landmarklar:** `[11, 12, 13, 14, 15, 16, 23, 24]`
- 11: Sol omuz (left shoulder)
- 12: Sağ omuz (right shoulder)
- 13: Sol dirsek (left elbow)
- 14: Sağ dirsek (right elbow)
- 15: Sol bilek (left wrist)
- 16: Sağ bilek (right wrist)
- 23: Sol kalça (left hip)
- 24: Sağ kalça (right hip)

**Minimum Görünürlük Oranı:** %75 (8 landmark'tan en az 6'sı görünür olmalı)
**Kalibrasyon Mesajı:** "Upper body must be visible (shoulders, arms, waist)"

---

### Squats
**Zorunlu Landmarklar:** `[11, 12, 23, 24, 25, 26, 27, 28]`
- 11, 12: Omuzlar (reference için)
- 23, 24: Kalçalar
- 25, 26: Dizler
- 27, 28: Ayak bilekleri

**Minimum Görünürlük Oranı:** %75 (8 landmark'tan en az 6'sı görünür olmalı)
**Kalibrasyon Mesajı:** "Full body must be visible (shoulders to feet)"

---

### Lunges
**Zorunlu Landmarklar:** `[11, 12, 23, 24, 25, 26, 27, 28]`
- Squats ile aynı

**Minimum Görünürlük Oranı:** %75
**Kalibrasyon Mesajı:** "Full body must be visible (shoulders to feet)"

---

### Pushups
**Zorunlu Landmarklar:** `[11, 12, 13, 14, 15, 16, 23, 24, 25, 26]`
- 11, 12: Omuzlar
- 13, 14: Dirsekler
- 15, 16: Bilekler
- 23, 24: Kalçalar
- 25, 26: Dizler

**Minimum Görünürlük Oranı:** %75 (10 landmark'tan en az 7-8'i görünür olmalı)
**Kalibrasyon Mesajı:** "Side view required (shoulders, arms, hips, knees)"

---

### Lateral Shoulder Raises
**Zorunlu Landmarklar:** `[11, 12, 13, 14, 15, 16, 23, 24]`
- Bicep curls ile aynı

**Minimum Görünürlük Oranı:** %75
**Kalibrasyon Mesajı:** "Upper body must be visible (shoulders, arms, waist)"

---

### Tricep Extensions
**Zorunlu Landmarklar:** `[11, 12, 13, 14, 15, 16]`
- 11, 12: Omuzlar
- 13, 14: Dirsekler
- 15, 16: Bilekler
- **Not:** Kalçalar zorunlu değil (sadece kollar)

**Minimum Görünürlük Oranı:** %75 (6 landmark'tan en az 4-5'i görünür olmalı)
**Kalibrasyon Mesajı:** "Upper body must be visible (shoulders and arms)"

---

### Dumbbell Rows
**Zorunlu Landmarklar:** `[11, 12, 13, 14, 15, 16, 23, 24]`
- Bicep curls ile aynı

**Minimum Görünürlük Oranı:** %75
**Kalibrasyon Mesajı:** "Upper body must be visible (shoulders, arms, waist)"

---

### Dumbbell Shoulder Press
**Zorunlu Landmarklar:** `[11, 12, 13, 14, 15, 16, 23, 24]`
- Bicep curls ile aynı

**Minimum Görünürlük Oranı:** %75
**Kalibrasyon Mesajı:** "Upper body must be visible (shoulders, arms, waist)"

---

## 🔄 Kalibrasyon Süreci

### 1. Kalibrasyon Koşulları

**Görünürlük Eşiği:** `CALIBRATION_VISIBILITY_THRESHOLD = 0.5`
- Bir landmark'ın görünür sayılması için `visibility >= 0.5` olmalı

**Minimum Görünürlük Oranı:** `CALIBRATION_MIN_VISIBILITY_RATIO = 0.75`
- Zorunlu landmarkların en az %75'i görünür olmalı (örn: 8 landmark'tan 6'sı)

**Minimum Frame Sayısı:** `CALIBRATION_FRAMES = 20`
- 20 frame toplanmalı (20Hz throttling ile yaklaşık 1 saniye)

**Timeout:** `CALIBRATION_TIMEOUT = 8.0` saniye
- 8 saniye içinde 20 frame toplanamazsa kalibrasyon başarısız

### 2. Kalibrasyon Algoritması

```python
# Her frame için:
1. Zorunlu landmarkların görünürlüğünü kontrol et
2. Eğer %75'i görünürse → frame'i ekle
3. 20 frame toplandığında → ortalamaları hesapla

# Ortalama hesaplama (33 landmark için):
for i in range(33):
    visible_frames = [f[i] for f in frames if f[i].visibility >= 0.5]
    if len(visible_frames) > 0:
        avg[i] = {
            'x': average_x,
            'y': average_y,
            'calibrated': True,
            'visible_frames': len(visible_frames)
        }
    else:
        avg[i] = {
            'x': 0.0,
            'y': 0.0,
            'calibrated': False,
            'visible_frames': 0
        }
```

### 3. Vücut Oranları Hesaplama

Vücut oranları sadece ilgili landmarklar kalibre edildiyse hesaplanır:

**Shoulder Width:**
- Gereksinim: 11 ve 12 (sol ve sağ omuz) kalibre edilmeli
- Hesaplama: `abs(landmark[11]['x'] - landmark[12]['x'])`
- Eğer kalibre değilse: `self.shoulder_width = None`

**Hip Width:**
- Gereksinim: 23 ve 24 (sol ve sağ kalça) kalibre edilmeli
- Hesaplama: `abs(landmark[23]['x'] - landmark[24]['x'])`
- Eğer kalibre değilse: `self.hip_width = None`

**Torso Height:**
- Gereksinim: 11, 12, 23, 24 (tümü) kalibre edilmeli
- Hesaplama: `abs((shoulder_avg_y) - (hip_avg_y))`
- Eğer kalibre değilse: `self.torso_height = None`

**Kol ve Bacak Uzunlukları:**
- Benzer şekilde, ilgili landmarklar kalibre değilse `None` olarak kalır

### 4. İlk Pozisyonlar (Initial Positions)

Sadece kalibre edilmiş landmarklar `initial_positions` dictionary'sine eklenir:

```python
# Örnek: Dirsek kalibre edilmişse
if avg[13].get('calibrated', False):
    self.initial_positions['left_elbow'] = {'x': avg[13]['x'], 'y': avg[13]['y']}
# Kalibre edilmemişse → initial_positions'da olmaz
```

**İlk pozisyonları kaydedilen landmarklar:**
- left_shoulder, right_shoulder (11, 12)
- left_elbow, right_elbow (13, 14)
- left_wrist, right_wrist (15, 16)
- left_hip, right_hip (23, 24)
- left_knee, right_knee (25, 26)
- left_ankle, right_ankle (27, 28)
- spine_center (hesaplanmış: 11, 12, 23, 24 ortalaması)

---

## 💾 Veri Loglama

### 1. Kalibrasyon Sırasında

**Loglanan Veriler:**
- **TÜM 33 landmark** için ortalama pozisyonlar hesaplanır
- Kalibre edilen landmarklar: `calibrated: True`, gerçek (x, y) pozisyonları
- Kalibre edilmeyen landmarklar: `calibrated: False`, (0.0, 0.0) pozisyonları
- Her landmark için `visible_frames` sayısı kaydedilir

**Terminal Çıktısı:**
```
📊 Calibration complete: 28/33 landmarks calibrated
⚠️  Warning: Shoulders not calibrated (left: True, right: True)  # Eğer bir sorun varsa
```

### 2. Workout Sırasında

**Kaydedilen Veriler:**
- **TÜM 33 landmark** her frame'de kaydedilir (görünür olsun ya da olmasın)
- `landmarks_sequence`: Her frame için 33 landmark listesi
- Her landmark: `{'x': float, 'y': float, 'z': float, 'visibility': float}`

**Kayıt Formatı (samples.json):**
```json
{
  "rep_number": 1,
  "landmarks_sequence": [
    [  // Frame 1
      {"x": 0.5, "y": 0.3, "z": -0.2, "visibility": 0.99},  // Landmark 0
      {"x": 0.51, "y": 0.31, "z": -0.21, "visibility": 0.98},  // Landmark 1
      ...  // 33 landmark total
    ],
    [  // Frame 2
      ...  // 33 landmark
    ]
  ]
}
```

**Önemli Notlar:**
- Kalibre edilmemiş landmarklar da kaydedilir (visibility değerleriyle birlikte)
- Görünürlük değeri (`visibility`) her zaman loglanır (0.0-1.0 arası)
- Eğer bir landmark görünür değilse, MediaPipe hala bir (x, y) pozisyonu tahmin eder (ancak visibility düşük olur)

---

## ⚠️ Etkileri ve Sonuçları

### 1. Form Analizi Üzerindeki Etkileri

#### Kalibre Edilmemiş Landmarklar:

**Drift Kontrolü:**
- `_check_drift()` fonksiyonu kalibre edilmemiş landmarklar için `None` döndürür
- Örnek: Eğer sol dirsek (13) kalibre edilmemişse, "Sol dirsek oynuyor" hatası tespit edilmez
- Kod:
  ```python
  def _check_drift(self, current, initial, tolerance, label):
      if initial is None:  # Landmark kalibre edilmemişse
          return None  # Kontrolü atla
      # ... drift kontrolü
  ```

**Vücut Oranı Hesaplamaları:**
- Eğer `shoulder_width`, `torso_height`, veya `hip_width` `None` ise, bu oranlara dayalı kontroller atlanır
- Örnek: Eğer `torso_height` `None` ise, omuz kalkması (shoulder rise) kontrolü yapılmaz
- Kod:
  ```python
  if self.torso_height is not None:
      # Omuz kalkması kontrolü yap
  else:
      # Kontrolü atla
  ```

**Açı Hesaplamaları:**
- Açı hesaplamaları (örn: dirsek açısı) için kullanılan landmarklar görünür olmalı
- Eğer bir landmark görünür değilse, açı hesaplanamaz ve form analizi eksik kalır

### 2. Skor Hesaplama Üzerindeki Etkileri

**Bölgesel Skorlar:**
- Eğer bir bölge için kritik landmarklar kalibre edilmemişse, o bölgenin skorları eksik kalabilir
- Örnek: Eğer omuzlar (11, 12) kalibre edilmemişse, kollar bölgesi için bazı kontroller atlanır
- Ancak diğer kontroller (örn: dirsek açısı) hala çalışır

**Genel Skor:**
- Genel skor, mevcut kontrollerin ortalamasından hesaplanır
- Kalibre edilmemiş landmarklar, o kontrollerin yapılamamasına neden olur
- Bu, skorun daha az hassas olmasına neden olabilir (ancak yanlış yönlendirme yapmaz)

### 3. Rep Sayma Üzerindeki Etkileri

**Rep Sayma Algoritması:**
- Rep sayma, açı hesaplamalarına dayanır
- Eğer rep sayma için gerekli landmarklar görünür değilse, rep sayılamaz
- Örnek: Bicep curls için dirsekler (13, 14) ve bilekler (15, 16) görünür olmalı

### 4. Veri Seti Kalitesi Üzerindeki Etkileri

**ML Model Eğitimi:**
- Kalibre edilmemiş landmarklar hala kaydedilir (visibility değerleriyle)
- ML modeli, visibility değerlerine bakarak hangi landmarkların güvenilir olduğunu öğrenebilir
- Ancak, kalibre edilmemiş landmarkların (x, y) pozisyonları anlamlı olmayabilir

**Veri Temizleme:**
- Düşük visibility değerlerine sahip landmarklar, veri temizleme sırasında filtrelenebilir
- Örnek: `visibility < 0.5` olan landmarklar eğitim sırasında kullanılmayabilir

---

## 📊 Örnek Senaryolar

### Senaryo 1: Tüm Zorunlu Landmarklar Kalibre Edildi

**Durum:**
- Bicep curls için 8 zorunlu landmark'ın hepsi görünür
- Kalibrasyon başarılı: 28/33 landmark kalibre edildi

**Sonuçlar:**
- ✅ Tüm form kontrolleri aktif
- ✅ Drift kontrolleri yapılabilir
- ✅ Vücut oranları hesaplanabilir
- ✅ Skorlar tam olarak hesaplanır

---

### Senaryo 2: Bazı Zorunlu Landmarklar Kalibre Edilmedi

**Durum:**
- Bicep curls için sol bilek (15) kalibre edilemedi (görünürlük < 0.5)
- Kalibrasyon yine de başarılı (7/8 zorunlu landmark görünür, %75 eşiğini geçti)
- Kalibrasyon sonucu: 27/33 landmark kalibre edildi

**Sonuçlar:**
- ✅ Rep sayma çalışır (diğer landmarklar yeterli)
- ⚠️ Sol bilek için drift kontrolü yapılmaz (`initial_positions['left_wrist']` yok)
- ✅ Sağ bilek için drift kontrolü yapılır
- ✅ Diğer form kontrolleri normal çalışır
- ✅ Workout sırasında sol bilek verileri hala kaydedilir (visibility ile birlikte)

---

### Senaryo 3: Yetersiz Zorunlu Landmark Görünürlüğü

**Durum:**
- Bicep curls için sadece 5/8 zorunlu landmark görünür (%62.5 < %75)
- Kalibrasyon başarısız (timeout sonrası)

**Sonuçlar:**
- ❌ Kalibrasyon tamamlanamaz
- ❌ Workout başlatılamaz
- ✅ Kullanıcıya uyarı mesajı gösterilir: "Upper body must be visible (shoulders, arms, waist)"

---

### Senaryo 4: Omuzlar Kalibre Edilmedi (Bicep Curls)

**Durum:**
- Omuzlar (11, 12) kalibre edilemedi
- Diğer landmarklar kalibre edildi

**Sonuçlar:**
- ❌ `shoulder_width = None` (hesaplanamaz)
- ❌ `torso_height = None` (hesaplanamaz)
- ❌ Omuz kalkması (shoulder rise) kontrolü yapılmaz
- ✅ Dirsek drift kontrolü yapılır (eğer dirsekler kalibre edildiyse)
- ✅ Dirsek açısı kontrolleri yapılır
- ⚠️ Bazı form kontrolleri atlanır, skor daha az hassas olur

---

## 📝 Özet Tablosu

| Özellik | Zorunlu Landmarklar | Opsiyonel Landmarklar |
|---------|---------------------|----------------------|
| **Kalibrasyon Gereksinimi** | %75'i görünür olmalı | Görünür olması zorunlu değil |
| **İlk Pozisyon Kaydı** | ✅ Kaydedilir | ❌ Kalibre edilmezse kaydedilmez |
| **Workout Sırasında Kayıt** | ✅ Her frame'de kaydedilir | ✅ Her frame'de kaydedilir (visibility ile) |
| **Form Analizi** | ✅ Kontroller yapılır | ⚠️ Kalibre edilmezse kontroller atlanır |
| **Drift Kontrolü** | ✅ Yapılır | ❌ Kalibre edilmezse yapılmaz |
| **Vücut Oranları** | ✅ Hesaplanır (ilgili landmarklar varsa) | ⚠️ Kullanılmazsa hesaplanmaz |
| **Rep Sayma** | ✅ Gerekli landmarklar varsa çalışır | ⚠️ Kullanılmazsa etkilenmez |

---

## 🔍 Kod Referansları

**Kalibrasyon Kontrolü:**
- `CameraV2/api_server.py`: `FormAnalyzer.calibrate()` (satır 322-480)
- `CameraV2/api_server.py`: `check_required_landmarks()` (satır 153-182)

**Form Analizi:**
- `CameraV2/api_server.py`: `FormAnalyzer.check_form()` (satır 498+)
- `CameraV2/api_server.py`: `FormAnalyzer._check_drift()` (satır 486-496)

**Veri Kaydetme:**
- `CameraV2/dataset_collector.py`: `RepSample.landmarks_sequence` (satır 25)
- `CameraV2/dataset_collector.py`: `save_session()` (satır 189+)

---

## 🎯 Öneriler

1. **İdeal Kalibrasyon:** Tüm zorunlu landmarkların görünür olması en iyi sonuçları verir
2. **Minimum Gereksinim:** Zorunlu landmarkların en az %75'i görünür olmalı
3. **Veri Kalitesi:** ML modeli eğitimi için, yüksek visibility değerlerine sahip landmarklar tercih edilmelidir
4. **Kullanıcı Uyarıları:** Kalibrasyon sırasında hangi landmarkların görünür olmadığı kullanıcıya bildirilebilir

---

**Son Güncelleme:** 1 Ocak 2025
