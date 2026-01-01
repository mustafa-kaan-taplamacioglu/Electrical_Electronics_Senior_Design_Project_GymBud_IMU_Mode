# Performans Metrikleri ve Skor Hesaplama Açıklaması

Bu dokümanda, sistemin hangi metrikleri kullanarak skorları hesapladığı detaylı olarak açıklanmaktadır.

## Genel Skor Hesaplama Mantığı

### 1. Bölgesel Skorlar (Regional Scores)

Her egzersiz için **4 bölge** analiz edilir:
- **Kollar (Arms)**
- **Bacaklar (Legs)**
- **Gövde (Core)**
- **Kafa (Head)**

Her bölge için birden fazla metrik kontrol edilir ve her metrik için bir skor üretilir. Bölgesel skor, o bölgeye ait **tüm metrik skorlarının ortalaması** alınarak hesaplanır:

```python
arms_score = sum(arms_scores) / len(arms_scores) if arms_scores else 100
legs_score = sum(legs_scores) / len(legs_scores) if legs_scores else 100
core_score = sum(core_scores) / len(core_scores) if core_scores else 100
head_score = sum(head_scores) / len(head_scores) if head_scores else 100
```

### 2. Genel Form Skoru (Final Score)

Genel form skoru, bölgesel skorların **egzersiz tipine göre ağırlıklı ortalaması** alınarak hesaplanır:

**Bicep Curls için:**
- Kollar: %50
- Gövde: %30
- Kafa: %10
- Bacaklar: %10

```python
final_score = (arms_score * 0.5 + core_score * 0.3 + head_score * 0.1 + legs_score * 0.1)
```

---

## Bicep Curls - Detaylı Metrikler

**Bölgesel Ağırlıklar:**
- 💪 **Kollar (Arms): %50** (En önemli)
- 🏋️ **Gövde (Core): %30** (İkinci en önemli)
- 👤 **Kafa (Head): %10**
- 🦵 **Bacaklar (Legs): %10**

**Genel Skor Hesaplama:**
```
final_score = (arms_score * 0.5) + (core_score * 0.3) + (head_score * 0.1) + (legs_score * 0.1)
```

### 💪 Kollar (Arms) - Metrikler (Ağırlık: %50):

1. **Üst Kol Açısı (Upper Arm Angle from Vertical)**
   - **Metrik:** Üst kolun dikey eksenden sapma açısı
   - **Hedef:** 0-20° (dikey olmalı)
   - **Eşik:** >30° ise sorun
   - **Skor:** `max(50, 100 - açı)`
   - **Örnek:** 35° sapma → 65 puan

2. **Dirsek Oynama (Elbow Drift)**
   - **Metrik:** Dirseğin kalibrasyon sırasındaki başlangıç pozisyonundan sapma mesafesi
   - **Eşik:** Omuz genişliğinin %15'inden fazla
   - **Skor:** `max(40, 100 - (sapma/eşik) * 30)`
   - **Örnek:** "Sol dirsek oynuyor" → 40-70 puan arası

3. **Omuz Kalkması (Shoulder Rise)**
   - **Metrik:** Omuzun başlangıç pozisyonundan yukarı çıkma miktarı
   - **Eşik:** Gövde yüksekliğinin %8'i
   - **Skor:** `max(50, 100 - (çıkma/eşik) * 25)`
   - **Örnek:** "Sol omuz kalkıyor" → 50-75 puan arası

4. **Dirsek Omuz Üstünde (Critical)**
   - **Metrik:** Dirseğin omuzun üstünde olması (çok kritik hata)
   - **Eşik:** Dirsek Y < Omuz Y - 0.03
   - **Skor:** Sabit 20 puan (çok düşük!)
   - **Örnek:** "Sol dirsek omuzun üstünde!" → 20 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %30):

1. **Omuz Seviyesi (Shoulders Level)**
   - **Metrik:** Omuzların yatay eksenden sapma açısı
   - **Hedef:** 0° (tam yatay)
   - **Eşik:** >15° ise sorun
   - **Skor:** `max(60, 100 - açı * 2)`
   - **Örnek:** 20° eğik → 60 puan

2. **Gövde Eğilmesi (Torso Lean)**
   - **Metrik:** Gövdenin dikey eksenden sapma açısı
   - **Hedef:** <20° (dik durmalı)
   - **Eşik:** >20° ise sorun
   - **Skor:** `max(55, 100 - açı * 2)`
   - **Örnek:** 25° eğik → 50 puan

3. **Kalça Kayması (Hip Shift)**
   - **Metrik:** Kalçanın merkez çizgiden sapma mesafesi
   - **Eşik:** Kalça genişliğinin %10'undan fazla
   - **Skor:** `max(60, 100 - (sapma/kalça_genişliği) * 100)`
   - **Örnek:** "Kalça kayıyor" → 60-100 puan arası

4. **Omuz Rotasyonu (Shoulder Rotation)**
   - **Metrik:** Sol ve sağ omuzlar arasındaki Y ekseni farkı
   - **Eşik:** Gövde yüksekliğinin %12'sinden fazla
   - **Skor:** `max(50, 100 - (fark/gövde_yüksekliği) * 150)`
   - **Örnek:** "Omuzlar dönüyor" → 50-100 puan arası

### 🦵 Bacaklar (Legs) - Metrikler (Ağırlık: %10):

1. **Diz Açısı (Knee Angle)**
   - **Metrik:** Üst bacak ve alt bacak arasındaki açı
   - **Hedef:** 160-180° (neredeyse düz)
   - **Eşik:** <160° ise sorun
   - **Skor:** Sabit 60 puan (orta öncelik)
   - **Örnek:** "Bacaklar düz tutulmalı" → 60 puan

2. **Bacak Asimetrisi (Leg Asymmetry)**
   - **Metrik:** Sol ve sağ diz açıları arasındaki fark
   - **Eşik:** >15° fark
   - **Skor:** Sabit 65 puan
   - **Örnek:** "Bacaklar asimetrik" → 65 puan

### 👤 Kafa (Head) - Metrikler (Ağırlık: %10):

1. **Kafa Pozisyonu (Head Position - Forward/Backward)**
   - **Metrik:** Kafanın omuz seviyesine göre Y pozisyonu
   - **Hedef:** Omuz seviyesinde (±0.1)
   - **Eşik:** Omuz seviyesinden 0.15 birimden fazla aşağıda veya 0.1 birimden fazla yukarıda
   - **Skor:** 
     - Çok öne eğik: 70 puan
     - Çok geride: 75 puan
   - **Örnek:** "Kafan çok öne eğik" → 70 puan

2. **Kafa Hizalanması (Head Alignment - Left/Right)**
   - **Metrik:** Kafanın omuz merkezine göre X pozisyonu
   - **Hedef:** Omuz merkezinde (±0.08)
   - **Eşik:** >0.08 birim sapma
   - **Skor:** Sabit 75 puan
   - **Örnek:** "Kafa merkezde değil" → 75 puan

---

## Skor Hesaplama Örneği (Bicep Curls)

### Senaryo: Bir frame'de tespit edilen sorunlar:
1. Sol üst kol 35° açık → Arms: 65 puan
2. Sol dirsek oynuyor → Arms: 55 puan
3. Omuzlar 20° eğik → Core: 60 puan
4. Kalça kayıyor → Core: 70 puan
5. Kafa merkezde değil → Head: 75 puan
6. Bacaklar asimetrik → Legs: 65 puan

### Hesaplama:

**Bölgesel Skorlar:**
- **Arms:** (65 + 55) / 2 = **60.0 puan**
- **Core:** (60 + 70) / 2 = **65.0 puan**
- **Head:** 75 / 1 = **75.0 puan**
- **Legs:** 65 / 1 = **65.0 puan**

**Genel Form Skoru:**
```
final_score = (60.0 * 0.5) + (65.0 * 0.3) + (75.0 * 0.1) + (65.0 * 0.1)
            = 30.0 + 19.5 + 7.5 + 6.5
            = 63.5 puan
```

---

## Özel Durumlar

### 1. Kritik Hatalar (Critical Issues)
Eğer herhangi bir metrik skoru ≤30 ise, genel skor maksimum 40 ile sınırlandırılır:
```python
if any(s <= 30 for s in scores):
    final_score = min(final_score, 40)
```

### 2. Sorun Yoksa (No Issues)
Hiçbir sorun tespit edilmezse, tüm skorlar 88'e ayarlanır:
```python
if not scores:
    final_score = 88
    arms_score = 88
    legs_score = 88
    core_score = 88
    head_score = 88
```

### 3. Kalibre Edilmemişse (Not Calibrated)
Eğer kalibrasyon tamamlanmamışsa, tüm skorlar 100'e ayarlanır (hiçbir değerlendirme yapılmaz).

---

---

## Squats - Detaylı Metrikler

### 🦵 Bacaklar (Legs) - Metrikler (Ağırlık: %50):

1. **Üst Bacak Açısı (Thigh Angle from Horizontal)**
   - **Metrik:** Üst bacağın yatay eksenden açısı (0° = paralel = iyi derinlik)
   - **Hedef:** 0-20° (paralele yakın)
   - **Not:** Açı ne kadar küçükse, squat o kadar derin

2. **Alt Bacak Açısı (Shin Angle from Vertical)**
   - **Metrik:** Alt bacağın dikey eksenden sapma açısı
   - **Hedef:** <35° (dikey olmalı)
   - **Eşik:** >35° ise sorun
   - **Skor:** `max(50, 100 - açı)`

3. **Diz Takibi (Knee Tracking)**
   - **Metrik:** Diz genişliği / Ayak bileği genişliği oranı
   - **Hedef:** Dizler ayakların üstünde, içe çökmesin
   - **Eşik:** Diz genişliği < Ayak genişliği * 0.8
   - **Skor:** Sabit 40 puan (çok kritik!)

4. **Diz Açısı Asimetrisi**
   - **Metrik:** Sol ve sağ diz açıları arasındaki fark
   - **Eşik:** >15° fark
   - **Skor:** Sabit 65 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %40):

1. **Gövde Açısı (Torso Angle from Vertical)**
   - **Metrik:** Gövdenin dikey eksenden sapma açısı
   - **Hedef:** <45° (dik durmalı)
   - **Eşik:** >45° ise sorun
   - **Skor:** `max(40, 100 - açı)`

2. **Kalça Kayması (Hip Shift)**
   - **Metrik:** Kalçanın merkez çizgiden sapma mesafesi
   - **Eşik:** Kalça genişliğinin %15'inden fazla
   - **Skor:** `max(50, 100 - (sapma/kalça_genişliği) * 100)`

3. **Omuz Seviyesi (Shoulders Level)**
   - **Metrik:** Omuzların yatay eksenden sapma açısı
   - **Eşik:** >10°
   - **Skor:** `max(70, 100 - açı * 2)`

### 👤 Kafa (Head) - Metrikler (Ağırlık: %5):

1. **Kafa Pozisyonu (Öne Eğik)**
   - **Eşik:** Omuz seviyesinden 0.2 birimden fazla aşağıda
   - **Skor:** Sabit 60 puan

### 💪 Kollar (Arms) - Metrikler (Ağırlık: %5):

1. **Kol Asimetrisi**
   - **Eşik:** Sol ve sağ kol açıları arasında >20° fark
   - **Skor:** Sabit 75 puan

---

## Lunges - Detaylı Metrikler

### 🦵 Bacaklar (Legs) - Metrikler (Ağırlık: %50):

1. **Ön Üst Bacak Açısı (Front Thigh Angle from Horizontal)**
   - **Metrik:** Ön bacağın üst kısmının yatay eksenden açısı
   - **Hedef:** Alt pozisyonda yataya yakın (0-20°)

2. **Ön Alt Bacak Açısı (Front Shin Angle from Vertical)**
   - **Metrik:** Ön bacağın alt kısmının dikey eksenden sapma açısı
   - **Hedef:** <25° (dikey olmalı)
   - **Eşik:** >25° ise sorun
   - **Skor:** `max(55, 100 - açı * 1.5)`

3. **Arka Üst Bacak Açısı (Back Thigh Angle from Vertical)**
   - **Metrik:** Arka bacağın üst kısmının dikey eksenden sapma açısı
   - **Hedef:** <40° (dik olmalı)
   - **Eşik:** >40° ise sorun
   - **Skor:** `max(60, 100 - açı)`

4. **Diz-Ayak Bileği Hizası (Knee Over Ankle)**
   - **Metrik:** Ön dizin ayak bileğini geçmemesi
   - **Eşik:** >0.08 birim sapma
   - **Skor:** Sabit 55 puan

5. **Diz Açısı Asimetrisi**
   - **Eşik:** >20° fark
   - **Skor:** Sabit 65 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %40):

1. **Gövde Dikliği (Torso Upright)**
   - **Metrik:** Gövdenin dikey eksenden sapma açısı
   - **Hedef:** <20° (dik durmalı)
   - **Eşik:** >20° ise sorun
   - **Skor:** `max(50, 100 - açı * 2)`

2. **Kalça Karesi (Hips Square)**
   - **Metrik:** Kalça çizgisinin yatay eksenden sapma açısı
   - **Hedef:** 0° (tam yatay)
   - **Eşik:** >15°
   - **Skor:** `max(65, 100 - açı * 2)`

3. **Kalça Kayması**
   - **Eşik:** Kalça genişliğinin %15'inden fazla
   - **Skor:** `max(50, 100 - (sapma/kalça_genişliği) * 100)`

4. **Omuz Seviyesi**
   - **Eşik:** >10°
   - **Skor:** `max(70, 100 - açı * 2)`

---

## Pushups - Detaylı Metrikler

### 💪 Kollar (Arms) - Metrikler (Ağırlık: %40):

1. **Üst Kol Açısı (Upper Arm Angle from Horizontal)**
   - **Metrik:** Alt pozisyonda üst kolun yatay eksenden açısı
   - **Hedef:** ~45° (vücuda yakın)
   - **Not:** Ölçülür ama eşik belirtilmemiş

2. **Dirsek Açısı Asimetrisi**
   - **Eşik:** Sol ve sağ dirsek açıları arasında >15° fark
   - **Skor:** Sabit 65 puan

3. **Dirsek Açılması (Elbow Flare)**
   - **Metrik:** Dirsek genişliği / Omuz genişliği oranı
   - **Hedef:** Dirsekler vücuda yakın (omuz genişliğinin 1.8 katından küçük)
   - **Eşik:** Dirsek genişliği > Omuz genişliği * 1.8
   - **Skor:** Sabit 50 puan

4. **Bilek Pozisyonu**
   - **Eşik:** Sol ve sağ bilek Y pozisyonu arasında >0.1 birim fark
   - **Skor:** Sabit 70 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %40):

1. **Vücut Çizgisi (Body Line)**
   - **Metrik:** Gövde ve üst bacak açıları arasındaki fark
   - **Hedef:** Fark <20° (düz çizgi)
   - **Eşik:** >20° fark
   - **Skor:** 
     - Kalça çöküyorsa: `max(40, 100 - fark * 2)`
     - Kalça çok yüksekse: `max(50, 100 - fark * 2)`

2. **Omuz Simetrisi**
   - **Eşik:** >12°
   - **Skor:** `max(60, 100 - açı * 3)`

### 🦵 Bacaklar (Legs) - Metrikler (Ağırlık: %15):

1. **Bacak Stabilitesi**
   - **Not:** Pushups için bacaklar daha az kritik

---

## Lateral Shoulder Raises - Detaylı Metrikler

### 💪 Kollar (Arms) - Metrikler (Ağırlık: %50):

1. **Üst Kol Açısı (Upper Arm Angle from Vertical)**
   - **Metrik:** Üst kolun dikey eksenden sapma açısı
   - **Hedef:** Üstte ~80-100° (yatay), altta ~0-15°
   - **Not:** Ölçülür ama eşik belirtilmemiş

2. **Kol Asimetrisi**
   - **Eşik:** >15° fark
   - **Skor:** `max(55, 100 - fark * 2)`

3. **Dirsek Bükülmesi (Elbow Bend)**
   - **Hedef:** 150-170° (hafif bükülü, kilitli değil)
   - **Eşik:** 
     - >175° (kilitli): 70 puan
     - <140° (çok bükülü): 65 puan

4. **Bilek Pozisyonu**
   - **Eşik:** Bilek dirseğin üstünde (>0.05 birim)
   - **Skor:** Sabit 55 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %30):

1. **Omuz Kalkması (Shoulder Shrug)**
   - **Metrik:** Omuzun başlangıç pozisyonundan yukarı çıkma miktarı
   - **Eşik:** Gövde yüksekliğinin %8'inden fazla
   - **Skor:** Sabit 50 puan

2. **Gövde Yana Kayma**
   - **Eşik:** Sol ve sağ gövde açıları arasında >10° fark
   - **Skor:** Sabit 60 puan

3. **Omuz Seviyesi**
   - **Eşik:** >10°
   - **Skor:** `max(70, 100 - açı * 2)`

4. **Kalça Kayması**
   - **Eşik:** Kalça genişliğinin %10'undan fazla
   - **Skor:** `max(60, 100 - (sapma/kalça_genişliği) * 100)`

---

## Tricep Extensions - Detaylı Metrikler

### 💪 Kollar (Arms) - Metrikler (Ağırlık: %50):

1. **Üst Kol Açısı (Upper Arm Angle from Vertical)**
   - **Metrik:** Üst kolun dikey eksenden sapma açısı
   - **Hedef:** <25° (kafaya yakın, dikey)
   - **Eşik:** >25° ise sorun
   - **Skor:** `max(50, 100 - açı * 2)`

2. **Kol Asimetrisi**
   - **Eşik:** >15° fark
   - **Skor:** `max(55, 100 - fark * 2)`

3. **Dirsek Açısı Asimetrisi**
   - **Eşik:** >15° fark
   - **Skor:** Sabit 65 puan

4. **Dirsek Kayması (Upper Arm Drift)**
   - **Metrik:** Dirseğin başlangıç pozisyonundan sapma mesafesi
   - **Eşik:** Omuz genişliğinin %15'inden fazla
   - **Skor:** `max(45, 100 - (sapma/eşik) * 40)`

5. **Dirsek Açılması (Elbow Flare)**
   - **Metrik:** Dirsek genişliği / Kafa genişliği oranı
   - **Eşik:** Dirsek genişliği > Kafa genişliği * 2.5
   - **Skor:** Sabit 55 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %30):

1. **Gövde Stabilitesi**
   - **Eşik:** >15°
   - **Skor:** `max(60, 100 - açı * 2)`

2. **Omuz Seviyesi**
   - **Eşik:** >10°
   - **Skor:** `max(70, 100 - açı * 2)`

3. **Kalça Kayması**
   - **Eşik:** Kalça genişliğinin %10'undan fazla
   - **Skor:** `max(60, 100 - (sapma/kalça_genişliği) * 100)`

---

## Dumbbell Rows - Detaylı Metrikler

### 💪 Kollar (Arms) - Metrikler (Ağırlık: %40):

1. **Kol Asimetrisi**
   - **Eşik:** >20° fark
   - **Skor:** `max(55, 100 - fark * 2)`

2. **Dirsek Pozisyonu (Elbow Position)**
   - **Metrik:** Dirseğin kalça ve omuz arasında olması
   - **Eşik:** Dirsek çok açık (kalça/omuz çizgisinin 0.1 birim dışında)
   - **Skor:** Sabit 55 puan

3. **Dirsek Açısı Asimetrisi**
   - **Eşik:** >15° fark
   - **Skor:** Sabit 65 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %45):

1. **Gövde Açısı (Torso Angle)**
   - **Metrik:** Gövdenin dikey eksenden sapma açısı
   - **Hedef:** 30-60° (öne eğik)
   - **Eşik:** 
     - <30° (yetersiz eğik): 65 puan
     - >60° (çok eğik): 60 puan

2. **Sırt Düzliği (Back Straight)**
   - **Eşik:** >15°
   - **Skor:** Sabit 55 puan

3. **Omuz Rotasyonu**
   - **Eşik:** Gövde yüksekliğinin %12'sinden fazla
   - **Skor:** `max(50, 100 - (fark/gövde_yüksekliği) * 150)`

4. **Kalça Kayması**
   - **Eşik:** Kalça genişliğinin %10'undan fazla
   - **Skor:** `max(60, 100 - (sapma/kalça_genişliği) * 100)`

### 👤 Kafa (Head) - Metrikler (Ağırlık: %10):

1. **Kafa Pozisyonu**
   - **Eşik:** 
     - Çok aşağıda (>0.15 birim): 65 puan
     - Çok yukarıda (>0.1 birim): 70 puan

2. **Kafa Hizalanması**
   - **Eşik:** >0.1 birim sapma
   - **Skor:** Sabit 75 puan

---

## Dumbbell Shoulder Press - Detaylı Metrikler

### 💪 Kollar (Arms) - Metrikler (Ağırlık: %50):

1. **Kol Asimetrisi**
   - **Eşik:** >15° fark
   - **Skor:** `max(55, 100 - fark * 2)`

2. **Dirsek Açısı**
   - **Hedef:** Üstte ~170-180° (neredeyse düz)
   - **Eşik:** >175° ise (tam açık, kabul edilebilir)
   - **Not:** Ölçülür ama skor belirtilmemiş

3. **Bilek Pozisyonu**
   - **Eşik:** El omuz seviyesinin 0.1 birimden fazla altında
   - **Skor:** Sabit 60 puan

4. **Dirsek Genişliği**
   - **Eşik:** Dirsek genişliği > Omuz genişliği * 2
   - **Skor:** Sabit 60 puan

5. **Bilek Stabilitesi**
   - **Eşik:** Sol ve sağ bilek Y pozisyonu arasında >0.1 birim fark
   - **Skor:** Sabit 70 puan

### 🏋️ Gövde (Core) - Metrikler (Ağırlık: %30):

1. **Gövde Geriye Eğilme**
   - **Eşik:** >15°
   - **Skor:** `max(45, 100 - açı * 3)`

2. **Omuz Seviyesi**
   - **Eşik:** >10°
   - **Skor:** `max(65, 100 - açı * 3)`

3. **Kalça Kayması**
   - **Eşik:** Kalça genişliğinin %10'undan fazla
   - **Skor:** `max(60, 100 - (sapma/kalça_genişliği) * 100)`

---

## Diğer Egzersizler için Ağırlıklar

### Lateral Shoulder Raises, Tricep Extensions, Shoulder Press:
- Kollar: %50, Gövde: %30, Kafa: %10, Bacaklar: %10

### Squats, Lunges:
- Bacaklar: %50, Gövde: %40, Kollar: %5, Kafa: %5

### Pushups:
- Gövde: %40, Kollar: %40, Bacaklar: %15, Kafa: %5

### Dumbbell Rows:
- Gövde: %45, Kollar: %40, Kafa: %10, Bacaklar: %5

---

## Özet

**Metrikler:**
- Açı ölçümleri (angle measurements)
- Mesafe ölçümleri (distance measurements)
- Pozisyon karşılaştırmaları (position comparisons)
- Asimetri kontrolleri (symmetry checks)
- Drift/hareket kontrolleri (drift/movement checks)

**Skor Hesaplama:**
1. Her metrik için bir skor üretilir (0-100 arası)
2. Her bölge için metrik skorlarının ortalaması alınır
3. Genel skor, bölgesel skorların egzersiz tipine göre ağırlıklı ortalamasıdır
4. Kritik hatalar genel skoru sınırlandırır

**Görünen Skorlar:**
- Arayüzde gösterilen skorlar, **tüm rep'ler boyunca** hesaplanan bölgesel skorların ortalamasıdır
- Her rep için frame'ler üzerinden hesaplanan skorlar, o rep için ortalamaya dahil edilir
- Session sonunda, tüm rep'ler için bölgesel skorların ortalaması alınır

