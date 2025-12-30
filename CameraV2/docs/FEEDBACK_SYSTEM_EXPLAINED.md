# Feedback Sistemi Açıklaması

## 🎯 Sistem Nasıl Çalışıyor?

**ÖNEMLİ:** Sistem OpenAI API kullanmıyor! Tamamen **rule-based (kural tabanlı)** bir sistem.

## 📊 Feedback Oluşturma Süreci

### 1. Form Analizi (`FormAnalyzer.check_form()`)

MediaPipe landmark'larından (33 nokta) gerçek zamanlı olarak form analizi yapılıyor:

```python
# Örnek: Bicep Curls için
- Dirsek pozisyonu kontrolü
- Omuz pozisyonu kontrolü  
- Gövde açısı kontrolü
- Kalça stabilitesi kontrolü
- vs.
```

### 2. Issue Detection (Sorun Tespiti)

Her kontrol için **eşik değerleri (thresholds)** var. Eğer eşik aşılırsa, bir "issue" oluşturuluyor:

**Örnek Kurallar (Bicep Curls):**

```python
# Dirsek hareket toleransı
drift_tolerance = shoulder_width * 0.15  # Omuz genişliğinin %15'i

# Omuz kalkma toleransı  
rise_tolerance = torso_height * 0.08     # Gövde yüksekliğinin %8'i

# Üst kol açısı
if upper_arm_angle > 30°:
    issue = "Sol üst kol çok açık"
    
# Omuz eğikliği
if shoulders_angle > 15°:
    issue = "Omuzlar eğik - düz dur"
```

### 3. Feedback Mesajı Oluşturma

Tespit edilen issues'lara göre `get_rule_based_regional_feedback()` ve `get_rule_based_overall_feedback()` fonksiyonları Türkçe feedback mesajları üretiyor.

## 🔧 Nasıl Ayarlanır?

### Örnek: Dirsek Hareket Toleransını Değiştirmek

`api_server.py` dosyasında `FormAnalyzer.check_form()` metodunda:

```python
# BICEP CURLS bölümünde (satır ~454)
drift_tolerance = self.shoulder_width * 0.15  # ← Bu değeri değiştir

# Daha sıkı kontrol için:
drift_tolerance = self.shoulder_width * 0.10  # %15 → %10

# Daha gevşek kontrol için:
drift_tolerance = self.shoulder_width * 0.20  # %15 → %20
```

### Örnek: Omuz Eğiklik Eşiğini Değiştirmek

```python
# Satır ~507
if shoulders_angle > 15:  # ← Bu eşik değeri
    core_issues.append('Omuzlar eğik - düz dur')
    
# Daha sıkı:
if shoulders_angle > 10:  # 15° → 10°
    
# Daha gevşek:
if shoulders_angle > 20:  # 15° → 20°
```

### Örnek: Açı Eşiklerini Değiştirmek

```python
# Satır ~461
if left_upper_arm_angle > 30:  # ← Bu açı eşiği
    
# Daha sıkı:
if left_upper_arm_angle > 25:  # 30° → 25°
    
# Daha gevşek:
if left_upper_arm_angle > 35:  # 30° → 35°
```

## 📋 Mevcut Eşik Değerleri (Bicep Curls)

| Kontrol | Eşik Değeri | Kod Konumu | Açıklama |
|---------|-------------|------------|----------|
| **Dirsek Drift** | `shoulder_width * 0.15` | Satır 454 | Dirsek başlangıç pozisyonundan %15 shoulder_width kadar kayabilir |
| **Omuz Kalkması** | `torso_height * 0.08` | Satır 485 | Omuz %8 torso_height kadar kalkabilir |
| **Üst Kol Açısı** | `30°` | Satır 461 | Üst kol 30°'den fazla açılırsa issue |
| **Omuz Eğikliği** | `15°` | Satır 507 | Omuzlar 15°'den fazla eğikse issue |
| **Gövde Eğikliği** | `20°` | Satır 515 | Gövde 20°'den fazla eğilirse issue |
| **Kalça Kayması** | `hip_width * 0.1` | Satır 521 | Kalça %10 hip_width kadar kayabilir |

## ✅ Feedback Doğruluğu

**Artıları:**
- ✅ Gerçek zamanlı, hızlı feedback
- ✅ API dependency yok (ücretsiz, sınırsız)
- ✅ Tutarlı sonuçlar
- ✅ MediaPipe verilerine dayalı (objektif)

**Sınırlamalar:**
- ⚠️ Sadece tanımlı kurallara göre çalışır
- ⚠️ Karmaşık form problemlerini yakalayamayabilir
- ⚠️ Eşik değerleri manuel ayarlanmalı

## 🔍 Feedback Mesajları Nasıl Oluşturuluyor?

1. **Issue tespiti** → `check_form()` issues listesi oluşturur
2. **Regional feedback** → `get_rule_based_regional_feedback()` issue'a göre Türkçe mesaj oluşturur
3. **Overall feedback** → `get_rule_based_overall_feedback()` genel skor ve issues'a göre mesaj oluşturur

**Örnek:**
```python
# Issue tespit edildi:
issues = ['Sol dirsek oynuyor', 'Omuzlar eğik - düz dur']

# Feedback fonksiyonu:
if 'dirsek' in issue and 'sol' in issue:
    return "Sol dirseğini gövdene sabitle, daha az oynatmalısın."
```

## 🛠️ Özelleştirme Önerileri

Eğer feedback'ler çok sıkı veya çok gevşekse:

1. **Eşik değerlerini ayarlayın** (yukarıdaki tabloya bakın)
2. **Yeni kurallar ekleyin** (`check_form()` metoduna)
3. **Feedback mesajlarını değiştirin** (`get_rule_based_regional_feedback()` metodunda)

