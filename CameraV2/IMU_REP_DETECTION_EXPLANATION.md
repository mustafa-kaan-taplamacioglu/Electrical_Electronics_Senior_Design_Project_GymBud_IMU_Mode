# IMU-Only Modda Biceps Curl Rep Sayma Mantığı

## Genel Bakış

IMU-only modda biceps curl rep sayma, `IMUPeriodicRepDetector` sınıfı kullanılarak yapılır. Bu sınıf, gyroscope magnitude (açısal hız büyüklüğü) verilerini kullanarak periyodik hareketleri tespit eder ve her periyodu bir rep olarak sayar.

## Teknik Detaylar

### 1. Gyroscope Magnitude Hesaplama

**Konum:** `_calculate_gyro_magnitude()` metodu (satır 1618-1665)

Biceps curl için:
- **Kullanılan sensörler:** Her iki wrist (sol ve sağ bilek) node'ları
- **Hesaplama:** Her wrist'ten gelen `gx`, `gy`, `gz` (gyroscope x, y, z eksenleri) değerleri kullanılarak magnitude hesaplanır:
  ```
  magnitude = sqrt(gx² + gy² + gz²)
  ```
- **Ortalama:** Her iki wrist'in magnitude değerlerinin ortalaması alınır (daha güvenilir tespit için)

**Örnek:**
- Sol wrist: gx=50, gy=30, gz=20 → magnitude = √(50² + 30² + 20²) = 61.6 deg/s
- Sağ wrist: gx=45, gy=35, gz=25 → magnitude = √(45² + 35² + 25²) = 61.5 deg/s
- **Kullanılan değer:** (61.6 + 61.5) / 2 = 61.55 deg/s

### 2. Sliding Window Buffer

**Konum:** `add_imu_sample()` metodu (satır 1557-1616)

- Her yeni IMU sample geldiğinde gyroscope magnitude hesaplanır
- Son 90 sample (yaklaşık 5.8 saniye, 15.5 Hz sample rate'de) bir buffer'da tutulur
- Buffer boyutu dinamik: Maksimum 180 sample (2 × window_size)

### 3. Rep Tespiti: Peak Detection

**Konum:** `_detect_period()` metodu (satır 1667-1780)

Rep tespiti, **adaptive peak detection** algoritması kullanılarak yapılır:

#### a) Signal Normalizasyonu
- Signal'in ortalaması çıkarılır (zero-mean normalization)
- Bu, farklı hızlardaki rep'leri daha iyi tespit etmeye yardımcı olur

#### b) Adaptive Threshold Hesaplama

**Threshold = signal_median + signal_std × threshold_factor**

- **signal_median:** Son 90 sample'ın medyan değeri
- **signal_std:** Son 90 sample'ın standart sapması
- **threshold_factor:** Son rep'lerin sürelerine göre adaptif olarak ayarlanır:
  - Hızlı rep'ler (kısa süre) → düşük threshold
  - Yavaş rep'ler (uzun süre) → yüksek threshold
  - Varsayılan: 0.7 (biceps curl için)

**Minimum Peak Height:**
- Base minimum: 60.0 deg/s (biceps curl için gerçek veri analizinden çıkarılmış)
- Adaptive: `max(60.0, signal_median × 0.7) × threshold_factor`

#### c) Peak Detection Algoritması

1. **Son N sample'ı kontrol et:** Son `min_period_samples` (29 sample, ~1.9 saniye) içinde bir peak aranır

2. **Local Maximum Tespiti:**
   - Bir sample'ın hem öncesi hem sonrasından büyük olması gerekir
   - Peak yüksekliği threshold'u geçmelidir
   - Peak yüksekliği minimum peak height'ı geçmelidir

3. **Double Counting Önleme:**
   - Son peak'ten bu yana yeterince zaman geçmiş olmalı
   - Adaptive minimum distance: Son rep'lerin ortalama süresinin %80'i
   - Biceps curl için: Ortalama 1.92 saniye → minimum distance ≈ 23 sample (1.92s × 15.5Hz × 0.8)

4. **State Machine:**
   - **is_tracking = False:** İlk peak tespit edildi → `is_tracking = True`, periyod başlar
   - **is_tracking = True:** İkinci peak tespit edildi → `is_tracking = False`, rep tamamlandı!

### 4. Rep Süresi ve Hız Tespiti

**Konum:** `add_imu_sample()` metodu, rep_duration hesaplama (satır 1595-1603)

- Her rep tamamlandığında, rep süresi hesaplanır:
  ```
  rep_duration = timestamp (ikinci peak) - current_period_start_time (ilk peak)
  ```
- Son 10 rep'in süreleri saklanır (adaptive thresholding için)
- Bu süreler, gelecekteki rep'lerin threshold'larını ayarlamak için kullanılır

### 5. Parameter Ayarları (Biceps Curl için)

**Konum:** `__init__()` metodu (satır 1524-1555)

- **window_size:** 90 sample (~5.8 saniye, 15.5 Hz'de)
- **min_period_samples:** 29 sample (~1.9 saniye, biceps curl ortalama süresi)
- **min_peak_height_base:** 60.0 deg/s (gerçek veri analizinden: mean=95.4, median=88.6, 75th percentile=149.4)

### 6. Çalışma Akışı

1. **IMU data gelir** → `add_imu_sample()` çağrılır
2. **Gyroscope magnitude hesaplanır** → Buffer'a eklenir
3. **Yeterli sample var mı?** → En az 58 sample (2 × min_period_samples) gerekir
4. **Peak detection çalıştırılır** → `_detect_period()` çağrılır
5. **Rep tamamlandı mı?** → İkinci peak tespit edildiyse:
   - `rep_count` artırılır
   - `rep_duration` hesaplanır
   - Rep bilgileri döndürülür
   - Frontend'e bildirim gönderilir

## Örnek Senaryo

**Biceps Curl - 1 Rep:**

```
Zaman (s) | Gyro Mag (deg/s) | Durum
----------|-------------------|--------
0.0       | 45                | Buffer'a ekleniyor
0.5       | 50                | Buffer'a ekleniyor
1.0       | 80                | ⚠️ İlk Peak! (threshold=70) → is_tracking=True
1.5       | 75                | Periyod devam ediyor
2.0       | 60                | Periyod devam ediyor
2.5       | 85                | ⚠️ İkinci Peak! (threshold=70) → Rep tamamlandı!
         |                   | rep_count=1, duration=1.5s
```

## Avantajlar

1. **Kamera gerektirmez:** Sadece IMU sensörleri yeterli
2. **Hız adaptasyonu:** Farklı hızlardaki rep'leri tespit edebilir
3. **Double counting önleme:** Adaptive minimum distance ile
4. **Robust tespit:** Median ve std-based threshold kullanımı
5. **Gerçek veri bazlı:** Biceps curl veri setinden optimize edilmiş parametreler

## Sınırlamalar

1. **Minimum süre:** ~1.9 saniyeden kısa rep'ler tespit edilemeyebilir
2. **Çok hızlı rep'ler:** Sample rate limitleri (15.5 Hz) nedeniyle
3. **Gürültü:** Çok fazla gürültülü sinyallerde yanlış tespitler olabilir
4. **Adaptive threshold:** İlk birkaç rep'te threshold henüz optimize olmamış olabilir

