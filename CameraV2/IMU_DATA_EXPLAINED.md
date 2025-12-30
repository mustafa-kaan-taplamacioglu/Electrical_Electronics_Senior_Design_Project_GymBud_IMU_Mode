# 📡 IMU Sensor Verileri: Quaternion ve Euler Açıları Açıklaması

## 🎯 Genel Bakış

IMU (Inertial Measurement Unit) sensor'lerinden gelen **quaternion** ve **Euler açıları**, sensor'ün **3D uzaydaki yönelimini (orientation)** temsil eder.

---

## 📐 Quaternion Nedir?

### **Matematiksel Tanım:**

Quaternion, 3D rotasyonları temsil etmek için kullanılan 4 bileşenli bir sayıdır:

```
q = w + xi + yj + zk
```

**Bileşenler:**
- **w** = Skaler (scalar) bileşen → Rotasyon açısının cos değeri
- **x, y, z** = Vektörel (vector) bileşenler → Rotasyon ekseninin yönü

### **Özellikleri:**

1. **Gimbal Lock Problemi Yok:** Quaternion'lar 3D rotasyonları temsil ederken gimbal lock (eksen kilitleme) problemi yaşamazlar
2. **Smooth Interpolation:** İki quaternion arasında yumuşak geçiş yapılabilir (SLERP)
3. **Hesaplama Verimliliği:** Rotasyon hesaplamalarında matrislerden daha hızlıdır
4. **Normalizasyon:** Genellikle `w² + x² + y² + z² = 1` olacak şekilde normalize edilir

### **Kayıt Formatımız:**

```json
"quaternion": {
  "w": 0.998,    // Skaler bileşen (rotation angle cos)
  "x": 0.012,    // X ekseni vektör bileşeni
  "y": 0.034,    // Y ekseni vektör bileşeni
  "z": 0.056     // Z ekseni vektör bileşeni
}
```

**Yorumlama:**
- **w ≈ 1.0:** Sensor neredeyse orijinal pozisyonda (küçük rotasyon)
- **x, y, z ≈ 0:** Sensor düz duruyor
- **w, x, y, z değerleri:** Sensor'ün 3D uzayda hangi açıyla döndüğünü gösterir

---

## 🔄 Euler Açıları Nedir?

### **Tanım:**

Euler açıları, 3D rotasyonları **3 ayrı açı** ile temsil eder. Her açı, bir eksen etrafında dönüşü gösterir.

### **Eksenler (Roll, Pitch, Yaw):**

```
       Y (Pitch - Öne/Arkaya)
       ↑
       |
       |
Z ←----+----→ X (Roll - Sağa/Sola)
       |
       |
       ↓
  (Yaw - Dönüş)
```

**3 Ana Açı:**

1. **Roll (Yatış):** X ekseni etrafında dönüş
   - Sensor'ün sağa/sola yatışı
   - Örnek: Kolu yan tarafa kaldırma

2. **Pitch (Yunuslama):** Y ekseni etrafında dönüş
   - Sensor'ün öne/arkaya eğilmesi
   - Örnek: Kolu öne/arkaya hareket ettirme

3. **Yaw (Sapma):** Z ekseni etrafında dönüş
   - Sensor'ün dikey eksen etrafında dönmesi
   - Örnek: Kolu dairesel hareket ettirme

### **Kayıt Formatımız:**

```json
"euler": {
  "roll": 5.2,     // X ekseni etrafında dönüş (derece)
  "pitch": 12.5,   // Y ekseni etrafında dönüş (derece)
  "yaw": 178.3     // Z ekseni etrafında dönüş (derece)
}
```

**Yorumlama:**
- **Roll = 0°:** Sensor düz (yatay)
- **Pitch = 0°:** Sensor düz (dikey eksen)
- **Yaw = 0°:** Sensor ön yönünde
- **Değerler:** -180° ile +180° arası (veya 0° ile 360° arası)

---

## 🔗 Quaternion ↔ Euler Dönüşümü

Quaternion ve Euler açıları **birbirine dönüştürülebilir**:

### **Quaternion → Euler:**

```python
import math

def quaternion_to_euler(w, x, y, z):
    # Roll (X ekseni)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (Y ekseni)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (Z ekseni)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
```

### **Euler → Quaternion:**

```python
def euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return w, x, y, z
```

---

## 📊 Bizim Sistemde Ne Kaydediyoruz?

### **IMU Sensor Verileri:**

Her IMU sensor'den (left_wrist, right_wrist, chest) şunları kaydediyoruz:

```json
{
  "left_wrist": {
    "node_id": 1,
    "timestamp": 1703123456.789,
    
    // 1. ACCELEROMETER (İvmeölçer)
    "accel": {
      "x": 0.0,      // X ekseni ivmesi (g - yerçekimi birimi)
      "y": -0.5144,  // Y ekseni ivmesi
      "z": 0.8808    // Z ekseni ivmesi
    },
    // → Ne ölçüyor? Sensor'ün ivmesini (hızlanma/yavaşlama)
    // → Birim: g (yerçekimi, 1g = 9.81 m/s²)
    
    // 2. GYROSCOPE (Jiroskop)
    "gyro": {
      "x": -1.26,    // X ekseni açısal hızı (derece/saniye)
      "y": -5.39,    // Y ekseni açısal hızı
      "z": -0.56     // Z ekseni açısal hızı
    },
    // → Ne ölçüyor? Sensor'ün dönüş hızını (angular velocity)
    // → Birim: derece/saniye (deg/s)
    
    // 3. QUATERNION (Yönelim - Madgwick Filter'den)
    "quaternion": {
      "w": 0.998,    // Skaler bileşen
      "x": 0.012,    // X vektör bileşeni
      "y": 0.034,    // Y vektör bileşeni
      "z": 0.056     // Z vektör bileşeni
    },
    // → Ne ölçüyor? Sensor'ün 3D uzaydaki yönelimini (orientation)
    // → Nasıl hesaplanıyor? Accelerometer + Gyroscope → Madgwick Filter → Quaternion
    // → Kullanım: Sensor'ün tam rotasyon bilgisi (gimbal lock yok)
    
    // 4. EULER ANGLES (Açılar - Quaternion'dan dönüştürülmüş)
    "euler": {
      "roll": 5.2,    // X ekseni etrafında dönüş (derece)
      "pitch": 12.5,  // Y ekseni etrafında dönüş (derece)
      "yaw": 178.3    // Z ekseni etrafında dönüş (derece)
    }
    // → Ne ölçüyor? Sensor'ün 3 eksen etrafındaki dönüş açıları
    // → Nasıl hesaplanıyor? Quaternion'dan dönüştürülür
    // → Kullanım: İnsan tarafından anlaşılması kolay (3 ayrı açı)
  }
}
```

---

## 🏋️ Egzersiz Örnekleri

### **Bicep Curls (Biceps Curl):**

**Sol Bilek (Left Wrist) IMU:**

```
Başlangıç Pozisyonu:
  quaternion: {w: 0.998, x: 0.012, y: 0.034, z: 0.056}
  euler: {roll: 5.2°, pitch: 12.5°, yaw: 178.3°}

Kolu Yukarı Kaldırırken (Curl Up):
  quaternion: {w: 0.950, x: 0.012, y: 0.280, z: 0.145}
  euler: {roll: 5.0°, pitch: 35.0°, yaw: 170.0°}
  → Pitch artıyor (öne doğru bükülme)

Kolu Aşağı İndirirken (Curl Down):
  quaternion: {w: 0.998, x: 0.012, y: 0.034, z: 0.056}
  euler: {roll: 5.2°, pitch: 12.5°, yaw: 178.3°}
  → Pitch azalıyor (başlangıç pozisyonuna dönüş)
```

**Ne Kaydediyoruz?**
- Her frame'de quaternion ve Euler açıları
- Zaman içindeki değişimleri takip ediyoruz
- ML model için feature extraction yapıyoruz (ROM, velocity, acceleration)

---

## 🧮 Madgwick Filter (Sensor Fusion)

### **Ne Yapıyor?**

Accelerometer ve Gyroscope verilerini birleştirip **daha doğru orientation** hesaplıyor:

```
Accelerometer (ivme) + Gyroscope (açısal hız)
        ↓
   Madgwick Filter
        ↓
   Quaternion (yönelim)
        ↓
   Euler Açıları (dönüştürme)
```

### **Neden Gerekli?**

1. **Gyroscope Drift:** Gyroscope zamanla sapma (drift) yapar
2. **Accelerometer Noise:** Accelerometer gürültülü (noisy) veri üretir
3. **Sensor Fusion:** İki sensor'ü birleştirerek daha doğru sonuç alırız

### **Kodda Nerede?**

`CameraV2/gymbud_imu_bridge.py` dosyasında `MadgwickFilter` sınıfı:

```python
class MadgwickFilter:
    """Madgwick filter for sensor fusion (accel + gyro → quaternion)"""
    
    def update(self, accel, gyro, dt):
        # Accelerometer ve gyroscope'u birleştir
        # Quaternion hesapla
        return quaternion
```

---

## 📈 ML Model İçin Kullanım

### **Feature Extraction:**

IMU verilerinden şu özellikleri çıkarıyoruz:

```python
# Her node için (left_wrist, right_wrist, chest):

# 1. Euler Angles Features
- roll_mean, roll_std, roll_min, roll_max, roll_range
- pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range
- yaw_mean, yaw_std, yaw_min, yaw_max, yaw_range

# 2. Quaternion Features
- quat_w_mean, quat_w_std, quat_w_min, quat_w_max
- quat_x_mean, quat_x_std, quat_x_min, quat_x_max
- quat_y_mean, quat_y_std, quat_y_min, quat_y_max
- quat_z_mean, quat_z_std, quat_z_min, quat_z_max

# 3. Accelerometer Features
- accel_x_mean, accel_x_std, accel_x_min, accel_x_max, accel_x_range
- accel_y_mean, accel_y_std, accel_y_min, accel_y_max, accel_y_range
- accel_z_mean, accel_z_std, accel_z_min, accel_z_max, accel_z_range

# 4. Gyroscope Features
- gyro_x_mean, gyro_x_std, gyro_x_min, gyro_x_max, gyro_x_range
- gyro_y_mean, gyro_y_std, gyro_y_min, gyro_y_max, gyro_y_range
- gyro_z_mean, gyro_z_std, gyro_z_min, gyro_z_max, gyro_z_range
```

**Toplam:** Her node için ~45 feature × 3 node = **~135 IMU feature**

---

## ✅ Özet

### **Quaternion:**
- **Ne?** 3D rotasyonu temsil eden 4 bileşenli sayı (w, x, y, z)
- **Ne Kaydediyor?** Sensor'ün 3D uzaydaki tam yönelimi
- **Avantaj:** Gimbal lock yok, smooth interpolation
- **Kullanım:** Rotasyon hesaplamaları, sensor fusion

### **Euler Açıları:**
- **Ne?** 3 ayrı açı ile rotasyon temsili (roll, pitch, yaw)
- **Ne Kaydediyor?** Sensor'ün 3 eksen etrafındaki dönüş açıları
- **Avantaj:** İnsan tarafından anlaşılması kolay
- **Kullanım:** ML feature extraction, görselleştirme

### **Bizim Sistemde:**
1. **Accelerometer:** İvme ölçer (g)
2. **Gyroscope:** Açısal hız ölçer (deg/s)
3. **Madgwick Filter:** Sensor fusion → Quaternion
4. **Quaternion → Euler:** İnsan tarafından anlaşılır formata dönüştürme
5. **Her İkisini Kaydediyoruz:** Hem quaternion hem Euler (her ikisinin avantajlarından yararlanmak için)

