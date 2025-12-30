# 🧭 Quaternion ve Yön Vektörü İlişkisi

## 🤔 Quaternion = Normal Vektör mü?

**Kısa Cevap:** Tam olarak değil, ama **benzer şekilde düşünülebilir** bazı durumlarda!

---

## 📐 Normal Vektör Nedir?

### **Tanım:**

Normal vektör, bir **düzlemin dik yönünü** gösteren 3D vektördür:

```
Düzlem: z = 0 (xy düzlemi)
Normal Vektör: n = (0, 0, 1)  → Z ekseni yönünde
```

### **Özellikler:**

1. **Yön:** Düzleme dik (perpendicular)
2. **Büyüklük:** Genellikle birim vektör (length = 1)
3. **3D:** (x, y, z) → 3 bileşen

---

## 🔄 Quaternion Nedir?

### **Tanım:**

Quaternion, bir **rotasyonu (dönüşü)** temsil eden 4D sayıdır:

```
q = w + xi + yj + zk
```

### **Özellikler:**

1. **Rotasyon:** 3D uzayda döndürmeyi temsil eder
2. **4D:** (w, x, y, z) → 4 bileşen
3. **Birim Vektör:** Quaternion'dan **yön vektörü** çıkarılabilir

---

## 🔗 Quaternion → Yön Vektörü

Quaternion'dan **bir yön vektörü** çıkarılabilir, ama bu **normal vektör değildir** - bu **rotasyon sonrası birim vektördür**.

### **Nasıl Çıkarılır?**

Bir quaternion, **bir birim vektörü döndürür**. Örneğin, **forward direction (ileri yön)** veya **up direction (yukarı yön)**.

```python
import numpy as np

def quaternion_to_forward_vector(qw, qx, qy, qz):
    """
    Quaternion'dan forward direction (ileri yön) vektörü çıkar.
    
    Quaternion bir rotasyonu temsil eder. Forward vektör (0, 0, 1) 
    bu rotasyonla döndürülür.
    """
    # Forward vector: (0, 0, 1) - z ekseni yönünde
    forward = np.array([0, 0, 1])
    
    # Quaternion ile döndür
    # Quaternion rotasyon matrisi kullanarak
    # (veya quaternion multiplication)
    
    # Basitleştirilmiş versiyon:
    # q * [0, 0, 1] * q^(-1) → rotated vector
    
    # Rotasyon sonrası forward vektör
    rotated_forward = quaternion_rotate(qw, qx, qy, qz, forward)
    
    return rotated_forward  # 3D vektör (x, y, z)

def quaternion_to_up_vector(qw, qx, qy, qz):
    """
    Quaternion'dan up direction (yukarı yön) vektörü çıkar.
    
    Up vector: (0, 1, 0) - y ekseni yönünde
    """
    up = np.array([0, 1, 0])
    rotated_up = quaternion_rotate(qw, qx, qy, qz, up)
    return rotated_up
```

---

## 🎯 IMU Sensor Örneği

### **Sensor Yönelimi:**

IMU sensor'ün **yüzeyi** bir düzlemdir. Bu düzlemin **normal vektörü**, sensor'ün **"up" yönünü** gösterir.

```
IMU Sensor (Düzlem):
  Yüzey normali → Sensor'ün hangi yöne baktığı
  
Quaternion → Rotasyon matrisi → Yön vektörü
```

### **Bizim Sistemde:**

```json
{
  "left_wrist": {
    "quaternion": {
      "w": 0.998,
      "x": 0.012,
      "y": 0.034,
      "z": 0.056
    },
    "euler": {
      "roll": 5.2,
      "pitch": 12.5,
      "yaw": 178.3
    }
  }
}
```

**Yorumlama:**
- Quaternion → Sensor'ün **rotasyonu**
- Rotasyon → Sensor yüzeyinin **normal vektörünü** hesaplayabiliriz
- Normal vektör → Sensor'ün **hangi yöne baktığını** gösterir

---

## 📊 Karşılaştırma

### **Normal Vektör (Düzlem):**
```
Düzlem: Sensor yüzeyi
Normal Vektör: n = (nx, ny, nz)
→ Düzleme dik yön
→ Büyüklük: 1 (birim vektör)
```

### **Quaternion (Rotasyon):**
```
Quaternion: q = (w, x, y, z)
→ Bir rotasyonu temsil eder
→ Rotasyon → Yön vektörü çıkarılabilir
→ Çıkarılan vektör → Sensor'ün "forward" veya "up" yönü
```

### **İlişki:**

```
Quaternion (q) 
  ↓ (rotasyon uygula)
Birim Vektör (v) çıkarılabilir
  ↓
Bu vektör sensor'ün yönünü gösterir
  ↓
Bazı durumlarda bu = Sensor yüzeyinin normal vektörü
```

---

## ✅ Kısmen Doğru Yorumlama

### **Evet, Kısmen:**
✅ Quaternion'dan **yön vektörü** çıkarılabilir
✅ Bu vektör sensor'ün **hangi yöne baktığını** gösterir
✅ Sensor yüzeyinin **normal vektörü** olarak düşünülebilir (sensor yüzeyi bir düzlemse)

### **Ama Tam Olarak Değil:**
❌ Quaternion kendisi **normal vektör değildir** (4D, normal vektör 3D)
❌ Quaternion bir **rotasyon** temsil eder
❌ Normal vektör, quaternion'dan **hesaplanabilir** (ama aynı şey değil)

---

## 🎭 Pratik Örnek

### **Sol Bilek IMU Sensor:**

**Durum 1: Avuç içi yukarı bakıyor**
```
Quaternion: {w: 0.707, x: 0, y: 0.707, z: 0}
Euler: {roll: 0°, pitch: 90°, yaw: 0°}

Yorumlama:
  → Sensor yüzeyi (avuç içi) yukarı bakıyor
  → Normal vektör: n ≈ (0, 0, 1)  (z ekseni yönünde)
  → Quaternion'dan çıkarılabilir: up_vector ≈ (0, 0, 1)
  
✅ Bu durumda quaternion → normal vektör olarak düşünülebilir!
```

**Durum 2: Avuç içi öne bakıyor**
```
Quaternion: {w: 0.707, x: 0.707, y: 0, z: 0}
Euler: {roll: 90°, pitch: 0°, yaw: 0°}

Yorumlama:
  → Sensor yüzeyi (avuç içi) öne bakıyor
  → Normal vektör: n ≈ (1, 0, 0)  (x ekseni yönünde)
  → Quaternion'dan çıkarılabilir: forward_vector ≈ (1, 0, 0)
  
✅ Bu durumda da quaternion → normal vektör olarak düşünülebilir!
```

---

## 🔬 Matematiksel Açıklama

### **Quaternion → Yön Vektörü:**

Quaternion `q = (w, x, y, z)` bir **birim vektörü** döndürmek için kullanılır:

```
v_rotated = q * v * q^(-1)

Örnek:
  v = (0, 0, 1)  → Forward vektör (z ekseni)
  q = Quaternion (sensor rotasyonu)
  v_rotated = q * (0, 0, 1) * q^(-1)  → Sensor'ün forward yönü
```

### **Normal Vektör Olarak:**

Eğer sensor yüzeyinin **normal vektörü** istiyorsak:
- Sensor yüzeyi genellikle **xy düzlemi** (z = 0)
- Normal vektör: **z ekseni** yönünde → (0, 0, 1)
- Quaternion ile döndürülür → Sensor'ün **gerçek normal vektörü**

```
n_sensor = q * (0, 0, 1) * q^(-1)

→ Sensor yüzeyinin normal vektörü!
```

---

## ✅ Sonuç

### **Kısmen Doğru:**
✅ Quaternion'dan **yön vektörü** çıkarılabilir
✅ Bu vektör sensor yüzeyinin **normal vektörü** olarak **düşünülebilir**
✅ Sensor'ün **hangi yöne baktığını** gösterir

### **Ama Dikkat:**
⚠️ Quaternion kendisi normal vektör **değildir** (4D vs 3D)
⚠️ Quaternion bir **rotasyon** temsil eder
⚠️ Normal vektör, quaternion'dan **hesaplanır** (ama aynı şey değil)

### **Pratik Kullanım:**
```
IMU Sensor Quaternion:
  → Sensor'ün rotasyonunu gösterir
  → Rotasyon → Yön vektörü çıkarılabilir
  → Yön vektörü → Sensor yüzeyinin normal vektörü olarak düşünülebilir
  → "Sensor hangi yöne bakıyor?" sorusunu cevaplar
```

**Özet:** Evet, quaternion'dan çıkarılan yön vektörünü **normal vektör** olarak yorumlayabilirsiniz, çünkü sensor yüzeyinin **hangi yöne baktığını** gösterir! 🎯

