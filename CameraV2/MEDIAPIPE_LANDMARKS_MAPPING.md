# 📍 MediaPipe Pose 33 Landmark Mapping

## 🎯 Tüm 33 Landmark Noktası ve Vücut Bölgeleri

MediaPipe Pose, insan vücudunu **33 nokta (landmark)** ile temsil eder. Her nokta bir vücut bölgesine karşılık gelir.

---

## 📊 Landmark Listesi (0-32)

### **YÜZ (Face) - 0-10**

| Index | İsim | Türkçe | Açıklama |
|---|---|---|---|
| **0** | `nose` | Burun | Burun ucunun merkezi |
| **1** | `left_eye_inner` | Sol göz iç | Sol gözün iç köşesi (buruna yakın) |
| **2** | `left_eye` | Sol göz | Sol gözün merkezi |
| **3** | `left_eye_outer` | Sol göz dış | Sol gözün dış köşesi |
| **4** | `right_eye_inner` | Sağ göz iç | Sağ gözün iç köşesi (buruna yakın) |
| **5** | `right_eye` | Sağ göz | Sağ gözün merkezi |
| **6** | `right_eye_outer` | Sağ göz dış | Sağ gözün dış köşesi |
| **7** | `left_ear` | Sol kulak | Sol kulağın merkezi |
| **8** | `right_ear` | Sağ kulak | Sağ kulağın merkezi |
| **9** | `mouth_left` | Sol ağız köşesi | Ağzın sol köşesi |
| **10** | `mouth_right` | Sağ ağız köşesi | Ağzın sağ köşesi |

---

### **ÜST GÖVDE (Upper Body) - 11-16**

| Index | İsim | Türkçe | Açıklama |
|---|---|---|---|
| **11** | `left_shoulder` | Sol omuz | Sol omuz eklemi |
| **12** | `right_shoulder` | Sağ omuz | Sağ omuz eklemi |
| **13** | `left_elbow` | Sol dirsek | Sol dirsek eklemi |
| **14** | `right_elbow` | Sağ dirsek | Sağ dirsek eklemi |
| **15** | `left_wrist` | Sol bilek | Sol el bileği |
| **16** | `right_wrist` | Sağ bilek | Sağ el bileği |

---

### **ELLER (Hands) - 17-22**

| Index | İsim | Türkçe | Açıklama |
|---|---|---|---|
| **17** | `left_pinky` | Sol küçük parmak | Sol el küçük parmak (bileğe yakın) |
| **18** | `right_pinky` | Sağ küçük parmak | Sağ el küçük parmak (bileğe yakın) |
| **19** | `left_index` | Sol işaret parmağı | Sol el işaret parmağı (bileğe yakın) |
| **20** | `right_index` | Sağ işaret parmağı | Sağ el işaret parmağı (bileğe yakın) |
| **21** | `left_thumb` | Sol başparmak | Sol el başparmak (bileğe yakın) |
| **22** | `right_thumb` | Sağ başparmak | Sağ el başparmak (bileğe yakın) |

**Not:** Bu noktalar **bileğe yakın** el noktalarıdır, tam el landmark'ları değildir (tam el landmark'ları MediaPipe Hand modeli ile alınır).

---

### **ALT GÖVDE (Torso) - 23-24**

| Index | İsim | Türkçe | Açıklama |
|---|---|---|---|
| **23** | `left_hip` | Sol kalça | Sol kalça eklemi |
| **24** | `right_hip` | Sağ kalça | Sağ kalça eklemi |

---

### **BACAKLAR (Legs) - 25-28**

| Index | İsim | Türkçe | Açıklama |
|---|---|---|---|
| **25** | `left_knee` | Sol diz | Sol diz eklemi |
| **26** | `right_knee` | Sağ diz | Sağ diz eklemi |
| **27** | `left_ankle` | Sol ayak bileği | Sol ayak bileği |
| **28** | `right_ankle` | Sağ ayak bileği | Sağ ayak bileği |

---

### **AYAKLAR (Feet) - 29-32**

| Index | İsim | Türkçe | Açıklama |
|---|---|---|---|
| **29** | `left_heel` | Sol topuk | Sol ayağın topuk noktası |
| **30** | `right_heel` | Sağ topuk | Sağ ayağın topuk noktası |
| **31** | `left_foot_index` | Sol ayak parmak ucu | Sol ayağın parmak ucu |
| **32** | `right_foot_index` | Sağ ayak parmak ucu | Sağ ayağın parmak ucu |

```
0-10:   Yüz (Face) - 11 nokta
11-16:  Üst Gövde (Upper Body) - 6 nokta
  11-12: Omuzlar (Shoulders)
  13-14: Dirsekler (Elbows)
  15-16: Bilekler (Wrists)
17-22:  Eller (Hands) - 6 nokta (bileğe yakın el noktaları)
  17-18: Küçük parmaklar (Pinky)
  19-20: İşaret parmakları (Index)
  21-22: Başparmaklar (Thumb)
23-32:  Alt Gövde + Bacaklar + Ayaklar - 10 nokta
  23-24: Kalçalar (Hips)
  25-26: Dizler (Knees)
  27-28: Ayak bilekleri (Ankles)
  29-30: Topuklar (Heels)
  31-32: Ayak parmak uçları (Foot Index)
```

---

## 📐 Görsel Düzen

### **Vücut İskeleti (Skeleton):**

```
                   0 (nose)
                  /   \
           1-2-3 (left_eye)  4-5-6 (right_eye)
            |                 |
            7 (left_ear)  8 (right_ear)
               9              10
          (mouth_left)  (mouth_right)
                   
         11 (left_shoulder)  12 (right_shoulder)
                |                    |
         13 (left_elbow)      14 (right_elbow)
                |                    |
         15 (left_wrist)      16 (right_wrist)
         17 (left_pinky)      18 (right_pinky)
         19 (left_index)      20 (right_index)
         21 (left_thumb)      22 (right_thumb)
               
         23 (left_hip)        24 (right_hip)
                |                    |
         25 (left_knee)       26 (right_knee)
                |                    |
         27 (left_ankle)      28 (right_ankle)
                |                    |
         29 (left_heel)       30 (right_heel)
                |                    |
         31 (left_foot_index) 32 (right_foot_index)
```

---

## 💻 Kod İçinde Kullanım

### **Landmark Adları (Dictionary):**

```python
LANDMARK_NAMES = {
    # Face
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    
    # Upper Body
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    
    # Hands
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    
    # Lower Body
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}
```

### **Örnek Kullanım:**

```python
# Landmark dizisinden bir nokta al
landmarks = pose_landmarks  # 33 elemanlı liste

# Burun
nose = landmarks[0]  # {x, y, z, visibility}

# Sol omuz
left_shoulder = landmarks[11]  # {x, y, z, visibility}

# Sol dirsek
left_elbow = landmarks[13]  # {x, y, z, visibility}

# Sol bilek
left_wrist = landmarks[15]  # {x, y, z, visibility}

# Sol kalça
left_hip = landmarks[23]  # {x, y, z, visibility}

# Sol diz
left_knee = landmarks[25]  # {x, y, z, visibility}

# Sol ayak bileği
left_ankle = landmarks[27]  # {x, y, z, visibility}
```

---

## 🏋️ Egzersiz Analizi İçin Önemli Noktalar

### **Bicep Curls:**
- **11, 12:** Omuzlar (shoulder stability)
- **13, 14:** Dirsekler (angle measurement)
- **15, 16:** Bilekler (movement tracking)
- **23, 24:** Kalçalar (posture check)

### **Squats:**
- **23, 24:** Kalçalar (hip depth)
- **25, 26:** Dizler (knee angle)
- **27, 28:** Ayak bilekleri (ankle position)
- **29, 30, 31, 32:** Ayaklar (foot contact)

### **Pushups:**
- **11, 12:** Omuzlar (shoulder position)
- **13, 14:** Dirsekler (elbow angle)
- **15, 16:** Bilekler (wrist alignment)
- **23, 24:** Kalçalar (body line)

---

## 📊 Landmark Formatı

Her landmark şu formatta gelir:

```json
{
  "x": 0.5,           // X koordinatı (0-1 arası, görüntü genişliğine normalize)
  "y": 0.3,           // Y koordinatı (0-1 arası, görüntü yüksekliğine normalize)
  "z": 0.1,           // Z koordinatı (derinlik, yakınlık)
  "visibility": 0.9   // Görünürlük skoru (0-1 arası, 1 = tamamen görünür)
}
```

---

## ✅ Özet Tablo (Hızlı Referans)

| Index | İsim | Vücut Bölgesi | Kullanım |
|---|---|---|---|
| 0 | nose | Yüz | Baş pozisyonu |
| 1-6 | eyes | Yüz | Baş yönü |
| 7-8 | ears | Yüz | Baş rotasyonu |
| 9-10 | mouth | Yüz | Yüz oryantasyonu |
| 11-12 | shoulders | Üst Gövde | Omuz pozisyonu, gövde eğimi |
| 13-14 | elbows | Üst Gövde | Kol açıları, hareket menzili |
| 15-16 | wrists | Üst Gövde | El pozisyonu, kol hareketi |
| 23-24 | hips | Alt Gövde | Kalça pozisyonu, gövde dengesi |
| 25-26 | knees | Bacaklar | Diz açıları, squat derinliği |
| 27-28 | ankles | Bacaklar | Ayak bileği pozisyonu |
| 29-30 | heels | Ayaklar | Topuk teması |
| 31-32 | foot_index | Ayaklar | Ayak ucu pozisyonu |

---

## 🎯 Frontend'de Kullanım

Frontend'de landmark'lar şu şekilde etiketleniyor:

```typescript
// WorkoutSessionWithIMU.tsx veya WorkoutSession.tsx
landmarks.forEach((landmark, index) => {
  // Canvas'a numara çiz (0-32)
  ctx.fillText(index.toString(), x, y);
});
```

**Görsel:** Her landmark noktasının üzerinde **kırmızı nokta** ve **siyah numara (0-32)** görünür.

---

## 📚 Referans

- MediaPipe Pose Documentation: https://google.github.io/mediapipe/solutions/pose.html
- Landmark Model: https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmark_upper_body_heavy.tflite

---

## ✅ Sonuç

**Toplam 33 landmark noktası:**

- **0-10:** Yüz (11 nokta)
- **11-16:** Üst gövde (6 nokta) - **11-12 (omuzlar), 13-14 (dirsekler), 15-16 (bilekler)**
- **17-22:** ❌ Kullanılmaz (6 boş)
- **23-32:** Alt gövde + bacaklar + ayaklar (10 nokta) - **23-24 (kalçalar), 25-26 (dizler), 27-28 (ayak bilekleri), 29-30 (topuklar), 31-32 (ayak parmak uçları)**

**Toplam:** 11 (yüz) + 6 (üst gövde) + 6 (eller) + 10 (alt gövde+bacaklar+ayaklar) = **33 aktif landmark**

