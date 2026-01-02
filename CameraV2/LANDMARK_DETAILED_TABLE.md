# MediaPipe Pose - 33 Landmark Detaylı Tablo

Bu doküman, MediaPipe Pose modelinin 33 landmark'ının her birinin hangi vücut parçasına ve bölgesine denk geldiğini detaylı bir tablo halinde sunar.

---

## 📊 Tüm Landmarklar - Detaylı Tablo

| Landmark No | İngilizce İsim | Türkçe İsim | Vücut Bölgesi | Anatomik Bölge | Açıklama | Form Analizinde Kullanımı |
|-------------|----------------|-------------|---------------|----------------|----------|---------------------------|
| **0** | `nose` | Burun | Yüz | Baş | Burun ucunun merkezi | Kafa pozisyonu kontrolü, baş hizalanması |
| **1** | `left_eye_inner` | Sol göz iç | Yüz | Baş | Sol gözün iç köşesi (buruna yakın) | Baş yönü, göz hizası kontrolü |
| **2** | `left_eye` | Sol göz | Yüz | Baş | Sol gözün merkezi | Baş pozisyonu, göz hizası |
| **3** | `left_eye_outer` | Sol göz dış | Yüz | Baş | Sol gözün dış köşesi | Baş rotasyonu, göz genişliği |
| **4** | `right_eye_inner` | Sağ göz iç | Yüz | Baş | Sağ gözün iç köşesi (buruna yakın) | Baş yönü, göz hizası kontrolü |
| **5** | `right_eye` | Sağ göz | Yüz | Baş | Sağ gözün merkezi | Baş pozisyonu, göz hizası |
| **6** | `right_eye_outer` | Sağ göz dış | Yüz | Baş | Sağ gözün dış köşesi | Baş rotasyonu, göz genişliği |
| **7** | `left_ear` | Sol kulak | Yüz | Baş | Sol kulağın merkezi | Baş rotasyonu, yüz yönü |
| **8** | `right_ear` | Sağ kulak | Yüz | Baş | Sağ kulağın merkezi | Baş rotasyonu, yüz yönü |
| **9** | `mouth_left` | Sol ağız köşesi | Yüz | Baş | Ağzın sol köşesi | Yüz oryantasyonu, baş pozisyonu |
| **10** | `mouth_right` | Sağ ağız köşesi | Yüz | Baş | Ağzın sağ köşesi | Yüz oryantasyonu, baş pozisyonu |
| **11** | `left_shoulder` | Sol omuz | Üst Gövde | Kollar | Sol omuz eklemi | Omuz pozisyonu, gövde stabilitesi, vücut oranları (shoulder_width) |
| **12** | `right_shoulder` | Sağ omuz | Üst Gövde | Kollar | Sağ omuz eklemi | Omuz pozisyonu, gövde stabilitesi, vücut oranları (shoulder_width) |
| **13** | `left_elbow` | Sol dirsek | Üst Gövde | Kollar | Sol dirsek eklemi | Dirsek açısı, kol hareketi, drift kontrolü, rep sayma |
| **14** | `right_elbow` | Sağ dirsek | Üst Gövde | Kollar | Sağ dirsek eklemi | Dirsek açısı, kol hareketi, drift kontrolü, rep sayma |
| **15** | `left_wrist` | Sol bilek | Üst Gövde | Kollar | Sol el bileği | Bilek pozisyonu, kol hareketi, rep sayma, IMU senkronizasyonu |
| **16** | `right_wrist` | Sağ bilek | Üst Gövde | Kollar | Sağ el bileği | Bilek pozisyonu, kol hareketi, rep sayma, IMU senkronizasyonu |
| **17** | `left_pinky` | Sol küçük parmak | Üst Gövde | Kollar | Sol el küçük parmak (bileğe yakın) | El pozisyonu (nadiren kullanılır) |
| **18** | `right_pinky` | Sağ küçük parmak | Üst Gövde | Kollar | Sağ el küçük parmak (bileğe yakın) | El pozisyonu (nadiren kullanılır) |
| **19** | `left_index` | Sol işaret parmağı | Üst Gövde | Kollar | Sol el işaret parmağı (bileğe yakın) | El pozisyonu (nadiren kullanılır) |
| **20** | `right_index` | Sağ işaret parmağı | Üst Gövde | Kollar | Sağ el işaret parmağı (bileğe yakın) | El pozisyonu (nadiren kullanılır) |
| **21** | `left_thumb` | Sol başparmak | Üst Gövde | Kollar | Sol el başparmak (bileğe yakın) | El pozisyonu (nadiren kullanılır) |
| **22** | `right_thumb` | Sağ başparmak | Üst Gövde | Kollar | Sağ el başparmak (bileğe yakın) | El pozisyonu (nadiren kullanılır) |
| **23** | `left_hip` | Sol kalça | Alt Gövde | Gövde | Sol kalça eklemi | Kalça pozisyonu, gövde stabilitesi, vücut oranları (hip_width, torso_height) |
| **24** | `right_hip` | Sağ kalça | Alt Gövde | Gövde | Sağ kalça eklemi | Kalça pozisyonu, gövde stabilitesi, vücut oranları (hip_width, torso_height) |
| **25** | `left_knee` | Sol diz | Alt Gövde | Bacaklar | Sol diz eklemi | Diz açısı, squat derinliği, bacak hareketi, rep sayma |
| **26** | `right_knee` | Sağ diz | Alt Gövde | Bacaklar | Sağ diz eklemi | Diz açısı, squat derinliği, bacak hareketi, rep sayma |
| **27** | `left_ankle` | Sol ayak bileği | Alt Gövde | Bacaklar | Sol ayak bileği | Ayak bileği pozisyonu, bacak hizalanması, squat kontrolü |
| **28** | `right_ankle` | Sağ ayak bileği | Alt Gövde | Bacaklar | Sağ ayak bileği | Ayak bileği pozisyonu, bacak hizalanması, squat kontrolü |
| **29** | `left_heel` | Sol topuk | Alt Gövde | Ayaklar | Sol ayağın topuk noktası | Ayak teması, denge kontrolü, squat kontrolü |
| **30** | `right_heel` | Sağ topuk | Alt Gövde | Ayaklar | Sağ ayağın topuk noktası | Ayak teması, denge kontrolü, squat kontrolü |
| **31** | `left_foot_index` | Sol ayak parmak ucu | Alt Gövde | Ayaklar | Sol ayağın parmak ucu | Ayak pozisyonu, denge kontrolü, squat kontrolü |
| **32** | `right_foot_index` | Sağ ayak parmak ucu | Alt Gövde | Ayaklar | Sağ ayağın parmak ucu | Ayak pozisyonu, denge kontrolü, squat kontrolü |

---

## 🎯 Vücut Bölgelerine Göre Gruplandırma

### 📍 Yüz (Face) - Landmark 0-10 (11 nokta)

**Kullanım:** Baş pozisyonu, baş hizalanması, baş rotasyonu kontrolü

| No | İsim | Kullanım |
|---|---|---------|
| 0 | nose | Kafa pozisyonu kontrolü (en önemli yüz landmark'ı) |
| 1-6 | eyes | Baş yönü, göz hizası |
| 7-8 | ears | Baş rotasyonu, yüz yönü |
| 9-10 | mouth | Yüz oryantasyonu |

**Form Analizi:** Genellikle kafa pozisyonu kontrolü için kullanılır (örn: "Kafan çok öne eğik", "Kafan çok geride")

---

### 💪 Üst Gövde (Upper Body) - Landmark 11-16 (6 nokta)

**Kullanım:** Üst vücut egzersizleri için kritik (bicep curls, shoulder press, tricep extensions, vb.)

| No | İsim | Vücut Oranı Hesaplaması | Form Kontrolleri |
|---|---|------------------------|------------------|
| 11-12 | shoulders | `shoulder_width = abs(11.x - 12.x)` | Omuz seviyesi, omuz kalkması, omuz rotasyonu |
| 13-14 | elbows | - | Dirsek açısı, dirsek drift, dirsek pozisyonu, rep sayma |
| 15-16 | wrists | - | Bilek pozisyonu, kol hareketi, rep sayma, IMU senkronizasyonu |

**Vücut Oranları:**
- **shoulder_width:** 11 ve 12 kullanılarak hesaplanır
- **torso_height:** 11, 12, 23, 24 kullanılarak hesaplanır
- **upper_arm_length:** 11 ve 13 kullanılarak hesaplanır
- **forearm_length:** 13 ve 15 kullanılarak hesaplanır

---

### ✋ Eller (Hands) - Landmark 17-22 (6 nokta)

**Kullanım:** El pozisyonu (nadiren kullanılır, MediaPipe Hand modeli daha detaylı el analizi için kullanılır)

| No | İsim | Not |
|---|---|-----|
| 17-18 | pinky | Sol ve sağ küçük parmak (bileğe yakın) |
| 19-20 | index | Sol ve sağ işaret parmağı (bileğe yakın) |
| 21-22 | thumb | Sol ve sağ başparmak (bileğe yakın) |

**Önemli Not:** Bu noktalar bileğe yakın el noktalarıdır, tam el landmark'ları değildir. Detaylı el analizi için MediaPipe Hand modeli kullanılır.

---

### 🏋️ Alt Gövde (Lower Body) - Landmark 23-24 (2 nokta)

**Kullanım:** Gövde stabilitesi, kalça pozisyonu, vücut oranları

| No | İsim | Vücut Oranı Hesaplaması | Form Kontrolleri |
|---|---|------------------------|------------------|
| 23-24 | hips | `hip_width = abs(23.x - 24.x)`, `torso_height` (11,12,23,24 ile) | Kalça pozisyonu, kalça kayması, gövde stabilitesi |

**Vücut Oranları:**
- **hip_width:** 23 ve 24 kullanılarak hesaplanır
- **torso_height:** 11, 12, 23, 24 kullanılarak hesaplanır

---

### 🦵 Bacaklar (Legs) - Landmark 25-28 (4 nokta)

**Kullanım:** Alt vücut egzersizleri için kritik (squats, lunges, vb.)

| No | İsim | Vücut Oranı Hesaplaması | Form Kontrolleri |
|---|---|------------------------|------------------|
| 25-26 | knees | - | Diz açısı, diz pozisyonu, squat derinliği, rep sayma |
| 27-28 | ankles | - | Ayak bileği pozisyonu, bacak hizalanması, diz-ayak bileği hizası |

**Vücut Oranları:**
- **thigh_length:** 23 ve 25 kullanılarak hesaplanır
- **shin_length:** 25 ve 27 kullanılarak hesaplanır

---

### 👣 Ayaklar (Feet) - Landmark 29-32 (4 nokta)

**Kullanım:** Ayak teması, denge kontrolü, squat kontrolü

| No | İsim | Form Kontrolleri |
|---|---|------------------|
| 29-30 | heels | Topuk teması, denge kontrolü |
| 31-32 | foot_index | Ayak pozisyonu, denge kontrolü |

---

## 🎯 Egzersiz Bazında Kullanılan Landmarklar

### Bicep Curls
**Zorunlu:** 11, 12, 13, 14, 15, 16, 23, 24
- **Kollar:** 11-16 (omuzlar, dirsekler, bilekler)
- **Gövde:** 23-24 (kalçalar)
- **Kafa:** 0 (baş pozisyonu kontrolü)

### Squats
**Zorunlu:** 11, 12, 23, 24, 25, 26, 27, 28
- **Bacaklar:** 23-28 (kalçalar, dizler, ayak bilekleri)
- **Ayaklar:** 29-32 (topuklar, ayak parmak uçları - opsiyonel)
- **Omuzlar:** 11-12 (referans için)

### Lunges
**Zorunlu:** 11, 12, 23, 24, 25, 26, 27, 28
- Squats ile aynı

### Pushups
**Zorunlu:** 11, 12, 13, 14, 15, 16, 23, 24, 25, 26
- **Üst Vücut:** 11-16 (omuzlar, dirsekler, bilekler)
- **Gövde:** 23-24 (kalçalar)
- **Bacaklar:** 25-26 (dizler - vücut çizgisi kontrolü için)

### Lateral Shoulder Raises
**Zorunlu:** 11, 12, 13, 14, 15, 16, 23, 24
- Bicep curls ile aynı

### Tricep Extensions
**Zorunlu:** 11, 12, 13, 14, 15, 16
- **Sadece Kollar:** 11-16 (omuzlar, dirsekler, bilekler)

### Dumbbell Rows
**Zorunlu:** 11, 12, 13, 14, 15, 16, 23, 24
- Bicep curls ile aynı

### Dumbbell Shoulder Press
**Zorunlu:** 11, 12, 13, 14, 15, 16, 23, 24
- Bicep curls ile aynı

---

## 📐 Vücut İskeleti Görsel Düzeni

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

### Python (Backend)

```python
# Landmark dizisinden bir nokta al
landmarks = pose_landmarks  # 33 elemanlı liste

# Burun
nose = landmarks[0]  # {'x': float, 'y': float, 'z': float, 'visibility': float}

# Sol omuz
left_shoulder = landmarks[11]

# Sol dirsek
left_elbow = landmarks[13]

# Sol bilek
left_wrist = landmarks[15]

# Sol kalça
left_hip = landmarks[23]

# Sol diz
left_knee = landmarks[25]

# Sol ayak bileği
left_ankle = landmarks[27]
```

### TypeScript (Frontend)

```typescript
// Landmark dizisinden bir nokta al
const landmarks: Landmark[] = poseLandmarks; // 33 elemanlı dizi

// Burun
const nose = landmarks[0]; // {x: number, y: number, z: number, visibility: number}

// Sol omuz
const leftShoulder = landmarks[11];

// Sol dirsek
const leftElbow = landmarks[13];
```

---

## 📊 Özet İstatistikler

- **Toplam Landmark Sayısı:** 33
- **Yüz Landmarkları:** 11 (0-10)
- **Üst Gövde Landmarkları:** 6 (11-16)
- **El Landmarkları:** 6 (17-22)
- **Alt Gövde Landmarkları:** 2 (23-24)
- **Bacak Landmarkları:** 4 (25-28)
- **Ayak Landmarkları:** 4 (29-32)

**En Çok Kullanılan Landmarklar (Form Analizi):**
1. 11-12 (Omuzlar) - Vücut oranları, gövde stabilitesi
2. 13-14 (Dirsekler) - Kol açıları, rep sayma
3. 15-16 (Bilekler) - Kol hareketi, rep sayma, IMU senkronizasyonu
4. 23-24 (Kalçalar) - Gövde stabilitesi, vücut oranları
5. 25-26 (Dizler) - Squat derinliği, rep sayma
6. 0 (Burun) - Kafa pozisyonu kontrolü

---

## 📚 Referanslar

- MediaPipe Pose Documentation: https://google.github.io/mediapipe/solutions/pose.html
- MediaPipe Pose Landmark Model: https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark
- Proje Dokümantasyonu: `MEDIAPIPE_LANDMARKS_MAPPING.md`

---

**Son Güncelleme:** 1 Ocak 2025
