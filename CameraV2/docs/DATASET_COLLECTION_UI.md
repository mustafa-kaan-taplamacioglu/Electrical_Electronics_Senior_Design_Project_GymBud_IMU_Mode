# 📊 Dataset Collection UI Kullanım Kılavuzu

## ✅ Dataset Collection Özelliği Eklendi!

Artık **UI üzerinden** dataset collection'ı açıp kapatabilirsiniz!

---

## 🎯 Nasıl Kullanılır?

### 1. **Antrenmana Başla**
- Normal şekilde antrenman seçip, kamerayı başlatın

### 2. **Dataset Collection Toggle'ı Bul**
- Camera selection ekranında, **"📊 Dataset Collection (ML Training)"** bölümünü göreceksiniz
- Sağdaki toggle switch'i kullanarak açıp kapatabilirsiniz

### 3. **Collection'ı Aktif Et**
- Toggle'ı **ON** (yeşil) yapın
- Artık her rep otomatik olarak kaydedilecek
- Toplanan rep sayısı gerçek zamanlı olarak gösterilecek

### 4. **Antrenman Yap**
- Normal şekilde antrenman yapın
- Her rep tamamlandığında otomatik olarak dataset'e eklenir
- Toplanan rep sayısı artarak gösterilir: `✅ Collecting data... • 5 rep(s) collected`

### 5. **Session Bitince**
- **Otomatik kayıt:** Session bittiğinde dataset otomatik olarak kaydedilir
- **Manuel kayıt:** Toggle'ı **OFF** yaparak da manuel kaydedebilirsiniz

---

## 📍 UI'da Nerede?

### Camera Selection Ekranı

```
┌─────────────────────────────────────────┐
│  🔄 Sensor Fusion Mode                  │
│  [Camera + IMU Enhancement ▼]          │
│                                         │
│  📊 Dataset Collection (ML Training)   │
│  [                       ] ← Toggle    │
│  ✅ Collecting data... • 5 reps        │
└─────────────────────────────────────────┘
```

---

## 🎛️ Toggle Durumları

### 🔴 OFF (Kırmızı/Gri)
- Dataset collection **kapalı**
- Rep'ler kaydedilmez
- Mesaj: `"Data collection disabled. Enable to save rep data for training."`

### 🟢 ON (Yeşil)
- Dataset collection **açık**
- Her rep otomatik kaydedilir
- Mesaj: `"✅ Collecting data for ML model training • X rep(s) collected"`

---

## 📂 Veriler Nereye Kaydediliyor?

Dataset'ler `CameraV2/dataset/` klasörüne kaydedilir:

```
dataset/
├── bicep_curls_20250101_120000/
│   ├── samples.json          # Tüm rep verileri
│   └── summary.csv           # Özet tablo
├── squats_20250101_130000/
│   ├── samples.json
│   └── summary.csv
└── ...
```

---

## 🔄 Real-Time Güncellemeler

- **Rep sayısı:** Her rep tamamlandığında otomatik artar
- **Status:** Collection durumu gerçek zamanlı gösterilir
- **Kayıt:** Session bitince veya toggle OFF yapınca otomatik kaydedilir

---

## ⚙️ Backend Entegrasyonu

### WebSocket Mesajları

**Frontend → Backend:**
```json
{
  "type": "start_collection"
}
```

```json
{
  "type": "stop_collection",
  "auto_label_perfect": true
}
```

**Backend → Frontend:**
```json
{
  "type": "dataset_collection_status",
  "status": "collecting",
  "collected_reps": 5
}
```

---

## 📊 Örnek Kullanım Senaryosu

### Senaryo 1: Normal Antrenman (Collection Kapalı)
1. Antrenman seç → Camera başlat
2. Toggle **OFF** (default)
3. Antrenman yap
4. Rep'ler sayılır ama kaydedilmez

### Senaryo 2: Dataset Toplama
1. Antrenman seç → Camera başlat
2. Toggle **ON** yap
3. Antrenman yap (10-20 rep)
4. Toggle **OFF** yap → Dataset kaydedilir
5. `dataset/` klasöründe görünür

### Senaryo 3: Session Bitince Otomatik Kayıt
1. Toggle **ON** yap
2. Antrenman yap
3. "Finish" butonuna tıkla
4. Dataset otomatik kaydedilir
5. Mesaj: `"💾 Dataset saved successfully!"`

---

## ⚠️ Önemli Notlar

1. **Toggle sadece antrenman başladıktan sonra aktif olur**
   - Camera selection ekranında disabled görünebilir
   - Antrenman başladıktan sonra toggle çalışır

2. **WebSocket bağlantısı gerekli**
   - Backend çalışıyor olmalı (`python -m uvicorn api_server:app`)
   - Bağlantı yoksa toggle çalışmaz

3. **Her rep otomatik kaydedilir**
   - Collection açıkken her rep için:
     - Landmarks sequence
     - Form skorları
     - Regional skorlar
     - Açılar
     - Issues
   - Otomatik olarak kaydedilir

4. **Memory yönetimi**
   - Her rep için son 100 frame saklanır
   - Session bitince otomatik temizlenir

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Backend'i başlat
cd CameraV2
python -m uvicorn api_server:app --reload --port 8000

# 2. Frontend'i başlat (başka terminal)
cd CameraV2/frontend
npm run dev

# 3. Tarayıcıda aç
# http://localhost:5173

# 4. Dataset Collection'ı aktif et
# - Antrenman seç
# - Camera başlat
# - Toggle ON yap
# - Antrenman yap!
```

---

## ❓ Sorun Giderme

### Toggle çalışmıyor?
- ✅ Backend çalışıyor mu? (`http://localhost:8000`)
- ✅ WebSocket bağlantısı kuruldu mu? (Console'da "Backend WebSocket connected" mesajı)
- ✅ Antrenman başlatıldı mı? (Camera başlatıldı mı?)

### Rep'ler kaydedilmiyor?
- ✅ Toggle ON mu?
- ✅ Backend console'da mesajlar var mı?
- ✅ `dataset/` klasörü oluştu mu?

### Dataset dosyası bulunamıyor?
- ✅ `CameraV2/dataset/` klasörüne bakın
- ✅ Session ID ile klasör oluşmuş mu? (`exercise_YYYYMMDD_HHMMSS/`)

---

**Artık UI'dan kolayca dataset toplayabilirsiniz! 🎉**

