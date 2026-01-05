# Sistem Çalıştırma Komutları (Run System Commands)

## 📋 Gereksinimler (Requirements)

1. **Python 3.8+** (backend için)
2. **Node.js 16+** (frontend için)
3. **gymbud_imu_bridge** (IMU verisi için - opsiyonel)

---

## 🚀 Sistem Başlatma (System Startup)

### **1. Backend Server (API Server)**

```bash
cd CameraV2
python3 -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

**Alternatif (eğer uvicorn yoksa):**
```bash
cd CameraV2
python3 api_server.py
```

**Not:** Backend `http://localhost:8000` adresinde çalışacak.

---

### **2. Frontend (React + Vite)**

**Yeni terminal'de:**
```bash
cd CameraV2/frontend
npm install  # İlk kurulumda gerekli
npm run dev
```

**Not:** Frontend genellikle `http://localhost:5173` adresinde çalışır.

---

### **3. IMU Bridge (Opsiyonel - IMU verisi için)**

**Yeni terminal'de (eğer IMU kullanacaksan):**
```bash
cd CameraV2
python3 gymbud_imu_bridge.py
```

**Not:** IMU bridge `ws://localhost:8765` adresinde çalışır.

---

## 📝 Hızlı Başlatma (Quick Start)

### **Tüm sistemi başlatmak için:**

**Terminal 1 - Backend:**
```bash
cd /Users/kaantaplamacioglu/Desktop/Elec-491-separate-training-workflow/CameraV2
python3 -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd /Users/kaantaplamacioglu/Desktop/Elec-491-separate-training-workflow/CameraV2/frontend
npm run dev
```

**Terminal 3 - IMU Bridge (opsiyonel):**
```bash
cd /Users/kaantaplamacioglu/Desktop/Elec-491-separate-training-workflow/CameraV2
python3 gymbud_imu_bridge.py
```

---

## 🔍 Port Kontrolü

Sistemin çalışıp çalışmadığını kontrol et:

```bash
# Backend kontrolü
curl http://localhost:8000/docs

# Frontend kontrolü (tarayıcıda)
open http://localhost:5173

# IMU Bridge kontrolü (opsiyonel)
curl http://localhost:8765
```

---

## ⚠️ Sorun Giderme (Troubleshooting)

### **Backend başlamıyorsa:**
```bash
# Port 8000 kullanımda mı kontrol et
lsof -i :8000

# Eğer kullanımdaysa, process'i öldür
kill -9 <PID>

# Veya farklı port kullan
python3 -m uvicorn api_server:app --reload --port 8001
```

### **Frontend başlamıyorsa:**
```bash
# node_modules'ü sil ve yeniden kur
cd CameraV2/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### **IMU Bridge başlamıyorsa:**
- IMU cihazının bağlı olduğundan emin ol
- Serial port'un doğru olduğundan emin ol
- `gymbud_imu_bridge.py` dosyasındaki port ayarlarını kontrol et

---

## 📊 Sistem Durumu

Sistem başladıktan sonra:

1. **Backend:** Terminal'de `Application startup complete` mesajını görmelisin
2. **Frontend:** Terminal'de `Local: http://localhost:5173` mesajını görmelisin
3. **IMU Bridge:** Terminal'de `IMU Bridge started on ws://localhost:8765` mesajını görmelisin

---

## 🎯 Kullanım Akışı

1. Backend'i başlat
2. Frontend'i başlat
3. (Opsiyonel) IMU Bridge'i başlat
4. Tarayıcıda `http://localhost:5173` adresine git
5. Hareket seç (bicep_curls, squats, vb.)
6. ML Training modunu seç
7. IMU+Camera modunu seç (otomatik)
8. Antrenmana başla!

---

## 🔄 Durdurma (Stop)

Her terminal'de `Ctrl+C` ile durdur.

