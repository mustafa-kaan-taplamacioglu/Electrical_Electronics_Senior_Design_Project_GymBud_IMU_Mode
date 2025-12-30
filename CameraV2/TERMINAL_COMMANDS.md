# Terminal Komutları - Sırayla Çalıştırın

## 🔌 Terminal 1: IMU Bridge (WebSocket Server)

```bash
cd /Users/kaantaplamacioglu/Desktop/github_repo_elec_491/Elec-491/CameraV2
python3 gymbud_imu_bridge.py
```

**Ne yapar:**
- Serial port'tan IMU verilerini okur (`/dev/cu.usbmodem101`)
- WebSocket üzerinden frontend'e gönderir (port: 8765)
- CSV loglarına yazar (`logs/` klasörüne)

---

## 🌐 Terminal 2: API Server (FastAPI Backend)

```bash
cd /Users/kaantaplamacioglu/Desktop/github_repo_elec_491/Elec-491/CameraV2
python3 api_server.py
```

**Ne yapar:**
- Camera verilerini işler (MediaPipe pose detection)
- FastAPI backend'i çalıştırır (port: 8000)
- ML model inference yapar
- Dataset collection yönetir

---

## 💻 Terminal 3: Frontend (React App)

```bash
cd /Users/kaantaplamacioglu/Desktop/github_repo_elec_491/Elec-491/CameraV2/frontend
npm run dev
```

**Ne yapar:**
- React frontend'i çalıştırır (genellikle port: 5173 veya 3000)
- Browser'da otomatik açılır veya `http://localhost:5173` adresine gidin

---

## ✅ Çalışma Sırası

1. **Önce Terminal 1'i başlatın** (IMU Bridge)
   - Serial port bağlantısını kontrol eder
   - IMU verilerini bekler

2. **Sonra Terminal 2'yi başlatın** (API Server)
   - Camera pipeline'ını başlatır
   - Backend hazır olur

3. **Son olarak Terminal 3'ü başlatın** (Frontend)
   - Web arayüzü açılır
   - Her iki servise de bağlanır

---

## 🔍 Kontrol Etmek İçin

**IMU Bridge çalışıyor mu?**
- Terminal 1'de IMU verilerini görmelisiniz (node_id, sample_number, ax, ay, az, gx, gy, gz)

**API Server çalışıyor mu?**
- Terminal 2'de "Application startup complete" mesajını görmelisiniz
- Browser'da `http://localhost:8000/docs` adresine gidip API dokümantasyonunu görebilirsiniz

**Frontend çalışıyor mu?**
- Browser'da workout interface açılmalı
- Camera ve IMU verileri görünmelidir

---

## ⚠️ Sorun Giderme

**Serial port bulunamadı:**
```bash
ls -la /dev/cu.usbmodem*
```
- Arduino Central Hub bağlı mı kontrol edin
- Port numarası değişmişse `gymbud_imu_bridge.py` içindeki `SERIAL_PORT` değişkenini güncelleyin

**Port zaten kullanılıyor:**
- Eski process'leri kapatın: `pkill -f gymbud_imu_bridge.py` veya `pkill -f api_server.py`

**Dependencies eksik:**
```bash
cd /Users/kaantaplamacioglu/Desktop/github_repo_elec_491/Elec-491/CameraV2
pip3 install -r requirements.txt
```
