# Hızlı Kill Komutları - Port Dolu Olduğunda

## 🔥 Port Bazlı Kill Komutları

### **Tek Komutla Port'taki Process'i Durdur:**

```bash
# Backend (Port 8000)
kill -9 $(lsof -ti:8000)

# Frontend (Port 5173)
kill -9 $(lsof -ti:5173)

# IMU Bridge (Port 8765)
kill -9 $(lsof -ti:8765)
```

---

## ⚡ Tek Komutla Hepsini Durdur

```bash
kill -9 $(lsof -ti:8000) $(lsof -ti:5173) $(lsof -ti:8765) 2>/dev/null
```

---

## 🔨 pkill ile (Process İsmi ile)

```bash
# Backend
pkill -9 -f uvicorn
pkill -9 -f api_server

# Frontend
pkill -9 -f vite
pkill -9 -f node

# IMU Bridge
pkill -9 -f gymbud_imu_bridge
```

---

## 📋 Kopyala-Yapıştır Komutlar

### **Backend Durdur:**
```bash
kill -9 $(lsof -ti:8000) 2>/dev/null; echo "✅ Backend durduruldu"
```

### **Frontend Durdur:**
```bash
kill -9 $(lsof -ti:5173) 2>/dev/null; echo "✅ Frontend durduruldu"
```

### **IMU Bridge Durdur:**
```bash
kill -9 $(lsof -ti:8765) 2>/dev/null; echo "✅ IMU Bridge durduruldu"
```

### **Hepsini Durdur:**
```bash
kill -9 $(lsof -ti:8000) $(lsof -ti:5173) $(lsof -ti:8765) 2>/dev/null; echo "✅ Tüm process'ler durduruldu"
```

---

## 🔍 Port Kontrolü (Process Var mı?)

```bash
# Port 8000 kontrol
lsof -i:8000

# Port 5173 kontrol
lsof -i:5173

# Port 8765 kontrol
lsof -i:8765
```

---

## 📝 Script Kullanımı

**Oluşturulan script:**
```bash
cd CameraV2
./kill_ports.sh
```

---

## 💡 Alternatif Yöntemler

### **1. Process ID Bulup Kill:**
```bash
# Process ID'yi bul
lsof -ti:8000

# Bulunan PID'yi kullan
kill -9 <PID>
```

### **2. Tüm Python Process'leri:**
```bash
pkill -9 python3
```

### **3. Tüm Node Process'leri:**
```bash
pkill -9 node
```

---

## ⚠️ Notlar

- `kill -9` = Force kill (process'i zorla durdurur)
- `2>/dev/null` = Hata mesajlarını gizler (port boşsa hata vermez)
- `lsof -ti:PORT` = Port'taki process ID'yi döndürür
- `pkill -9 -f PATTERN` = Process isminde pattern arar ve force kill yapar

