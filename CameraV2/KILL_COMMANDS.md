# Process Durdurma Komutları (Kill Commands)

## 🛑 Process'leri Durdurma

### **Yöntem 1: Port Bazlı Durdurma (Önerilen)**

**Backend (Port 8000):**
```bash
lsof -ti:8000 | xargs kill -9
```

**Frontend (Port 5173):**
```bash
lsof -ti:5173 | xargs kill -9
```

**IMU Bridge (Port 8765):**
```bash
lsof -ti:8765 | xargs kill -9
```

---

### **Yöntem 2: Process İsmi ile Durdurma**

**Backend (uvicorn/api_server):**
```bash
pkill -f "uvicorn api_server:app"
# veya
pkill -f "api_server.py"
```

**Frontend (vite):**
```bash
pkill -f "vite"
# veya
pkill -f "npm run dev"
```

**IMU Bridge:**
```bash
pkill -f "gymbud_imu_bridge.py"
```

---

### **Yöntem 3: Tümünü Tek Seferde Durdurma**

```bash
# Backend, Frontend ve IMU Bridge'i hepsini durdur
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
lsof -ti:8765 | xargs kill -9 2>/dev/null
```

**Veya bash script olarak:**
```bash
pkill -f "uvicorn api_server:app"
pkill -f "vite"
pkill -f "gymbud_imu_bridge.py"
```

---

### **Yöntem 4: Process ID Bulup Durdurma**

**1. Process ID'yi bul:**
```bash
# Backend için
lsof -i:8000

# Frontend için
lsof -i:5173

# IMU Bridge için
lsof -i:8765
```

**2. Process ID'yi kullanarak durdur:**
```bash
kill -9 <PID>
```

---

## 🔍 Process Kontrolü

**Hangi portlar kullanımda?**
```bash
lsof -i:8000  # Backend
lsof -i:5173  # Frontend
lsof -i:8765  # IMU Bridge
```

**Tüm Python process'leri:**
```bash
ps aux | grep python
```

**Tüm Node process'leri:**
```bash
ps aux | grep node
```

---

## ⚡ Hızlı Komutlar (Kopyala-Yapıştır)

### **Tek Komutla Hepsini Durdur:**
```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null; lsof -ti:5173 | xargs kill -9 2>/dev/null; lsof -ti:8765 | xargs kill -9 2>/dev/null; echo "✅ Tüm process'ler durduruldu"
```

### **Python Process'lerini Durdur:**
```bash
pkill -f "uvicorn api_server:app"; pkill -f "gymbud_imu_bridge.py"; echo "✅ Python process'ler durduruldu"
```

### **Node Process'lerini Durdur:**
```bash
pkill -f "vite"; echo "✅ Node process'ler durduruldu"
```

---

## 📝 Bash Script Oluşturma

**Kill script'i oluştur:**

```bash
cd /Users/kaantaplamacioglu/Desktop/Elec-491-separate-training-workflow/CameraV2
cat > kill_all.sh << 'EOF'
#!/bin/bash
echo "🛑 Process'leri durduruyorum..."

# Backend
if lsof -ti:8000 > /dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9
    echo "✅ Backend (port 8000) durduruldu"
else
    echo "ℹ️  Backend zaten durdurulmuş"
fi

# Frontend
if lsof -ti:5173 > /dev/null 2>&1; then
    lsof -ti:5173 | xargs kill -9
    echo "✅ Frontend (port 5173) durduruldu"
else
    echo "ℹ️  Frontend zaten durdurulmuş"
fi

# IMU Bridge
if lsof -ti:8765 > /dev/null 2>&1; then
    lsof -ti:8765 | xargs kill -9
    echo "✅ IMU Bridge (port 8765) durduruldu"
else
    echo "ℹ️  IMU Bridge zaten durdurulmuş"
fi

echo "✅ Tamamlandı!"
EOF

chmod +x kill_all.sh
```

**Kullanım:**
```bash
./kill_all.sh
```

---

## ⚠️ Notlar

1. **`kill -9`** = Force kill (process'i zorla durdurur)
2. **`2>/dev/null`** = Hata mesajlarını gizler (process yoksa hata vermez)
3. **`lsof -ti:PORT`** = Port'ta çalışan process ID'yi döndürür
4. **`pkill -f PATTERN`** = Process isminde pattern arar ve durdurur

---

## 🔄 Yeniden Başlatma

**Durdur ve yeniden başlat:**
```bash
# Durdur
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null
lsof -ti:8765 | xargs kill -9 2>/dev/null

# Yeniden başlat (yeni terminal'lerde)
# Terminal 1:
cd CameraV2 && python3 -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2:
cd CameraV2/frontend && npm run dev

# Terminal 3:
cd CameraV2 && python3 gymbud_imu_bridge.py
```

