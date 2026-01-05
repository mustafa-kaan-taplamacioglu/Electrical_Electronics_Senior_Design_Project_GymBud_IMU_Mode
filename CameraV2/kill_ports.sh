#!/bin/bash
# Port bazlı kill komutları - portlar dolu olduğunda kullan

echo "🛑 Port'lardaki process'leri durduruyorum..."

# Backend - Port 8000
PID_8000=$(lsof -ti:8000)
if [ ! -z "$PID_8000" ]; then
    kill -9 $PID_8000 2>/dev/null
    echo "✅ Port 8000 (Backend) durduruldu - PID: $PID_8000"
else
    echo "ℹ️  Port 8000 boş"
fi

# Frontend - Port 5173
PID_5173=$(lsof -ti:5173)
if [ ! -z "$PID_5173" ]; then
    kill -9 $PID_5173 2>/dev/null
    echo "✅ Port 5173 (Frontend) durduruldu - PID: $PID_5173"
else
    echo "ℹ️  Port 5173 boş"
fi

# IMU Bridge - Port 8765
PID_8765=$(lsof -ti:8765)
if [ ! -z "$PID_8765" ]; then
    kill -9 $PID_8765 2>/dev/null
    echo "✅ Port 8765 (IMU Bridge) durduruldu - PID: $PID_8765"
else
    echo "ℹ️  Port 8765 boş"
fi

echo ""
echo "✅ Tamamlandı!"
