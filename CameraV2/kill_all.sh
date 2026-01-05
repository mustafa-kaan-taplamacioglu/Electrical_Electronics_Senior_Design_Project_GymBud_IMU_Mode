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
