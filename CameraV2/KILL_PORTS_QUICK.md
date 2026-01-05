# Quick Kill Commands for Ports

Terminale yazabileceğiniz hızlı kill komutları:

## Tek Komutla Tüm Portları Kill Et:

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null; lsof -ti:8765 | xargs kill -9 2>/dev/null; lsof -ti:3000 | xargs kill -9 2>/dev/null; echo "✅ All ports killed"
```

## Ayrı Ayrı Kill Komutları:

### API Server (port 8000)
```bash
lsof -ti:8000 | xargs kill -9
```

### IMU Bridge (port 8765)
```bash
lsof -ti:8765 | xargs kill -9
```

### Frontend (port 3000)
```bash
lsof -ti:3000 | xargs kill -9
```

## Process Name ile Kill (Alternatif):

### API Server
```bash
pkill -9 -f "uvicorn.*api_server"
```

### IMU Bridge
```bash
pkill -9 -f "gymbud_imu_bridge"
```

### Frontend
```bash
pkill -9 -f "react-scripts"
```

## Tüm Python/Node Process'lerini Kill Et (Dikkatli kullanın!):

```bash
pkill -9 python3
pkill -9 node
```

