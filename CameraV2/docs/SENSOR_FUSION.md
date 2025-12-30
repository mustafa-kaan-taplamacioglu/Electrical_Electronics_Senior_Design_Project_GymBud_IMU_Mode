# GymBud Sensor Fusion System

## Overview

This system fuses data from two sources for enhanced avatar animation:
1. **Camera (MediaPipe)** - Full body pose detection
2. **IMU Sensors (BLE)** - Precise orientation for wrists and chest

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              WorkoutSessionWithIMU.tsx                   │    │
│  │  ┌──────────────────┐  ┌─────────────────────────────┐  │    │
│  │  │  MediaPipe Pose  │  │    HumanAvatar.tsx          │  │    │
│  │  │   Detection      │  │  (3D Avatar + Fusion)       │  │    │
│  │  └────────┬─────────┘  └─────────────────────────────┘  │    │
│  │           │                       ▲                      │    │
│  │           ▼                       │                      │    │
│  │   ┌───────────────┐      ┌────────┴────────┐            │    │
│  │   │   Camera      │      │   useIMU Hook   │            │    │
│  │   │   Landmarks   │      │   (WebSocket)   │            │    │
│  │   └───────────────┘      └────────┬────────┘            │    │
│  └────────────────────────────────────│────────────────────┘    │
│                                       │                          │
└───────────────────────────────────────│──────────────────────────┘
                                        │
                              WebSocket │ ws://localhost:8765
                                        │
┌───────────────────────────────────────│──────────────────────────┐
│                    Python Backend                                 │
│  ┌────────────────────────────────────▼─────────────────────┐    │
│  │              gymbud_imu_bridge.py                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │    │
│  │  │  Madgwick    │  │  WebSocket   │  │  CSV         │    │    │
│  │  │  Filter      │  │  Server      │  │  Logger      │    │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │    │
│  │           ▲                                               │    │
│  │           │ Serial                                        │    │
│  │  ┌────────┴──────────────────────────────────────────┐   │    │
│  │  │            Arduino Central Hub                     │   │    │
│  │  │  (GymBud_Central_Hub.ino)                         │   │    │
│  │  └───────────────────────────────────────────────────┘   │    │
│  └───────────────────────────────────────────────────────────┘   │
│                         ▲  ▲  ▲                                   │
│                  BLE    │  │  │                                   │
│          ┌──────────────┘  │  └──────────────┐                   │
│  ┌───────┴───────┐ ┌──────┴──────┐ ┌────────┴───────┐           │
│  │   Left Wrist  │ │ Right Wrist │ │     Chest      │           │
│  │   Node (ID:1) │ │  Node (ID:2)│ │   Node (ID:3)  │           │
│  └───────────────┘ └─────────────┘ └────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
```

## Node Configuration

| Node ID | Name | Location | Purpose |
|---------|------|----------|---------|
| 1 | Left Wrist | LW | Forearm orientation |
| 2 | Right Wrist | RW | Forearm orientation |
| 3 | Chest | CH | Torso orientation |

## Fusion Modes

| Mode | Description |
|------|-------------|
| `camera_only` | Only use MediaPipe camera data |
| `camera_primary` | Camera + IMU enhancement (default) |
| `imu_primary` | IMU dominates, camera for fallback |
| `imu_only` | Only use IMU data (no camera) |

## Quick Start

### 1. Install Dependencies

```bash
cd CameraV2
pip install -r requirements.txt
```

### 2. Start the IMU Bridge

```bash
# Edit gymbud_imu_bridge.py to set your serial port:
# SERIAL_PORT = "/dev/cu.usbmodem101"  # macOS
# SERIAL_PORT = "COM5"                 # Windows

python gymbud_imu_bridge.py
```

You should see:
```
╔═══════════════════════════════════════════════════╗
║          GymBud IMU-WebSocket Bridge              ║
╚═══════════════════════════════════════════════════╝

🚀 Starting GymBud IMU Bridge
   Serial: /dev/cu.usbmodem101 @ 115200
   WebSocket: ws://0.0.0.0:8765
   
✅ Serial port opened
✅ WebSocket server started on ws://localhost:8765
📊 Node 1 (left_wrist): Roll=0.5° Pitch=-2.3° Yaw=10.1° [48.5 Hz]
```

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Start the Backend API (optional, for form analysis)

```bash
python api_server.py
```

## Using the Enhanced WorkoutSession

Import and use `WorkoutSessionWithIMU` instead of `WorkoutSession`:

```tsx
import { WorkoutSessionWithIMU } from './components/WorkoutSessionWithIMU';

// In your component:
<WorkoutSessionWithIMU
  exercise="bicep_curls"
  apiKey={apiKey}
  avatarUrl="/models/avatar-male.glb"
  onEnd={() => setSelectedExercise(null)}
/>
```

## HumanAvatar Props (Updated)

```tsx
<HumanAvatar 
  // Existing props
  landmarks={currentLandmarks} 
  width={400} 
  height={480} 
  modelUrl="/models/avatar-male.glb"
  showSkeleton={true}
  isCalibrated={true}
  
  // NEW: IMU props
  imuData={{
    leftWrist: leftWristData,   // IMUNodeData
    rightWrist: rightWristData, // IMUNodeData
    chest: chestData            // IMUNodeData
  }}
  fusionMode="camera_primary"   // SensorFusionMode
  showIMUDebug={true}          // Show debug overlay
/>
```

## IMU Data Format

Each IMU node sends data in this format:

```typescript
interface IMUNodeData {
  node_id: number;          // 1, 2, or 3
  timestamp: number;        // Unix timestamp
  accel: { x, y, z };      // Accelerometer (g)
  gyro: { x, y, z };       // Gyroscope (deg/s)
  quaternion: { w, x, y, z }; // Orientation quaternion
  euler: {                  // Euler angles (degrees)
    roll: number;
    pitch: number;
    yaw: number;
  };
}
```

## Calibration

The system automatically calibrates when:
1. Camera detects full body and holds for 30 frames
2. IMU orientations are captured at the same moment as "rest pose"

Press **"Reset IMU"** button to recalibrate IMU orientation to current position.

## Fusion Algorithm

### Camera Primary Mode (Recommended)

```
Final Rotation = SLERP(Camera Rotation, IMU Rotation, weight)

Weights:
- Left Wrist: 70% IMU, 30% Camera
- Right Wrist: 70% IMU, 30% Camera  
- Chest: 50% IMU, 50% Camera
```

### Madgwick Filter

The bridge uses a Madgwick orientation filter to estimate orientation from raw IMU data:
- `BETA = 0.1` - Filter gain (lower = smoother, higher = faster)
- `SAMPLE_RATE = 50` - Expected IMU sample rate in Hz

## File Summary

| File | Purpose |
|------|---------|
| `CameraV2/gymbud_imu_bridge.py` | Serial → WebSocket bridge with Madgwick filter |
| `CameraV2/frontend/src/hooks/useIMU.ts` | React hook for IMU WebSocket |
| `CameraV2/frontend/src/utils/sensorFusion.ts` | Fusion algorithms and helpers |
| `CameraV2/frontend/src/components/HumanAvatar.tsx` | 3D avatar with fusion support |
| `CameraV2/frontend/src/components/WorkoutSessionWithIMU.tsx` | Complete workout session with IMU |
| `CameraV2/frontend/src/types.ts` | TypeScript types for IMU data |

## Troubleshooting

### IMU Not Connecting
1. Check serial port: `ls /dev/cu.*` (macOS) or Device Manager (Windows)
2. Ensure Arduino is running GymBud_Central_Hub.ino
3. Check WebSocket port 8765 is not in use

### Avatar Not Moving with IMU
1. Ensure calibration completed (hold still for 1 second)
2. Check fusion mode is not `camera_only`
3. Verify IMU data is being received (check debug panel)

### Jittery Movement
1. Increase smoothing in `sensorFusion.ts`: `SMOOTHING.rotation = 0.5`
2. Reduce Madgwick beta in `gymbud_imu_bridge.py`: `BETA = 0.05`
3. Increase fusion weight toward camera

