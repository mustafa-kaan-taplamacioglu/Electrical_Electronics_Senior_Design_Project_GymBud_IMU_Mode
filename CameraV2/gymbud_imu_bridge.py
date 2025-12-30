#!/usr/bin/env python3
"""
GymBud IMU-WebSocket Bridge
============================
Reads IMU data from BLE nodes via serial and broadcasts to frontend via WebSocket.

Node IDs:
- 1 = Left Wrist (LW)
- 2 = Right Wrist (RW)

This enables sensor fusion with the camera-based MediaPipe avatar.
"""

import asyncio
import json
import time
import re
import csv
import math
import os
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Set
from dataclasses import dataclass, asdict

import serial
import websockets
from websockets.server import serve

# ==================== CONFIGURATION ====================

SERIAL_PORT = "/dev/cu.usbmodem101"  # Change to your port (found: 1101 or 1201)
BAUD_RATE = 115200

WEBSOCKET_HOST = "0.0.0.0"
WEBSOCKET_PORT = 8765

# Node configuration
NODE_NAMES = {
    1: "left_wrist",
    2: "right_wrist",
    3: "chest"
}

# Madgwick filter settings for orientation estimation
BETA = 0.1  # Filter gain (0.01 - 0.5, higher = faster response, more noise)
SAMPLE_RATE = 20  # Hz (20 samples/sec per node, 60 total for 3 nodes)

# CSV logging
LOG_TO_CSV = True
CSV_DIR = "logs"

# ==================== DATA CLASSES ====================

@dataclass
class IMUSample:
    """Single IMU sample from a node."""
    node_id: int
    node_name: str
    timestamp: float
    # Accelerometer (g)
    ax: float
    ay: float
    az: float
    # Gyroscope (deg/s)
    gx: float
    gy: float
    gz: float
    # Calculated orientation (quaternion)
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    # Euler angles (degrees) for easier use
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class FusedIMUData:
    """All IMU nodes fused into a single message for frontend."""
    timestamp: float
    left_wrist: Optional[dict] = None
    right_wrist: Optional[dict] = None
    chest: Optional[dict] = None
    # Raw sensor data (exact as received from serial) for each node
    raw_data: Optional[dict] = None  # {node_id: "1,10898,0.0,-0.5144,0.8808,-1.26,-5.39,-0.56"}


# ==================== MADGWICK FILTER ====================

class MadgwickFilter:
    """
    Madgwick's gradient descent orientation filter for IMU data.
    Estimates orientation from accelerometer and gyroscope data.
    """
    
    def __init__(self, beta: float = 0.1, sample_period: float = 0.02):
        self.beta = beta
        self.sample_period = sample_period
        # Quaternion: [w, x, y, z]
        self.q = [1.0, 0.0, 0.0, 0.0]
    
    def update(self, gx: float, gy: float, gz: float, 
               ax: float, ay: float, az: float) -> tuple:
        """
        Update orientation estimate.
        
        Args:
            gx, gy, gz: Gyroscope data in rad/s
            ax, ay, az: Accelerometer data (normalized)
            
        Returns:
            (qw, qx, qy, qz): Orientation quaternion
        """
        q0, q1, q2, q3 = self.q
        
        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)
        
        # Normalize accelerometer measurement
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0.001:
            ax /= norm
            ay /= norm
            az /= norm
            
            # Gradient descent algorithm corrective step
            _2q0 = 2.0 * q0
            _2q1 = 2.0 * q1
            _2q2 = 2.0 * q2
            _2q3 = 2.0 * q3
            _4q0 = 4.0 * q0
            _4q1 = 4.0 * q1
            _4q2 = 4.0 * q2
            _8q1 = 8.0 * q1
            _8q2 = 8.0 * q2
            q0q0 = q0 * q0
            q1q1 = q1 * q1
            q2q2 = q2 * q2
            q3q3 = q3 * q3
            
            # Gradient
            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
            s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
            s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
            s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay
            
            # Normalize gradient
            norm = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            if norm > 0.001:
                s0 /= norm
                s1 /= norm
                s2 /= norm
                s3 /= norm
                
                # Apply feedback step
                qDot1 -= self.beta * s0
                qDot2 -= self.beta * s1
                qDot3 -= self.beta * s2
                qDot4 -= self.beta * s3
        
        # Integrate to get quaternion
        q0 += qDot1 * self.sample_period
        q1 += qDot2 * self.sample_period
        q2 += qDot3 * self.sample_period
        q3 += qDot4 * self.sample_period
        
        # Normalize quaternion
        norm = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if norm > 0.001:
            self.q = [q0 / norm, q1 / norm, q2 / norm, q3 / norm]
        
        return tuple(self.q)
    
    def get_euler(self) -> tuple:
        """
        Get Euler angles (roll, pitch, yaw) from quaternion.
        
        Returns:
            (roll, pitch, yaw) in degrees
        """
        q0, q1, q2, q3 = self.q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw)
        )
    
    def reset(self):
        """Reset orientation to identity."""
        self.q = [1.0, 0.0, 0.0, 0.0]


# ==================== IMU BRIDGE ====================

class IMUBridge:
    """
    Bridges serial IMU data to WebSocket for frontend consumption.
    """
    
    # Regex to parse Arduino output - format: "ID:1 ax:0.08 ay:-0.68 az:0.74 gx:1.12 gy:-4.83 gz:-0.42"
    LINE_RE = re.compile(
        r"ID:(?P<id>\d+)\s+ax:(?P<ax>-?\d+(\.\d+)?)\s+ay:(?P<ay>-?\d+(\.\d+)?)\s+az:(?P<az>-?\d+(\.\d+)?)\s+"
        r"gx:(?P<gx>-?\d+(\.\d+)?)\s+gy:(?P<gy>-?\d+(\.\d+)?)\s+gz:(?P<gz>-?\d+(\.\d+)?)"
    )
    
    # CSV format: "node_id,sample_number,ax,ay,az,gx,gy,gz"
    # Example: "2,10874,-0.1088,0.2621,0.9877,0.63,-1.96,0.77"
    CSV_RE = re.compile(
        r"^(?P<id>[123]),(?P<sample>\d+),(?P<ax>-?\d+\.?\d*),(?P<ay>-?\d+\.?\d*),(?P<az>-?\d+\.?\d*),"
        r"(?P<gx>-?\d+\.?\d*),(?P<gy>-?\d+\.?\d*),(?P<gz>-?\d+\.?\d*)\s*$"
    )
    
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.serial: Optional[serial.Serial] = None
        
        # Connected WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Madgwick filters for each node
        self.filters: Dict[int, MadgwickFilter] = {
            node_id: MadgwickFilter(beta=BETA, sample_period=1.0/SAMPLE_RATE)
            for node_id in NODE_NAMES.keys()
        }
        
        # Latest data from each node
        self.latest_data: Dict[int, IMUSample] = {}
        
        # Raw data strings (exact as received) for each node - for exact frontend display
        self.raw_data_strings: Dict[int, str] = {}
        
        # Data history for each node (for smoothing)
        self.history: Dict[int, deque] = {
            node_id: deque(maxlen=10)
            for node_id in NODE_NAMES.keys()
        }
        
        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        
        # Stats (per-node tracking)
        self.sample_count = 0
        self.node_sample_counts: Dict[int, int] = {nid: 0 for nid in NODE_NAMES.keys()}
        self.node_last_time: Dict[int, float] = {}
        self.node_rates: Dict[int, float] = {nid: 0.0 for nid in NODE_NAMES.keys()}
        self.start_time = None
        
        # Running flag
        self.running = False
        
        # Callbacks for external systems (e.g., api_server.py for training data collection)
        # Callback signature: callback(node_id: int, sample: IMUSample, raw_values: dict)
        self.sample_callbacks: list = []
    
    def start_logging(self):
        """Start CSV logging - ML-friendly formatted data."""
        if not LOG_TO_CSV:
            return
        
        os.makedirs(CSV_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{CSV_DIR}/gymbud_fusion_{timestamp}.csv"
        
        self.csv_file = open(filename, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # ML-friendly CSV format: timestamp, node_id, ax, ay, az, gx, gy, gz, qw, qx, qy, qz, roll, pitch, yaw
        self.csv_writer.writerow([
            "timestamp", "node_id", "node_name",
            "ax", "ay", "az", "gx", "gy", "gz",
            "qw", "qx", "qy", "qz",
            "roll", "pitch", "yaw"
        ])
        print(f"📝 Logging ML-formatted data to: {filename}")
    
    def stop_logging(self):
        """Stop CSV logging."""
        if self.csv_file:
            self.csv_file.flush()  # Ensure all data is written before closing
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
    
    def parse_line(self, line: str) -> Optional[tuple]:
        """
        Parse a line of serial data.
        
        Supported formats:
        1. CSV: "node_id,sample_number,ax,ay,az,gx,gy,gz" (e.g., "2,10874,-0.1088,0.2621,0.9877,0.63,-1.96,0.77")
        2. ID: "ID:1 ax:0.08 ay:-0.68 az:0.74 gx:1.12 gy:-4.83 gz:-0.42"
        
        Returns:
            (node_id, sample_number, ax, ay, az, gx, gy, gz, raw_line) or None if parse failed
            sample_number will be None if not in the input format
            raw_line is the original unmodified line string
        """
        original_line = line  # Keep original for exact output
        line = line.strip()
        if not line:
            return None
        
        # Try CSV format first (most common): "1,10875,0.001,-0.5168,0.8808,2.03,-4.48,-0.42"
        m = self.CSV_RE.match(line)
        if m:
            return (
                int(m["id"]),
                int(m["sample"]),  # Include sample_number
                float(m["ax"]), float(m["ay"]), float(m["az"]),
                float(m["gx"]), float(m["gy"]), float(m["gz"]),
                original_line  # Return original raw line for exact output
            )
        
        # Try ID: format (alternative) - no sample_number in this format
        m = self.LINE_RE.search(line)
        if m:
            return (
                int(m["id"]),
                None,  # No sample_number in ID format
                float(m["ax"]), float(m["ay"]), float(m["az"]),
                float(m["gx"]), float(m["gy"]), float(m["gz"]),
                original_line  # Return original raw line for exact output
            )
        
        return None
    
    def process_sample(self, node_id: int, ax: float, ay: float, az: float,
                       gx: float, gy: float, gz: float) -> IMUSample:
        """
        Process an IMU sample: apply Madgwick filter and create IMUSample.
        """
        timestamp = time.time()
        
        # Get or create filter
        if node_id not in self.filters:
            self.filters[node_id] = MadgwickFilter(beta=BETA, sample_period=1.0/SAMPLE_RATE)
        
        filt = self.filters[node_id]
        
        # Convert gyro from deg/s to rad/s
        gx_rad = math.radians(gx)
        gy_rad = math.radians(gy)
        gz_rad = math.radians(gz)
        
        # Update Madgwick filter
        qw, qx, qy, qz = filt.update(gx_rad, gy_rad, gz_rad, ax, ay, az)
        roll, pitch, yaw = filt.get_euler()
        
        # Create sample
        sample = IMUSample(
            node_id=node_id,
            node_name=NODE_NAMES.get(node_id, f"unknown_{node_id}"),
            timestamp=timestamp,
            ax=ax, ay=ay, az=az,
            gx=gx, gy=gy, gz=gz,
            qw=qw, qx=qx, qy=qy, qz=qz,
            roll=roll, pitch=pitch, yaw=yaw
        )
        
        # Store latest
        self.latest_data[node_id] = sample
        self.history[node_id].append(sample)
        
        # CSV logging is done in serial_reader() with ML-formatted data
        # This method processes IMU filter data for frontend WebSocket
        
        return sample
    
    def register_sample_callback(self, callback):
        """
        Register a callback function to be called for each IMU sample.
        Callback signature: callback(node_id: int, sample: IMUSample, raw_values: dict)
        where raw_values contains: {'ax': float, 'ay': float, 'az': float, 'gx': float, 'gy': float, 'gz': float}
        """
        if callback not in self.sample_callbacks:
            self.sample_callbacks.append(callback)
    
    def unregister_sample_callback(self, callback):
        """Unregister a callback function."""
        if callback in self.sample_callbacks:
            self.sample_callbacks.remove(callback)
    
    def get_fused_data(self) -> dict:
        """
        Get fused data from all nodes for frontend.
        """
        # Calculate average sample rate across all nodes
        total_rate = sum(self.node_rates.values())
        avg_rate = total_rate / len(self.node_rates) if self.node_rates else 0
        
        data = {
            "type": "imu_update",
            "timestamp": time.time(),
            "sample_rate": avg_rate,  # Average Hz across nodes
            "nodes": {},
            "raw_data": {}  # Raw CSV strings (exact as received from serial)
        }
        
        for node_id, sample in self.latest_data.items():
            data["nodes"][sample.node_name] = {
                "node_id": node_id,
                "timestamp": sample.timestamp,
                "sample_rate": self.node_rates.get(node_id, 0),  # Per-node rate
                # Raw IMU
                "accel": {"x": sample.ax, "y": sample.ay, "z": sample.az},
                "gyro": {"x": sample.gx, "y": sample.gy, "z": sample.gz},
                # Orientation quaternion
                "quaternion": {
                    "w": sample.qw,
                    "x": sample.qx,
                    "y": sample.qy,
                    "z": sample.qz
                },
                # Euler angles (easier for frontend)
                "euler": {
                    "roll": sample.roll,
                    "pitch": sample.pitch,
                    "yaw": sample.yaw
                }
            }
            
            # Include formatted data string (ML-friendly format) for frontend display
            if node_id in self.raw_data_strings:
                data["raw_data"][sample.node_name] = self.raw_data_strings[node_id]
        
        return data
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected WebSocket clients."""
        if not self.clients:
            return
        
        msg_json = json.dumps(message)
        
        # Remove disconnected clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(msg_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        self.clients -= disconnected
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle a WebSocket client connection."""
        self.clients.add(websocket)
        client_id = id(websocket)
        print(f"🔌 Client connected: {client_id} (Total: {len(self.clients)})")
        
        try:
            # Send initial state
            await websocket.send(json.dumps({
                "type": "init",
                "nodes": list(NODE_NAMES.values()),
                "sample_rate": SAMPLE_RATE
            }))
            
            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "reset_orientation":
                        # Reset all filters
                        for filt in self.filters.values():
                            filt.reset()
                        await websocket.send(json.dumps({
                            "type": "orientation_reset",
                            "success": True
                        }))
                    
                    elif data.get("type") == "calibrate":
                        # Could add calibration logic here
                        pass
                        
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"🔌 Client disconnected: {client_id} (Total: {len(self.clients)})")
    
    async def serial_reader(self):
        """Read from serial port and process IMU data."""
        print(f"🔌 Opening serial port: {self.port} @ {self.baud}")
        
        try:
            self.serial = serial.Serial(
                self.port, 
                self.baud, 
                timeout=0.01,  # Reduced timeout for faster reading
                write_timeout=0.01,
                inter_byte_timeout=None
            )
            time.sleep(0.5)  # Reduced wait time for Arduino reset
            self.serial.reset_input_buffer()
            print("✅ Serial port opened")
        except Exception as e:
            print(f"❌ Failed to open serial port: {e}")
            return
        
        self.start_time = time.time()
        last_broadcast = time.time()
        BROADCAST_INTERVAL = 1.0 / SAMPLE_RATE  # Match sensor rate: 1/20 = 0.05s (20 Hz per sensor)
        last_data_warning = 0
        WARNING_INTERVAL = 5.0  # Warn every 5 seconds if no data
        
        print("📡 Waiting for IMU data from serial port...")
        print("   Make sure Arduino Central Hub is connected and IMU sensors are paired.")
        
        try:
            while self.running:
                # Read line (timeout is low, so this won't block long)
                try:
                    line_bytes = self.serial.readline()
                    if not line_bytes:
                        # No data available, check if we should warn
                        now = time.time()
                        if now - last_data_warning >= WARNING_INTERVAL:
                            print(f"⏳ Still waiting for data... (check Arduino connection)")
                            last_data_warning = now
                        await asyncio.sleep(0.001)  # Very short sleep when no data
                        continue
                    
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    
                except Exception as e:
                    print(f"Serial read error: {e}")
                    await asyncio.sleep(0.01)  # Reduced sleep on error
                    continue
                
                # Parse line
                parsed = self.parse_line(line)
                if parsed:
                    node_id, sample_number, ax, ay, az, gx, gy, gz, raw_line = parsed
                    
                    # Reset warning timer when we get data
                    last_data_warning = time.time()
                    
                    # Increment internal sample counter for this node (for rate tracking)
                    self.node_sample_counts[node_id] += 1
                    
                    # Use parsed sample_number if available, otherwise use internal counter
                    if sample_number is None:
                        sample_number = self.node_sample_counts[node_id]
                    
                    # Process sample for IMU filter (MUST process every sample for 20 Hz rate)
                    # This uses the parsed float values for Madgwick filter calculations
                    sample = self.process_sample(node_id, ax, ay, az, gx, gy, gz)
                    self.sample_count += 1
                    
                    # Store formatted string for frontend display (ML-friendly format)
                    node_name = NODE_NAMES.get(node_id, f"node_{node_id}")
                    formatted_line = f"{node_id},{sample_number},{ax:.4f},{ay:.4f},{az:.4f},{gx:.2f},{gy:.2f},{gz:.2f}"
                    self.raw_data_strings[node_id] = formatted_line
                    
                    # Terminal output: ML-friendly formatted data (every sample, no skipping)
                    # Format: node_id,sample_number,ax,ay,az,gx,gy,gz
                    print(formatted_line)
                    
                    # CSV logging: ML-friendly formatted data with all computed values
                    # Includes: timestamp, node info, raw IMU, quaternion, euler angles
                    if self.csv_writer:
                        self.csv_writer.writerow([
                            sample.timestamp, node_id, node_name,
                            ax, ay, az, gx, gy, gz,
                            sample.qw, sample.qx, sample.qy, sample.qz,
                            sample.roll, sample.pitch, sample.yaw
                        ])
                        # Flush every 20 samples to reduce I/O overhead (but still process every sample)
                        if self.sample_count % 20 == 0:
                            self.csv_file.flush()
                    
                    # Call registered callbacks (e.g., for api_server.py training data collection)
                    # Pass raw IMU values along with processed sample
                    raw_values = {'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz}
                    for callback in self.sample_callbacks:
                        try:
                            callback(node_id, sample, raw_values)
                        except Exception as e:
                            print(f"⚠️  Error in IMU sample callback: {e}")
                    
                    # Track per-node rate
                    now = time.time()
                    if node_id in self.node_last_time:
                        dt = now - self.node_last_time[node_id]
                        if dt > 0:
                            # Exponential moving average for smooth rate
                            instant_rate = 1.0 / dt
                            self.node_rates[node_id] = 0.9 * self.node_rates[node_id] + 0.1 * instant_rate
                    self.node_last_time[node_id] = now
                
                # Broadcast to WebSocket clients (throttled)
                now = time.time()
                if now - last_broadcast >= BROADCAST_INTERVAL:
                    fused = self.get_fused_data()
                    await self.broadcast(fused)
                    last_broadcast = now
                
                # No sleep here - process as fast as possible when data is available
                
        finally:
            if self.serial:
                self.serial.close()
                print("🔌 Serial port closed")
    
    async def run(self):
        """Run the IMU bridge."""
        self.running = True
        self.start_logging()
        
        print(f"🚀 Starting GymBud IMU Bridge")
        print(f"   Serial: {self.port} @ {self.baud}")
        print(f"   WebSocket: ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
        print(f"   Nodes: {NODE_NAMES}")
        print()
        
        # Start WebSocket server
        server = await serve(
            self.handle_client,
            WEBSOCKET_HOST,
            WEBSOCKET_PORT
        )
        print(f"✅ WebSocket server started on ws://localhost:{WEBSOCKET_PORT}")
        
        try:
            # Run serial reader
            await self.serial_reader()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
        finally:
            self.running = False
            self.stop_logging()
            server.close()
            await server.wait_closed()
            print("👋 Goodbye!")


# ==================== GLOBAL INSTANCE ====================
# Global instance for external access (e.g., from api_server.py)
_global_bridge_instance: Optional['IMUBridge'] = None

def get_global_bridge() -> Optional['IMUBridge']:
    """Get the global IMUBridge instance if it exists."""
    return _global_bridge_instance

# ==================== MAIN ====================

async def main():
    global _global_bridge_instance
    bridge = IMUBridge(SERIAL_PORT, BAUD_RATE)
    _global_bridge_instance = bridge
    await bridge.run()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║          GymBud IMU-WebSocket Bridge              ║
    ║                                                   ║
    ║  Node 1: Left Wrist (LW)                         ║
    ║  Node 2: Right Wrist (RW)                        ║
    ║  Node 3: Chest (CH)                              ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())

