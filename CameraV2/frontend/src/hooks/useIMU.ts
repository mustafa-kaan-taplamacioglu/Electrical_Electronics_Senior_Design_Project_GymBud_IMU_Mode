/**
 * useIMU Hook
 * ============
 * WebSocket hook for receiving real-time IMU data from GymBud sensors.
 * 
 * Nodes:
 * - 1 = Left Wrist (LW)
 * - 2 = Right Wrist (RW)
 * - 3 = Chest (CH)
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import type { IMUFusedData, IMUNodeData } from '../types';

const IMU_WEBSOCKET_URL = 'ws://localhost:8765';

interface UseIMUOptions {
  autoConnect?: boolean;
  reconnectInterval?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

interface UseIMUResult {
  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  
  // Data
  data: IMUFusedData | null;
  leftWrist: IMUNodeData | null;
  rightWrist: IMUNodeData | null;
  chest: IMUNodeData | null;
  
  // Stats
  sampleRate: number;
  latency: number;
  
  // Actions
  connect: () => void;
  disconnect: () => void;
  resetOrientation: () => void;
}

export function useIMU(options: UseIMUOptions = {}): UseIMUResult {
  const {
    autoConnect = false,
    reconnectInterval = 3000,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<IMUFusedData | null>(null);
  const [sampleRate, setSampleRate] = useState(0);
  const [latency, setLatency] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sampleCountRef = useRef(0);
  const lastRateCheckRef = useRef(Date.now());

  // Calculate sample rate
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      const elapsed = (now - lastRateCheckRef.current) / 1000;
      if (elapsed > 0) {
        setSampleRate(sampleCountRef.current / elapsed);
        sampleCountRef.current = 0;
        lastRateCheckRef.current = now;
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setIsConnecting(true);
    setError(null);

    const ws = new WebSocket(IMU_WEBSOCKET_URL);

    ws.onopen = () => {
      console.log('🎯 IMU WebSocket connected');
      setIsConnected(true);
      setIsConnecting(false);
      setError(null);
      onConnect?.();
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === 'imu_update') {
          const imuData = message as IMUFusedData;
          setData(imuData);
          sampleCountRef.current++;

          // Calculate latency
          const now = Date.now() / 1000;
          if (imuData.timestamp) {
            setLatency((now - imuData.timestamp) * 1000);
          }
        } else if (message.type === 'init') {
          console.log('🎯 IMU initialized:', message.nodes);
        } else if (message.type === 'orientation_reset') {
          console.log('🎯 Orientation reset:', message.success);
        }
      } catch (e) {
        console.error('Failed to parse IMU message:', e);
      }
    };

    ws.onerror = (event) => {
      console.error('🎯 IMU WebSocket error:', event);
      setError('Connection error');
      onError?.(event);
    };

    ws.onclose = () => {
      console.log('🎯 IMU WebSocket disconnected');
      setIsConnected(false);
      setIsConnecting(false);
      onDisconnect?.();

      // Auto-reconnect
      if (reconnectInterval > 0) {
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('🎯 Attempting IMU reconnection...');
          connect();
        }, reconnectInterval);
      }
    };

    wsRef.current = ws;
  }, [onConnect, onDisconnect, onError, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  const resetOrientation = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset_orientation' }));
    }
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Extract node data (chest is optional, defaults to null if not present)
  const leftWrist = data?.nodes?.left_wrist || null;
  const rightWrist = data?.nodes?.right_wrist || null;
  const chest = data?.nodes?.chest || null;

  return {
    isConnected,
    isConnecting,
    error,
    data,
    leftWrist,
    rightWrist,
    chest,
    sampleRate,
    latency,
    connect,
    disconnect,
    resetOrientation,
  };
}

export default useIMU;

