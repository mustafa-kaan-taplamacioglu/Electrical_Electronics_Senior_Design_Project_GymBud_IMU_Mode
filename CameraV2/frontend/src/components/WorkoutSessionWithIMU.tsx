/**
 * WorkoutSessionWithIMU
 * ======================
 * Enhanced workout session component that integrates both:
 * - MediaPipe camera-based pose detection
 * - BLE IMU sensor data (Left Wrist, Right Wrist, Chest)
 * 
 * This component demonstrates sensor fusion for more accurate avatar animation.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Pose, Results } from '@mediapipe/pose';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import type { ExerciseType, RepData, AIFeedback, SensorFusionMode, IMUNodeData, MLMode } from '../types';
import { EXERCISES } from '../config/exercises';
import { HumanAvatar } from './HumanAvatar';
import { useIMU } from '../hooks/useIMU';

interface Landmark3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface CameraDevice {
  deviceId: string;
  label: string;
}

interface Props {
  exercise: ExerciseType;
  apiKey: string;
  avatarUrl?: string;
  mlMode: MLMode;
  onEnd: () => void;
}

type SessionState = 'camera_select' | 'connecting' | 'detecting' | 'calibrating' | 'ready' | 'countdown' | 'tracking' | 'resting' | 'finished';

export const WorkoutSessionWithIMU = ({ exercise, apiKey, avatarUrl, mlMode, onEnd }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<Pose | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [state, setState] = useState<SessionState>('camera_select');
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [cameraStarted, setCameraStarted] = useState(false);
  const [visibilityMessage, setVisibilityMessage] = useState<string>('');
  const [debugLogs, setDebugLogs] = useState<string[]>([]);
  
  // Sensor fusion mode
  const [fusionMode, setFusionMode] = useState<SensorFusionMode>('camera_primary');
  const [showIMUDebug, setShowIMUDebug] = useState(true);
  
  // Dataset collection
  const [datasetCollectionEnabled, setDatasetCollectionEnabled] = useState(false);
  const [datasetCollectionStatus, setDatasetCollectionStatus] = useState<'idle' | 'collecting' | 'saved'>('idle');
  const [collectedRepsCount, setCollectedRepsCount] = useState(0);
  
  // Training dialog (only for ML Training Mode)
  const [showTrainingDialog, setShowTrainingDialog] = useState(false);
  const [trainingAction, setTrainingAction] = useState<'none' | 'save_only' | 'save_and_train' | 'skip'>('none');
  const [trainingStatus, setTrainingStatus] = useState<'idle' | 'training' | 'completed' | 'error'>('idle');
  const [trainingMessage, setTrainingMessage] = useState<string>('');
  const [performanceMetrics, setPerformanceMetrics] = useState<any>(null);
  const [trainingSampleCount, setTrainingSampleCount] = useState<number>(0);
  
  // Raw IMU data stream (last 15 samples for display)
  const [rawDataStream, setRawDataStream] = useState<string[]>([]);
  const rawDataRef = useRef<string[]>([]);
  
  // Responsive avatar size for IMU-only mode
  const avatarWrapperRef = useRef<HTMLDivElement>(null);
  const [avatarSize, setAvatarSize] = useState({ width: 400, height: 480 });
  
  // Update avatar size on window resize (for IMU-only mode)
  useEffect(() => {
    const updateAvatarSize = () => {
      if (fusionMode === 'imu_only' && avatarWrapperRef.current) {
        const containerWidth = avatarWrapperRef.current.clientWidth;
        const width = Math.min(containerWidth || window.innerWidth * 0.9, 600);
        const height = width * 0.75; // 4:3 aspect ratio
        setAvatarSize({ width, height });
      } else {
        setAvatarSize({ width: 400, height: 480 });
      }
    };
    
    updateAvatarSize();
    window.addEventListener('resize', updateAvatarSize);
    return () => window.removeEventListener('resize', updateAvatarSize);
  }, [fusionMode]);
  
  // IMU hook
  const {
    isConnected: imuConnected,
    isConnecting: imuConnecting,
    data: imuRawData,
    leftWrist,
    rightWrist,
    chest,
    sampleRate: imuSampleRate,
    connect: connectIMU,
    disconnect: disconnectIMU,
    resetOrientation
  } = useIMU({
    autoConnect: false,
    reconnectInterval: 3000
  });
  
  // Convert IMU data to the format HumanAvatar expects (chest is optional)
  const imuData = imuRawData ? {
    leftWrist: leftWrist || undefined,
    rightWrist: rightWrist || undefined,
    chest: chest || undefined
  } : null;
  
  // Update raw data stream when IMU data changes - use EXACT raw data from backend
  useEffect(() => {
    if (!imuRawData?.raw_data) return;
    
    const newLines: string[] = [];
    const rawData = imuRawData.raw_data;
    
    // Use EXACT raw data strings from backend (no formatting, no changes)
    if (rawData.left_wrist) {
      newLines.push(rawData.left_wrist);
    }
    if (rawData.right_wrist) {
      newLines.push(rawData.right_wrist);
    }
    if (rawData.chest) {
      newLines.push(rawData.chest);
    }
    
    if (newLines.length > 0) {
      rawDataRef.current = [...rawDataRef.current, ...newLines].slice(-15);
      setRawDataStream([...rawDataRef.current]);
    }
  }, [imuRawData?.raw_data]);
  
  // Debug logger
  const addLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] ${msg}`);
    setDebugLogs(prev => [...prev.slice(-20), `[${timestamp}] ${msg}`]);
  };
  
  const [repCount, setRepCount] = useState(0);
  const [formScore, setFormScore] = useState(100);
  const [issues, setIssues] = useState<string[]>([]);
  const [currentAngle, setCurrentAngle] = useState(0);
  const [phase, setPhase] = useState<'up' | 'down'>('down');
  const [feedbacks, setFeedbacks] = useState<AIFeedback[]>([]);
  const [regionalFeedbacks, setRegionalFeedbacks] = useState<{
    arms: string;
    legs: string;
    core: string;
    head: string;
  } | null>(null);
  const [reps, setReps] = useState<RepData[]>([]);
  const [sessionFeedback, setSessionFeedback] = useState<string>('');
  const [avgFormScore, setAvgFormScore] = useState<number>(0);
  const [regionalScores, setRegionalScores] = useState<{
    arms: number;
    legs: number;
    core: number;
    head: number;
  } | null>(null);
  const [currentLandmarks, setCurrentLandmarks] = useState<Landmark3D[] | null>(null);
  const [startTime] = useState(new Date());
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [countdownNumber, setCountdownNumber] = useState<number | null>(null);
  
  // Workout configuration (sets, reps, rest)
  const [numberOfSets, setNumberOfSets] = useState<number>(3);
  const [repsPerSet, setRepsPerSet] = useState<number>(10);
  const [restTimeSeconds, setRestTimeSeconds] = useState<number>(60);
  
  // Workout progress tracking
  const [currentSet, setCurrentSet] = useState<number>(1);
  const [currentRepInSet, setCurrentRepInSet] = useState<number>(0);
  const [restCountdown, setRestCountdown] = useState<number>(0);

  const exerciseConfig = EXERCISES[exercise];

  // Get available cameras on mount
  useEffect(() => {
    const getCameras = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices
          .filter(device => device.kind === 'videoinput')
          .map((device, index) => ({
            deviceId: device.deviceId,
            label: device.label || `Camera ${index + 1}`,
          }));
        
        setCameras(videoDevices);
        if (videoDevices.length > 0) {
          setSelectedCamera(videoDevices[0].deviceId);
        }
      } catch (error) {
        console.error('Error getting cameras:', error);
      }
    };
    
    getCameras();
  }, []);

  // Start camera with selected device (or skip camera for IMU-only mode)
  const startCamera = async () => {
    if (cameraStarted) return;
    
    // IMU-only mode: Skip camera, connect to IMU and backend directly
    if (fusionMode === 'imu_only') {
      addLog('IMU-only mode: Skipping camera, connecting to backend...');
      setState('connecting');
      setCameraStarted(true);  // Set to true to trigger WebSocket connection
      
      // Connect to IMU if not connected
      if (!imuConnected && !imuConnecting) {
        addLog('Connecting to IMU sensors...');
        connectIMU();
      }
      return;  // Skip camera initialization
    }
    
    // Camera modes: Require camera selection
    if (!selectedCamera) return;
    
    addLog(`Starting camera...`);
    setState('connecting');
    setCameraStarted(true);
    
    // Also connect to IMU if not connected
    if (!imuConnected && !imuConnecting) {
      addLog('Connecting to IMU sensors...');
      connectIMU();
    }
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: selectedCamera },
          width: 640,
          height: 480,
        },
      });
      
      if (!videoRef.current) return;
      
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      
      const pose = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      pose.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      pose.onResults(onResults);
      poseRef.current = pose;
      
      const detectPose = async () => {
        if (poseRef.current && videoRef.current) {
          try {
            await poseRef.current.send({ image: videoRef.current });
          } catch (e) {
            console.error('Pose detection error:', e);
          }
          requestAnimationFrame(detectPose);
        }
      };
      
      await pose.initialize();
      addLog('Pose model loaded!');
      detectPose();
      
    } catch (error) {
      addLog(`ERROR: ${error}`);
    }
  };

  // Connect to backend WebSocket
  useEffect(() => {
    if (!cameraStarted) return;
    
    // Prevent multiple connections
    if (wsRef.current && (wsRef.current.readyState === WebSocket.CONNECTING || wsRef.current.readyState === WebSocket.OPEN)) {
      console.log('⚠️ WebSocket already connecting/connected, skipping...');
      return;
    }
    
    // Close existing connection if any
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (e) {
        console.error('Error closing existing WebSocket:', e);
      }
      wsRef.current = null;
    }
    
    console.log('🔌 Opening new WebSocket connection...');
    const ws = new WebSocket(`ws://localhost:8000/ws/${exercise}`);
    
    ws.onopen = () => {
      console.log('✅ Backend WebSocket connected');
      addLog('Backend WebSocket connected');
      const initMessage: any = { 
        type: 'init', 
        api_key: apiKey,
        ml_mode: mlMode,
        fusion_mode: fusionMode,  // Send user-selected fusion mode
        workout_config: {
          numberOfSets: numberOfSets,
          repsPerSet: repsPerSet,
          restTimeSeconds: restTimeSeconds
        }
      };
      
      try {
        ws.send(JSON.stringify(initMessage));
        
        // Dataset collection is automatic in both usage and train modes
        if (mlMode === 'usage' || mlMode === 'train') {
          ws.send(JSON.stringify({ type: 'start_collection' }));
          setDatasetCollectionStatus('collecting');
          setDatasetCollectionEnabled(true); // Set state for UI display
        }
      } catch (e) {
        console.error('Error sending init message:', e);
      }
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type !== 'update' && data.type !== 'calibration_progress') {
        addLog(`WS: ${data.type} ${data.state || ''}`);
      }
      
      if (data.type === 'ready') {
        setState('detecting');
      } else if (data.type === 'countdown') {
        setState('countdown');
        if (data.number !== undefined) {
          setCountdownNumber(data.number);
        }
        // Clear countdown after START is shown
        if (data.number === 0) {
          setTimeout(() => {
            setCountdownNumber(null);
          }, 500);
        }
      } else if (data.type === 'state') {
        setState(data.state);
        if (data.message) {
          setVisibilityMessage(data.message);
        }
      } else if (data.type === 'visibility') {
        setVisibilityMessage(data.message + (data.missing ? ` (Missing: ${data.missing})` : ''));
      } else if (data.type === 'calibration_progress') {
        setCalibrationProgress(data.progress);
        setVisibilityMessage('');
      } else if (data.type === 'update') {
        setCurrentAngle(data.angle);
        setPhase(data.phase);
        setRepCount(data.rep_count);
        setFormScore(data.form_score);
        setIssues(data.issues || []);
        
        // Update set/rep tracking
        if (data.current_set !== undefined) {
          setCurrentSet(data.current_set);
        }
        if (data.current_rep_in_set !== undefined) {
          setCurrentRepInSet(data.current_rep_in_set);
        }
        
        if (data.feedback) {
          setFeedbacks(prev => [...prev, {
            message: data.feedback,
            type: data.form_score >= 80 ? 'success' : data.form_score >= 60 ? 'warning' : 'error',
            timestamp: new Date(),
          }]);
        }
        
        // Handle regional feedback
        if (data.regional_feedback) {
          setRegionalFeedbacks(data.regional_feedback);
        }
        
        if (data.rep_completed) {
          const repInfo = typeof data.rep_completed === 'object' ? data.rep_completed : {};
          setReps(prev => [...prev, {
            repNumber: repInfo.rep || data.rep_count,
            formScore: repInfo.form_score || data.form_score || 0,
            issues: data.issues || [],
            timestamp: new Date(),
          }]);
          
          // Update collected reps count if data recording is active (both usage and train modes)
          if ((mlMode === 'usage' || mlMode === 'train') && repInfo) {
            setCollectedRepsCount(prev => prev + 1);
          }
        }
      } else if (data.type === 'rest_countdown') {
        setState('resting');
        if (data.remaining !== undefined) {
          setRestCountdown(data.remaining);
        }
      } else if (data.type === 'rep_feedback') {
        // Async AI feedback received (non-blocking)
        if (data.feedback) {
          setFeedbacks(prev => [...prev, {
            message: data.feedback,
            type: 'success',
            timestamp: new Date(),
          }]);
        }
        if (data.regional_feedback) {
          setRegionalFeedbacks(data.regional_feedback);
        }
      } else if (data.type === 'dataset_collection_status') {
        setDatasetCollectionStatus(data.status || 'idle');
        if (data.collected_reps !== undefined) {
          setCollectedRepsCount(data.collected_reps);
        }
      } else if (data.type === 'ml_training_status') {
        // ML training status updates
        if (data.status === 'started') {
          console.log('🤖 ML training started:', data.message);
        } else if (data.status === 'completed') {
          console.log('✅ ML training completed:', data.message);
          // Show success notification
          alert(`✅ ML Model Training Completed!\n\n${data.message}`);
        } else if (data.status === 'error') {
          console.error('❌ ML training error:', data.error);
          // Show error notification
          alert(`⚠️ ML Training Error\n\n${data.error}`);
        }
      } else if (data.type === 'session_summary') {
        console.log('📥 Received session_summary:', data);
        console.log('   Feedback:', data.feedback);
        console.log('   Regional scores:', data.regional_scores);
        console.log('   Regional feedback:', data.regional_feedback);
        console.log('   Rep list:', data.rep_list);
        
        // Set feedback immediately
        if (data.feedback) {
          setSessionFeedback(data.feedback);
          console.log('✅ Session feedback set:', data.feedback.substring(0, 50) + '...');
        } else {
          console.warn('⚠️ No feedback in session_summary');
          setSessionFeedback('Harika antrenman! Devam et! 💪');
        }
        
        setRepCount(data.total_reps);
        setAvgFormScore(data.avg_form || 0);
        setRegionalScores(data.regional_scores || null);
        
        // Set rep list if provided
        if (data.rep_list && Array.isArray(data.rep_list)) {
          // Convert rep_list to RepData format
          const repDataList: RepData[] = data.rep_list.map((rep: any) => ({
            repNumber: rep.rep_number || 0,
            formScore: rep.form_score || 0,
            duration: rep.duration || 0,
            speedClass: rep.speed_class || 'medium',
            speedLabel: rep.speed_label || 'Orta Hız',
            speedEmoji: rep.speed_emoji || '✅',
            isValid: rep.is_valid !== false,
            issues: rep.issues || [],
            regionalScores: rep.regional_scores || {}
          }));
          setReps(repDataList);
          console.log('✅ Rep list set:', repDataList.length, 'reps');
        }
        
        // Set regional feedback if provided
        if (data.regional_feedback) {
          setRegionalFeedbacks(data.regional_feedback);
          console.log('✅ Regional feedback set:', data.regional_feedback);
        }
        
        // Show data collection dialog for both usage and train modes
        // Don't close WebSocket yet - we need it for training_action
        setShowTrainingDialog(true);
        console.log('📊 Showing training dialog...');
      } else if (data.type === 'training_status') {
        setTrainingStatus(data.status);
        setTrainingMessage(data.message || '');
        console.log(`📊 Training status: ${data.status}, message: ${data.message}`);
        
        // Store performance metrics and sample count if available
        if (data.performance_metrics) {
          setPerformanceMetrics(data.performance_metrics);
          console.log('📈 Performance metrics:', data.performance_metrics);
        }
        if (data.sample_count) {
          setTrainingSampleCount(data.sample_count);
        }
        
        if (data.status === 'completed' || data.status === 'error') {
          // Keep dialog open to show result, user can close it
          // Don't auto-close - let user see the result and click to continue
        }
      }
    };
    
    ws.onerror = (error) => {
      addLog(`WS ERROR: ${error}`);
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = (event) => {
      addLog('WebSocket closed');
      console.log('WebSocket closed:', event.code, event.reason);
      // Don't try to reconnect automatically - let user restart if needed
    };
    
    wsRef.current = ws;
    
    return () => {
      // Cleanup: Close WebSocket connection
      if (wsRef.current) {
        console.log('🧹 Cleaning up WebSocket connection...');
        try {
          // Stop dataset collection on cleanup
          if ((mlMode === 'usage' || mlMode === 'train') && wsRef.current.readyState === WebSocket.OPEN) {
            try {
              wsRef.current.send(JSON.stringify({ type: 'stop_collection' }));
            } catch (e) {
              console.error('Error sending stop_collection:', e);
            }
          }
          if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
            wsRef.current.close();
          }
        } catch (e) {
          console.error('Error closing WebSocket:', e);
        }
        wsRef.current = null;
      }
    };
  }, [exercise, apiKey, cameraStarted, mlMode, fusionMode, numberOfSets, repsPerSet, restTimeSeconds]);
  
  // Toggle dataset collection
  const toggleDatasetCollection = () => {
    // Allow toggling even if WebSocket not yet open (will be sent when open)
    const newState = !datasetCollectionEnabled;
    setDatasetCollectionEnabled(newState);
    
    // If WebSocket is open, send message immediately
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      if (newState) {
        wsRef.current.send(JSON.stringify({ type: 'start_collection' }));
        setDatasetCollectionStatus('collecting');
        setCollectedRepsCount(0);
        addLog('Dataset collection started');
      } else {
        wsRef.current.send(JSON.stringify({ type: 'stop_collection' }));
        setDatasetCollectionStatus('idle');
        addLog('Dataset collection stopped');
      }
    } else {
      // Will be sent when WebSocket opens (handled in useEffect)
      addLog(newState ? 'Dataset collection will start when connected' : 'Dataset collection will stop when connected');
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
      poseRef.current?.close();
      disconnectIMU();
    };
  }, [disconnectIMU]);

  const onResults = useCallback((results: Results) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.poseLandmarks) {
      drawConnectors(ctx, results.poseLandmarks, Pose.POSE_CONNECTIONS, {
        color: '#00FF00',
        lineWidth: 2,
      });
      drawLandmarks(ctx, results.poseLandmarks, {
        color: '#FF0000',
        lineWidth: 1,
        radius: 3,
      });
      
      // Draw landmark labels (numbers 0-32) - ALL landmarks, regardless of visibility
      ctx.fillStyle = '#FFFFFF';
      ctx.strokeStyle = '#000000';
      ctx.font = 'bold 10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      results.poseLandmarks.forEach((landmark, index) => {
        const x = landmark.x * canvas.width;
        const y = landmark.y * canvas.height;
        // Draw text with outline for better visibility
        ctx.lineWidth = 2;
        ctx.strokeText(index.toString(), x, y - 8);
        ctx.fillText(index.toString(), x, y - 8);
      });

      const landmarks = results.poseLandmarks.map((l) => ({
        x: l.x,
        y: l.y,
        z: l.z,
        visibility: l.visibility,
      }));
      
      setCurrentLandmarks(landmarks);
      
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Send pose data with IMU data if available
        const poseData: any = { type: 'pose', landmarks };
        if (imuData && (leftWrist || rightWrist || chest)) {
          // Convert nested IMU format (from gymbud_imu_bridge) to flat format (expected by backend)
          const convertNodeData = (nodeData: any) => {
            if (!nodeData) return null;
            return {
              ax: nodeData.accel?.x ?? null,
              ay: nodeData.accel?.y ?? null,
              az: nodeData.accel?.z ?? null,
              gx: nodeData.gyro?.x ?? null,
              gy: nodeData.gyro?.y ?? null,
              gz: nodeData.gyro?.z ?? null,
              qw: nodeData.quaternion?.w ?? null,
              qx: nodeData.quaternion?.x ?? null,
              qy: nodeData.quaternion?.y ?? null,
              qz: nodeData.quaternion?.z ?? null,
              roll: nodeData.euler?.roll ?? null,
              pitch: nodeData.euler?.pitch ?? null,
              yaw: nodeData.euler?.yaw ?? null,
            };
          };
          
          poseData.imu_data = {
            left_wrist: convertNodeData(leftWrist),
            right_wrist: convertNodeData(rightWrist),
            chest: convertNodeData(chest),
            timestamp: Date.now() / 1000.0  // Convert to seconds (Unix timestamp)
          };
          // Also include 'imu' for backward compatibility
          poseData.imu = poseData.imu_data;
        }
        wsRef.current.send(JSON.stringify(poseData));
      }
    }

    ctx.restore();
  }, [imuData, leftWrist, rightWrist, chest]);

  const handleEnd = async () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_session' }));
    }
    
    disconnectIMU();
    setState('finished');
  };

  const getStatusText = () => {
    switch (state) {
      case 'camera_select':
        return '📷 Select Camera';
      case 'connecting':
        return '🔌 Connecting...';
      case 'detecting':
        return visibilityMessage || '🔍 Detecting body...';
      case 'calibrating':
        return `⏳ Calibrating: ${(calibrationProgress * 100).toFixed(0)}%`;
      case 'ready':
        return '✅ Ready!';
      case 'countdown':
        return countdownNumber === 0 ? 'START!' : countdownNumber?.toString() || '';
      case 'tracking':
        return `🏋️ ${exerciseConfig.displayName}`;
      case 'resting':
        return `⏸️ Rest: ${restCountdown}s`;
      case 'finished':
        return '🎉 Workout Complete!';
    }
  };

  const getFormColor = () => {
    if (formScore >= 80) return '#22c55e';
    if (formScore >= 60) return '#eab308';
    return '#ef4444';
  };

  // Camera selection screen
  if (state === 'camera_select') {
    return (
      <div className="camera-select-screen">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="camera-select-card"
        >
          <h2>📷 Select Camera & Sensors</h2>
          <p className="exercise-info">
            Selected exercise: <strong>{exerciseConfig.displayName}</strong>
          </p>
          
          {/* Camera selection */}
          <div className="camera-list">
            <h4>Camera</h4>
            {cameras.length === 0 ? (
              <p className="no-camera">No camera found...</p>
            ) : (
              cameras.map((camera, index) => (
                <label
                  key={camera.deviceId}
                  className={`camera-option ${selectedCamera === camera.deviceId ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="camera"
                    value={camera.deviceId}
                    checked={selectedCamera === camera.deviceId}
                    onChange={(e) => setSelectedCamera(e.target.value)}
                  />
                  <span className="camera-icon">📹</span>
                  <span className="camera-name">{camera.label}</span>
                  <span className="camera-index">#{index + 1}</span>
                </label>
              ))
            )}
          </div>
          
          {/* IMU Status */}
          <div className="imu-status" style={{ marginTop: '20px' }}>
            <h4>🎛️ IMU Sensors</h4>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '10px',
              padding: '10px',
              background: imuConnected ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)',
              borderRadius: '8px'
            }}>
              <span style={{ 
                width: '12px', 
                height: '12px', 
                borderRadius: '50%', 
                background: imuConnected ? '#22c55e' : '#ef4444'
              }}></span>
              <span>{imuConnected ? 'Connected' : imuConnecting ? 'Connecting...' : 'Disconnected'}</span>
              {!imuConnected && (
                <button 
                  onClick={connectIMU}
                  style={{ 
                    marginLeft: 'auto', 
                    padding: '5px 10px',
                    background: '#3b82f6',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Connect
                </button>
              )}
            </div>
            <p style={{ fontSize: '12px', color: '#888', marginTop: '5px' }}>
              Run gymbud_imu_bridge.py to enable IMU sensors
            </p>
          </div>
          
          {/* Check IMU Activity - Real-time Data Panel */}
          {imuConnected && (
            <div className="imu-activity" style={{ marginTop: '20px' }}>
              <h4>📊 Check IMU Activity</h4>
              <div style={{
                background: '#0a0a0f',
                border: '1px solid #22c55e',
                borderRadius: '12px',
                padding: '15px',
                fontFamily: 'monospace',
                fontSize: '12px'
              }}>
                {/* Header */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 1fr 1fr 1fr',
                  gap: '10px',
                  marginBottom: '10px',
                  paddingBottom: '10px',
                  borderBottom: '1px solid #333',
                  color: '#888',
                  fontWeight: 'bold'
                }}>
                  <span>Node</span>
                  <span>Roll (°)</span>
                  <span>Pitch (°)</span>
                  <span>Yaw (°)</span>
                </div>
                
                {/* Left Wrist */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 1fr 1fr 1fr',
                  gap: '10px',
                  marginBottom: '8px',
                  alignItems: 'center'
                }}>
                  <span style={{ color: '#3b82f6' }}>🤚 LW</span>
                  <span style={{ color: leftWrist ? '#22c55e' : '#666' }}>
                    {leftWrist ? leftWrist.euler.roll.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: leftWrist ? '#22c55e' : '#666' }}>
                    {leftWrist ? leftWrist.euler.pitch.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: leftWrist ? '#22c55e' : '#666' }}>
                    {leftWrist ? leftWrist.euler.yaw.toFixed(1) : '---'}
                  </span>
                </div>
                
                {/* Right Wrist */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 1fr 1fr 1fr',
                  gap: '10px',
                  marginBottom: '8px',
                  alignItems: 'center'
                }}>
                  <span style={{ color: '#ef4444' }}>✋ RW</span>
                  <span style={{ color: rightWrist ? '#22c55e' : '#666' }}>
                    {rightWrist ? rightWrist.euler.roll.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: rightWrist ? '#22c55e' : '#666' }}>
                    {rightWrist ? rightWrist.euler.pitch.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: rightWrist ? '#22c55e' : '#666' }}>
                    {rightWrist ? rightWrist.euler.yaw.toFixed(1) : '---'}
                  </span>
                </div>
                
                {/* Chest */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 1fr 1fr 1fr',
                  gap: '10px',
                  marginBottom: '8px',
                  alignItems: 'center'
                }}>
                  <span style={{ color: '#f59e0b' }}>👕 CH</span>
                  <span style={{ color: chest ? '#22c55e' : '#666' }}>
                    {chest ? chest.euler.roll.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: chest ? '#22c55e' : '#666' }}>
                    {chest ? chest.euler.pitch.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: chest ? '#22c55e' : '#666' }}>
                    {chest ? chest.euler.yaw.toFixed(1) : '---'}
                  </span>
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 1fr 1fr 1fr',
                  gap: '10px',
                  marginBottom: '8px',
                  alignItems: 'center'
                }}>
                  <span style={{ color: '#a855f7' }}>✋ RW</span>
                  <span style={{ color: rightWrist ? '#22c55e' : '#666' }}>
                    {rightWrist ? rightWrist.euler.roll.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: rightWrist ? '#22c55e' : '#666' }}>
                    {rightWrist ? rightWrist.euler.pitch.toFixed(1) : '---'}
                  </span>
                  <span style={{ color: rightWrist ? '#22c55e' : '#666' }}>
                    {rightWrist ? rightWrist.euler.yaw.toFixed(1) : '---'}
                  </span>
                </div>
                
                {/* Sample Rate */}
                <div style={{
                  marginTop: '10px',
                  paddingTop: '10px',
                  borderTop: '1px solid #333',
                  display: 'flex',
                  justifyContent: 'space-between',
                  color: '#888'
                }}>
                  <span>Sample Rate: <span style={{ color: '#22c55e' }}>{imuSampleRate.toFixed(0)} Hz</span></span>
                  <button
                    onClick={resetOrientation}
                    style={{
                      padding: '4px 8px',
                      background: '#6b21a8',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '11px'
                    }}
                  >
                    🔄 Reset Zero
                  </button>
                </div>
                
                {/* Raw Data Stream */}
                <div style={{ marginTop: '15px' }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '8px'
                  }}>
                    <span style={{ color: '#888', fontSize: '11px' }}>📡 Raw Data Stream (node,ts,ax,ay,az,gx,gy,gz)</span>
                  </div>
                  <div style={{
                    background: '#000',
                    border: '1px solid #333',
                    borderRadius: '6px',
                    padding: '8px',
                    maxHeight: '180px',
                    overflowY: 'auto',
                    fontFamily: 'Monaco, Consolas, monospace',
                    fontSize: '10px',
                    lineHeight: '1.4'
                  }}>
                    {rawDataStream.length === 0 ? (
                      <div style={{ color: '#666', textAlign: 'center' }}>Waiting for data...</div>
                    ) : (
                      rawDataStream.map((line, idx) => {
                        const nodeId = line.charAt(0);
                        let color = '#888';
                        if (nodeId === '1') color = '#3b82f6'; // LW - blue
                        if (nodeId === '2') color = '#a855f7'; // RW - purple
                        if (nodeId === '3') color = '#f59e0b'; // CH - orange
                        return (
                          <div key={idx} style={{ color, opacity: 0.6 + (idx / rawDataStream.length) * 0.4 }}>
                            {line}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Fusion Mode */}
          <div className="fusion-mode" style={{ marginTop: '20px' }}>
            <h4>🔄 Sensor Fusion Mode</h4>
            <select 
              value={fusionMode}
              onChange={(e) => setFusionMode(e.target.value as SensorFusionMode)}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: '8px',
                border: '1px solid #333',
                background: '#1a1a1a',
                color: 'white'
              }}
            >
              <option value="camera_only">Camera Only (No IMU)</option>
              <option value="camera_primary">Camera + IMU Enhancement</option>
              <option value="imu_only">IMU Only (No Camera)</option>
            </select>
          </div>
          
          {/* Dataset Collection Info (automatic in both modes) */}
          {(mlMode === 'usage' || mlMode === 'train') ? (
          <div className="dataset-collection" style={{ marginTop: '20px', padding: '15px', background: '#1a1a1a', borderRadius: '8px', border: '1px solid #333' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h4 style={{ margin: 0 }}>
                {mlMode === 'usage' ? '📊 Data Recording' : '📚 ML Training Dataset Collection'}
              </h4>
              <span style={{ color: '#22c55e', fontSize: '14px', fontWeight: 'bold' }}>✅ Enabled</span>
            </div>
            {mlMode === 'train' ? (
              <div>
                <p style={{ fontSize: '12px', color: '#888', margin: '5px 0' }}>
                  <span style={{ color: '#22c55e' }}>
                    Synchronized collection enabled for ML training:
                  </span>
                </p>
                <div style={{ marginTop: '8px', padding: '8px', background: '#0a0a0a', borderRadius: '4px' }}>
                  <div style={{ fontSize: '11px', color: '#3b82f6', marginBottom: '4px' }}>
                    📹 Camera: MediaPipe landmarks → MLTRAINCAMERA
                  </div>
                  <div style={{ fontSize: '11px', color: '#a855f7', marginBottom: '4px' }}>
                    📡 IMU: Sensor data → MLTRAINIMU
                  </div>
                  {datasetCollectionStatus === 'collecting' && collectedRepsCount > 0 && (
                    <div style={{ fontSize: '11px', color: '#22c55e', marginTop: '6px', fontWeight: 'bold' }}>
                      {collectedRepsCount} rep(s) collected to both datasets
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <p style={{ fontSize: '12px', color: '#888', margin: '5px 0' }}>
                <span style={{ color: '#22c55e' }}>
                  All workout data is automatically saved for later analysis.
                  {datasetCollectionStatus === 'collecting' && collectedRepsCount > 0 && (
                    <span> • {collectedRepsCount} rep(s) recorded</span>
                  )}
                </span>
              </p>
            )}
            {datasetCollectionStatus === 'saved' && (
              <p style={{ fontSize: '12px', color: '#22c55e', margin: '5px 0' }}>
                💾 {mlMode === 'train' ? 'Training datasets saved successfully!' : 'Dataset saved successfully!'}
              </p>
            )}
          </div>
          ) : null}
          
          {/* Mode Info */}
          <div style={{ 
            marginTop: '20px', 
            padding: '15px', 
            background: mlMode === 'usage' 
              ? 'rgba(34, 197, 94, 0.2)' 
              : 'rgba(59, 130, 246, 0.2)', 
            borderRadius: '8px', 
            border: `1px solid ${mlMode === 'usage' ? '#22c55e' : '#3b82f6'}` 
          }}>
            <h4 style={{ margin: '0 0 5px 0' }}>
              {mlMode === 'usage' ? '⚡ Usage Mode' : '📚 ML Training Mode'}
            </h4>
            <p style={{ fontSize: '12px', color: '#ccc', margin: 0 }}>
              {mlMode === 'usage'
                ? 'Angle-based rep counting and form analysis. All workout data is automatically saved for later analysis.'
                : 'Collecting data and training ML models. Dataset collection is automatically enabled.'}
            </p>
          </div>
          
          {/* Workout Configuration */}
          <div className="workout-config" style={{ marginTop: '20px', padding: '15px', background: '#1a1a1a', borderRadius: '8px', border: '1px solid #333' }}>
            <h4 style={{ margin: '0 0 15px 0' }}>🏋️ Workout Configuration</h4>
            
            {/* Number of Sets */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', color: '#ccc' }}>
                Number of Sets:
              </label>
              <input
                type="number"
                min="1"
                max="20"
                value={numberOfSets}
                onChange={(e) => setNumberOfSets(Math.max(1, parseInt(e.target.value) || 1))}
                style={{
                  width: '100%',
                  padding: '10px',
                  borderRadius: '6px',
                  border: '1px solid #444',
                  background: '#0a0a0a',
                  color: 'white',
                  fontSize: '16px'
                }}
              />
            </div>
            
            {/* Reps Per Set */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', color: '#ccc' }}>
                Reps Per Set:
              </label>
              <input
                type="number"
                min="1"
                max="100"
                value={repsPerSet}
                onChange={(e) => setRepsPerSet(Math.max(1, parseInt(e.target.value) || 1))}
                style={{
                  width: '100%',
                  padding: '10px',
                  borderRadius: '6px',
                  border: '1px solid #444',
                  background: '#0a0a0a',
                  color: 'white',
                  fontSize: '16px'
                }}
              />
            </div>
            
            {/* Rest Time Between Sets */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px', fontSize: '14px', color: '#ccc' }}>
                Rest Time Between Sets (seconds):
              </label>
              <input
                type="number"
                min="0"
                max="600"
                value={restTimeSeconds}
                onChange={(e) => setRestTimeSeconds(Math.max(0, parseInt(e.target.value) || 0))}
                style={{
                  width: '100%',
                  padding: '10px',
                  borderRadius: '6px',
                  border: '1px solid #444',
                  background: '#0a0a0a',
                  color: 'white',
                  fontSize: '16px'
                }}
              />
            </div>
            
            {/* Dumbbell rows instruction (only right side) */}
            {exercise === 'dumbbell_rows' && (
              <div style={{ marginBottom: '15px', padding: '12px', borderRadius: '6px', background: '#1e3a8a20', border: '1px solid #3b82f6' }}>
                <p style={{ fontSize: '14px', color: '#3b82f6', margin: 0, textAlign: 'center' }}>
                  📹 Kameranın karşısına <strong>sağ yanınızı</strong> dönün (90° açıdan, vücudun sağ tarafı görünecek)
                </p>
              </div>
            )}
          </div>
          
          <div className="button-group" style={{ marginTop: '20px' }}>
            <button className="back-button" onClick={onEnd}>
              ← Back
            </button>
            <button 
              className="start-button"
              onClick={startCamera}
              disabled={fusionMode !== 'imu_only' && !selectedCamera}
            >
              Start →
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

  // Data Collection Dialog (shown for both usage and train modes after session)
  // IMPORTANT: Check dialog BEFORE finished state, so dialog takes priority
  if (showTrainingDialog) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10000
      }}>
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          style={{
            background: '#1a1a1a',
            borderRadius: '20px',
            padding: 'clamp(24px, 4vw, 40px)',
            maxWidth: 'min(90vw, 700px)',
            width: '90%',
            maxHeight: '90vh',
            border: '2px solid rgba(59, 130, 246, 0.3)',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.7), 0 0 40px rgba(59, 130, 246, 0.1)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          }}
        >
          <div style={{
            overflowY: 'auto',
            overflowX: 'hidden',
            paddingRight: '8px',
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: '20px'
          }}>
            <div style={{ 
              borderBottom: '2px solid rgba(59, 130, 246, 0.2)', 
              paddingBottom: '20px',
              marginBottom: '20px'
            }}>
              <h2 style={{ 
                margin: '0 0 12px 0', 
                color: '#fff', 
                fontSize: 'clamp(24px, 4vw, 32px)',
                fontWeight: 700,
                background: 'linear-gradient(135deg, #3b82f6 0%, #00ccff 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text'
              }}>
                📊 Eğitim Seti Seçenekleri
              </h2>
              <p style={{ 
                color: '#9ca3af', 
                margin: 0, 
                fontSize: 'clamp(14px, 2vw, 16px)',
                lineHeight: '1.5'
              }}>
                Toplanan verilerle ne yapmak istersiniz?
              </p>
            </div>

            {/* Show feedback - always show if available */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(0, 204, 255, 0.1) 100%)',
              border: '1px solid rgba(59, 130, 246, 0.4)',
              borderRadius: '16px',
              padding: 'clamp(16px, 3vw, 24px)',
              boxShadow: '0 4px 12px rgba(59, 130, 246, 0.1)'
            }}>
              <h3 style={{ 
                color: '#fff', 
                fontSize: 'clamp(16px, 2.5vw, 20px)', 
                marginBottom: '12px',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                🤖 AI Coach Feedback
              </h3>
              {sessionFeedback ? (
                <p style={{ 
                  color: '#e5e7eb', 
                  fontSize: 'clamp(14px, 2vw, 15px)', 
                  lineHeight: '1.7', 
                  margin: 0 
                }}>
                  {sessionFeedback}
                </p>
              ) : (
                <p style={{ 
                  color: '#9ca3af', 
                  fontSize: 'clamp(13px, 2vw, 14px)', 
                  lineHeight: '1.6', 
                  margin: 0, 
                  fontStyle: 'italic' 
                }}>
                  Feedback yükleniyor...
                </p>
              )}
            </div>

            {/* Regional Scores - Only Arms with Average Form Score */}
            {regionalFeedbacks?.arms && (
              <div style={{
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(0, 204, 255, 0.08) 100%)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderRadius: '16px',
                padding: 'clamp(16px, 3vw, 20px)',
                boxShadow: '0 4px 12px rgba(59, 130, 246, 0.1)'
              }}>
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center', 
                  marginBottom: '12px',
                  flexWrap: 'wrap',
                  gap: '8px'
                }}>
                  <h3 style={{ 
                    color: '#fff', 
                    fontSize: 'clamp(16px, 2.5vw, 18px)',
                    margin: 0,
                    fontWeight: 600,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    💪 Kollar
                  </h3>
                  <div style={{
                    background: 'rgba(59, 130, 246, 0.2)',
                    padding: '6px 16px',
                    borderRadius: '20px',
                    border: '1px solid rgba(59, 130, 246, 0.4)'
                  }}>
                    <span style={{ 
                      color: '#fff', 
                      fontSize: 'clamp(16px, 2.5vw, 18px)', 
                      fontWeight: 'bold',
                      background: 'linear-gradient(135deg, #3b82f6 0%, #00ccff 100%)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text'
                    }}>
                      {reps.length > 0
                        ? (reps.reduce((sum, r) => sum + r.formScore, 0) / reps.length).toFixed(1)
                        : avgFormScore.toFixed(1)}%
                    </span>
                  </div>
                </div>
                <p style={{ 
                  color: '#e5e7eb', 
                  fontSize: 'clamp(14px, 2vw, 15px)', 
                  margin: 0,
                  lineHeight: '1.6'
                }}>
                  {regionalFeedbacks.arms}
                </p>
              </div>
            )}

            {trainingStatus === 'idle' && (
              <div style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                gap: 'clamp(12px, 2vw, 16px)',
                marginTop: 'auto',
                paddingTop: '20px',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <button
                onClick={async () => {
                  setTrainingAction('save_only');
                  // Keep WebSocket open for training_action
                  if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({
                      type: 'training_action',
                      action: 'save_only'
                    }));
                    // Wait a bit for response, then close
                    setTimeout(() => {
                      if (wsRef.current) {
                        wsRef.current.close();
                      }
                      // Close dialog first
                      setShowTrainingDialog(false);
                      // Reset all state
                      setState('camera_select');
                      setRepCount(0);
                      setReps([]);
                      setSessionFeedback('');
                      setAvgFormScore(0);
                      setRegionalScores(null);
                      setRegionalFeedbacks(null);
                      setIssues([]);
                      setFeedbacks([]);
                      setFormScore(100);
                      // Don't disconnect IMU - it will be reused for next workout
                      // Return to exercise selection
                      setTimeout(() => {
                        onEnd();
                      }, 100);
                    }, 500);
                  } else {
                    // WebSocket already closed, just close dialog and return
                    setShowTrainingDialog(false);
                    // Reset all state
                    setState('camera_select');
                    setRepCount(0);
                    setReps([]);
                    setSessionFeedback('');
                    setAvgFormScore(0);
                    setRegionalScores(null);
                    setRegionalFeedbacks(null);
                    setIssues([]);
                    setFeedbacks([]);
                    setFormScore(100);
                    // Don't disconnect IMU - it will be reused for next workout
                    // Return to exercise selection
                    setTimeout(() => {
                      onEnd();
                    }, 100);
                  }
                }}
                  style={{
                    padding: 'clamp(16px, 3vw, 20px)',
                    background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(0, 204, 255, 0.15) 100%)',
                    border: '2px solid rgba(59, 130, 246, 0.5)',
                    borderRadius: '14px',
                    color: '#fff',
                    fontSize: 'clamp(16px, 2.5vw, 18px)',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    textAlign: 'left',
                    boxShadow: '0 4px 12px rgba(59, 130, 246, 0.2)',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.35) 0%, rgba(0, 204, 255, 0.25) 100%)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 6px 20px rgba(59, 130, 246, 0.3)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.25) 0%, rgba(0, 204, 255, 0.15) 100%)';
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.2)';
                  }}
                >
                  <div style={{ 
                    fontWeight: 'bold', 
                    marginBottom: '8px',
                    fontSize: 'clamp(16px, 2.5vw, 18px)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    💾 Eğitim Setini Kaydet
                  </div>
                  <div style={{ 
                    fontSize: 'clamp(13px, 2vw, 14px)', 
                    color: '#d1d5db',
                    lineHeight: '1.5'
                  }}>
                    Verileri kaydet. Model eğitimi için train_ml_models.py kullanın (bkz: TRAINING_GUIDE.md)
                  </div>
                </button>


              <button
                onClick={async () => {
                  setTrainingAction('skip');
                  // Keep WebSocket open for training_action
                  if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({
                      type: 'training_action',
                      action: 'skip'
                    }));
                    // Wait a bit for response, then close
                    setTimeout(() => {
                      if (wsRef.current) {
                        wsRef.current.close();
                      }
                      // Close dialog first
                      setShowTrainingDialog(false);
                      // Reset all state
                      setState('camera_select');
                      setRepCount(0);
                      setReps([]);
                      setSessionFeedback('');
                      setAvgFormScore(0);
                      setRegionalScores(null);
                      setRegionalFeedbacks(null);
                      setIssues([]);
                      setFeedbacks([]);
                      setFormScore(100);
                      // Don't disconnect IMU - it will be reused for next workout
                      // Return to exercise selection
                      setTimeout(() => {
                        onEnd();
                      }, 100);
                    }, 500);
                  } else {
                    // WebSocket already closed, just close dialog and return
                    setShowTrainingDialog(false);
                    // Reset all state
                    setState('camera_select');
                    setRepCount(0);
                    setReps([]);
                    setSessionFeedback('');
                    setAvgFormScore(0);
                    setRegionalScores(null);
                    setRegionalFeedbacks(null);
                    setIssues([]);
                    setFeedbacks([]);
                    setFormScore(100);
                    // Don't disconnect IMU - it will be reused for next workout
                    // Return to exercise selection
                    setTimeout(() => {
                      onEnd();
                    }, 100);
                  }
                }}
                  style={{
                    padding: 'clamp(16px, 3vw, 20px)',
                    background: 'linear-gradient(135deg, rgba(107, 114, 128, 0.2) 0%, rgba(75, 85, 99, 0.15) 100%)',
                    border: '2px solid rgba(107, 114, 128, 0.4)',
                    borderRadius: '14px',
                    color: '#fff',
                    fontSize: 'clamp(16px, 2.5vw, 18px)',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    textAlign: 'left',
                    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.2)'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = 'linear-gradient(135deg, rgba(107, 114, 128, 0.3) 0%, rgba(75, 85, 99, 0.25) 100%)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 6px 20px rgba(0, 0, 0, 0.3)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'linear-gradient(135deg, rgba(107, 114, 128, 0.2) 0%, rgba(75, 85, 99, 0.15) 100%)';
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.2)';
                  }}
                >
                  <div style={{ 
                    fontWeight: 'bold', 
                    marginBottom: '8px',
                    fontSize: 'clamp(16px, 2.5vw, 18px)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}>
                    ⏭️ Eğitim Setini Direkt Kaydetme
                  </div>
                  <div style={{ 
                    fontSize: 'clamp(13px, 2vw, 14px)', 
                    color: '#d1d5db',
                    lineHeight: '1.5'
                  }}>
                    Verileri kaydetme, direkt geç
                  </div>
                </button>
              </div>
            )}

            {trainingStatus === 'training' && (
              <div style={{ textAlign: 'center', padding: 'clamp(20px, 3vw, 30px)' }}>
                <div style={{ fontSize: 'clamp(40px, 6vw, 48px)', marginBottom: '20px' }}>⏳</div>
                <div style={{ color: '#fff', fontSize: 'clamp(16px, 2.5vw, 18px)', marginBottom: '10px', fontWeight: 600 }}>
                  Training ML Model...
                </div>
                <div style={{ color: '#9ca3af', fontSize: 'clamp(13px, 2vw, 14px)' }}>
                  {trainingMessage || 'This may take a few minutes...'}
                </div>
              </div>
            )}

            {trainingStatus === 'completed' && (
              <div style={{ textAlign: 'center', padding: 'clamp(20px, 3vw, 30px)' }}>
                <div style={{ fontSize: 'clamp(40px, 6vw, 48px)', marginBottom: '20px' }}>✅</div>
                <div style={{ 
                  color: '#22c55e', 
                  fontSize: 'clamp(16px, 2.5vw, 18px)', 
                  marginBottom: '12px', 
                  fontWeight: 'bold' 
                }}>
                  İşlem Tamamlandı!
                </div>
                
                {/* Main message */}
                <div style={{ 
                  color: '#e5e7eb', 
                  fontSize: 'clamp(13px, 2vw, 14px)', 
                  marginBottom: '20px', 
                  whiteSpace: 'pre-line', 
                  textAlign: 'center',
                  lineHeight: '1.6'
                }}>
                  {trainingMessage || 'Veriler başarıyla kaydedildi.'}
                </div>
              
              {/* Performance Metrics Box */}
              {performanceMetrics && (
                <div style={{ 
                  background: 'rgba(59, 130, 246, 0.15)', 
                  border: '1px solid rgba(59, 130, 246, 0.4)', 
                  borderRadius: '10px', 
                  padding: '15px', 
                  marginBottom: '20px',
                  textAlign: 'left'
                }}>
                  <div style={{ color: '#3b82f6', fontSize: '16px', fontWeight: 'bold', marginBottom: '12px', textAlign: 'center' }}>
                    📊 Model Performans Metrikleri
                  </div>
                  
                  {trainingSampleCount > 0 && (
                    <div style={{ color: '#d1d5db', fontSize: '13px', marginBottom: '10px', textAlign: 'center', paddingBottom: '10px', borderBottom: '1px solid rgba(59, 130, 246, 0.3)' }}>
                      📦 Eğitim Örnekleri: <span style={{ color: '#fff', fontWeight: 'bold' }}>{trainingSampleCount}</span>
                    </div>
                  )}
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                    <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '10px', borderRadius: '6px' }}>
                      <div style={{ color: '#9ca3af', fontSize: '12px', marginBottom: '4px' }}>Test R² (Açıklama Oranı)</div>
                      <div style={{ color: '#22c55e', fontSize: '18px', fontWeight: 'bold' }}>
                        {(performanceMetrics.test_r2 * 100).toFixed(1)}%
                      </div>
                    </div>
                    
                    <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '10px', borderRadius: '6px' }}>
                      <div style={{ color: '#9ca3af', fontSize: '12px', marginBottom: '4px' }}>Test MAE (Ortalama Hata)</div>
                      <div style={{ color: '#f59e0b', fontSize: '18px', fontWeight: 'bold' }}>
                        {performanceMetrics.test_mae?.toFixed(2) || 'N/A'}
                      </div>
                    </div>
                    
                    <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '10px', borderRadius: '6px' }}>
                      <div style={{ color: '#9ca3af', fontSize: '12px', marginBottom: '4px' }}>Train R²</div>
                      <div style={{ color: '#3b82f6', fontSize: '18px', fontWeight: 'bold' }}>
                        {(performanceMetrics.train_r2 * 100).toFixed(1)}%
                      </div>
                    </div>
                    
                    <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '10px', borderRadius: '6px' }}>
                      <div style={{ color: '#9ca3af', fontSize: '12px', marginBottom: '4px' }}>Test MSE</div>
                      <div style={{ color: '#ef4444', fontSize: '18px', fontWeight: 'bold' }}>
                        {performanceMetrics.test_mse?.toFixed(2) || 'N/A'}
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <button
                onClick={() => {
                  if (wsRef.current) {
                    wsRef.current.close();
                    wsRef.current = null; // Clear reference
                  }
                  // Close dialog first
                  setShowTrainingDialog(false);
                  // Reset all state
                  setState('camera_select');
                  setRepCount(0);
                  setReps([]);
                  setSessionFeedback('');
                  setAvgFormScore(0);
                  setRegionalScores(null);
                  setRegionalFeedbacks(null);
                  setIssues([]);
                  setFeedbacks([]);
                  setFormScore(100);
                  // Don't disconnect IMU - it will be reused for next workout
                  // Return to exercise selection
                  setTimeout(() => {
                    onEnd();
                  }, 100);
                }}
                style={{
                  padding: 'clamp(12px, 2vw, 14px) clamp(20px, 3vw, 24px)',
                  background: 'linear-gradient(135deg, #3b82f6 0%, #00ccff 100%)',
                  border: 'none',
                  borderRadius: '10px',
                  color: '#fff',
                  fontSize: 'clamp(14px, 2vw, 16px)',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  transition: 'all 0.2s',
                  boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 6px 16px rgba(59, 130, 246, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.3)';
                }}
              >
                Tamam
              </button>
            </div>
          )}

            {trainingStatus === 'error' && (
              <div style={{ textAlign: 'center', padding: 'clamp(20px, 3vw, 30px)' }}>
                <div style={{ fontSize: 'clamp(40px, 6vw, 48px)', marginBottom: '20px' }}>❌</div>
                <div style={{ 
                  color: '#ef4444', 
                  fontSize: 'clamp(16px, 2.5vw, 18px)', 
                  marginBottom: '12px',
                  fontWeight: 600
                }}>
                  Hata Oluştu
                </div>
                <div style={{ 
                  color: '#9ca3af', 
                  fontSize: 'clamp(13px, 2vw, 14px)', 
                  marginBottom: '24px',
                  lineHeight: '1.6'
                }}>
                  {trainingMessage || 'Bir hata oluştu.'}
                </div>
              <button
                onClick={() => {
                  if (wsRef.current) {
                    wsRef.current.close();
                    wsRef.current = null; // Clear reference
                  }
                  // Close dialog first
                  setShowTrainingDialog(false);
                  // Reset all state
                  setState('camera_select');
                  setRepCount(0);
                  setReps([]);
                  setSessionFeedback('');
                  setAvgFormScore(0);
                  setRegionalScores(null);
                  setRegionalFeedbacks(null);
                  setIssues([]);
                  setFeedbacks([]);
                  setFormScore(100);
                  // Don't disconnect IMU - it will be reused for next workout
                  // Return to exercise selection
                  setTimeout(() => {
                    onEnd();
                  }, 100);
                }}
                style={{
                  padding: 'clamp(12px, 2vw, 14px) clamp(20px, 3vw, 24px)',
                  background: '#ef4444',
                  border: 'none',
                  borderRadius: '10px',
                  color: '#fff',
                  fontSize: 'clamp(14px, 2vw, 16px)',
                  cursor: 'pointer',
                  fontWeight: 'bold',
                  transition: 'all 0.2s',
                  boxShadow: '0 4px 12px rgba(239, 68, 68, 0.3)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = '#dc2626';
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 6px 16px rgba(239, 68, 68, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = '#ef4444';
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(239, 68, 68, 0.3)';
                }}
              >
                Kapat
              </button>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    );
  }

  // Session finished screen
  if (state === 'finished') {
    return (
      <div className="session-finished">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="finish-card"
        >
          <h1>🎉 Congratulations!</h1>
          <div className="stats">
            <div className="stat">
              <span className="value">{repCount}</span>
              <span className="label">Reps</span>
            </div>
            <div className="stat">
              <span className="value">
                {reps.length > 0
                  ? (reps.reduce((sum, r) => sum + r.formScore, 0) / reps.length).toFixed(0)
                  : avgFormScore.toFixed(0)}
                %
              </span>
              <span className="label">Avg. Form</span>
            </div>
          </div>

          {/* Arms Feedback */}
          {regionalFeedbacks?.arms && (
            <div className="arms-feedback" style={{
              background: 'rgba(59, 130, 246, 0.1)',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              borderRadius: '12px',
              padding: '16px',
              marginBottom: '24px',
              textAlign: 'left'
            }}>
              <h3 style={{ 
                fontSize: '1rem', 
                fontWeight: 600, 
                marginBottom: '8px',
                color: '#3b82f6',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                💪 Kollar Feedback
              </h3>
              <p style={{ 
                fontSize: '0.9375rem', 
                lineHeight: '1.6', 
                color: '#e5e7eb',
                margin: 0
              }}>
                {regionalFeedbacks.arms}
              </p>
            </div>
          )}

          {/* Rep List */}
          {reps.length > 0 && (
            <div className="rep-list" style={{
              background: 'rgba(255, 255, 255, 0.03)',
              borderRadius: '16px',
              padding: '20px',
              marginBottom: '24px',
              maxHeight: '400px',
              overflowY: 'auto'
            }}>
              <h3 style={{
                fontSize: '1rem',
                fontWeight: 600,
                marginBottom: '16px',
                color: '#fff',
                textAlign: 'left'
              }}>
                📊 Rep Detayları
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {reps.map((rep, index) => (
                  <div
                    key={index}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      padding: '12px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      borderRadius: '8px',
                      border: '1px solid rgba(255, 255, 255, 0.1)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flex: 1 }}>
                      <span style={{ 
                        fontSize: '0.875rem', 
                        fontWeight: 600,
                        color: '#9ca3af',
                        minWidth: '50px'
                      }}>
                        Rep #{rep.repNumber}
                      </span>
                      <div style={{
                        fontSize: '1.25rem',
                        fontWeight: 'bold',
                        color: rep.formScore >= 80 ? '#22c55e' : rep.formScore >= 60 ? '#eab308' : '#ef4444',
                        minWidth: '50px'
                      }}>
                        {rep.formScore.toFixed(0)}%
                      </div>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        fontSize: '0.875rem',
                        color: '#9ca3af'
                      }}>
                        <span>{rep.speedEmoji}</span>
                        <span>{rep.speedLabel}</span>
                        <span style={{ color: '#6b7280' }}>({rep.duration.toFixed(1)}s)</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="ai-feedback">
            <h3>🤖 AI Coach Feedback</h3>
            {sessionFeedback ? (
              <p>{sessionFeedback}</p>
            ) : (
              <p>Loading feedback...</p>
            )}
          </div>

          <button className="back-button" onClick={() => {
            // Close WebSocket if open
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              try {
                wsRef.current.close();
              } catch (e) {
                console.error('Error closing WebSocket:', e);
              }
            }
            
            // Disconnect IMU
            try {
              disconnectIMU();
            } catch (e) {
              console.error('Error disconnecting IMU:', e);
            }
            
            // Reset all state
            setState('camera_select');
            setRepCount(0);
            setReps([]);
            setSessionFeedback('');
            setAvgFormScore(0);
            setRegionalScores(null);
            setRegionalFeedbacks(null);
            setIssues([]);
            setFeedbacks([]);
            setFormScore(100);
            
            // Call onEnd to return to exercise selection
            setTimeout(() => {
              onEnd();
            }, 100);
          }}>
            ← Back to Exercise Selection
          </button>
        </motion.div>
      </div>
    );
  }

  // Rest overlay for resting state
  const restOverlay = state === 'resting' && restCountdown > 0 ? (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.85)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 10000,
      pointerEvents: 'none'
    }}>
      <div style={{
        fontSize: '100px',
        fontWeight: 'bold',
        color: '#f97316',
        textShadow: '0 0 40px rgba(249, 115, 22, 0.8)',
        animation: 'pulse 1s ease-in-out',
        fontFamily: 'Arial, sans-serif',
        marginBottom: '20px'
      }}>
        {restCountdown}
      </div>
      <div style={{
        fontSize: '24px',
        color: '#ccc',
        fontFamily: 'Arial, sans-serif'
      }}>
        Rest Time
      </div>
      <div style={{
        fontSize: '16px',
        color: '#888',
        marginTop: '10px'
      }}>
        Set {currentSet} complete! Next set starting in...
      </div>
    </div>
  ) : null;

  // Countdown overlay for countdown state
  const countdownOverlay = state === 'countdown' && countdownNumber !== null ? (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 10000,
      pointerEvents: 'none'
    }}>
      <div style={{
        fontSize: countdownNumber === 0 ? '120px' : '200px',
        fontWeight: 'bold',
        color: countdownNumber === 0 ? '#22c55e' : '#3b82f6',
        textShadow: '0 0 40px rgba(59, 130, 246, 0.8)',
        animation: 'pulse 0.5s ease-in-out',
        fontFamily: 'Arial, sans-serif'
      }}>
        {countdownNumber === 0 ? 'START!' : countdownNumber}
      </div>
    </div>
  ) : null;

  return (
    <div className="workout-session">
      {countdownOverlay}
      {restOverlay}
      <div className="header">
        <button className="back-button" onClick={onEnd}>
          ← Back
        </button>
        <h2>{getStatusText()}</h2>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {/* IMU Status Indicator */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            padding: '5px 10px',
            background: imuConnected ? 'rgba(34, 197, 94, 0.3)' : 'rgba(100, 100, 100, 0.3)',
            borderRadius: '15px',
            fontSize: '12px'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: imuConnected ? '#22c55e' : '#666'
            }}></span>
            IMU {imuConnected ? `${imuSampleRate.toFixed(0)}Hz` : 'Off'}
          </div>
          <button className="end-button" onClick={handleEnd}>
            Finish
          </button>
        </div>
      </div>

      <div className="main-content" style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: 'clamp(16px, 2vw, 24px)',
        padding: 'clamp(12px, 2vw, 20px)',
        height: '100%',
        overflow: 'hidden'
      }}>
        {/* LEFT COLUMN: 3D Grid + IMU Activity Monitor */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'clamp(12px, 2vw, 16px)',
          overflow: 'hidden'
        }}>
          {/* 3D Avatar/Grid */}
          <div className={`video-avatar-container ${fusionMode === 'imu_only' ? 'imu-only-mode' : ''}`} style={{
            flex: '1 1 auto',
            minHeight: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
          {/* Camera View - Hidden in IMU-only mode */}
          {fusionMode !== 'imu_only' && (
            <div className="video-container">
              <video ref={videoRef} autoPlay playsInline style={{ display: 'none' }} />
              <canvas ref={canvasRef} width={640} height={480} />
            </div>
          )}
          
          {/* 3D Avatar with IMU Fusion */}
          <div 
            ref={avatarWrapperRef}
            className="avatar-wrapper" 
            style={{
              width: '100%',
              maxWidth: fusionMode === 'imu_only' ? 'min(90vw, 600px)' : '400px',
              aspectRatio: '4/3',
              height: 'auto',
              transition: 'all 0.3s ease',
              margin: '0 auto'
            }}
          >
            <HumanAvatar 
              landmarks={currentLandmarks} 
              width={avatarSize.width} 
              height={avatarSize.height} 
              modelUrl={avatarUrl}
              showSkeleton={true}
              isCalibrated={state === 'ready' || state === 'countdown' || state === 'tracking'}
              imuData={imuData}
              fusionMode={fusionMode}
              showIMUDebug={showIMUDebug}
            />
            <div className="avatar-label">
              3D Avatar 
              {fusionMode !== 'camera_only' && imuConnected && (
                <span style={{ color: '#22c55e', marginLeft: '5px' }}>+ IMU</span>
              )}
            </div>
          </div>
        </div>
          
          {/* Fusion Mode Toggle */}
          <div style={{
            display: 'flex',
            gap: 'clamp(6px, 1.5vw, 10px)',
            justifyContent: 'center',
            flexWrap: 'wrap'
          }}>
          <button
            onClick={() => setFusionMode('camera_only')}
            style={{
              padding: 'clamp(6px, 1.5vw, 8px) clamp(12px, 2vw, 16px)',
              background: fusionMode === 'camera_only' ? '#3b82f6' : '#333',
              border: 'none',
              borderRadius: '20px',
              color: 'white',
              cursor: 'pointer',
              fontSize: 'clamp(10px, 1.5vw, 12px)',
              transition: 'all 0.2s',
              fontWeight: 500
            }}
            onMouseEnter={(e) => {
              if (fusionMode !== 'camera_only') {
                e.currentTarget.style.background = '#444';
              }
            }}
            onMouseLeave={(e) => {
              if (fusionMode !== 'camera_only') {
                e.currentTarget.style.background = '#333';
              }
            }}
          >
            📷 Camera Only
          </button>
          <button
            onClick={() => setFusionMode('camera_primary')}
            style={{
              padding: 'clamp(6px, 1.5vw, 8px) clamp(12px, 2vw, 16px)',
              background: fusionMode === 'camera_primary' ? '#3b82f6' : '#333',
              border: 'none',
              borderRadius: '20px',
              color: 'white',
              cursor: 'pointer',
              fontSize: 'clamp(10px, 1.5vw, 12px)',
              transition: 'all 0.2s',
              fontWeight: 500
            }}
            onMouseEnter={(e) => {
              if (fusionMode !== 'camera_primary') {
                e.currentTarget.style.background = '#444';
              }
            }}
            onMouseLeave={(e) => {
              if (fusionMode !== 'camera_primary') {
                e.currentTarget.style.background = '#333';
              }
            }}
          >
            🔄 Fusion
          </button>
          <button
            onClick={() => setFusionMode('imu_only')}
            style={{
              padding: 'clamp(6px, 1.5vw, 8px) clamp(12px, 2vw, 16px)',
              background: fusionMode === 'imu_only' ? '#3b82f6' : '#333',
              border: 'none',
              borderRadius: '20px',
              color: 'white',
              cursor: 'pointer',
              fontSize: 'clamp(10px, 1.5vw, 12px)',
              transition: 'all 0.2s',
              fontWeight: 500
            }}
            onMouseEnter={(e) => {
              if (fusionMode !== 'imu_only') {
                e.currentTarget.style.background = '#444';
              }
            }}
            onMouseLeave={(e) => {
              if (fusionMode !== 'imu_only') {
                e.currentTarget.style.background = '#333';
              }
            }}
          >
            🎛️ IMU Only
          </button>
          <button
            onClick={resetOrientation}
            style={{
              padding: 'clamp(6px, 1.5vw, 8px) clamp(12px, 2vw, 16px)',
              background: '#6b21a8',
              border: 'none',
              borderRadius: '20px',
              color: 'white',
              cursor: 'pointer',
              fontSize: 'clamp(10px, 1.5vw, 12px)',
              transition: 'all 0.2s',
              fontWeight: 500
            }}
            title="Reset IMU orientation"
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#7c3aed';
              e.currentTarget.style.transform = 'scale(1.05)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = '#6b21a8';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            🔄 Reset IMU
          </button>
          <label style={{
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            fontSize: 'clamp(10px, 1.5vw, 12px)',
            color: '#888',
            cursor: 'pointer'
          }}>
            <input
              type="checkbox"
              checked={showIMUDebug}
              onChange={(e) => setShowIMUDebug(e.target.checked)}
            />
            Debug
          </label>
          </div>
          
          {/* Real-time IMU Activity Panel (during workout) */}
          {showIMUDebug && imuConnected && (
            <div style={{
              width: '100%',
              background: 'rgba(0, 0, 0, 0.8)',
              border: '1px solid #22c55e',
              borderRadius: '12px',
              padding: 'clamp(10px, 2vw, 12px) clamp(12px, 2.5vw, 16px)',
              fontFamily: 'monospace',
              fontSize: 'clamp(9px, 1.5vw, 11px)',
              boxShadow: '0 4px 12px rgba(34, 197, 94, 0.2)',
              maxHeight: '40vh',
              overflowY: 'auto'
            }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '8px',
              paddingBottom: '8px',
              borderBottom: '1px solid #333'
            }}>
              <span style={{ 
                color: '#22c55e', 
                fontWeight: 'bold',
                fontSize: 'clamp(11px, 1.8vw, 13px)'
              }}>📊 IMU Activity Monitor</span>
              <span style={{ 
                color: '#888',
                fontSize: 'clamp(10px, 1.5vw, 12px)'
              }}>{imuSampleRate.toFixed(0)} Hz</span>
            </div>
            <div style={{ 
              display: 'flex', 
              gap: 'clamp(12px, 3vw, 20px)', 
              justifyContent: 'space-between', 
              flexWrap: 'wrap' 
            }}>
              {/* Left Wrist */}
              <div style={{ 
                flex: '1 1 min(100%, 280px)', 
                textAlign: 'center', 
                minWidth: 'min(100%, 200px)',
                padding: '8px',
                background: 'rgba(59, 130, 246, 0.05)',
                borderRadius: '8px',
                border: '1px solid rgba(59, 130, 246, 0.2)'
              }}>
                <div style={{ 
                  color: '#3b82f6', 
                  marginBottom: '8px', 
                  fontWeight: 'bold',
                  fontSize: 'clamp(10px, 1.8vw, 12px)'
                }}>🤚 Left Wrist (LW)</div>
                
                {/* Orientation (Euler) */}
                <div style={{ 
                  marginBottom: '8px', 
                  padding: 'clamp(4px, 1vw, 6px)', 
                  background: 'rgba(59, 130, 246, 0.1)', 
                  borderRadius: '6px',
                  transition: 'background 0.2s'
                }}>
                  <div style={{ 
                    fontSize: 'clamp(9px, 1.3vw, 10px)', 
                    color: '#888', 
                    marginBottom: '4px',
                    fontWeight: 600
                  }}>Orientation:</div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Roll: {leftWrist ? leftWrist.euler.roll.toFixed(1) : '--'}°
                  </div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Pitch: {leftWrist ? leftWrist.euler.pitch.toFixed(1) : '--'}°
                  </div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Yaw: {leftWrist ? leftWrist.euler.yaw.toFixed(1) : '--'}°
                  </div>
                </div>
                
                {/* Accelerometer (XYZ) */}
                <div style={{ 
                  marginBottom: '8px', 
                  padding: 'clamp(4px, 1vw, 6px)', 
                  background: 'rgba(34, 197, 94, 0.1)', 
                  borderRadius: '6px',
                  transition: 'background 0.2s'
                }}>
                  <div style={{ 
                    fontSize: 'clamp(9px, 1.3vw, 10px)', 
                    color: '#888', 
                    marginBottom: '4px',
                    fontWeight: 600
                  }}>Accel (g):</div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    X: {leftWrist ? leftWrist.accel.x.toFixed(2) : '--'}
                  </div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Y: {leftWrist ? leftWrist.accel.y.toFixed(2) : '--'}
                  </div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Z: {leftWrist ? leftWrist.accel.z.toFixed(2) : '--'}
                  </div>
                </div>
                
                {/* Gyroscope (XYZ) */}
                <div style={{ 
                  marginBottom: '8px', 
                  padding: 'clamp(4px, 1vw, 6px)', 
                  background: 'rgba(249, 115, 22, 0.1)', 
                  borderRadius: '6px',
                  transition: 'background 0.2s'
                }}>
                  <div style={{ 
                    fontSize: 'clamp(9px, 1.3vw, 10px)', 
                    color: '#888', 
                    marginBottom: '4px',
                    fontWeight: 600
                  }}>Gyro (deg/s):</div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    X: {leftWrist ? leftWrist.gyro.x.toFixed(1) : '--'}
                  </div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Y: {leftWrist ? leftWrist.gyro.y.toFixed(1) : '--'}
                  </div>
                  <div style={{ 
                    color: leftWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Z: {leftWrist ? leftWrist.gyro.z.toFixed(1) : '--'}
                  </div>
                </div>
                
                {/* Unit Vectors */}
                {leftWrist?.unit_vectors && (
                  <div style={{ 
                    marginBottom: '8px', 
                    padding: 'clamp(4px, 1vw, 6px)', 
                    background: 'rgba(168, 85, 247, 0.1)', 
                    borderRadius: '6px',
                    transition: 'background 0.2s'
                  }}>
                    <div style={{ 
                      fontSize: 'clamp(9px, 1.3vw, 10px)', 
                      color: '#888', 
                      marginBottom: '4px',
                      fontWeight: 600
                    }}>Unit Vectors:</div>
                    <div style={{ 
                      color: '#fff', 
                      fontSize: 'clamp(8px, 1.2vw, 9px)',
                      lineHeight: '1.4',
                      fontFamily: 'monospace'
                    }}>
                      <div>Normal: ({leftWrist.unit_vectors.normal.x.toFixed(2)}, {leftWrist.unit_vectors.normal.y.toFixed(2)}, {leftWrist.unit_vectors.normal.z.toFixed(2)})</div>
                      <div>Tangent: ({leftWrist.unit_vectors.tangent.x.toFixed(2)}, {leftWrist.unit_vectors.tangent.y.toFixed(2)}, {leftWrist.unit_vectors.tangent.z.toFixed(2)})</div>
                      <div>Binormal: ({leftWrist.unit_vectors.binormal.x.toFixed(2)}, {leftWrist.unit_vectors.binormal.y.toFixed(2)}, {leftWrist.unit_vectors.binormal.z.toFixed(2)})</div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Right Wrist */}
              <div style={{ 
                flex: '1 1 min(100%, 280px)', 
                textAlign: 'center', 
                minWidth: 'min(100%, 200px)',
                padding: '8px',
                background: 'rgba(168, 85, 247, 0.05)',
                borderRadius: '8px',
                border: '1px solid rgba(168, 85, 247, 0.2)'
              }}>
                <div style={{ 
                  color: '#a855f7', 
                  marginBottom: '8px', 
                  fontWeight: 'bold',
                  fontSize: 'clamp(10px, 1.8vw, 12px)'
                }}>✋ Right Wrist (RW)</div>
                
                {/* Orientation (Euler) */}
                <div style={{ 
                  marginBottom: '8px', 
                  padding: 'clamp(4px, 1vw, 6px)', 
                  background: 'rgba(168, 85, 247, 0.1)', 
                  borderRadius: '6px',
                  transition: 'background 0.2s'
                }}>
                  <div style={{ 
                    fontSize: 'clamp(9px, 1.3vw, 10px)', 
                    color: '#888', 
                    marginBottom: '4px',
                    fontWeight: 600
                  }}>Orientation:</div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Roll: {rightWrist ? rightWrist.euler.roll.toFixed(1) : '--'}°
                  </div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Pitch: {rightWrist ? rightWrist.euler.pitch.toFixed(1) : '--'}°
                  </div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Yaw: {rightWrist ? rightWrist.euler.yaw.toFixed(1) : '--'}°
                  </div>
                </div>
                
                {/* Accelerometer (XYZ) */}
                <div style={{ 
                  marginBottom: '8px', 
                  padding: 'clamp(4px, 1vw, 6px)', 
                  background: 'rgba(34, 197, 94, 0.1)', 
                  borderRadius: '6px',
                  transition: 'background 0.2s'
                }}>
                  <div style={{ 
                    fontSize: 'clamp(9px, 1.3vw, 10px)', 
                    color: '#888', 
                    marginBottom: '4px',
                    fontWeight: 600
                  }}>Accel (g):</div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    X: {rightWrist ? rightWrist.accel.x.toFixed(2) : '--'}
                  </div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Y: {rightWrist ? rightWrist.accel.y.toFixed(2) : '--'}
                  </div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Z: {rightWrist ? rightWrist.accel.z.toFixed(2) : '--'}
                  </div>
                </div>
                
                {/* Gyroscope (XYZ) */}
                <div style={{ 
                  marginBottom: '8px', 
                  padding: 'clamp(4px, 1vw, 6px)', 
                  background: 'rgba(249, 115, 22, 0.1)', 
                  borderRadius: '6px',
                  transition: 'background 0.2s'
                }}>
                  <div style={{ 
                    fontSize: 'clamp(9px, 1.3vw, 10px)', 
                    color: '#888', 
                    marginBottom: '4px',
                    fontWeight: 600
                  }}>Gyro (deg/s):</div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    X: {rightWrist ? rightWrist.gyro.x.toFixed(1) : '--'}
                  </div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Y: {rightWrist ? rightWrist.gyro.y.toFixed(1) : '--'}
                  </div>
                  <div style={{ 
                    color: rightWrist ? '#fff' : '#666', 
                    fontSize: 'clamp(9px, 1.3vw, 10px)',
                    lineHeight: '1.4'
                  }}>
                    Z: {rightWrist ? rightWrist.gyro.z.toFixed(1) : '--'}
                  </div>
                </div>
                
                {/* Unit Vectors */}
                {rightWrist?.unit_vectors && (
                  <div style={{ 
                    marginBottom: '8px', 
                    padding: 'clamp(4px, 1vw, 6px)', 
                    background: 'rgba(168, 85, 247, 0.1)', 
                    borderRadius: '6px',
                    transition: 'background 0.2s'
                  }}>
                    <div style={{ 
                      fontSize: 'clamp(9px, 1.3vw, 10px)', 
                      color: '#888', 
                      marginBottom: '4px',
                      fontWeight: 600
                    }}>Unit Vectors:</div>
                    <div style={{ 
                      color: '#fff', 
                      fontSize: 'clamp(8px, 1.2vw, 9px)',
                      lineHeight: '1.4',
                      fontFamily: 'monospace'
                    }}>
                      <div>Normal: ({rightWrist.unit_vectors.normal.x.toFixed(2)}, {rightWrist.unit_vectors.normal.y.toFixed(2)}, {rightWrist.unit_vectors.normal.z.toFixed(2)})</div>
                      <div>Tangent: ({rightWrist.unit_vectors.tangent.x.toFixed(2)}, {rightWrist.unit_vectors.tangent.y.toFixed(2)}, {rightWrist.unit_vectors.tangent.z.toFixed(2)})</div>
                      <div>Binormal: ({rightWrist.unit_vectors.binormal.x.toFixed(2)}, {rightWrist.unit_vectors.binormal.y.toFixed(2)}, {rightWrist.unit_vectors.binormal.z.toFixed(2)})</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        </div>
        
        {/* RIGHT COLUMN: Rep Details + Feedback */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 'clamp(12px, 2vw, 16px)',
          overflow: 'hidden',
          height: '100%'
        }}>
          {/* Rep List */}
          {reps.length > 0 && (
            <div style={{
              padding: 'clamp(12px, 2vw, 16px)',
              background: 'rgba(255, 255, 255, 0.03)',
              borderRadius: '12px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              maxHeight: '50vh',
              overflowY: 'auto',
              flex: '1 1 auto'
            }}>
              <h3 style={{ 
                fontSize: 'clamp(0.9rem, 2vw, 1rem)', 
                fontWeight: 600, 
                marginBottom: '12px', 
                color: '#fff' 
              }}>
                📊 Rep Detayları
              </h3>
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}>
                <AnimatePresence>
                  {reps.map((rep, index) => (
                    <motion.div
                      key={rep.repNumber || index}
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '10px',
                        background: rep.formScore >= 80 
                          ? 'rgba(34, 197, 94, 0.15)' 
                          : rep.formScore >= 60 
                          ? 'rgba(234, 179, 8, 0.15)' 
                          : 'rgba(239, 68, 68, 0.15)',
                        borderRadius: '8px',
                        border: `1px solid ${
                          rep.formScore >= 80 
                            ? 'rgba(34, 197, 94, 0.3)' 
                            : rep.formScore >= 60 
                            ? 'rgba(234, 179, 8, 0.3)' 
                            : 'rgba(239, 68, 68, 0.3)'
                        }`
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, flexWrap: 'wrap' }}>
                        <span style={{ 
                          fontSize: 'clamp(11px, 1.5vw, 13px)', 
                          fontWeight: 600,
                          color: '#9ca3af',
                          minWidth: '50px'
                        }}>
                          Rep #{rep.repNumber}
                        </span>
                        <div style={{
                          fontSize: 'clamp(14px, 2vw, 16px)',
                          fontWeight: 'bold',
                          color: rep.formScore >= 80 ? '#22c55e' : rep.formScore >= 60 ? '#eab308' : '#ef4444',
                          minWidth: '45px'
                        }}>
                          {rep.formScore.toFixed(0)}%
                        </div>
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: 'clamp(10px, 1.3vw, 12px)',
                          color: '#9ca3af'
                        }}>
                          <span>{rep.speedEmoji || '⚡'}</span>
                          <span>{rep.speedLabel || 'Normal'}</span>
                          <span style={{ color: '#6b7280' }}>({rep.duration?.toFixed(1) || '0.0'}s)</span>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}
          
          {/* Stats Overlay */}
          <div className="stats-bar" style={{
            marginTop: 'auto'
          }}>
          <div className="stat-item">
            <span className="stat-value">Set: {currentSet}/{numberOfSets}</span>
            <span className="stat-label">SET</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">Rep: {currentRepInSet}/{repsPerSet}</span>
            <span className="stat-label">REP</span>
          </div>
          <div className="stat-item">
            <div className="form-bar-mini">
              <motion.div
                className="form-fill"
                animate={{ width: `${formScore}%` }}
                style={{ backgroundColor: getFormColor() }}
              />
            </div>
            <span className="stat-label">{formScore.toFixed(0)}% Form</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{currentAngle.toFixed(0)}°</span>
            <span className="stat-label">{(phase || 'down').toUpperCase()}</span>
          </div>
          </div>

          {/* Issues display */}
          <AnimatePresence>
            {issues.length > 0 && (
              <motion.div
                className="issues-bar"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                {issues.map((issue, i) => (
                  <span key={i} className="issue">
                    ⚠️ {issue}
                  </span>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* AI Feedback panel - Tips and Regional Feedback */}
          <div className="feedback-panel" style={{
            maxHeight: '30vh',
            overflowY: 'auto'
          }}>
            {/* Regional Feedback */}
            {regionalFeedbacks && (
              <div className="regional-feedback" style={{
                marginTop: 'clamp(12px, 2vw, 20px)',
                padding: 'clamp(12px, 2vw, 15px)',
                background: 'rgba(255, 255, 255, 0.03)',
                borderRadius: '12px',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <h4 style={{ 
                  fontSize: 'clamp(0.8rem, 2vw, 0.875rem)', 
                  fontWeight: 600, 
                  marginBottom: '12px', 
                  color: '#fff' 
                }}>
                  📍 Bölgesel Feedback
                </h4>
                <div style={{ display: 'grid', gap: 'clamp(8px, 1.5vw, 10px)' }}>
                  {regionalFeedbacks.arms && (
                    <div style={{
                      padding: 'clamp(8px, 1.5vw, 10px)',
                      background: 'rgba(59, 130, 246, 0.1)',
                      borderRadius: '8px',
                      borderLeft: '3px solid #3b82f6',
                      fontSize: 'clamp(0.75rem, 1.8vw, 0.8rem)',
                      lineHeight: '1.5'
                    }}>
                      <strong style={{ color: '#3b82f6' }}>💪 Kollar:</strong> {regionalFeedbacks.arms}
                    </div>
                  )}
                  {regionalFeedbacks.legs && (
                    <div style={{
                      padding: 'clamp(8px, 1.5vw, 10px)',
                      background: 'rgba(168, 85, 247, 0.1)',
                      borderRadius: '8px',
                      borderLeft: '3px solid #a855f7',
                      fontSize: 'clamp(0.75rem, 1.8vw, 0.8rem)',
                      lineHeight: '1.5'
                    }}>
                      <strong style={{ color: '#a855f7' }}>🦵 Bacaklar:</strong> {regionalFeedbacks.legs}
                    </div>
                  )}
                  {regionalFeedbacks.core && (
                    <div style={{
                      padding: 'clamp(8px, 1.5vw, 10px)',
                      background: 'rgba(249, 115, 22, 0.1)',
                      borderRadius: '8px',
                      borderLeft: '3px solid #f97316',
                      fontSize: 'clamp(0.75rem, 1.8vw, 0.8rem)',
                      lineHeight: '1.5'
                    }}>
                      <strong style={{ color: '#f97316' }}>🏋️ Gövde:</strong> {regionalFeedbacks.core}
                    </div>
                  )}
                  {regionalFeedbacks.head && (
                    <div style={{
                      padding: 'clamp(8px, 1.5vw, 10px)',
                      background: 'rgba(34, 197, 94, 0.1)',
                      borderRadius: '8px',
                      borderLeft: '3px solid #22c55e',
                      fontSize: 'clamp(0.75rem, 1.8vw, 0.8rem)',
                      lineHeight: '1.5'
                    }}>
                      <strong style={{ color: '#22c55e' }}>👤 Kafa:</strong> {regionalFeedbacks.head}
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="tips">
              <h4>💡 Tips</h4>
              {exerciseConfig.formTips.map((tip, i) => (
                <div key={i} className="tip">
                  • {tip}
                </div>
              ))}
            </div>
          </div>
          
          {/* Debug Panel */}
          {showIMUDebug && (
            <div className="debug-panel" style={{
              maxHeight: '20vh',
              overflowY: 'auto',
              marginTop: 'clamp(12px, 2vw, 16px)'
            }}>
              <h4>🔧 Debug Logs</h4>
              <div className="debug-info">
                <span>State: <b>{state}</b></span>
                <span>Camera: <b>{selectedCamera ? 'Selected' : 'None'}</b></span>
                <span>WS: <b>{wsRef.current?.readyState === 1 ? 'Open' : 'Closed'}</b></span>
                <span>IMU: <b>{imuConnected ? 'Connected' : 'Disconnected'}</b></span>
                <span>Fusion: <b>{fusionMode}</b></span>
              </div>
              <div className="debug-logs">
                {debugLogs.map((log, i) => (
                  <div key={i} className="log-line">{log}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WorkoutSessionWithIMU;

