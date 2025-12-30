import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Pose, Results } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import type { ExerciseType, RepData, AIFeedback } from '../types';
import { EXERCISES } from '../config/exercises';
import { HumanAvatar } from './HumanAvatar';

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
  onEnd: () => void;
}

type SessionState = 'camera_select' | 'connecting' | 'detecting' | 'calibrating' | 'ready' | 'tracking' | 'finished';

export const WorkoutSession = ({ exercise, apiKey, avatarUrl, onEnd }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<Pose | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const [state, setState] = useState<SessionState>('camera_select');
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>('');
  const [cameraStarted, setCameraStarted] = useState(false);
  const [visibilityMessage, setVisibilityMessage] = useState<string>('');
  const [debugLogs, setDebugLogs] = useState<string[]>([]);
  
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
  const [reps, setReps] = useState<RepData[]>([]);
  const [sessionFeedback, setSessionFeedback] = useState<string>('');
  const [currentLandmarks, setCurrentLandmarks] = useState<Landmark3D[] | null>(null);
  const [startTime] = useState(new Date());
  const [calibrationProgress, setCalibrationProgress] = useState(0);

  const exerciseConfig = EXERCISES[exercise];

  // Get available cameras on mount
  useEffect(() => {
    const getCameras = async () => {
      try {
        // Request permission first
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

  // Start camera with selected device
  const startCamera = async () => {
    const log = (msg: string) => {
      const timestamp = new Date().toLocaleTimeString();
      console.log(`[${timestamp}] ${msg}`);
      setDebugLogs(prev => [...prev.slice(-20), `[${timestamp}] ${msg}`]);
    };
    
    if (!selectedCamera) {
      log('ERROR: No camera selected');
      return;
    }
    if (cameraStarted) {
      log('WARN: Camera already started');
      return;
    }
    
    log(`Starting camera: ${cameras.find(c => c.deviceId === selectedCamera)?.label}`);
    setState('connecting');
    setCameraStarted(true);
    
    try {
      // Get camera stream first
      log('Requesting camera stream...');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: { exact: selectedCamera },
          width: 640,
          height: 480,
        },
      });
      log('Camera stream obtained');
      
      if (!videoRef.current) {
        log('ERROR: Video ref is null');
        return;
      }
      
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      log('Video element playing');
      
      // Initialize MediaPipe Pose
      log('Initializing MediaPipe Pose...');
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
      log('MediaPipe Pose configured');
      
      // Start the pose detection loop
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
      
      // Wait for pose model to load
      log('Loading pose model (may take a few seconds)...');
      await pose.initialize();
      log('Pose model loaded! Starting detection...');
      
      detectPose();
      
    } catch (error) {
      log(`ERROR: ${error}`);
      console.error('Camera start error:', error);
    }
  };

  // Connect to WebSocket API when camera starts
  useEffect(() => {
    if (!cameraStarted) return;
    
    const ws = new WebSocket(`ws://localhost:8000/ws/${exercise}`);
    
    ws.onopen = () => {
      addLog('WebSocket connected');
      // Send API key
      ws.send(JSON.stringify({ type: 'init', api_key: apiKey }));
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Don't log every update (too spammy)
      if (data.type !== 'update' && data.type !== 'calibration_progress') {
        const timestamp = new Date().toLocaleTimeString();
        console.log(`[${timestamp}] WS: ${data.type} ${data.state || ''}`);
        setDebugLogs(prev => [...prev.slice(-20), `[${timestamp}] WS: ${data.type} ${data.state || data.message || ''}`]);
      }
      
      if (data.type === 'ready') {
        setState('detecting');
      } else if (data.type === 'state') {
        setState(data.state);
        // Show calibration timeout message
        if (data.message) {
          setVisibilityMessage(data.message);
        }
      } else if (data.type === 'visibility') {
        // Show what body parts need to be visible
        setVisibilityMessage(data.message + (data.missing ? ` (Missing: ${data.missing})` : ''));
      } else if (data.type === 'calibration_progress') {
        setCalibrationProgress(data.progress);
        setVisibilityMessage('');  // Clear message during calibration
      } else if (data.type === 'update') {
        setCurrentAngle(data.angle);
        setPhase(data.phase);
        setRepCount(data.rep_count);
        setFormScore(data.form_score);
        setIssues(data.issues || []);
        
        if (data.feedback) {
          setFeedbacks(prev => [...prev, {
            message: data.feedback,
            type: data.form_score >= 80 ? 'success' : data.form_score >= 60 ? 'warning' : 'error',
            timestamp: new Date(),
          }]);
        }
        
        if (data.rep_completed) {
          setReps(prev => [...prev, {
            repNumber: data.rep_completed.rep,
            formScore: data.rep_completed.form_score,
            issues: data.issues || [],
            timestamp: new Date(),
          }]);
        }
      } else if (data.type === 'session_summary') {
        // AI generated session feedback
        setSessionFeedback(data.feedback);
        setRepCount(data.total_reps);
        // Close connection after receiving summary
        ws.close();
      }
    };
    
    ws.onerror = (error) => {
      addLog(`WS ERROR: ${error}`);
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      addLog('WebSocket closed');
    };
    
    wsRef.current = ws;
    
    return () => {
      ws.close();
    };
  }, [exercise, apiKey, cameraStarted]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
      poseRef.current?.close();
    };
  }, []);

  const onResults = useCallback((results: Results) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    // Draw video frame
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    // Draw pose if landmarks exist
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

      // Convert landmarks
      const landmarks = results.poseLandmarks.map((l) => ({
        x: l.x,
        y: l.y,
        z: l.z,
        visibility: l.visibility,
      }));
      
      // Update state for 3D avatar
      setCurrentLandmarks(landmarks);
      
      // Send landmarks to Python backend via WebSocket
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'pose', landmarks }));
      }
    }

    ctx.restore();
  }, []);

  const handleEnd = async () => {
    // Stop camera first
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    
    // Request session summary from backend before closing
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_session' }));
      
      // Wait for response before transitioning
      // The session_summary message will be handled in onmessage
    }
    
    setState('finished');
  };

  const getStatusText = () => {
    switch (state) {
      case 'camera_select':
        return '📷 Select Camera';
      case 'connecting':
        return '🔌 Connecting to server...';
      case 'detecting':
        return visibilityMessage || '🔍 Detecting body...';
      case 'calibrating':
        return `⏳ Calibrating: ${(calibrationProgress * 100).toFixed(0)}%`;
      case 'ready':
        return '✅ Ready! Start moving';
      case 'tracking':
        return `🏋️ ${exerciseConfig.displayName}`;
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
          <h2>📷 Select Camera</h2>
          <p className="exercise-info">
            Selected exercise: <strong>{exerciseConfig.displayName}</strong>
          </p>
          
          <div className="camera-list">
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
          
          <div className="button-group">
            <button className="back-button" onClick={onEnd}>
              ← Back
            </button>
            <button 
              className="start-button"
              onClick={startCamera}
              disabled={!selectedCamera}
            >
              Start →
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

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
                  : 0}
                %
              </span>
              <span className="label">Avg. Form</span>
            </div>
          </div>

          <div className="ai-feedback">
            <h3>🤖 AI Coach Feedback</h3>
            {sessionFeedback ? (
              <p>{sessionFeedback}</p>
            ) : (
              <p>Loading feedback...</p>
            )}
          </div>

          <button className="back-button" onClick={onEnd}>
            ← Back to Exercise Selection
          </button>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="workout-session">
      <div className="header">
        <button className="back-button" onClick={onEnd}>
          ← Back
        </button>
        <h2>{getStatusText()}</h2>
        <button className="end-button" onClick={handleEnd}>
          Finish
        </button>
      </div>

      <div className="main-content">
        <div className="video-avatar-container">
          {/* Camera View */}
          <div className="video-container">
            <video ref={videoRef} autoPlay playsInline style={{ display: 'none' }} />
            <canvas ref={canvasRef} width={640} height={480} />
          </div>
          
          {/* 3D Avatar */}
          <div className="avatar-wrapper">
            <HumanAvatar 
              landmarks={currentLandmarks} 
              width={400} 
              height={480} 
              modelUrl={avatarUrl}
              showSkeleton={true}
              isCalibrated={state === 'ready' || state === 'tracking'}
            />
            <div className="avatar-label">3D Avatar</div>
          </div>
        </div>
        
        {/* Stats Overlay */}
        {/* Stats */}
        <div className="stats-bar">
            <div className="stat-item">
              <span className="stat-value">{repCount}</span>
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
              <span className="stat-label">{phase.toUpperCase()}</span>
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
        )}

        {/* AI Feedback panel */}
          <div className="feedback-panel">
            <h3>🤖 AI Feedback</h3>
            <div className="feedback-list">
              <AnimatePresence>
                {feedbacks.slice(-5).map((fb, i) => (
                  <motion.div
                    key={i}
                    className={`feedback-item ${fb.type}`}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    {fb.message}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

            <div className="tips">
              <h4>💡 Tips</h4>
              {exerciseConfig.formTips.map((tip, i) => (
                <div key={i} className="tip">
                  • {tip}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Debug Panel */}
        <div className="debug-panel">
          <h4>🔧 Debug Logs</h4>
          <div className="debug-info">
            <span>State: <b>{state}</b></span>
            <span>Camera: <b>{selectedCamera ? 'Selected' : 'None'}</b></span>
            <span>WS: <b>{wsRef.current?.readyState === 1 ? 'Open' : 'Closed'}</b></span>
          </div>
          <div className="debug-logs">
            {debugLogs.map((log, i) => (
              <div key={i} className="log-line">{log}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
