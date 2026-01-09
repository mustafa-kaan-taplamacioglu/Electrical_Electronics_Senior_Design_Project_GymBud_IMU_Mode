export type ExerciseType = 
  | 'bicep_curls'
  | 'squats'
  | 'lateral_shoulder_raises'
  | 'tricep_extensions'
  | 'dumbbell_rows'
  | 'dumbbell_shoulder_press';

export type ExerciseConfig = {
  name: string;
  displayName: string;
  description: string;
  primaryJoints: string[];
  repThreshold: {
    up: number;
    down: number;
  };
  formTips: string[];
}

export type RepData = {
  repNumber: number;
  formScore: number;
  issues: string[];
  timestamp: Date;
  duration?: number;
  speedClass?: string;
  speedLabel?: string;
  speedEmoji?: string;
  isValid?: boolean;
  regionalScores?: Record<string, any>;
}

export type SessionData = {
  exercise: ExerciseType;
  reps: RepData[];
  startTime: Date;
  endTime?: Date;
}

export type Landmark = {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export type PoseData = {
  landmarks: Landmark[];
  timestamp: number;
}

export type AIFeedback = {
  message: string;
  type: 'success' | 'warning' | 'error' | 'info';
  timestamp: Date;
}

// ==================== IMU Sensor Types ====================

export type Vector3 = {
  x: number;
  y: number;
  z: number;
}

export type Quaternion = {
  w: number;
  x: number;
  y: number;
  z: number;
}

export type EulerAngles = {
  roll: number;   // X-axis rotation (degrees)
  pitch: number;  // Y-axis rotation (degrees)
  yaw: number;    // Z-axis rotation (degrees)
}

export type UnitVectors = {
  normal: Vector3;    // Sensor's "up" direction (perpendicular to sensor surface)
  tangent: Vector3;   // Forward direction (movement direction)
  binormal: Vector3;   // Lateral direction (right side)
}

export type IMUNodeData = {
  node_id: number;
  timestamp: number;
  accel: Vector3;     // Accelerometer data (g)
  gyro: Vector3;      // Gyroscope data (deg/s)
  quaternion: Quaternion;
  euler: EulerAngles;
  unit_vectors?: UnitVectors;  // Optional: Unit vectors for orientation visualization
}

export type IMUFusedData = {
  type: 'imu_update';
  timestamp: number;
  nodes: {
    left_wrist?: IMUNodeData;
    right_wrist?: IMUNodeData;
    chest?: IMUNodeData;
  };
  raw_data?: {
    left_wrist?: string;  // Raw CSV string: "1,10898,0.0,-0.5144,0.8808,-1.26,-5.39,-0.56"
    right_wrist?: string;
    chest?: string;
  };
}

export type SensorFusionMode = 
  | 'camera_only'      // MediaPipe camera only
  | 'imu_only'         // IMU sensors only
  | 'camera_primary'   // Camera with IMU enhancement
  | 'imu_primary';     // IMU with camera fallback

export type MLMode = 
  | 'usage'           // Usage mode: Basic rep counting + form analysis + data recording (no ML inference)
  | 'train';          // ML Training mode: Data collection + ML training only (no inference)

export type FusedPoseData = {
  // Camera landmarks (MediaPipe)
  landmarks: Landmark[] | null;
  
  // IMU sensor data
  imu: IMUFusedData | null;
  
  // Fusion mode
  mode: SensorFusionMode;
  
  // Timestamps
  cameraTimestamp: number | null;
  imuTimestamp: number | null;
  
  // Connection status
  cameraConnected: boolean;
  imuConnected: boolean;
}
