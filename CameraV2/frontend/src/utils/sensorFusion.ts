/**
 * Sensor Fusion Utilities
 * ========================
 * Combines MediaPipe camera landmarks with IMU sensor data for enhanced
 * avatar animation accuracy.
 * 
 * Fusion Strategy:
 * 1. Camera provides overall body position and major joint locations
 * 2. IMU provides precise orientation for wrists and chest
 * 3. Complementary filter blends both data sources
 */

import * as THREE from 'three';
import type { IMUNodeData, Quaternion, EulerAngles } from '../types';

// ==================== CONSTANTS ====================

// Fusion weights (0 = camera only, 1 = IMU only)
export const FUSION_WEIGHTS = {
  leftWrist: 0.7,   // IMU more trusted for wrist rotation
  rightWrist: 0.7,
  chest: 0.6,        // IMU for chest/spine rotation (optional)
} as const;

// Smoothing factors (higher = more smoothing)
export const SMOOTHING = {
  position: 0.3,    // Camera position smoothing
  rotation: 0.4,    // IMU rotation smoothing
  fusion: 0.35,     // Combined smoothing
} as const;

// IMU to avatar bone axis mapping
// This maps IMU sensor orientation to avatar bone rotation axes
export const IMU_AXIS_MAPPING = {
  leftWrist: {
    roll: 'z',   // IMU roll → bone Z rotation
    pitch: 'x',  // IMU pitch → bone X rotation
    yaw: 'y',    // IMU yaw → bone Y rotation
    signs: { x: -1, y: 1, z: 1 }  // Axis direction corrections
  },
  rightWrist: {
    roll: 'z',
    pitch: 'x',
    yaw: 'y',
    signs: { x: 1, y: -1, z: -1 }  // Mirrored for right side
  },
  chest: {
    roll: 'z',   // IMU roll → spine Z rotation (lean left/right)
    pitch: 'x',  // IMU pitch → spine X rotation (lean forward/back)
    yaw: 'y',    // IMU yaw → spine Y rotation (twist)
    signs: { x: 1, y: 1, z: 1 }  // Chest orientation mapping
  }
} as const;

// ==================== QUATERNION UTILITIES ====================

/**
 * Convert IMU quaternion to THREE.js Quaternion
 */
export function imuQuatToThree(quat: Quaternion): THREE.Quaternion {
  return new THREE.Quaternion(quat.x, quat.y, quat.z, quat.w);
}

/**
 * Convert IMU euler angles (degrees) to THREE.js Euler (radians)
 */
export function imuEulerToThree(euler: EulerAngles, order: THREE.EulerOrder = 'XYZ'): THREE.Euler {
  return new THREE.Euler(
    THREE.MathUtils.degToRad(euler.pitch),
    THREE.MathUtils.degToRad(euler.yaw),
    THREE.MathUtils.degToRad(euler.roll),
    order
  );
}

/**
 * Spherical linear interpolation between two quaternions
 */
export function slerpQuat(
  a: THREE.Quaternion,
  b: THREE.Quaternion,
  t: number
): THREE.Quaternion {
  return a.clone().slerp(b, t);
}

/**
 * Normalize quaternion to unit length
 */
export function normalizeQuat(q: THREE.Quaternion): THREE.Quaternion {
  return q.clone().normalize();
}

// ==================== FUSION ALGORITHMS ====================

export interface CameraJointData {
  position: THREE.Vector3;
  rotation?: THREE.Quaternion;
}

export interface FusedJointData {
  position: THREE.Vector3;
  rotation: THREE.Quaternion;
  confidence: number;
}

/**
 * Complementary filter for fusing camera and IMU data
 * 
 * @param cameraData - Position from camera
 * @param imuData - Orientation from IMU sensor
 * @param weight - Blend weight (0 = camera, 1 = IMU)
 * @param previousFused - Previous fused result for smoothing
 */
export function complementaryFusion(
  cameraData: CameraJointData | null,
  imuData: IMUNodeData | null,
  weight: number,
  previousFused?: FusedJointData
): FusedJointData {
  // Default result
  let result: FusedJointData = {
    position: new THREE.Vector3(0, 0, 0),
    rotation: new THREE.Quaternion(),
    confidence: 0
  };

  // Camera only
  if (cameraData && !imuData) {
    result = {
      position: cameraData.position.clone(),
      rotation: cameraData.rotation?.clone() || new THREE.Quaternion(),
      confidence: 0.6
    };
  }
  // IMU only
  else if (!cameraData && imuData) {
    const imuQuat = imuQuatToThree(imuData.quaternion);
    result = {
      position: previousFused?.position.clone() || new THREE.Vector3(0, 0, 0),
      rotation: imuQuat,
      confidence: 0.7
    };
  }
  // Both sources available - fuse!
  else if (cameraData && imuData) {
    const cameraRot = cameraData.rotation || new THREE.Quaternion();
    const imuRot = imuQuatToThree(imuData.quaternion);
    
    // Blend rotations using SLERP
    const fusedRot = slerpQuat(cameraRot, imuRot, weight);
    
    result = {
      position: cameraData.position.clone(),
      rotation: fusedRot,
      confidence: 0.95
    };
  }

  // Apply smoothing if we have previous data
  if (previousFused) {
    result.position.lerp(previousFused.position, SMOOTHING.fusion);
    result.rotation = slerpQuat(previousFused.rotation, result.rotation, 1 - SMOOTHING.fusion);
  }

  return result;
}

// ==================== BONE ROTATION HELPERS ====================

/**
 * Apply IMU euler angles to a bone with proper axis mapping
 */
export function applyIMUToBone(
  bone: THREE.Bone,
  imuData: IMUNodeData,
  nodeType: 'leftWrist' | 'rightWrist' | 'chest',
  smoothingFactor: number = SMOOTHING.rotation
): void {
  const mapping = IMU_AXIS_MAPPING[nodeType];
  const euler = imuData.euler;
  
  // Get target rotation from IMU
  const targetRot = new THREE.Euler(
    THREE.MathUtils.degToRad(euler.pitch * mapping.signs.x),
    THREE.MathUtils.degToRad(euler.yaw * mapping.signs.y),
    THREE.MathUtils.degToRad(euler.roll * mapping.signs.z),
    'XYZ'
  );
  
  const targetQuat = new THREE.Quaternion().setFromEuler(targetRot);
  
  // Smooth interpolation
  bone.quaternion.slerp(targetQuat, 1 - smoothingFactor);
}

/**
 * Blend camera-based bone rotation with IMU data
 */
export function blendBoneRotation(
  bone: THREE.Bone,
  cameraRotation: THREE.Euler,
  imuData: IMUNodeData | null,
  nodeType: 'leftWrist' | 'rightWrist' | 'chest',
  imuWeight: number = FUSION_WEIGHTS[nodeType]
): void {
  const cameraQuat = new THREE.Quaternion().setFromEuler(cameraRotation);
  
  if (imuData) {
    const mapping = IMU_AXIS_MAPPING[nodeType];
    const euler = imuData.euler;
    
    const imuRot = new THREE.Euler(
      THREE.MathUtils.degToRad(euler.pitch * mapping.signs.x),
      THREE.MathUtils.degToRad(euler.yaw * mapping.signs.y),
      THREE.MathUtils.degToRad(euler.roll * mapping.signs.z),
      'XYZ'
    );
    const imuQuat = new THREE.Quaternion().setFromEuler(imuRot);
    
    // Blend camera and IMU rotations
    const blendedQuat = slerpQuat(cameraQuat, imuQuat, imuWeight);
    
    // Smooth application
    bone.quaternion.slerp(blendedQuat, 1 - SMOOTHING.fusion);
  } else {
    // Camera only
    bone.quaternion.slerp(cameraQuat, 1 - SMOOTHING.position);
  }
}

// ==================== CALIBRATION ====================

export interface CalibrationData {
  leftWristOffset: THREE.Quaternion;
  rightWristOffset: THREE.Quaternion;
  chestOffset: THREE.Quaternion;
}

/**
 * Calculate calibration offsets to align IMU with avatar rest pose
 */
export function calculateCalibrationOffsets(
  leftWrist: IMUNodeData | null,
  rightWrist: IMUNodeData | null,
  chest: IMUNodeData | null = null
): CalibrationData {
  const identity = new THREE.Quaternion();
  
  return {
    leftWristOffset: leftWrist 
      ? imuQuatToThree(leftWrist.quaternion).invert()
      : identity.clone(),
    rightWristOffset: rightWrist
      ? imuQuatToThree(rightWrist.quaternion).invert()
      : identity.clone(),
    chestOffset: chest
      ? imuQuatToThree(chest.quaternion).invert()
      : identity.clone(),
  };
}

/**
 * Apply calibration offset to IMU quaternion
 */
export function applyCalibratedRotation(
  imuQuat: THREE.Quaternion,
  offset: THREE.Quaternion
): THREE.Quaternion {
  return offset.clone().multiply(imuQuat);
}

// ==================== DEBUG HELPERS ====================

/**
 * Create a debug visualization object for IMU orientation
 */
export function createIMUDebugAxes(size: number = 0.2): THREE.AxesHelper {
  return new THREE.AxesHelper(size);
}

/**
 * Update debug axes to match IMU orientation
 */
export function updateDebugAxes(
  axes: THREE.AxesHelper,
  imuData: IMUNodeData
): void {
  const quat = imuQuatToThree(imuData.quaternion);
  axes.quaternion.copy(quat);
}

/**
 * Get IMU data as formatted string for debugging
 */
export function formatIMUData(data: IMUNodeData | null): string {
  if (!data) return 'No data';
  
  const e = data.euler;
  return `R:${e.roll.toFixed(1)}° P:${e.pitch.toFixed(1)}° Y:${e.yaw.toFixed(1)}°`;
}

