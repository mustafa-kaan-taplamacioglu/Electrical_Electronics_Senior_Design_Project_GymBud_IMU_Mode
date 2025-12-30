import { useRef, useEffect, Suspense, useMemo, useState } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, useGLTF, Environment, Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { TextureLoader } from 'three';
import * as SkeletonUtils from 'three/examples/jsm/utils/SkeletonUtils.js';
import type { IMUNodeData, SensorFusionMode } from '../types';
import { 
  imuQuatToThree, 
  imuEulerToThree, 
  FUSION_WEIGHTS,
  SMOOTHING,
  IMU_AXIS_MAPPING 
} from '../utils/sensorFusion';

interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

// IMU sensor data interface
interface IMUData {
  leftWrist?: IMUNodeData;
  rightWrist?: IMUNodeData;
  chest?: IMUNodeData;
}

interface Props {
  landmarks: Landmark[] | null;
  width?: number;
  height?: number;
  modelUrl?: string;
  showSkeleton?: boolean;
  backgroundUrl?: string;
  isCalibrated?: boolean;
  // NEW: IMU sensor data
  imuData?: IMUData | null;
  // NEW: Sensor fusion mode
  fusionMode?: SensorFusionMode;
  // NEW: Show IMU debug info
  showIMUDebug?: boolean;
}

// MediaPipe landmark indices
const MP = {
  NOSE: 0,
  LEFT_EYE: 2,
  RIGHT_EYE: 5,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

// Skeleton connections for visualization
const BONE_CONNECTIONS = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso
  [23, 24], // hips
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
];

// Skeleton visualization component - Dots attached to avatar body
function Skeleton({ landmarks }: { landmarks: Landmark[] | null }) {
  const groupRef = useRef<THREE.Group>(null);
  const jointsRef = useRef<THREE.Mesh[]>([]);
  const bonesRef = useRef<THREE.Line[]>([]);

  // Convert MediaPipe coords to 3D space - aligned with avatar body
  const toVec3 = (lm: Landmark): THREE.Vector3 => {
    // MediaPipe: x=0-1 (left to right), y=0-1 (top to bottom), z=depth
    // Avatar: centered at x=0, y=0 is ground, height ~1.7, facing z+
    
    // Mirror X and scale to avatar width (~0.5 on each side)
    const x = -(lm.x - 0.5) * 1.0;
    
    // Flip Y and map to avatar height (ground at -0.02, head at ~1.65)
    // MediaPipe y: 0=top of frame, 1=bottom
    // Avatar y: 0=ground, 1.65=head
    const y = (1 - lm.y) * 1.5 + 0.15;
    
    // Z: Position slightly in front of avatar body surface
    const z = 0.12 - lm.z * 0.3;
    
    return new THREE.Vector3(x, y, z);
  };

  useFrame(() => {
    if (!landmarks || !groupRef.current) return;

    // Update joint positions
    jointsRef.current.forEach((joint, i) => {
      if (landmarks[i] && landmarks[i].visibility && landmarks[i].visibility! > 0.5) {
        const pos = toVec3(landmarks[i]);
        joint.position.lerp(pos, 0.3);
        joint.visible = true;
      } else {
        joint.visible = false;
      }
    });

    // Update bone positions
    BONE_CONNECTIONS.forEach((conn, i) => {
      const bone = bonesRef.current[i];
      if (!bone) return;

      const start = jointsRef.current[conn[0]];
      const end = jointsRef.current[conn[1]];

      if (start?.visible && end?.visible) {
        const positions = bone.geometry.attributes.position as THREE.BufferAttribute;
        positions.setXYZ(0, start.position.x, start.position.y, start.position.z);
        positions.setXYZ(1, end.position.x, end.position.y, end.position.z);
        positions.needsUpdate = true;
        bone.visible = true;
      } else {
        bone.visible = false;
      }
    });
  });

  return (
    <group ref={groupRef}>
      {/* Joints - Small dots on avatar body */}
      {Array.from({ length: 33 }).map((_, i) => (
        <mesh
          key={`joint-${i}`}
          ref={(el) => { if (el) jointsRef.current[i] = el; }}
          visible={false}
        >
          <sphereGeometry args={[0.018, 8, 8]} />
          <meshStandardMaterial 
            color="#00ff88" 
            emissive="#00ff88"
            emissiveIntensity={0.8}
            transparent
            opacity={0.9}
          />
        </mesh>
      ))}

      {/* Bones - Thin lines connecting joints */}
      {BONE_CONNECTIONS.map((_, i) => (
        <line
          key={`bone-${i}`}
          ref={(el) => { if (el) bonesRef.current[i] = el as unknown as THREE.Line; }}
          visible={false}
        >
          <bufferGeometry attach="geometry">
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array(6)}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ccff" linewidth={1} transparent opacity={0.6} />
        </line>
      ))}
    </group>
  );
}

// IMU Sensor Marker - Colored sphere with label attached to bone position
function IMUSensorMarker({ 
  position, 
  label, 
  color = '#ff3333',
  isActive = false 
}: { 
  position: [number, number, number]; 
  label: string; 
  color?: string;
  isActive?: boolean;
}) {
  const markerRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  
  // Pulse animation when active
  useFrame((state) => {
    if (markerRef.current && isActive) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 5) * 0.3;
      markerRef.current.scale.setScalar(scale);
    }
    // Make marker always face camera
    if (meshRef.current) {
      meshRef.current.lookAt(state.camera.position);
    }
  });
  
  return (
    <group ref={markerRef} position={position}>
      {/* Larger colored sphere marker */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.045, 16, 16]} />
        <meshStandardMaterial 
          color={color} 
          emissive={color}
          emissiveIntensity={isActive ? 1.0 : 0.5}
          transparent
          opacity={0.9}
        />
      </mesh>
      {/* Outer ring for visibility */}
      <mesh>
        <ringGeometry args={[0.05, 0.065, 32]} />
        <meshBasicMaterial color={color} side={THREE.DoubleSide} transparent opacity={0.7} />
      </mesh>
      {/* Label */}
      <Billboard follow={true} lockX={false} lockY={false} lockZ={false}>
        <Text
          position={[0, 0.09, 0]}
          fontSize={0.06}
          color="#ffffff"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.006}
          outlineColor={color}
        >
          {label}
        </Text>
      </Billboard>
    </group>
  );
}

// IMU Sensor Markers Component - Attaches to avatar bones
function IMUSensorMarkers({ 
  bonesRef, 
  imuData,
  show = true 
}: { 
  bonesRef: React.RefObject<{ [key: string]: THREE.Bone }>;
  imuData?: { leftWrist?: any; rightWrist?: any; chest?: any } | null;
  show?: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const initializedRef = useRef(false);
  
  // Default positions (T-pose avatar approximate positions)
  const [lwPos, setLwPos] = useState<[number, number, number]>([-0.55, 0.95, 0.1]);
  const [rwPos, setRwPos] = useState<[number, number, number]>([0.55, 0.95, 0.1]);
  const [chPos, setChPos] = useState<[number, number, number]>([0, 1.25, 0.15]);
  
  useFrame(() => {
    if (!show) return;
    
    const bones = bonesRef.current;
    if (!bones || Object.keys(bones).length === 0) {
      // Use default positions if no bones available
      return;
    }
    
    // Try to get Left Wrist position
    const lwBones = ['LeftHand', 'LeftForeArm', 'LeftArm'];
    for (const name of lwBones) {
      if (bones[name]) {
        const pos = new THREE.Vector3();
        bones[name].getWorldPosition(pos);
        if (name === 'LeftForeArm') pos.x -= 0.08;
        if (name === 'LeftArm') pos.x -= 0.2;
        pos.z += 0.05; // Forward offset
        setLwPos([pos.x, pos.y, pos.z]);
        break;
      }
    }
    
    // Try to get Right Wrist position
    const rwBones = ['RightHand', 'RightForeArm', 'RightArm'];
    for (const name of rwBones) {
      if (bones[name]) {
        const pos = new THREE.Vector3();
        bones[name].getWorldPosition(pos);
        if (name === 'RightForeArm') pos.x += 0.08;
        if (name === 'RightArm') pos.x += 0.2;
        pos.z += 0.05; // Forward offset
        setRwPos([pos.x, pos.y, pos.z]);
        break;
      }
    }
    
    // Try to get Chest position (center of chest/upper spine)
    const chBones = ['Spine2', 'Spine1', 'Spine', 'Chest'];
    for (const name of chBones) {
      if (bones[name]) {
        const pos = new THREE.Vector3();
        bones[name].getWorldPosition(pos);
        pos.z += 0.1; // Forward offset
        setChPos([pos.x, pos.y, pos.z]);
        break;
      }
    }
    
    initializedRef.current = true;
  });
  
  if (!show) return null;
  
  return (
    <group ref={groupRef}>
      <group position={lwPos}>
        <IMUSensorMarker 
          position={[0, 0, 0]} 
          label="LW" 
          color="#3b82f6"
          isActive={!!imuData?.leftWrist}
        />
      </group>
      <group position={rwPos}>
        <IMUSensorMarker 
          position={[0, 0, 0]} 
          label="RW" 
          color="#a855f7"
          isActive={!!imuData?.rightWrist}
        />
      </group>
      <group position={chPos}>
        <IMUSensorMarker 
          position={[0, 0, 0]} 
          label="CH" 
          color="#f59e0b"
          isActive={!!imuData?.chest}
        />
      </group>
    </group>
  );
}

// Ready Player Me Avatar with bone animation + IMU sensor fusion
function RPMAvatar({ 
  landmarks, 
  modelUrl,
  isCalibrated = false,
  imuData = null,
  fusionMode = 'camera_primary'
}: { 
  landmarks: Landmark[] | null;
  modelUrl: string;
  isCalibrated?: boolean;
  imuData?: IMUData | null;
  fusionMode?: SensorFusionMode;
}) {
  const { scene } = useGLTF(modelUrl);
  const groupRef = useRef<THREE.Group>(null);
  const bonesRef = useRef<{ [key: string]: THREE.Bone }>({});
  const modelRef = useRef<THREE.Object3D | null>(null);
  const frameCountRef = useRef(0);
  
  // Calibration: store rest pose
  const restPoseRef = useRef<{ [key: number]: { x: number; y: number; z: number } } | null>(null);
  const calibrationFramesRef = useRef<Landmark[][]>([]);
  const isAvatarCalibratedRef = useRef(false);
  
  // Smoothing: store previous values for low-pass filter
  const smoothedLandmarksRef = useRef<{ [key: number]: { x: number; y: number; z: number } }>({});
  const LANDMARK_SMOOTHING = 0.4; // 0 = no smoothing, 1 = frozen (0.3-0.5 is good)
  
  // IMU calibration offsets (captured during calibration)
  const imuCalibrationRef = useRef<{
    leftWrist: THREE.Quaternion | null;
    rightWrist: THREE.Quaternion | null;
    chest: THREE.Quaternion | null;
  }>({
    leftWrist: null,
    rightWrist: null,
    chest: null
  });
  
  // Track if IMU is calibrated
  const isIMUCalibratedRef = useRef(false);

  // Clone the scene properly for skinned meshes
  const clonedModel = useMemo(() => {
    const clone = SkeletonUtils.clone(scene);
    return clone;
  }, [scene]);

  useEffect(() => {
    if (!clonedModel) {
      console.log('❌ No cloned model');
      return;
    }
    
    console.log('✅ Cloned model ready, traversing for bones...');
    modelRef.current = clonedModel;
    
    // Find and store bone references
    let boneCount = 0;
    let meshCount = 0;
    
    clonedModel.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        meshCount++;
        child.castShadow = true;
        child.receiveShadow = true;
      }
      if ((child as THREE.Bone).isBone) {
        boneCount++;
        bonesRef.current[child.name] = child as THREE.Bone;
      }
    });

    console.log(`🦴 RPM Avatar - Found ${boneCount} bones, ${meshCount} meshes`);
    console.log('🦴 Bone names:', Object.keys(bonesRef.current).slice(0, 10), '...');
    
    // Check for specific bones we need
    const requiredBones = ['LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'Spine', 'Hips'];
    const foundRequired = requiredBones.filter(b => bonesRef.current[b]);
    console.log(`🦴 Required bones found: ${foundRequired.length}/${requiredBones.length}:`, foundRequired);
  }, [clonedModel]);

  useFrame(() => {
    if (!landmarks || Object.keys(bonesRef.current).length === 0) return;
    
    frameCountRef.current++;
    
    const bones = bonesRef.current;
    
    // === CALIBRATION PHASE ===
    // When backend calibration is done, capture rest pose for avatar
    if (isCalibrated && !isAvatarCalibratedRef.current) {
      calibrationFramesRef.current.push([...landmarks]);
      
      if (calibrationFramesRef.current.length >= 20) {
        // Average the calibration frames to get REST POSE
        const avgPose: { [key: number]: { x: number; y: number; z: number } } = {};
        
        for (let i = 0; i < 33; i++) {
          let sumX = 0, sumY = 0, sumZ = 0, count = 0;
          calibrationFramesRef.current.forEach(frame => {
            if (frame[i] && frame[i].visibility && frame[i].visibility! > 0.5) {
              sumX += frame[i].x;
              sumY += frame[i].y;
              sumZ += frame[i].z;
              count++;
            }
          });
          if (count > 0) {
            avgPose[i] = { x: sumX / count, y: sumY / count, z: sumZ / count };
          }
        }
        
        restPoseRef.current = avgPose;
        isAvatarCalibratedRef.current = true;
        
        // === ALSO CALIBRATE IMU SENSORS ===
        // Capture current IMU orientations as "rest pose" offsets
        if (imuData) {
          if (imuData.leftWrist) {
            const quat = imuQuatToThree(imuData.leftWrist.quaternion);
            imuCalibrationRef.current.leftWrist = quat.invert();
          }
          if (imuData.rightWrist) {
            const quat = imuQuatToThree(imuData.rightWrist.quaternion);
            imuCalibrationRef.current.rightWrist = quat.invert();
          }
          if (imuData.chest) {
            const quat = imuQuatToThree(imuData.chest.quaternion);
            imuCalibrationRef.current.chest = quat.invert();
          }
          isIMUCalibratedRef.current = true;
          console.log('🎯 IMU sensors calibrated!');
        }
        
        // Log calibration data
        console.log('🎯 Avatar calibrated! Rest pose captured.');
        console.log('   Rest Left Shoulder:', avgPose[MP.LEFT_SHOULDER]);
        console.log('   Rest Left Elbow:', avgPose[MP.LEFT_ELBOW]);
        console.log('   Rest Nose:', avgPose[MP.NOSE]);
      }
      return; // Don't animate during calibration
    }
    
    // Don't animate if not calibrated
    if (!isAvatarCalibratedRef.current || !restPoseRef.current) return;
    
    const restPose = restPoseRef.current;
    
    // Debug log every 120 frames
    if (frameCountRef.current % 120 === 0) {
      console.log('🎭 Avatar animating with calibration data');
    }
    
    // Helper to get landmark with smoothing
    const getRawLm = (idx: number): { x: number; y: number; z: number } | null => {
      const lm = landmarks[idx];
      if (!lm || (lm.visibility !== undefined && lm.visibility < 0.5)) return null;
      
      let raw = { x: lm.x, y: lm.y, z: lm.z };
      
      // Apply smoothing
      const prev = smoothedLandmarksRef.current[idx];
      if (prev) {
        raw.x = prev.x * LANDMARK_SMOOTHING + raw.x * (1 - LANDMARK_SMOOTHING);
        raw.y = prev.y * LANDMARK_SMOOTHING + raw.y * (1 - LANDMARK_SMOOTHING);
        raw.z = prev.z * LANDMARK_SMOOTHING + raw.z * (1 - LANDMARK_SMOOTHING);
      }
      smoothedLandmarksRef.current[idx] = { ...raw };
      
      return raw;
    };
    
    // === IMU SENSOR FUSION HELPERS ===
    
    /**
     * Apply IMU orientation to a bone with calibration offset
     */
    const applyIMUToBone = (
      bone: THREE.Bone,
      imuNode: IMUNodeData | undefined,
      nodeType: 'leftWrist' | 'rightWrist' | 'chest',
      calibrationOffset: THREE.Quaternion | null
    ) => {
      if (!imuNode) return;
      
      const mapping = IMU_AXIS_MAPPING[nodeType];
      const euler = imuNode.euler;
      
      // Convert IMU euler to THREE.js euler (with axis mapping and sign corrections)
      const imuRot = new THREE.Euler(
        THREE.MathUtils.degToRad(euler.pitch * mapping.signs.x),
        THREE.MathUtils.degToRad(euler.yaw * mapping.signs.y),
        THREE.MathUtils.degToRad(euler.roll * mapping.signs.z),
        'XYZ'
      );
      
      let targetQuat = new THREE.Quaternion().setFromEuler(imuRot);
      
      // Apply calibration offset if available
      if (calibrationOffset) {
        targetQuat = calibrationOffset.clone().multiply(targetQuat);
      }
      
      // Smooth interpolation to current bone rotation
      bone.quaternion.slerp(targetQuat, 1 - SMOOTHING.rotation);
    };
    
    /**
     * Blend camera-derived rotation with IMU orientation
     */
    const blendWithIMU = (
      bone: THREE.Bone,
      cameraRotX: number,
      cameraRotY: number,
      cameraRotZ: number,
      imuNode: IMUNodeData | undefined,
      nodeType: 'leftWrist' | 'rightWrist' | 'chest',
      calibrationOffset: THREE.Quaternion | null
    ) => {
      const cameraQuat = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(cameraRotX, cameraRotY, cameraRotZ, 'XYZ')
      );
      
      if (imuNode && (fusionMode === 'camera_primary' || fusionMode === 'imu_primary')) {
        const mapping = IMU_AXIS_MAPPING[nodeType];
        const euler = imuNode.euler;
        
        const imuRot = new THREE.Euler(
          THREE.MathUtils.degToRad(euler.pitch * mapping.signs.x),
          THREE.MathUtils.degToRad(euler.yaw * mapping.signs.y),
          THREE.MathUtils.degToRad(euler.roll * mapping.signs.z),
          'XYZ'
        );
        
        let imuQuat = new THREE.Quaternion().setFromEuler(imuRot);
        
        // Apply calibration offset
        if (calibrationOffset) {
          imuQuat = calibrationOffset.clone().multiply(imuQuat);
        }
        
        // Blend based on fusion mode
        const imuWeight = fusionMode === 'imu_primary' 
          ? 0.8 
          : FUSION_WEIGHTS[nodeType];
        
        const blendedQuat = cameraQuat.clone().slerp(imuQuat, imuWeight);
        bone.quaternion.slerp(blendedQuat, 0.2);
      } else {
        // Camera only
        bone.quaternion.slerp(cameraQuat, 0.2);
      }
    };
    
    // Get DELTA from rest pose (this is the key!)
    const getDelta = (idx: number): { dx: number; dy: number; dz: number } | null => {
      const current = getRawLm(idx);
      const rest = restPose[idx];
      if (!current || !rest) return null;
      
      return {
        dx: current.x - rest.x,  // Positive = moved right
        dy: current.y - rest.y,  // Positive = moved down
        dz: current.z - rest.z,  // Positive = moved away from camera
      };
    };

    // Helper to calculate angle between 3 points
    const calcAngle = (
      a: { x: number; y: number } | null, 
      b: { x: number; y: number } | null, 
      c: { x: number; y: number } | null
    ): number | null => {
      if (!a || !b || !c) return null;
      const ba = { x: a.x - b.x, y: a.y - b.y };
      const bc = { x: c.x - b.x, y: c.y - b.y };
      const dot = ba.x * bc.x + ba.y * bc.y;
      const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
      const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
      if (magBA === 0 || magBC === 0) return null;
      const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
      return Math.acos(cosAngle);
    };

    const lerp = (current: number, target: number, alpha: number) => {
      return current + (target - current) * alpha;
    };

    // === ARM ANIMATION USING DELTA FROM REST POSE ===
    // Calculate how much each joint has MOVED from calibration position
    
    const deltaLeftElbow = getDelta(MP.LEFT_ELBOW);
    const deltaRightElbow = getDelta(MP.RIGHT_ELBOW);
    const deltaLeftWrist = getDelta(MP.LEFT_WRIST);
    const deltaRightWrist = getDelta(MP.RIGHT_WRIST);
    const deltaLeftShoulder = getDelta(MP.LEFT_SHOULDER);
    const deltaRightShoulder = getDelta(MP.RIGHT_SHOULDER);
    
    // Sensitivity multipliers
    const ARM_SENS = 3.0;
    const FOREARM_SENS = 2.5;
    
    // User's LEFT arm → Avatar's RIGHT arm (mirror)
    if (deltaLeftElbow && bones['RightArm']) {
      // Elbow moved down (dy > 0) → arm rotates down
      // Elbow moved right (dx > 0) → arm rotates forward
      const armRotZ = -deltaLeftElbow.dy * ARM_SENS;
      const armRotX = deltaLeftElbow.dx * ARM_SENS * 0.5;
      
      bones['RightArm'].rotation.z = lerp(bones['RightArm'].rotation.z, armRotZ, 0.2);
      bones['RightArm'].rotation.x = lerp(bones['RightArm'].rotation.x, armRotX, 0.2);
    }
    
    // User's LEFT forearm → Avatar's RIGHT forearm
    // Enhanced with IMU sensor data from RIGHT WRIST (mirrored)
    if (deltaLeftWrist && deltaLeftElbow && bones['RightForeArm']) {
      // Camera-based: Wrist moved relative to elbow = forearm bend
      const bendY = (deltaLeftWrist.dy - deltaLeftElbow.dy) * FOREARM_SENS;
      
      if (imuData?.rightWrist && fusionMode !== 'camera_only') {
        // IMU FUSION: Use right wrist IMU for precise forearm rotation
        blendWithIMU(
          bones['RightForeArm'],
          0,  // camera X
          -bendY,  // camera Y
          0,  // camera Z
          imuData.rightWrist,
          'rightWrist',
          imuCalibrationRef.current.rightWrist
        );
      } else {
        // Camera only
        bones['RightForeArm'].rotation.y = lerp(bones['RightForeArm'].rotation.y, -bendY, 0.2);
      }
    }

    // User's RIGHT arm → Avatar's LEFT arm (mirror)
    if (deltaRightElbow && bones['LeftArm']) {
      const armRotZ = deltaRightElbow.dy * ARM_SENS;
      const armRotX = -deltaRightElbow.dx * ARM_SENS * 0.5;
      
      bones['LeftArm'].rotation.z = lerp(bones['LeftArm'].rotation.z, armRotZ, 0.2);
      bones['LeftArm'].rotation.x = lerp(bones['LeftArm'].rotation.x, armRotX, 0.2);
    }
    
    // User's RIGHT forearm → Avatar's LEFT forearm
    // Enhanced with IMU sensor data from LEFT WRIST (mirrored)
    if (deltaRightWrist && deltaRightElbow && bones['LeftForeArm']) {
      const bendY = (deltaRightWrist.dy - deltaRightElbow.dy) * FOREARM_SENS;
      
      if (imuData?.leftWrist && fusionMode !== 'camera_only') {
        // IMU FUSION: Use left wrist IMU for precise forearm rotation
        blendWithIMU(
          bones['LeftForeArm'],
          0,  // camera X
          bendY,  // camera Y
          0,  // camera Z
          imuData.leftWrist,
          'leftWrist',
          imuCalibrationRef.current.leftWrist
        );
      } else {
        // Camera only
        bones['LeftForeArm'].rotation.y = lerp(bones['LeftForeArm'].rotation.y, bendY, 0.2);
      }
    }
    
    // Get raw landmarks for angle calculations
    const leftShoulder = getRawLm(MP.LEFT_SHOULDER);
    const leftElbow = getRawLm(MP.LEFT_ELBOW);
    const leftWrist = getRawLm(MP.LEFT_WRIST);
    const rightShoulder = getRawLm(MP.RIGHT_SHOULDER);
    const rightElbow = getRawLm(MP.RIGHT_ELBOW);
    const rightWrist = getRawLm(MP.RIGHT_WRIST);
    const leftHip = getRawLm(MP.LEFT_HIP);
    const rightHip = getRawLm(MP.RIGHT_HIP);

    // === LEG ANIMATION ===
    // For upper body exercises, legs stay static
    // Only hips rotation affects leg orientation (turning left/right)
    // Legs follow the hips rotation but don't bend

    // === SPINE USING DELTA + CHEST IMU (if available) ===
    const deltaLeftHip = getDelta(MP.LEFT_HIP);
    const deltaRightHip = getDelta(MP.RIGHT_HIP);
    
    if (deltaLeftShoulder && deltaRightShoulder && bones['Spine']) {
      const SPINE_SENS = 2.0;
      
      // Camera-based calculations
      const shoulderLean = ((deltaLeftShoulder.dx + deltaRightShoulder.dx) / 2) * SPINE_SENS;
      const forwardLean = ((deltaLeftShoulder.dz + deltaRightShoulder.dz) / 2) * SPINE_SENS;
      const twist = (deltaRightShoulder.dz - deltaLeftShoulder.dz) * SPINE_SENS;
      
      // Chest IMU → Avatar's spine (optional, if chest IMU data available)
      if (imuData?.chest && fusionMode !== 'camera_only') {
        // Use chest IMU for spine rotation (blend with camera)
        blendWithIMU(
          bones['Spine'],
          forwardLean,  // camera X (forward/back lean)
          -shoulderLean,  // camera Y (left/right lean)
          twist,  // camera Z (twist)
          imuData.chest,
          'chest',
          imuCalibrationRef.current.chest
        );
      } else {
        // Camera-based spine rotation only
        bones['Spine'].rotation.z = lerp(bones['Spine'].rotation.z, -shoulderLean, 0.15);
        bones['Spine'].rotation.x = lerp(bones['Spine'].rotation.x, forwardLean, 0.15);
        
        if (bones['Spine1']) {
          bones['Spine1'].rotation.y = lerp(bones['Spine1'].rotation.y, twist, 0.15);
        }
      }
    }
    
    // === IMU-ONLY MODE: Use IMU data with a neutral base pose ===
    if (fusionMode === 'imu_only' && imuData) {
      // First, set a neutral standing pose for all bones
      // This prevents the avatar from collapsing
      
      // Reset spine to upright
      if (bones['Spine']) {
        bones['Spine'].quaternion.slerp(new THREE.Quaternion(), 0.1);
      }
      if (bones['Spine1']) {
        bones['Spine1'].quaternion.slerp(new THREE.Quaternion(), 0.1);
      }
      if (bones['Spine2']) {
        bones['Spine2'].quaternion.slerp(new THREE.Quaternion(), 0.1);
      }
      
      // Set arms to a relaxed position (slightly down from T-pose)
      if (bones['LeftArm']) {
        const relaxedLeftArm = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(0, 0, Math.PI * 0.4, 'XYZ') // Arm slightly down
        );
        bones['LeftArm'].quaternion.slerp(relaxedLeftArm, 0.1);
      }
      if (bones['RightArm']) {
        const relaxedRightArm = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(0, 0, -Math.PI * 0.4, 'XYZ') // Arm slightly down
        );
        bones['RightArm'].quaternion.slerp(relaxedRightArm, 0.1);
      }
      
      // Now apply IMU rotations ONLY to the tracked limbs
      // Left wrist IMU → Avatar's left forearm (relative rotation)
      if (imuData.leftWrist && bones['LeftForeArm']) {
        const euler = imuData.leftWrist.euler;
        // Only apply roll (forearm rotation) - limit pitch and yaw
        const forearmRot = new THREE.Euler(
          THREE.MathUtils.degToRad(euler.pitch * 0.3), // Reduced pitch
          THREE.MathUtils.degToRad(euler.yaw * 0.2),   // Minimal yaw
          THREE.MathUtils.degToRad(euler.roll * 0.5),  // Main rotation
          'XYZ'
        );
        const targetQuat = new THREE.Quaternion().setFromEuler(forearmRot);
        bones['LeftForeArm'].quaternion.slerp(targetQuat, 0.15);
      }
      
      // Right wrist IMU → Avatar's right forearm
      if (imuData.rightWrist && bones['RightForeArm']) {
        const euler = imuData.rightWrist.euler;
        const forearmRot = new THREE.Euler(
          THREE.MathUtils.degToRad(euler.pitch * 0.3),
          THREE.MathUtils.degToRad(euler.yaw * 0.2),
          THREE.MathUtils.degToRad(-euler.roll * 0.5), // Inverted for right side
          'XYZ'
        );
        const targetQuat = new THREE.Quaternion().setFromEuler(forearmRot);
        bones['RightForeArm'].quaternion.slerp(targetQuat, 0.15);
      }
      
    }

    // === HEAD USING DELTA ===
    const deltaNose = getDelta(MP.NOSE);
    const deltaLeftEar = getDelta(MP.LEFT_EAR);
    const deltaRightEar = getDelta(MP.RIGHT_EAR);
    
    if (deltaNose && bones['Head']) {
      const HEAD_SENS = 4.0;
      
      // Head turn: nose moved left (dx < 0) → avatar looks right (mirrored)
      const headTurnY = -deltaNose.dx * HEAD_SENS;
      bones['Head'].rotation.y = lerp(bones['Head'].rotation.y, headTurnY, 0.15);
      
      // Head tilt: nose moved down (dy > 0) → avatar looks down
      // Use Z for up/down tilt since face depth changes when tilting
      const headTiltX = deltaNose.dz * HEAD_SENS * 0.5;
      bones['Head'].rotation.x = lerp(bones['Head'].rotation.x, headTiltX, 0.15);
      
      // Debug
      if (frameCountRef.current % 60 === 0) {
        console.log(`👤 Head delta: dx=${deltaNose.dx.toFixed(3)}, dy=${deltaNose.dy.toFixed(3)}, dz=${deltaNose.dz.toFixed(3)}`);
      }
    }

    // === HIPS USING DELTA ===
    if (deltaLeftHip && deltaRightHip && bones['Hips']) {
      const HIP_SENS = 1.5;
      
      // Hip twist: one hip moved forward more than the other
      const hipTwist = (deltaLeftHip.dz - deltaRightHip.dz) * HIP_SENS;
      bones['Hips'].rotation.y = lerp(bones['Hips'].rotation.y, hipTwist, 0.15);
      
      // Weight shift: one hip dropped more than the other
      const hipLean = (deltaRightHip.dy - deltaLeftHip.dy) * HIP_SENS;
      bones['Hips'].rotation.z = lerp(bones['Hips'].rotation.z, hipLean, 0.15);
    }
  });

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      <primitive object={clonedModel} />
      {/* IMU Sensor Markers */}
      <IMUSensorMarkers 
        bonesRef={bonesRef} 
        imuData={imuData}
        show={fusionMode !== 'camera_only' && !!imuData}
      />
    </group>
  );
}

// Static Avatar (before calibration - just T-pose)
function StaticAvatar({ modelUrl }: { modelUrl: string }) {
  const { scene } = useGLTF(modelUrl);
  
  const clonedModel = useMemo(() => {
    return SkeletonUtils.clone(scene);
  }, [scene]);

  useEffect(() => {
    clonedModel.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });
  }, [clonedModel]);

  return (
    <group position={[0, 0, 0]}>
      <primitive object={clonedModel} />
    </group>
  );
}

// Background component
function GymBackground({ url }: { url?: string }) {
  const groundY = -0.02;
  
  if (!url) {
    return (
      <>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, groundY, 0]} receiveShadow>
          <circleGeometry args={[2.5, 32]} />
          <meshStandardMaterial color="#1a1a24" />
        </mesh>
        <gridHelper args={[5, 12, '#3a3a4a', '#2a2a3a']} position={[0, groundY + 0.01, 0]} />
      </>
    );
  }

  const texture = useLoader(TextureLoader, url);
  
  return (
    <>
      <mesh position={[0, 1.2, -2.5]} receiveShadow>
        <planeGeometry args={[6, 4]} />
        <meshBasicMaterial map={texture} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, groundY, 0]} receiveShadow>
        <planeGeometry args={[5, 5]} />
        <meshStandardMaterial color="#1a1a24" opacity={0.9} transparent />
      </mesh>
    </>
  );
}

// Scene component
function Scene({ 
  landmarks, 
  modelUrl, 
  showSkeleton = false,
  backgroundUrl,
  isCalibrated = false,
  imuData = null,
  fusionMode = 'camera_primary'
}: { 
  landmarks: Landmark[] | null; 
  modelUrl?: string;
  showSkeleton?: boolean;
  backgroundUrl?: string;
  isCalibrated?: boolean;
  imuData?: IMUData | null;
  fusionMode?: SensorFusionMode;
}) {
  return (
    <>
      <ambientLight intensity={0.8} />
      <directionalLight position={[5, 5, 5]} intensity={1.2} castShadow />
      <directionalLight position={[-3, 3, -3]} intensity={0.5} />
      <pointLight position={[0, 3, 2]} intensity={0.8} color="#ffffff" />

      <GymBackground url={backgroundUrl} />

      {/* Only show avatar after calibration */}
      {modelUrl && isCalibrated ? (
        <Suspense fallback={null}>
          <RPMAvatar 
            landmarks={landmarks} 
            modelUrl={modelUrl} 
            isCalibrated={isCalibrated}
            imuData={imuData}
            fusionMode={fusionMode}
          />
        </Suspense>
      ) : modelUrl ? (
        // Before calibration: show static avatar in T-pose
        <Suspense fallback={null}>
          <StaticAvatar modelUrl={modelUrl} />
        </Suspense>
      ) : null}
      
      {showSkeleton && <Skeleton landmarks={landmarks} />}

      <OrbitControls 
        enablePan={false}
        minDistance={1.5}
        maxDistance={4}
        target={[0, 0.9, 0]}
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={Math.PI / 1.5}
      />
      <Environment preset="studio" />
    </>
  );
}

// Loading placeholder
function LoadingAvatar() {
  return (
    <mesh position={[0, 0.9, 0]}>
      <capsuleGeometry args={[0.3, 1, 8, 16]} />
      <meshStandardMaterial color="#444" wireframe />
    </mesh>
  );
}

export const HumanAvatar = ({ 
  landmarks, 
  width = 400, 
  height = 480,
  modelUrl = '/models/avatar-female.glb',
  showSkeleton = true,
  backgroundUrl,
  isCalibrated = false,
  imuData = null,
  fusionMode = 'camera_primary',
  showIMUDebug = false
}: Props) => {
  return (
    <div 
      style={{
        width,
        height,
        borderRadius: '20px',
        overflow: 'hidden',
        border: '1px solid #2a2a3a',
        background: 'linear-gradient(180deg, #12121a 0%, #0a0a0f 100%)',
        position: 'relative',
      }}
    >
      <Canvas
        camera={{ position: [0, 1, 2.8], fov: 45 }}
        shadows
        style={{ background: 'transparent' }}
      >
        <Suspense fallback={<LoadingAvatar />}>
          <Scene 
            landmarks={landmarks} 
            modelUrl={modelUrl} 
            showSkeleton={showSkeleton}
            backgroundUrl={backgroundUrl}
            isCalibrated={isCalibrated}
            imuData={imuData}
            fusionMode={fusionMode}
          />
        </Suspense>
      </Canvas>
      
      {/* IMU Debug Overlay */}
      {showIMUDebug && imuData && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          background: 'rgba(0,0,0,0.7)',
          color: '#00ff88',
          padding: '8px 12px',
          borderRadius: '8px',
          fontSize: '11px',
          fontFamily: 'monospace',
        }}>
          <div style={{ marginBottom: '4px', color: '#fff', fontWeight: 'bold' }}>
            🎛️ IMU Sensors ({fusionMode})
          </div>
          {imuData.leftWrist && (
            <div>LW: R:{imuData.leftWrist.euler.roll.toFixed(0)}° P:{imuData.leftWrist.euler.pitch.toFixed(0)}° Y:{imuData.leftWrist.euler.yaw.toFixed(0)}°</div>
          )}
          {imuData.rightWrist && (
            <div>RW: R:{imuData.rightWrist.euler.roll.toFixed(0)}° P:{imuData.rightWrist.euler.pitch.toFixed(0)}° Y:{imuData.rightWrist.euler.yaw.toFixed(0)}°</div>
          )}
          {imuData.chest && (
            <div>CH: R:{imuData.chest.euler.roll.toFixed(0)}° P:{imuData.chest.euler.pitch.toFixed(0)}° Y:{imuData.chest.euler.yaw.toFixed(0)}°</div>
          )}
        </div>
      )}
    </div>
  );
};

// Preload avatars
useGLTF.preload('/models/avatar-female.glb');
useGLTF.preload('/models/avatar-male.glb');
