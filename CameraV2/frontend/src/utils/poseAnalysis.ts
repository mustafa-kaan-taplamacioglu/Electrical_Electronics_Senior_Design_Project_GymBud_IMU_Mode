import type { Landmark, ExerciseType } from '../types';
import { EXERCISES } from '../config/exercises';

// MediaPipe landmark indices
const LANDMARKS = {
  NOSE: 0,
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

// Ultra Strict Form Configuration - Professional Level
const ULTRA_STRICT_CONFIG = {
  bicep_curls: {
    max_elbow_drift: 0.06,    // Dirsek omuz genişliğinin %6'sından fazla oynamamalı
    wrist_neutral_limit: 165, // Bilek en fazla 15 derece bükülebilir (nötr pozisyon)
    shoulder_rise_limit: 0.04, // Omuz en fazla %4 yükselebilir
    perfect_min_angle: 40,    // Tam bicep sıkıştırma hedefi
    perfect_max_angle: 160    // Tam kol açma hedefi
  },
  squats: {
    max_knee_forward: 0.07,   // Dizler ayak parmaklarını çok geçmemeli
    min_torso_angle: 75,      // Sırt dikliği (dikeyden sapma)
    hip_depth_threshold: 95,  // Kalça diz hizasının altına inmeli
    knee_cave_limit: 0.85     // Dizler ayak bileği genişliğinin %85'inin altına inmemeli
  },
};

// Calculate angle between three points
export const calculateAngle = (
  a: Landmark,
  b: Landmark,
  c: Landmark
): number => {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
};

// Get primary angle for exercise
export const getPrimaryAngle = (
  landmarks: Landmark[],
  exercise: ExerciseType
): number => {
  switch (exercise) {
    case 'bicep_curls':
      // Average of both elbow angles
      const leftElbow = calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );
      const rightElbow = calculateAngle(
        landmarks[LANDMARKS.RIGHT_SHOULDER],
        landmarks[LANDMARKS.RIGHT_ELBOW],
        landmarks[LANDMARKS.RIGHT_WRIST]
      );
      return (leftElbow + rightElbow) / 2;

    case 'squats':
      // Average of both knee angles
      const leftKnee = calculateAngle(
        landmarks[LANDMARKS.LEFT_HIP],
        landmarks[LANDMARKS.LEFT_KNEE],
        landmarks[LANDMARKS.LEFT_ANKLE]
      );
      const rightKnee = calculateAngle(
        landmarks[LANDMARKS.RIGHT_HIP],
        landmarks[LANDMARKS.RIGHT_KNEE],
        landmarks[LANDMARKS.RIGHT_ANKLE]
      );
      return (leftKnee + rightKnee) / 2;

    case 'lateral_shoulder_raises':
      // Shoulder angle (arm-torso)
      const leftShoulder = calculateAngle(
        landmarks[LANDMARKS.LEFT_HIP],
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW]
      );
      const rightShoulder = calculateAngle(
        landmarks[LANDMARKS.RIGHT_HIP],
        landmarks[LANDMARKS.RIGHT_SHOULDER],
        landmarks[LANDMARKS.RIGHT_ELBOW]
      );
      return (leftShoulder + rightShoulder) / 2;

    case 'tricep_extensions':
      return calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );

    case 'dumbbell_rows':
      return calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );

    case 'dumbbell_shoulder_press':
      return calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );

    default:
      return 0;
  }
};

// Check form and return score + issues
export type FormCheckResult = {
  score: number;
  issues: string[];
}

export const checkForm = (
  landmarks: Landmark[],
  exercise: ExerciseType,
  calibrationData?: {
    shoulderWidth: number;
    torsoHeight: number;
    initialElbowX: { left: number; right: number };
    initialShoulderY: { left: number; right: number };
  }
): FormCheckResult => {
  const issues: string[] = [];
  const scores: number[] = [];

  // Check landmark visibility
  const requiredLandmarks = getRequiredLandmarks(exercise);
  const visibleCount = requiredLandmarks.filter(
    (idx) => landmarks[idx]?.visibility > 0.5
  ).length;
  
  if (visibleCount < requiredLandmarks.length * 0.7) {
    return { score: 0, issues: ['Vücut görünmüyor'] };
  }

  // Common checks for upper body exercises
  const checkUpperBodyStability = () => {
    if (!calibrationData) return;
    
    const config = (ULTRA_STRICT_CONFIG as any)[exercise] || {
      max_elbow_drift: 0.12,
      shoulder_rise_limit: 0.08
    };
    
    // Elbow drift check (Ultra Sensitive)
    const leftElbowDrift = Math.abs(
      landmarks[LANDMARKS.LEFT_ELBOW].x - calibrationData.initialElbowX.left
    );
    const rightElbowDrift = Math.abs(
      landmarks[LANDMARKS.RIGHT_ELBOW].x - calibrationData.initialElbowX.right
    );
    
    const driftTolerance = calibrationData.shoulderWidth * config.max_elbow_drift;
    
    if (leftElbowDrift > driftTolerance) {
      issues.push('Sol dirsek çok oynuyor - gövdene sabitle');
      scores.push(Math.max(30, 100 - (leftElbowDrift / driftTolerance) * 40));
    }
    
    if (rightElbowDrift > driftTolerance) {
      issues.push('Sağ dirsek çok oynuyor - gövdene sabitle');
      scores.push(Math.max(30, 100 - (rightElbowDrift / driftTolerance) * 40));
    }

    // Shoulder rise check (Trapezius usage detection)
    const leftShoulderRise = calibrationData.initialShoulderY.left - landmarks[LANDMARKS.LEFT_SHOULDER].y;
    const rightShoulderRise = calibrationData.initialShoulderY.right - landmarks[LANDMARKS.RIGHT_SHOULDER].y;
    const riseTolerance = calibrationData.torsoHeight * config.shoulder_rise_limit;

    if (leftShoulderRise > riseTolerance) {
      issues.push('Sol omuz kalkıyor - aşağıda tut');
      scores.push(Math.max(40, 100 - (leftShoulderRise / riseTolerance) * 30));
    }

    if (rightShoulderRise > riseTolerance) {
      issues.push('Sağ omuz kalkıyor - aşağıda tut');
      scores.push(Math.max(40, 100 - (rightShoulderRise / riseTolerance) * 30));
    }
  };

  // ============================================================
  // EXERCISE-SPECIFIC FORM CHECKS (Detaylı Form Analizi)
  // ============================================================
  
  switch (exercise) {
    
    // ==================== BICEP CURL ====================
    case 'bicep_curls': {
      checkUpperBodyStability();
      
      const config = ULTRA_STRICT_CONFIG.bicep_curls;
      
      // 1. Bilek Bükme Kontrolü (Wrist Position)
      // Bilek ön kolla aynı doğrultuda olmalı
      const leftWristAngle = calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );
      
      // Bilek aşırı içeri bükülürse (wrist curl) ön kol yorulur
      if (landmarks[LANDMARKS.LEFT_WRIST].visibility > 0.8) {
        // Bilek-parmak ucu açısı gibi düşünülebilir (MediaPipe wrist-index finger)
        // Basitlik için wrist stabilitesini x-y sapmasıyla kontrol edelim
        const leftWristDev = Math.abs(landmarks[LANDMARKS.LEFT_WRIST].x - landmarks[LANDMARKS.LEFT_ELBOW].x);
        if (leftWristDev > 0.25) {
          issues.push('Sol bileğini bükme, nötr tut');
          scores.push(70);
        }
      }

      // 2. Dirsek açısı kontrolü (tam bükülme)
      const leftElbowAngle = calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );
      const rightElbowAngle = calculateAngle(
        landmarks[LANDMARKS.RIGHT_SHOULDER],
        landmarks[LANDMARKS.RIGHT_ELBOW],
        landmarks[LANDMARKS.RIGHT_WRIST]
      );
      
      if (leftElbowAngle < config.perfect_min_angle - 10) {
        issues.push('Sol kolu aşırı büküyorsun - bilek omuza değmesin');
        scores.push(80);
      }
      
      // 3. Omuz stabilitesi
      const shoulderTilt = Math.abs(
        landmarks[LANDMARKS.LEFT_SHOULDER].y - landmarks[LANDMARKS.RIGHT_SHOULDER].y
      );
      if (shoulderTilt > 0.04) {
        issues.push('Omuzların eğik - dik dur ve omuzdan güç alma');
        scores.push(Math.max(50, 100 - shoulderTilt * 400));
      }
      
      break;
    }
    
    // ==================== SQUAT ====================
    case 'squats': {
      // 1. Diz açısı kontrolü
      const leftKneeAngle = calculateAngle(
        landmarks[LANDMARKS.LEFT_HIP],
        landmarks[LANDMARKS.LEFT_KNEE],
        landmarks[LANDMARKS.LEFT_ANKLE]
      );
      const rightKneeAngle = calculateAngle(
        landmarks[LANDMARKS.RIGHT_HIP],
        landmarks[LANDMARKS.RIGHT_KNEE],
        landmarks[LANDMARKS.RIGHT_ANKLE]
      );
      
      // Paralele inme kontrolü (90 derece civarı)
      const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
      if (avgKneeAngle > 120 && avgKneeAngle < 150) {
        issues.push('Daha aşağı in - paralele kadar');
        scores.push(75);
      }
      
      // 2. Diz-ayak hizası kontrolü
      const leftKneeOverToe = landmarks[LANDMARKS.LEFT_KNEE].x - landmarks[LANDMARKS.LEFT_ANKLE].x;
      const rightKneeOverToe = landmarks[LANDMARKS.RIGHT_KNEE].x - landmarks[LANDMARKS.RIGHT_ANKLE].x;
      
      if (Math.abs(leftKneeOverToe) > 0.08) {
        issues.push('Sol diz ayak hizasından çıkıyor');
        scores.push(70);
      }
      if (Math.abs(rightKneeOverToe) > 0.08) {
        issues.push('Sağ diz ayak hizasından çıkıyor');
        scores.push(70);
      }
      
      // 3. Sırt düzlüğü kontrolü (kalça-omuz hizası)
      const hipMidX = (landmarks[LANDMARKS.LEFT_HIP].x + landmarks[LANDMARKS.RIGHT_HIP].x) / 2;
      const shoulderMidX = (landmarks[LANDMARKS.LEFT_SHOULDER].x + landmarks[LANDMARKS.RIGHT_SHOULDER].x) / 2;
      const backLean = Math.abs(hipMidX - shoulderMidX);
      
      if (backLean > 0.15) {
        issues.push('Sırtını dik tut - öne eğilme');
        scores.push(65);
      }
      
      // 4. Diz içe çökmesi kontrolü
      const kneeWidth = Math.abs(landmarks[LANDMARKS.LEFT_KNEE].x - landmarks[LANDMARKS.RIGHT_KNEE].x);
      const ankleWidth = Math.abs(landmarks[LANDMARKS.LEFT_ANKLE].x - landmarks[LANDMARKS.RIGHT_ANKLE].x);
      
      if (kneeWidth < ankleWidth * 0.8) {
        issues.push('Dizler içe çöküyor - dışa it');
        scores.push(60);
      }
      
      // 5. Topuklardan kalkma kontrolü
      const heelRise = Math.abs(landmarks[LANDMARKS.LEFT_ANKLE].y - landmarks[LANDMARKS.RIGHT_ANKLE].y);
      if (heelRise > 0.05) {
        issues.push('Topuklar yerden kalkıyor');
        scores.push(70);
      }
      
      scores.push(100); // Base score
      break;
    }
    
    // ==================== LATERAL RAISE ====================
    case 'lateral_shoulder_raises': {
      checkUpperBodyStability();
      
      // 1. Kol yüksekliği kontrolü (omuz hizasına kadar)
      const leftWristY = landmarks[LANDMARKS.LEFT_WRIST].y;
      const rightWristY = landmarks[LANDMARKS.RIGHT_WRIST].y;
      const shoulderY = (landmarks[LANDMARKS.LEFT_SHOULDER].y + landmarks[LANDMARKS.RIGHT_SHOULDER].y) / 2;
      
      // Kollar omuz seviyesini geçmemeli
      if (leftWristY < shoulderY - 0.1) {
        issues.push('Sol kol çok yüksek - omuz hizasında tut');
        scores.push(70);
      }
      if (rightWristY < shoulderY - 0.1) {
        issues.push('Sağ kol çok yüksek - omuz hizasında tut');
        scores.push(70);
      }
      
      // 2. Kol simetrisi
      const heightDiff = Math.abs(leftWristY - rightWristY);
      if (heightDiff > 0.08) {
        issues.push('Kollar eşit yükseklikte olmalı');
        scores.push(Math.max(60, 100 - heightDiff * 250));
      }
      
      // 3. Dirsek hafif bükük olmalı
      const leftElbowAngle = calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );
      const rightElbowAngle = calculateAngle(
        landmarks[LANDMARKS.RIGHT_SHOULDER],
        landmarks[LANDMARKS.RIGHT_ELBOW],
        landmarks[LANDMARKS.RIGHT_WRIST]
      );
      
      if (leftElbowAngle > 175) {
        issues.push('Sol dirsek hafif bükük olmalı');
        scores.push(85);
      }
      if (rightElbowAngle > 175) {
        issues.push('Sağ dirsek hafif bükük olmalı');
        scores.push(85);
      }
      
      // 4. Gövde sallanması
      const shoulderTilt = Math.abs(landmarks[LANDMARKS.LEFT_SHOULDER].y - landmarks[LANDMARKS.RIGHT_SHOULDER].y);
      if (shoulderTilt > 0.05) {
        issues.push('Gövdeni sabit tut - sallanma');
        scores.push(70);
      }
      
      scores.push(100);
      break;
    }
    
    // ==================== TRICEPS PUSHDOWN ====================
    case 'tricep_extensions': {
      checkUpperBodyStability();
      
      // 1. Üst kol sabit olmalı
      if (calibrationData) {
        const leftUpperArmMove = Math.abs(
          landmarks[LANDMARKS.LEFT_ELBOW].y - landmarks[LANDMARKS.LEFT_SHOULDER].y
        );
        const rightUpperArmMove = Math.abs(
          landmarks[LANDMARKS.RIGHT_ELBOW].y - landmarks[LANDMARKS.RIGHT_SHOULDER].y
        );
        
        // Dirsek omuz hizasında kalmalı
        if (leftUpperArmMove > calibrationData.torsoHeight * 0.2) {
          issues.push('Sol üst kolu sabit tut');
          scores.push(70);
        }
        if (rightUpperArmMove > calibrationData.torsoHeight * 0.2) {
          issues.push('Sağ üst kolu sabit tut');
          scores.push(70);
        }
      }
      
      // 2. Tam uzatma kontrolü
      const leftElbowAngle = calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );
      const rightElbowAngle = calculateAngle(
        landmarks[LANDMARKS.RIGHT_SHOULDER],
        landmarks[LANDMARKS.RIGHT_ELBOW],
        landmarks[LANDMARKS.RIGHT_WRIST]
      );
      
      // Üst pozisyonda kol neredeyse düz olmalı
      if (leftElbowAngle < 150 && leftElbowAngle > 100) {
        issues.push('Sol kolu tam uzat');
        scores.push(80);
      }
      if (rightElbowAngle < 150 && rightElbowAngle > 100) {
        issues.push('Sağ kolu tam uzat');
        scores.push(80);
      }
      
      // 3. Dirsekler sabit pozisyonda
      const elbowSpread = Math.abs(landmarks[LANDMARKS.LEFT_ELBOW].x - landmarks[LANDMARKS.RIGHT_ELBOW].x);
      const shoulderWidth = Math.abs(landmarks[LANDMARKS.LEFT_SHOULDER].x - landmarks[LANDMARKS.RIGHT_SHOULDER].x);
      
      if (elbowSpread > shoulderWidth * 1.3) {
        issues.push('Dirsekleri vücuda yakın tut');
        scores.push(75);
      }
      
      scores.push(100);
      break;
    }
    
    // ==================== DUMBBELL ROW ====================
    case 'dumbbell_rows': {
      checkUpperBodyStability();
      
      // 1. Sırt düzlüğü (çok önemli)
      const hipY = (landmarks[LANDMARKS.LEFT_HIP].y + landmarks[LANDMARKS.RIGHT_HIP].y) / 2;
      const shoulderY = (landmarks[LANDMARKS.LEFT_SHOULDER].y + landmarks[LANDMARKS.RIGHT_SHOULDER].y) / 2;
      const backAngle = Math.abs(hipY - shoulderY);
      
      if (backAngle < 0.1) {
        issues.push('Sırtı yere paralel tut');
        scores.push(65);
      }
      
      // 2. Dirsek vücuda yakın olmalı
      const elbowToHip = Math.abs(landmarks[LANDMARKS.LEFT_ELBOW].x - landmarks[LANDMARKS.LEFT_HIP].x);
      if (elbowToHip > 0.2) {
        issues.push('Dirseği vücuda yakın tut');
        scores.push(70);
      }
      
      // 3. Çekiş yüksekliği (kalça hizasına kadar)
      const wristY = landmarks[LANDMARKS.LEFT_WRIST].y;
      if (wristY > hipY + 0.1) {
        issues.push('Ağırlığı kalça hizasına çek');
        scores.push(75);
      }
      
      // 4. Omuz kanatları sıkışmalı (omuzlar arkaya)
      const shoulderTilt = landmarks[LANDMARKS.LEFT_SHOULDER].z - landmarks[LANDMARKS.RIGHT_SHOULDER].z;
      if (Math.abs(shoulderTilt) > 0.1) {
        issues.push('Omuz kanatlarını sık');
        scores.push(80);
      }
      
      // 5. Baş nötr pozisyonda
      const noseY = landmarks[LANDMARKS.NOSE].y;
      if (noseY < shoulderY - 0.2) {
        issues.push('Başı yukarı kaldırma');
        scores.push(85);
      }
      
      scores.push(100);
      break;
    }
    
    // ==================== SHOULDER PRESS ====================
    case 'dumbbell_shoulder_press': {
      checkUpperBodyStability();
      
      // 1. Tam uzatma kontrolü
      const leftElbowAngle = calculateAngle(
        landmarks[LANDMARKS.LEFT_SHOULDER],
        landmarks[LANDMARKS.LEFT_ELBOW],
        landmarks[LANDMARKS.LEFT_WRIST]
      );
      const rightElbowAngle = calculateAngle(
        landmarks[LANDMARKS.RIGHT_SHOULDER],
        landmarks[LANDMARKS.RIGHT_ELBOW],
        landmarks[LANDMARKS.RIGHT_WRIST]
      );
      
      // Üst pozisyonda kollar düz olmalı
      if (leftElbowAngle < 160 && leftElbowAngle > 120) {
        issues.push('Sol kolu tam uzat');
        scores.push(80);
      }
      if (rightElbowAngle < 160 && rightElbowAngle > 120) {
        issues.push('Sağ kolu tam uzat');
        scores.push(80);
      }
      
      // 2. Kol simetrisi
      const leftWristY = landmarks[LANDMARKS.LEFT_WRIST].y;
      const rightWristY = landmarks[LANDMARKS.RIGHT_WRIST].y;
      const heightDiff = Math.abs(leftWristY - rightWristY);
      
      if (heightDiff > 0.08) {
        issues.push('Kollar eşit yükseklikte olmalı');
        scores.push(75);
      }
      
      // 3. Sırt kavisi kontrolü (aşırı kavis tehlikeli)
      if (calibrationData) {
        const hipMidX = (landmarks[LANDMARKS.LEFT_HIP].x + landmarks[LANDMARKS.RIGHT_HIP].x) / 2;
        const shoulderMidX = (landmarks[LANDMARKS.LEFT_SHOULDER].x + landmarks[LANDMARKS.RIGHT_SHOULDER].x) / 2;
        const backArch = Math.abs(hipMidX - shoulderMidX);
        
        if (backArch > calibrationData.shoulderWidth * 0.3) {
          issues.push('⚠️ Sırtı fazla kavislendirme!');
          scores.push(50);
        }
      }
      
      // 4. Core aktif olmalı
      const hipTilt = Math.abs(landmarks[LANDMARKS.LEFT_HIP].y - landmarks[LANDMARKS.RIGHT_HIP].y);
      if (hipTilt > 0.05) {
        issues.push('Core sıkı tut - sallanma');
        scores.push(70);
      }
      
      // 5. Dirsek pozisyonu (90 derece başlangıç)
      const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;
      if (avgElbowAngle > 90 && avgElbowAngle < 100) {
        // Alt pozisyon - dirsek 90 derece iyi
        scores.push(100);
      }
      
      scores.push(100);
      break;
    }

    default:
      scores.push(100);
  }

  const avgScore = scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 100;
  return { score: avgScore, issues };
};

// Get required landmarks for each exercise
const getRequiredLandmarks = (exercise: ExerciseType): number[] => {
  switch (exercise) {
    case 'bicep_curls':
    case 'tricep_extensions':
    case 'lateral_shoulder_raises':
    case 'dumbbell_shoulder_press':
      // Upper body exercises: Face (0-10) + Upper Body (11-16) + Hands (17-22) = 23 landmarks
      return Array.from({ length: 23 }, (_, i) => i);

    case 'squats':
    case 'dumbbell_rows':
      // Full body: All 33 landmarks (0-32)
      return Array.from({ length: 33 }, (_, i) => i);

    default:
      return [];
  }
};

// Rep counter state
export type RepCounterState = {
  phase: 'up' | 'down';
  count: number;
  validCount: number;      // Sadece geçerli (iyi formlu) repler
  lastAngle: number;
  minAngleReached: number; // O rep'te ulaşılan minimum açı
  maxAngleReached: number; // O rep'te ulaşılan maksimum açı
  formScores: number[];
  lastRepValid: boolean;   // Son rep geçerli miydi?
  repFeedback: string;     // Son rep için geri bildirim
}

// Minimum form skoru eşiği - bunun altında rep SAYILMAZ
const MIN_FORM_SCORE_FOR_VALID_REP = 60;

// Her egzersiz için açı gereksinimleri
const ANGLE_REQUIREMENTS: Record<string, { minAngle: number; maxAngle: number; tolerance: number }> = {
  bicep_curls: { minAngle: 35, maxAngle: 150, tolerance: 15 },      // Dirsek: 35°-150° arası hareket
  squats: { minAngle: 70, maxAngle: 170, tolerance: 15 },           // Diz: 70°-170° arası
  lateral_shoulder_raises: { minAngle: 15, maxAngle: 85, tolerance: 10 }, // Omuz: 15°-85° arası
  tricep_extensions: { minAngle: 45, maxAngle: 165, tolerance: 15 }, // Dirsek: 45°-165° arası
  dumbbell_rows: { minAngle: 45, maxAngle: 160, tolerance: 15 },    // Dirsek: 45°-160° arası
  dumbbell_shoulder_press: { minAngle: 75, maxAngle: 170, tolerance: 15 }, // Dirsek: 75°-170° arası
};

export const createRepCounter = (): RepCounterState => ({
  phase: 'down',
  count: 0,
  validCount: 0,
  lastAngle: 0,
  minAngleReached: 180,
  maxAngleReached: 0,
  formScores: [],
  lastRepValid: true,
  repFeedback: '',
});

export const updateRepCounter = (
  state: RepCounterState,
  angle: number,
  exercise: ExerciseType,
  formScore: number
): { 
  newState: RepCounterState; 
  repCompleted: boolean; 
  avgFormScore: number;
  isValidRep: boolean;
  repFeedback: string;
} => {
  const config = EXERCISES[exercise];
  const { up, down } = config.repThreshold;
  const newState = { ...state };
  let repCompleted = false;
  let avgFormScore = 0;
  let isValidRep = false;
  let repFeedback = '';

  // Track form scores
  newState.formScores.push(formScore);

  // Track min/max angles during the rep
  newState.minAngleReached = Math.min(newState.minAngleReached, angle);
  newState.maxAngleReached = Math.max(newState.maxAngleReached, angle);

  // Get angle requirements for this exercise
  const requirements = ANGLE_REQUIREMENTS[exercise] || { minAngle: 30, maxAngle: 160, tolerance: 15 };

  // ============================================================
  // VALIDATE REP FUNCTION
  // ============================================================
  const validateRep = (): { valid: boolean; feedback: string } => {
    const avgScore = newState.formScores.reduce((a, b) => a + b, 0) / newState.formScores.length;
    const rangeOfMotion = newState.maxAngleReached - newState.minAngleReached;
    const requiredRange = requirements.maxAngle - requirements.minAngle;
    
    // Check 1: Ultra Strict Form Skoru (Profesyonel seviye için %80+)
    if (avgScore < 75) {
      return { 
        valid: false, 
        feedback: `❌ Form kalitesi düşük (%${avgScore.toFixed(0)}) - Rep sayılmadı!` 
      };
    }
    
    // Check 2: Tam Hareket Menzili (ROM)
    // Hareketin en az %75'i tamamlanmalı
    if (rangeOfMotion < requiredRange * 0.75) {
      return { 
        valid: false, 
        feedback: `❌ Yarım hareket tespit edildi - Kasını tam esnet ve sıkıştır!` 
      };
    }
    
    // Check 3: Tepe Noktası (Contracted Position)
    const minTolerance = (ULTRA_STRICT_CONFIG as any)[exercise]?.perfect_min_angle ? 10 : 15;
    if (newState.minAngleReached > requirements.minAngle + minTolerance) {
      return { 
        valid: false, 
        feedback: `❌ Tepe noktasında yeterli sıkıştırma yok - Kolunu biraz daha bük!` 
      };
    }
    
    // Check 4: Alt Nokta (Extended Position)
    const maxTolerance = (ULTRA_STRICT_CONFIG as any)[exercise]?.perfect_max_angle ? 10 : 15;
    if (newState.maxAngleReached < requirements.maxAngle - maxTolerance) {
      return { 
        valid: false, 
        feedback: `❌ Alt noktada kolunu tam açmadın - Kası tam esnet!` 
      };
    }
    
    // All checks passed!
    if (avgScore >= 92) {
      return { valid: true, feedback: `🌟 Mükemmel Form! (%${avgScore.toFixed(0)})` };
    } else if (avgScore >= 85) {
      return { valid: true, feedback: `✅ Profesyonel Rep (%${avgScore.toFixed(0)})` };
    } else {
      return { valid: true, feedback: `👍 Geçerli (%${avgScore.toFixed(0)}) - Dirsek stabilitesine odaklan.` };
    }
  };

  // ============================================================
  // EXERCISE-SPECIFIC REP COUNTING LOGIC
  // ============================================================
  
  const completeRep = () => {
    const validation = validateRep();
    avgFormScore = newState.formScores.reduce((a, b) => a + b, 0) / newState.formScores.length;
    
    newState.count += 1; // Toplam rep (geçerli + geçersiz)
    
    if (validation.valid) {
      newState.validCount += 1; // Sadece geçerli repler
      isValidRep = true;
    }
    
    newState.lastRepValid = validation.valid;
    newState.repFeedback = validation.feedback;
    repFeedback = validation.feedback;
    
    // Reset for next rep
    newState.formScores = [];
    newState.minAngleReached = 180;
    newState.maxAngleReached = 0;
    repCompleted = true;
  };
  
  switch (exercise) {
    // Bicep curl: Açı AZALIR yukarı çıkarken (60° üst, 140° alt)
    case 'bicep_curls':
      if (state.phase === 'down' && angle < up) {
        newState.phase = 'up';
      } else if (state.phase === 'up' && angle > down) {
        newState.phase = 'down';
        completeRep();
      }
      break;
    
    // Triceps pushdown: Açı ARTAR yukarı iterken (160° üst, 60° alt)
    case 'tricep_extensions':
      if (state.phase === 'down' && angle > up) {
        newState.phase = 'up';
      } else if (state.phase === 'up' && angle < down) {
        newState.phase = 'down';
        completeRep();
      }
      break;
    
    // Squat: Diz açısı AZALIR aşağı inerken (160° üst, 90° alt)
    case 'squats':
      if (state.phase === 'up' && angle < down) {
        newState.phase = 'down';
      } else if (state.phase === 'down' && angle > up) {
        newState.phase = 'up';
        completeRep();
      }
      break;
    
    // Lateral raise: Omuz açısı ARTAR yukarı kaldırırken (80° üst, 20° alt)
    case 'lateral_shoulder_raises':
      if (state.phase === 'down' && angle > up) {
        newState.phase = 'up';
      } else if (state.phase === 'up' && angle < down) {
        newState.phase = 'down';
        completeRep();
      }
      break;
    
    // Dumbbell row: Dirsek açısı AZALIR yukarı çekerken (60° üst, 150° alt)
    case 'dumbbell_rows':
    if (state.phase === 'down' && angle < up) {
      newState.phase = 'up';
    } else if (state.phase === 'up' && angle > down) {
      newState.phase = 'down';
        completeRep();
      }
      break;
    
    // Shoulder press: Dirsek açısı ARTAR yukarı iterken (160° üst, 90° alt)
    case 'dumbbell_shoulder_press':
      if (state.phase === 'down' && angle > up) {
        newState.phase = 'up';
      } else if (state.phase === 'up' && angle < down) {
        newState.phase = 'down';
        completeRep();
      }
      break;
    
    default:
      // Generic fallback
    if (state.phase === 'down' && angle > up) {
      newState.phase = 'up';
    } else if (state.phase === 'up' && angle < down) {
      newState.phase = 'down';
        completeRep();
    }
  }

  newState.lastAngle = angle;
  return { newState, repCompleted, avgFormScore, isValidRep, repFeedback };
};

// Helper: Get valid rep count (only properly executed reps)
export const getValidRepCount = (state: RepCounterState): number => {
  return state.validCount;
};

// Helper: Get rep quality summary
export const getRepQualitySummary = (state: RepCounterState): string => {
  if (state.count === 0) return 'Henüz rep yok';
  const validPercent = (state.validCount / state.count * 100).toFixed(0);
  return `${state.validCount}/${state.count} geçerli rep (${validPercent}%)`;
};

