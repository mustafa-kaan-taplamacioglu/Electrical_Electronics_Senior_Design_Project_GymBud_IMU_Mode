/**
 * IMU Analysis Utilities
 * Functions for formatting and displaying IMU-only mode analysis results
 */

export interface IMUAnalysis {
  left_wrist: {
    pitch_feedback?: string;
    roll_feedback?: string;
    yaw_feedback?: string;
    accel_feedback?: string;
    gyro_feedback?: string;
    pitch_range?: number;
    roll_range?: number;
    yaw_range?: number;
    accel_max?: number;
    gyro_max?: number;
    pitch_status?: 'excellent' | 'good' | 'moderate' | 'insufficient';
    roll_status?: 'excessive' | 'moderate' | 'good';
    gyro_status?: 'too_fast' | 'fast' | 'controlled';
  };
  right_wrist: {
    pitch_feedback?: string;
    roll_feedback?: string;
    yaw_feedback?: string;
    accel_feedback?: string;
    gyro_feedback?: string;
    pitch_range?: number;
    roll_range?: number;
    yaw_range?: number;
    accel_max?: number;
    gyro_max?: number;
    pitch_status?: 'excellent' | 'good' | 'moderate' | 'insufficient';
    roll_status?: 'excessive' | 'moderate' | 'good';
    gyro_status?: 'too_fast' | 'fast' | 'controlled';
  };
  bilateral_symmetry: {
    feedback?: string;
    status?: 'excellent' | 'good' | 'moderate' | 'poor';
    pitch_diff_pct?: number;
  };
  movement_quality: {
    tempo?: {
      feedback?: string;
      status?: 'optimal' | 'too_fast' | 'too_slow' | 'acceptable';
    };
  };
}

/**
 * Format IMU-only analysis for short display
 * Example: "Sol: 145° ROM (Mükemmel! Tam ROM) | Sağ: 142° ROM (Mükemmel! Tam ROM) | Simetri: 2.1% fark (Mükemmel! Dengeli kas gelişimi) | Tempo: 2.1s (İdeal! TUT optimal)"
 */
export function formatIMUOnlyAnalysisShort(imuAnalysis: IMUAnalysis | null | undefined): string {
  if (!imuAnalysis) {
    return 'IMU analizi yapılıyor...';
  }

  const parts: string[] = [];

  // Sol bilek kısa özet
  const lw = imuAnalysis.left_wrist || {};
  if (lw.pitch_feedback) {
    // Extract the feedback text after colon
    const pitchText = lw.pitch_feedback.split(':').slice(1).join(':').trim();
    if (pitchText) {
      parts.push(`Sol: ${pitchText}`);
    }
  }

  // Sağ bilek kısa özet
  const rw = imuAnalysis.right_wrist || {};
  if (rw.pitch_feedback) {
    const pitchText = rw.pitch_feedback.split(':').slice(1).join(':').trim();
    if (pitchText) {
      parts.push(`Sağ: ${pitchText}`);
    }
  }

  // Simetri
  const symmetry = imuAnalysis.bilateral_symmetry || {};
  if (symmetry.feedback) {
    const symText = symmetry.feedback.split(':').slice(1).join(':').trim();
    if (symText) {
      parts.push(`Simetri: ${symText}`);
    }
  }

  // Tempo
  const tempo = imuAnalysis.movement_quality?.tempo || {};
  if (tempo.feedback) {
    parts.push(tempo.feedback);
  }

  return parts.join(' | ') || 'IMU analizi yapılıyor...';
}

/**
 * Get detailed IMU analysis summary for display
 */
export function getIMUAnalysisSummary(imuAnalysis: IMUAnalysis | null | undefined): {
  leftWrist: string;
  rightWrist: string;
  symmetry: string;
  tempo: string;
} {
  if (!imuAnalysis) {
    return {
      leftWrist: 'Analiz yapılıyor...',
      rightWrist: 'Analiz yapılıyor...',
      symmetry: 'Analiz yapılıyor...',
      tempo: 'Analiz yapılıyor...',
    };
  }

  const lw = imuAnalysis.left_wrist || {};
  const rw = imuAnalysis.right_wrist || {};
  const symmetry = imuAnalysis.bilateral_symmetry || {};
  const tempo = imuAnalysis.movement_quality?.tempo || {};

  return {
    leftWrist: lw.pitch_feedback || `Sol: ${lw.pitch_range || 0}° ROM`,
    rightWrist: rw.pitch_feedback || `Sağ: ${rw.pitch_range || 0}° ROM`,
    symmetry: symmetry.feedback || 'Simetri analizi yapılıyor...',
    tempo: tempo.feedback || 'Tempo analizi yapılıyor...',
  };
}

/**
 * Get color indicator for IMU status
 */
export function getIMUStatusColor(status?: string): string {
  switch (status) {
    case 'excellent':
      return '#10b981'; // green
    case 'good':
      return '#3b82f6'; // blue
    case 'moderate':
      return '#f59e0b'; // yellow
    case 'insufficient':
    case 'excessive':
    case 'too_fast':
    case 'poor':
      return '#ef4444'; // red
    default:
      return '#6b7280'; // gray
  }
}

