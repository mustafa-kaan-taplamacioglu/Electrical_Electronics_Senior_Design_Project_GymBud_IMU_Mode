import type { ExerciseConfig, ExerciseType } from '../types';

export const EXERCISES: Record<ExerciseType, ExerciseConfig> = {
  bicep_curls: {
    name: 'bicep_curls',
    displayName: 'Bicep Curl',
    description: 'Dumbbell bicep workout',
    primaryJoints: ['shoulder', 'elbow', 'wrist'],
    repThreshold: { up: 60, down: 140 },
    formTips: [
      'Keep your elbows steady',
      'Don\'t raise your shoulders',
      'Move with control',
    ],
  },
  squats: {
    name: 'squats',
    displayName: 'Squat',
    description: 'Leg and glute exercise',
    primaryJoints: ['hip', 'knee', 'ankle'],
    repThreshold: { up: 160, down: 90 },
    formTips: [
      'Knees shouldn\'t pass toes',
      'Keep your back straight',
      'Push through heels',
    ],
  },
  lateral_shoulder_raises: {
    name: 'lateral_shoulder_raises',
    displayName: 'Lateral Raise',
    description: 'Side shoulder raise',
    primaryJoints: ['shoulder', 'elbow'],
    repThreshold: { up: 80, down: 20 },
    formTips: [
      'Slight bend in arms',
      'Raise to shoulder level',
      'Lower with control',
    ],
  },
  triceps_pushdown: {
    name: 'triceps_pushdown',
    displayName: 'Triceps Pushdown',
    description: 'Triceps workout',
    primaryJoints: ['shoulder', 'elbow', 'wrist'],
    repThreshold: { up: 160, down: 60 },
    formTips: [
      'Keep upper arm stable',
      'Only elbow moves',
      'Full extension',
    ],
  },
  dumbbell_rows: {
    name: 'dumbbell_rows',
    displayName: 'Dumbbell Row',
    description: 'Back exercise',
    primaryJoints: ['shoulder', 'elbow', 'hip'],
    repThreshold: { up: 60, down: 150 },
    formTips: [
      'Keep back flat',
      'Elbow close to body',
      'Squeeze shoulder blades',
    ],
  },
  dumbbell_shoulder_press: {
    name: 'dumbbell_shoulder_press',
    displayName: 'Shoulder Press',
    description: 'Overhead shoulder press',
    primaryJoints: ['shoulder', 'elbow'],
    repThreshold: { up: 160, down: 90 },
    formTips: [
      'Engage core',
      'Keep back straight',
      'Full arm extension',
    ],
  },
};

export const EXERCISE_LIST = Object.values(EXERCISES);

// Avatar options
export const AVATARS = [
  { id: 'emma', name: 'Emma', modelUrl: '/models/avatar-female.glb', gender: 'female' },
  { id: 'alex', name: 'Alex', modelUrl: '/models/avatar-male.glb', gender: 'male' },
];
