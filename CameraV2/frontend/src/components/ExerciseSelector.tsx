import { motion } from 'framer-motion';
import type { ExerciseType } from '../types';
import { EXERCISE_LIST } from '../config/exercises';

interface Props {
  onSelect: (exercise: ExerciseType) => void;
}

const exerciseIcons: Record<ExerciseType, string> = {
  bicep_curls: '💪',
  squats: '🦵',
  lateral_shoulder_raises: '🙆',
  triceps_pushdown: '💪',
  dumbbell_rows: '🏋️',
  dumbbell_shoulder_press: '🏋️‍♂️',
};

export const ExerciseSelector = ({ onSelect }: Props) => {
  return (
    <div className="exercise-selector">
      <motion.h1
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="title"
      >
        Select Exercise
      </motion.h1>

      <div className="exercise-grid">
        {EXERCISE_LIST.map((exercise, index) => (
          <motion.button
            key={exercise.name}
            className="exercise-card"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => {
              console.log('Exercise selected:', exercise.name);
              onSelect(exercise.name as ExerciseType);
            }}
          >
            <span className="icon">{exerciseIcons[exercise.name as ExerciseType]}</span>
            <span className="name">{exercise.displayName}</span>
            <span className="description">{exercise.description}</span>
          </motion.button>
        ))}
      </div>
    </div>
  );
};

