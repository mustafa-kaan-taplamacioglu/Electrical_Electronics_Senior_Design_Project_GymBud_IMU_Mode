import { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import type { ExerciseType } from './types';
import { ExerciseSelector } from './components/ExerciseSelector';
import { WorkoutSessionWithIMU } from './components/WorkoutSessionWithIMU';
import { initOpenAI } from './services/openai';
import { AVATARS } from './config/exercises';
import './App.css';

// API Key from environment variable
const DEFAULT_API_KEY = import.meta.env.VITE_OPENAI_API_KEY || '';

function App() {
  const [apiKey, setApiKey] = useState<string>(DEFAULT_API_KEY);
  const [isConfigured, setIsConfigured] = useState(!!DEFAULT_API_KEY);
  const [selectedExercise, setSelectedExercise] = useState<ExerciseType | null>(null);
  const [selectedAvatar, setSelectedAvatar] = useState(AVATARS[0]);
  const [selectedMLMode, setSelectedMLMode] = useState<'usage' | 'train' | null>(null);
  
  // Auto-init OpenAI if key exists
  useEffect(() => {
    if (DEFAULT_API_KEY) {
      initOpenAI(DEFAULT_API_KEY);
      setIsConfigured(true);
    }
  }, []);

  const handleApiKeySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (apiKey.trim()) {
      initOpenAI(apiKey.trim());
      setIsConfigured(true);
    }
  };

  const handleExerciseSelect = (exercise: ExerciseType) => {
    setSelectedExercise(exercise);
    setSelectedMLMode(null); // Reset ML mode when exercise changes
  };

  const handleMLModeSelect = (mode: 'usage' | 'train') => {
    setSelectedMLMode(mode);
    console.log(`Mode selected: ${mode}`);
  };

  const handleSessionEnd = () => {
    setSelectedExercise(null);
    setSelectedMLMode(null);
  };

  // API Key configuration screen
  if (!isConfigured) {
    return (
      <div className="app">
        <div className="config-screen">
          <motion.div
            className="config-card"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <h1>🏋️ Fitness AI Coach</h1>
            <p>Enter your OpenAI API key</p>

            <form onSubmit={handleApiKeySubmit}>
              <input
                type="password"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="api-input"
              />
              <button type="submit" className="submit-button">
                Start
              </button>
            </form>

            <p className="hint">
              API key is stored locally and secure.
            </p>
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <AnimatePresence mode="wait">
        {selectedExercise && selectedMLMode ? (
          <motion.div
            key="session"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="full-screen"
          >
            <WorkoutSessionWithIMU
              exercise={selectedExercise}
              apiKey={apiKey}
              avatarUrl={selectedAvatar.modelUrl}
              mlMode={selectedMLMode}
              onEnd={handleSessionEnd}
            />
          </motion.div>
        ) : selectedExercise ? (
          <motion.div
            key="ml-mode-selector"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="ml-mode-selector-screen"
          >
            <div className="ml-mode-selector-card">
              <h2>🏋️ Select Mode</h2>
              <p className="exercise-info">
                Selected exercise: <strong>{selectedExercise}</strong>
              </p>
              
              <div className="ml-mode-options">
                <button
                  className="ml-mode-option"
                  onClick={() => handleMLModeSelect('usage')}
                  style={{ borderColor: '#22c55e' }}
                >
                  <div className="ml-mode-icon">⚡</div>
                  <div className="ml-mode-content">
                    <h3>Usage Mode</h3>
                    <p>Angle-based rep counting, form analysis, and data recording. All workout data is saved automatically.</p>
                  </div>
                </button>
                
                <button
                  className="ml-mode-option"
                  onClick={() => handleMLModeSelect('train')}
                  style={{ borderColor: '#3b82f6' }}
                >
                  <div className="ml-mode-icon">📚</div>
                  <div className="ml-mode-content">
                    <h3>ML Training Mode</h3>
                    <p>Collect data and train ML models. Dataset collection enabled automatically.</p>
                  </div>
                </button>
              </div>
              
              <button
                className="back-button"
                onClick={() => setSelectedExercise(null)}
                style={{ marginTop: '30px' }}
              >
                ← Back to Exercise Selection
              </button>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="selector"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Avatar Selection */}
            <div className="avatar-selection">
              <h3>Select Your Avatar</h3>
              <div className="avatar-options">
                {AVATARS.map((avatar) => (
                  <button
                    key={avatar.id}
                    className={`avatar-option ${selectedAvatar.id === avatar.id ? 'selected' : ''}`}
                    onClick={() => setSelectedAvatar(avatar)}
                  >
                    <span className="avatar-icon">
                      {avatar.gender === 'female' ? '👩' : '👨'}
                    </span>
                    <span className="avatar-name">{avatar.name}</span>
                  </button>
                ))}
              </div>
            </div>
            
            <ExerciseSelector onSelect={handleExerciseSelect} />
            
            {/* Update Training Section */}
            <div style={{
              marginTop: '40px',
              padding: '20px',
              background: 'rgba(59, 130, 246, 0.1)',
              borderRadius: '12px',
              border: '1px solid rgba(59, 130, 246, 0.3)',
              maxWidth: '800px',
              margin: '40px auto 0'
            }}>
              <h3 style={{ color: '#fff', marginBottom: '15px', fontSize: '20px' }}>
                🔄 Update Existing Models
              </h3>
              <p style={{ color: '#aaa', marginBottom: '15px', fontSize: '14px' }}>
                Update existing ML models using unused training data. This will use datasets that have been saved but not yet used for training.
              </p>
              <button
                onClick={async () => {
                  const exercise = prompt('Enter exercise name (e.g., bicep_curls, squats):');
                  if (exercise) {
                    try {
                      const response = await fetch(`http://localhost:8000/api/update_model/${exercise}`, {
                        method: 'POST'
                      });
                      const data = await response.json();
                      if (data.success) {
                        alert(`✅ Model updated successfully!\n\n${data.message}`);
                      } else {
                        alert(`❌ Update failed:\n\n${data.message}`);
                      }
                    } catch (error) {
                      alert(`❌ Error: ${error}`);
                    }
                  }
                }}
                style={{
                  padding: '12px 24px',
                  background: 'rgba(59, 130, 246, 0.3)',
                  border: '2px solid #3b82f6',
                  borderRadius: '8px',
                  color: '#fff',
                  fontSize: '16px',
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  fontWeight: 'bold'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(59, 130, 246, 0.5)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(59, 130, 246, 0.3)';
                }}
              >
                🔄 Update Training Models
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
