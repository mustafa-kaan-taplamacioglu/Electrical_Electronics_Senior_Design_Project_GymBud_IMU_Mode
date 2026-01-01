"""
IMU Dataset Collector for ML Training
Collects raw IMU sensor data synchronized with workout sessions
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class IMUDatasetCollector:
    """Collects IMU sensor data for ML training."""
    
    def __init__(self, dataset_dir: str = "MLTRAINIMU"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_session_id = None
        self.current_exercise = None  # Track current exercise
        self.current_samples: List[Dict] = []
        self.is_collecting = False
    
    def start_session(self, exercise: str, user_id: str = "default") -> str:
        """Start a new IMU data collection session."""
        session_id = f"{exercise}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session_id = session_id
        self.current_exercise = exercise  # Store exercise name
        self.current_samples = []
        self.is_collecting = True
        
        print(f"📡 Started IMU data collection session: {session_id}")
        return session_id
    
    def add_sample(self, imu_data: Dict, timestamp: Optional[float] = None):
        """Add a single IMU sample to current session."""
        if not self.is_collecting:
            raise ValueError("No active collection session. Call start_session() first.")
        
        if timestamp is None:
            timestamp = datetime.now().timestamp()
        
        sample = {
            'timestamp': timestamp,
            'imu_data': imu_data  # Should contain left_wrist, right_wrist, etc.
        }
        
        self.current_samples.append(sample)
    
    def add_rep_sequence(self, rep_number: int, imu_sequence: List[Dict], rep_start_time: float):
        """
        Add a complete rep's IMU sequence.
        
        Args:
            rep_number: Rep number (0 for session-level continuous data, >0 for counted reps)
            imu_sequence: List of IMU samples for this rep
            rep_start_time: Camera rep completion timestamp (from camera collector)
        """
        if not self.is_collecting:
            raise ValueError("No active collection session. Call start_session() first.")
        
        rep_data = {
            'rep_number': rep_number,
            'rep_start_time': rep_start_time,  # Camera rep completion timestamp (synchronized with camera data)
            'camera_rep_timestamp': rep_start_time,  # Alias for clarity (same value as rep_start_time)
            'samples': imu_sequence
        }
        
        self.current_samples.append(rep_data)
        print(f"✅ Added IMU rep #{rep_number} to session (total: {len(self.current_samples)} reps) [Samples: {len(imu_sequence)}, Camera timestamp: {rep_start_time}]")
    
    def save_session(self) -> Optional[str]:
        """Save current session to disk."""
        if not self.current_session_id:
            print("⚠️  No session to save")
            return None
        
        if len(self.current_samples) == 0:
            print("⚠️  No samples to save")
            return None
        
        # Create exercise-specific folder structure
        # Format: {dataset_dir}/{exercise}/{session_id}/
        if self.current_exercise:
            exercise_dir = self.dataset_dir / self.current_exercise
            exercise_dir.mkdir(exist_ok=True)
            session_dir = exercise_dir / self.current_session_id
        else:
            # Fallback to old structure (for backwards compatibility)
            session_dir = self.dataset_dir / self.current_session_id
        
        session_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        json_path = session_dir / "imu_samples.json"
        with open(json_path, 'w') as f:
            json.dump({
                'session_id': self.current_session_id,
                'total_reps': len(self.current_samples),
                'samples': self.current_samples
            }, f, indent=2)
        
        # Save summary CSV (rep-level summary)
        summary_csv_path = session_dir / "summary.csv"
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rep_number', 'num_samples', 'timestamp'])
            for sample in self.current_samples:
                if 'rep_number' in sample:
                    writer.writerow([
                        sample['rep_number'],
                        len(sample.get('samples', [])),
                        sample.get('rep_start_time', '')
                    ])
        
        # Save IMU data as CSV in gymbud_imu_bridge format: timestamp, node_id, node_name, ax, ay, az, gx, gy, gz, qw, qx, qy, qz, roll, pitch, yaw, rep_number
        imu_csv_path = session_dir / "imu_samples.csv"
        node_mapping = {
            'left_wrist': (1, 'left_wrist'),
            'right_wrist': (2, 'right_wrist'),
            'chest': (3, 'chest')
        }
        
        total_rows = 0
        empty_reps = 0
        with open(imu_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header in gymbud_imu_bridge format + rep_number
            writer.writerow(['timestamp', 'node_id', 'node_name', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'qw', 'qx', 'qy', 'qz', 'roll', 'pitch', 'yaw', 'rep_number'])
            
            # Process each rep (including rep_number=0 for session-level continuous data)
            for rep_data in self.current_samples:
                if 'rep_number' not in rep_data:
                    continue
                
                rep_number = rep_data.get('rep_number')
                samples = rep_data.get('samples', [])
                rep_start_time = rep_data.get('rep_start_time', 0)
                num_samples = len(samples)
                
                if num_samples == 0:
                    empty_reps += 1
                    continue
                
                # Process each sample in the rep (already synchronized with camera landmarks)
                for sample_idx, sample in enumerate(samples):
                    # Get timestamp from sample or estimate from rep_start_time
                    sample_timestamp = sample.get('timestamp', rep_start_time + (sample_idx * 0.05))  # Assume 20Hz = 50ms per sample
                    
                    # Get rep_number from sample if available (for session-level continuous data)
                    sample_rep_number = sample.get('rep_number', rep_number)
                    
                    # Process each node in the sample
                    # Sample structure: {timestamp: ..., 'left_wrist': {...}, 'right_wrist': {...}, 'chest': {...}, 'rep_number': ...}
                    for node_name, (node_id, node_display_name) in node_mapping.items():
                        if node_name in sample:
                            node_data = sample[node_name]
                            if isinstance(node_data, dict):
                                # Extract IMU data fields in gymbud_imu_bridge format + rep_number
                                row = [
                                    sample_timestamp,    # timestamp (when this IMU sample was collected)
                                    node_id,             # node_id
                                    node_display_name,   # node_name
                                    node_data.get('ax', ''),
                                    node_data.get('ay', ''),
                                    node_data.get('az', ''),
                                    node_data.get('gx', ''),
                                    node_data.get('gy', ''),
                                    node_data.get('gz', ''),
                                    node_data.get('qw', ''),
                                    node_data.get('qx', ''),
                                    node_data.get('qy', ''),
                                    node_data.get('qz', ''),
                                    node_data.get('roll', ''),
                                    node_data.get('pitch', ''),
                                    node_data.get('yaw', ''),
                                    sample_rep_number    # rep_number (0 for session-level, >0 for specific rep)
                                ]
                                writer.writerow(row)
                                total_rows += 1
        
        print(f"💾 Saved IMU dataset: {session_dir}")
        print(f"   - Total reps: {len(self.current_samples)}")
        print(f"   - Empty reps (no IMU data): {empty_reps}")
        print(f"   - Total CSV rows: {total_rows}")
        print(f"   - JSON: {json_path}")
        print(f"   - CSV: {imu_csv_path}")
        print(f"   - Summary CSV: {summary_csv_path}")
        
        return str(session_dir)
    
    def stop_session(self):
        """Stop collection session."""
        self.is_collecting = False
        self.current_exercise = None
    
    def load_session(self, session_id: str, exercise: str = None) -> Dict:
        """
        Load a saved session.
        
        Args:
            session_id: Session ID
            exercise: Exercise name (if None, searches in all exercise folders)
        """
        if exercise:
            session_dir = self.dataset_dir / exercise / session_id
        else:
            # Search in all exercise folders
            session_dir = None
            for exercise_dir in self.dataset_dir.iterdir():
                if not exercise_dir.is_dir():
                    continue
                potential_dir = exercise_dir / session_id
                if potential_dir.exists():
                    session_dir = potential_dir
                    break
        
        if session_dir is None or not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id} not found" + (f" for exercise {exercise}" if exercise else ""))
        
        json_path = session_dir / "imu_samples.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Session file {json_path} not found")
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def load_all_sessions(self, exercise: Optional[str] = None) -> List[Dict]:
        """
        Load all sessions, optionally filtered by exercise.
        
        Args:
            exercise: If provided, only load sessions for this exercise.
        """
        sessions = []
        
        if exercise:
            # Load from exercise-specific folder
            exercise_dir = self.dataset_dir / exercise
            if not exercise_dir.exists():
                return []
            
            for session_dir in exercise_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_id = session_dir.name
                try:
                    session_data = self.load_session(session_id, exercise=exercise)
                    sessions.append(session_data)
                except Exception as e:
                    print(f"⚠️  Failed to load session {session_id}: {e}")
        else:
            # Load from all exercises
            for exercise_dir in self.dataset_dir.iterdir():
                if not exercise_dir.is_dir():
                    continue
                
                for session_dir in exercise_dir.iterdir():
                    if not session_dir.is_dir():
                        continue
                    
                    session_id = session_dir.name
                    try:
                        session_data = self.load_session(session_id, exercise=exercise_dir.name)
                        sessions.append(session_data)
                    except Exception as e:
                        print(f"⚠️  Failed to load session {session_id}: {e}")
        
        return sessions
    
    def load_dataset(self, exercise: str = None) -> List[Dict]:
        """
        Load all IMU samples as a flat list (for ML training).
        
        Args:
            exercise: If provided, only load samples for this exercise.
        """
        all_samples = []
        
        sessions = self.load_all_sessions(exercise=exercise)
        
        for session_data in sessions:
            if 'samples' in session_data:
                # Each sample is a rep sequence
                for rep_sample in session_data['samples']:
                    if 'rep_number' in rep_sample and 'samples' in rep_sample:
                        # Convert to RepSample-like format
                        all_samples.append({
                            'rep_number': rep_sample['rep_number'],
                            'imu_sequence': rep_sample['samples'],
                            'session_id': session_data.get('session_id', 'unknown'),
                            'exercise': exercise or session_data.get('session_id', '').split('_')[0]
                        })
        
        print(f"📂 Loaded {len(all_samples)} IMU rep samples" + (f" (exercise: {exercise})" if exercise else ""))
        return all_samples

