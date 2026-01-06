# ML Model Training Status

## Overview
Her hareket ve her mod için ML model eğitimi yapılacak:
- **4 Hareket**: bicep_curls, lateral_shoulder_raises, squats, tricep_extensions
- **3 Mod**: Camera Only, IMU Only, Sensory Fusion (Camera + IMU)
- **Toplam**: 12 ML Model (4 x 3)

## Training Infrastructure

✅ **Completed:**
1. `train_all_exercises_all_modes.py` - Comprehensive training script created
   - Supports training all exercises and all modes
   - One-class classifier training integrated
   - CSV format support added to `imu_dataset_collector.py`

## Data Availability

| Exercise | Camera Sessions | IMU Sessions | Status |
|----------|----------------|--------------|--------|
| bicep_curls | 12 | 12 | ✅ Ready for all modes |
| lateral_shoulder_raises | 1 | 3 | ⚠️ Limited camera data |
| squats | 4 | 3 | ✅ Ready for all modes |
| tricep_extensions | 0 | 3 | ❌ Only IMU mode possible |

## Current Issues

1. **IMU Data Format**: `samples.json` files are in camera dataset format (not IMU format)
   - These files contain `imu_sequence` field but need conversion
   - Solution: Extract IMU sequences from `samples.json` files

2. **One-Class Classifier**: Needs training for each exercise and mode
   - IMU-based one-class: ✅ Infrastructure ready
   - Camera-based one-class: ⚠️ Not yet implemented

## Next Steps

1. Fix IMU dataset collector to handle `samples.json` format (extract `imu_sequence`)
2. Train all 12 ML models:
   ```bash
   python train_all_exercises_all_modes.py --all
   ```
3. Train one-class classifiers for each exercise and mode
4. Integrate rule-based ensemble (already exists in `exercise_ensemble.py`)
5. Integrate ML + rule-based fusion for rep counting and form scoring
6. Integrate LLM + rule-based + scientific feedback (already exists in `ai_service.py`)

## Training Commands

### Train all models for all exercises:
```bash
python train_all_exercises_all_modes.py --all
```

### Train specific exercise:
```bash
python train_all_exercises_all_modes.py --exercise bicep_curls
```

### Train specific mode:
```bash
python train_all_exercises_all_modes.py --all --mode imu
```

### Skip existing models:
```bash
python train_all_exercises_all_modes.py --all --skip-existing
```

### Single-output models:
```bash
python train_all_exercises_all_modes.py --all --single-output
```

