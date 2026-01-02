#!/bin/bash
# Train all 18 single-output models (6 exercises × 3 modes)

EXERCISES=("bicep_curls" "squats" "lateral_shoulder_raises" "triceps_pushdown" "dumbbell_rows" "dumbbell_shoulder_press")

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Training All 18 Single-Output Models"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "6 Exercises × 3 Modes (Camera, IMU, Fusion) = 18 Models"
echo ""

total_success=0
total_failed=0

for exercise in "${EXERCISES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Exercise: $exercise"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Camera model
    echo ""
    echo "📹 Training Camera Model..."
    if python3 train_ml_models.py --exercise "$exercise" --camera-only --single-output; then
        ((total_success++))
        echo "✅ Camera model trained successfully"
    else
        ((total_failed++))
        echo "❌ Camera model training failed"
    fi
    
    # IMU model
    echo ""
    echo "🎚️ Training IMU Model..."
    if python3 train_ml_models.py --exercise "$exercise" --imu-only --single-output; then
        ((total_success++))
        echo "✅ IMU model trained successfully"
    else
        ((total_failed++))
        echo "❌ IMU model training failed"
    fi
    
    # Fusion model
    echo ""
    echo "🔀 Training Fusion Model..."
    if python3 train_ml_models.py --exercise "$exercise" --fusion --single-output; then
        ((total_success++))
        echo "✅ Fusion model trained successfully"
    else
        ((total_failed++))
        echo "❌ Fusion model training failed"
    fi
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "📊 TRAINING SUMMARY"
echo "════════════════════════════════════════════════════════════════"
echo "Total Models Trained: $total_success"
echo "Total Failed: $total_failed"
echo "Total Attempted: $((total_success + total_failed))"
echo "════════════════════════════════════════════════════════════════"

if [ $total_failed -eq 0 ]; then
    echo "✅ All models trained successfully!"
    exit 0
else
    echo "⚠️  Some models failed to train"
    exit 1
fi

