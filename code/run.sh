#!/usr/bin/env bash
set -euo pipefail

# ── Start JACK audio server (dummy backend, required by TASCAR) ──────────
if ! pgrep -x jackd >/dev/null 2>&1; then
  jackd --no-realtime  -m -d dummy -r 16000 -p 16 &
  sleep 0.5
fi

cd /code

echo "Starting pipeline..."
echo ""

# Step 1: Generate training data
echo "Step 1: Generating data (01_generate_data_tr.py)..."
uv run 01_generate_data_tr.py
echo "✓ Data generation completed"
echo ""

# Step 2: Assess IRs
echo "Step 2: Assessing IRs (02a_ir_assessment.py)..."
uv run 02a_ir_assessment.py
echo "✓ IR assessment completed"
echo ""

# Step 3: Assess contralateral DTW alignment
echo "Step 3: Assessing contralateral DTW alignment (02b_dtw_assessment_clat.py)..."
uv run 02b_dtw_assessment_clat.py
echo "✓ Contralateral DTW alignment assessment completed"
echo ""

echo "Pipeline completed successfully!"
