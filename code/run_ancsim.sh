#!/bin/bash

# Script to run data generation and IR assessment

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

echo "Starting ANC simulation pipeline..."
echo ""

# Step 1: Render impulse responses
echo "Step 1: Rendering impulse responses (03a_generate_data_rtsim.py)..."
uv run 03a_generate_data_rtsim.py
echo "✓ Impulse response rendering completed"
echo ""

# Step 2: Calculate filter coefficients
echo "Step 2: Calculating filter coefficients (03b_calculate_irs.py)..."
uv run 03b_calculate_irs.py
echo "✓ Filter coefficient calculation completed"
echo ""

# Step 3: Build FAUST apps
echo "Step 3: Building FAUST apps (03c_build_faust_apps.py)..."
uv run 03c_build_faust_apps.py
echo "✓ FAUST app building completed"
echo ""

# Step 4: Create innovation signals
echo "Step 4: Creating innovation signals (03d_create_innovationsignal.py)..."
uv run 03d_create_innovationsignal.py
echo "✓ Innovation signal creation completed"
echo ""

# Step 5: Run ANC simulation
echo "Step 5: Running ANC simulation (03e_run_anc_simulation.py)..."
uv run 03e_run_anc_simulation.py
echo "✓ ANC simulation completed"
echo ""

# Step 6: Assess ANC performance
echo "Step 6: Assessing ANC performance (03f_analysis_anc_simulation.py)..."
uv run 03f_analysis_anc_simulation.py
echo "✓ ANC performance assessment completed"
echo ""

echo "ANC simulation pipeline completed successfully!"