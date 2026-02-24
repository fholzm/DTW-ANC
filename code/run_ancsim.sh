#!/bin/bash

# Script to run data generation and IR assessment

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

echo "Starting ANC simulation pipeline..."
echo ""

# Step 1: Calculate filter coefficients
echo "Step 1: Calculating filter coefficients (03a_calculate_irs.py)..."
uv run 03a_calculate_irs.py
echo "✓ Filter coefficient calculation completed"
echo ""

# Step 2: Build FAUST apps
echo "Step 2: Building FAUST apps (03b_build_faust_apps.py)..."
uv run 03b_build_faust_apps.py
echo "✓ FAUST app building completed"
echo ""

# Step 3: Create innovation signals
echo "Step 3: Creating innovation signals (03c_create_innovationsignal.py)..."
uv run 03c_create_innovationsignal.py
echo "✓ Innovation signal creation completed"
echo ""

# Step 4: Run ANC simulation
echo "Step 4: Running ANC simulation (03d_run_anc_simulation.py)..."
uv run 03d_run_anc_simulation.py
echo "✓ ANC simulation completed"
echo ""

# Step 5: Assess ANC performance
echo "Step 5: Assessing ANC performance (03e_analysis_anc_simulation.py)..."
uv run 03e_analysis_anc_simulation.py
echo "✓ ANC performance assessment completed"
echo ""

echo "ANC simulation pipeline completed successfully!"