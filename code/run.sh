#!/bin/bash

# Script to run data generation and IR assessment

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

echo "Starting pipeline..."
echo ""

# Step 1: Generate training data
echo "Step 1: Generating data (01_generate_data_tr.py)..."
python 01_generate_data_tr.py
echo "✓ Data generation completed"
echo ""

# Step 2: Assess IRs
echo "Step 2: Assessing IRs (02_ir_assessment.py)..."
python 02_ir_assessment.py
echo "✓ IR assessment completed"
echo ""

echo "Pipeline completed successfully!"
