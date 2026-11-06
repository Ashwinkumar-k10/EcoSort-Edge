@echo off
title EcoSort-Edge: Benchmark Evaluator
echo ============================================================================
echo   EcoSort-Edge: Holdout Test Benchmark (Deliverables D3 & D4)
echo ============================================================================
python evaluate_pipeline.py --image_dir data/processed/test/ --output_dir results/
pause
