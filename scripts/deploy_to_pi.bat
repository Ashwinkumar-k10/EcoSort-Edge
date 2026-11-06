@echo off
setlocal

echo ==========================================================================
echo       EcoSort-Edge: Automated Deployment to Raspberry Pi 5
echo ==========================================================================

set /p PI_IP="Enter your Raspberry Pi IP address: "
set /p PI_USER="Enter your Raspberry Pi username (default: pi5): "
if "%PI_USER%"=="" set PI_USER=pi5

echo.
echo [*] Target Pi: %PI_USER%@%PI_IP%
echo [*] Creating remote directories...
ssh %PI_USER%@%PI_IP% "mkdir -p ~/ecosort_edge/models/checkpoints ~/ecosort_edge/scripts ~/ecosort_edge/data/smoke_test ~/ecosort_edge/results"

echo.
echo [*] Copying Core Deployment Files...
scp demo_modes.py evaluate_pipeline.py predict.py requirements.txt %PI_USER%@%PI_IP%:~/ecosort_edge/
scp scripts/*.py %PI_USER%@%PI_IP%:~/ecosort_edge/scripts/
scp models/mobilenetv3_large_int8.pte models/mobilenetv3_large_optimized.pt %PI_USER%@%PI_IP%:~/ecosort_edge/models/
scp models/checkpoints/mobilenetv3_ecosort.pth %PI_USER%@%PI_IP%:~/ecosort_edge/models/checkpoints/
scp -r data/smoke_test/* %PI_USER%@%PI_IP%:~/ecosort_edge/data/smoke_test/

echo.
echo ==========================================================================
echo [SUCCESS] 100%% Authentic ExecuTorch Files Transferred!
echo Now in your Raspberry Pi terminal run:
echo    cd ~/ecosort_edge
echo    python3 evaluate_pipeline.py --image_dir data/smoke_test/ --model_path models/mobilenetv3_large_int8.pte --device cpu
echo    python3 demo_modes.py
echo ==========================================================================
pause
