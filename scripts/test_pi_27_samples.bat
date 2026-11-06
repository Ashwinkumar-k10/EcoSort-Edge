@echo off
set PI_IP=%1
if "%PI_IP%"=="" (
    echo Usage: test_pi_27_samples.bat ^<PI_IP^>
    echo Example: test_pi_27_samples.bat 172.16.12.xxx
    exit /b 1
)

echo ===========================================================================
echo   Deploying 27 Real Test Images and ExecuTorch INT8 Engine to Pi 5 (%PI_IP%)
echo ===========================================================================

echo [*] Creating target directories on Pi...
ssh -o ConnectTimeout=5 pi5@%PI_IP% "mkdir -p ~/ecosort_edge/data/test_27_samples ~/ecosort_edge/models ~/ecosort_edge/results"

echo [*] Transferring 4.31 MB INT8 ExecuTorch binary (.pte)...
scp models\mobilenetv3_large_int8.pte pi5@%PI_IP%:~/ecosort_edge/models/

echo [*] Transferring 27 real holdout test images and ground truth...
scp data\test_27_samples\*.* pi5@%PI_IP%:~/ecosort_edge/data/test_27_samples/

echo [*] Transferring evaluation scripts and decision engine...
scp evaluate_pipeline.py demo_modes.py predict.py pi5@%PI_IP%:~/ecosort_edge/
scp scripts\decision_engine.py pi5@%PI_IP%:~/ecosort_edge/scripts/

echo [*] Executing on-device benchmark on Raspberry Pi 5...
ssh pi5@%PI_IP% "cd ~/ecosort_edge && python3 evaluate_pipeline.py --image_dir data/test_27_samples --model_path models/mobilenetv3_large_int8.pte --device cpu --output_dir results/pi_results"

echo [*] Fetching results back to laptop...
scp pi5@%PI_IP%:~/ecosort_edge/results/pi_results/deliverable_d3_results.json results\pi_27_results.json

echo ===========================================================================
echo   Pi 5 Test Complete! Results saved to results\pi_27_results.json
echo ===========================================================================
