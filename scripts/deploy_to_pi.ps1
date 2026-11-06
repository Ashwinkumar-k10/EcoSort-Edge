param (
    [Parameter(Mandatory=$true, HelpMessage="Enter your Raspberry Pi IP address (e.g. 192.168.1.50)")]
    [string]$PiIP,

    [Parameter(Mandatory=$false)]
    [string]$User = "pi",

    [Parameter(Mandatory=$false)]
    [string]$RemoteDir = "~/ecosort_edge"
)

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host "      EcoSort-Edge: Automated Deployment to Raspberry Pi 5               " -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " • Target Pi IP   : $PiIP"
Write-Host " • Pi Username    : $User"
Write-Host " • Remote Path    : $RemoteDir"
Write-Host "=========================================================================="

# 1. Test Network Ping
Write-Host "`n[1/4] Pinging Raspberry Pi 5 at $PiIP..." -ForegroundColor Yellow
$ping = Test-Connection -ComputerName $PiIP -Count 2 -Quiet
if (-not $ping) {
    Write-Host "[!] Warning: Pi 5 does not respond to ping. Ensure Pi 5 and Laptop are on the same Wi-Fi / Ethernet." -ForegroundColor Red
} else {
    Write-Host "[+] Pi 5 is online and reachable!" -ForegroundColor Green
}

# 2. Create Remote Directory on Pi
Write-Host "`n[2/4] Setting up remote project directory on Pi 5..." -ForegroundColor Yellow
ssh ${User}@${PiIP} "mkdir -p ${RemoteDir}/models/checkpoints ${RemoteDir}/scripts ${RemoteDir}/data/smoke_test ${RemoteDir}/results"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] SSH connection failed. Check your username, password, or SSH enablement on Pi." -ForegroundColor Red
    exit 1
}

# 3. Transfer Core Deployment Files via SCP
Write-Host "`n[3/4] Copying Model Binaries, Scripts & Smoke Test Images..." -ForegroundColor Yellow
scp -r demo_modes.py evaluate_pipeline.py predict.py ${User}@${PiIP}:${RemoteDir}/
scp -r scripts/decision_engine.py scripts/llm_explainer.py ${User}@${PiIP}:${RemoteDir}/scripts/
scp -r models/mobilenetv3_large_int8.pte models/mobilenetv3_large_optimized.pt ${User}@${PiIP}:${RemoteDir}/models/
scp -r models/checkpoints/mobilenetv3_ecosort.pth ${User}@${PiIP}:${RemoteDir}/models/checkpoints/
scp -r data/smoke_test/* ${User}@${PiIP}:${RemoteDir}/data/smoke_test/

Write-Host "`n[4/4] Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " To run on your Raspberry Pi 5, execute in terminal:" -ForegroundColor White
Write-Host "   ssh ${User}@${PiIP}" -ForegroundColor Yellow
Write-Host "   cd ${RemoteDir}" -ForegroundColor Yellow
Write-Host "   python3 demo_modes.py" -ForegroundColor Yellow
Write-Host "   python3 evaluate_pipeline.py --image_dir data/smoke_test/ --device cpu" -ForegroundColor Yellow
Write-Host "==========================================================================" -ForegroundColor Cyan
