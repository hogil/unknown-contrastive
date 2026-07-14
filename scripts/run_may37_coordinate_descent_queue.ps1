param()

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$OutputRoot = Join-Path $Root "runs\may37_coordinate_descent"
$Log = Join-Path $OutputRoot "coordinate_queue.log"

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $Log -Append
}

function Wait-ForGpu {
    while ($true) {
        $busy = Get-CimInstance Win32_Process |
            Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "_ssl_methods\.py|run_may37" }
        $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
        if ($line -match "(\d+)\s*%.*?(\d+)\s*MiB") {
            $util = [int]$Matches[1]
            $mem = [int]$Matches[2]
            if ((-not $busy) -and $util -le 30 -and $mem -le 1200) { return }
        }
        Start-Sleep -Seconds 60
    }
}

trap {
    Write-QueueLog "FAIL: $($_.Exception.Message)"
    Write-QueueLog "STACK: $($_.ScriptStackTrace)"
    exit 1
}

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
Remove-Item Env:\PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "0"

# Coordinate descent follows the same paired-control order as the source
# reproduction: TAPT first, then the otherwise identical no-TAPT arm.
foreach ($backbone in @("cnn_tapt", "nocnn")) {
    Wait-ForGpu
    Write-QueueLog "START coordinate descent backbone=$backbone"
    & python -u scripts\run_may37_coordinate_descent.py --backbone $backbone *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "coordinate descent failed backbone=$backbone exit=$LASTEXITCODE" }
    Write-QueueLog "DONE coordinate descent backbone=$backbone"
}

Write-QueueLog "DONE all May coordinate-descent reproductions"
