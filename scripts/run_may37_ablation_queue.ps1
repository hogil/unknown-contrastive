param(
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$Log = Join-Path $Root "_may37_ablation_queue.log"
$KnownCnnPid = 24628
$Runner = Join-Path $Root "scripts\run_may37_ablation.py"
$ResultsRoot = Join-Path $Root "runs\may37_manifest_reproduction"

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $Log -Append
}

function Wait-ForGpu {
    if (-not $StartNow) {
        try {
            $process = Get-Process -Id $KnownCnnPid -ErrorAction Stop
            Write-QueueLog "Waiting for external known-CNN process PID=$($process.Id) to finish."
            Wait-Process -Id $KnownCnnPid
        } catch {
            Write-QueueLog "External known-CNN process PID=$KnownCnnPid is already absent."
        }
    }
    $quiet = 0
    while ($quiet -lt 3) {
        $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
        if ($line -match "(\d+)\s*%.*?(\d+)\s*MiB") {
            $util = [int]$Matches[1]
            $mem = [int]$Matches[2]
            if ($util -le 10 -and $mem -le 1800) {
                $quiet++
            } else {
                $quiet = 0
            }
            Write-QueueLog "GPU check util=$util% mem=${mem}MiB quiet=$quiet/3"
        }
        if ($quiet -lt 3) { Start-Sleep -Seconds 60 }
    }
}

function Test-Completed([string]$Backbone, [string]$Cell) {
    $suffix = "_may37_{0}_{1}" -f $Backbone, $Cell.ToLower()
    $matches = @(Get-ChildItem -Path $ResultsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "*$suffix" -and
            (Test-Path (Join-Path $_.FullName "contrastive\may37_epoch_metrics.csv"))
        })
    return $matches.Count -gt 0
}

Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:USE_LIBUV = "0"

Write-QueueLog "May-37 historical-equivalent ablation queue started."
Wait-ForGpu

$cells = @("FROZEN", "B0", "B1", "B2", "B3", "B4", "B5")
foreach ($backbone in @("cnn_tapt", "nocnn")) {
    foreach ($cell in $cells) {
        if (Test-Completed $backbone $cell) {
            Write-QueueLog "SKIP completed backbone=$backbone cell=$cell"
            continue
        }
        Write-QueueLog "START backbone=$backbone cell=$cell"
        & python -u $Runner --backbone $backbone --cell $cell *>> $Log
        if ($LASTEXITCODE -ne 0) {
            throw "failed backbone=$backbone cell=$cell exit=$LASTEXITCODE"
        }
        Write-QueueLog "DONE backbone=$backbone cell=$cell"
    }
}
Write-QueueLog "DONE all May-37 historical-equivalent ablation cells."
