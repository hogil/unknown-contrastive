$ErrorActionPreference = "Continue"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = "D:\project\unknown-contrastive"
$Python = "C:\Users\hgcho\AppData\Local\Programs\Python\Python313\python.exe"
$Protocol = "D:\project\unknown-contrastive\scripts\fcmae_fixed_protocol.py"
$Holdout = "D:\project\unknown-contrastive\scripts\run_fcmae_adapter_holdout_validation.py"
$Gate = "D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_HOLDOUT_260725\fcmae_adapter_holdout_260725_gate.json"
$Seed3Checkpoint = "D:\project\unknown-contrastive\runs\fcmae_adapter_ep4_three_seed_260721\embeddings\fcmae_ad1_s3_ckpt.pt"
$Log = "D:\project\unknown-contrastive\runs\fcmae_adapter_ep4_three_seed_260721\confirmation_chain.log"

New-Item -ItemType Directory -Force -Path "D:\project\unknown-contrastive\runs\fcmae_adapter_ep4_three_seed_260721" | Out-Null

function Write-ChainLog {
    param([string]$Message)
    Add-Content -LiteralPath $Log -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ssK"), $Message)
}

function Test-HoldoutGate {
    if (-not (Test-Path -LiteralPath $Gate -PathType Leaf)) {
        return $false
    }
    try {
        $payload = Get-Content -LiteralPath $Gate -Raw -Encoding UTF8 | ConvertFrom-Json
        return (
            -not [string]::IsNullOrWhiteSpace([string]$payload.protocol_id) -and
            $payload.gate.accepted -is [bool]
        )
    }
    catch {
        return $false
    }
}

function Test-GpuReady {
    try {
        $row = @(
            nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits 2>$null
        ) | Select-Object -First 1
        if ($null -eq $row) {
            return $false
        }
        $parts = [string]$row -split ","
        if ($parts.Count -lt 2) {
            return $false
        }
        return ([int]$parts[0].Trim() -lt 10 -and [int]$parts[1].Trim() -ge 6000)
    }
    catch {
        return $false
    }
}

Write-ChainLog "chain invoked"

if (Test-HoldoutGate) {
    Write-ChainLog "valid final holdout gate exists: $Gate; exit 0"
    exit 0
}

$CurrentPid = $PID
$RunningProtocol = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
    Where-Object {
        $_.ProcessId -ne $CurrentPid -and
        $_.CommandLine -match '(?i)[\\/]scripts[\\/]fcmae_fixed_protocol\.py["'']?\s+run-seeds(?:\s|$)'
    })

if ($RunningProtocol.Count -gt 0) {
    foreach ($Process in $RunningProtocol) {
        Write-ChainLog ("run-seeds already running: pid={0}; exit 0" -f $Process.ProcessId)
    }
    exit 0
}

$RunningSeedWorker = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
    Where-Object {
        $_.CommandLine -match '(?i)[\\/]_ssl_methods\.py["'']?(?:\s|$)' -and
        $_.CommandLine -match '(?i)--tag\s+["'']?fcmae_ad1_s(?:1|3|5)["'']?(?:\s|$)'
    })

if ($RunningSeedWorker.Count -gt 0) {
    foreach ($Process in $RunningSeedWorker) {
        Write-ChainLog ("seed worker already running: pid={0}; exit 0" -f $Process.ProcessId)
    }
    exit 0
}

if (-not (Test-GpuReady)) {
    Write-ChainLog "GPU busy or insufficient free memory; defer"
    exit 0
}

Write-ChainLog "starting fixed protocol: $Protocol run-seeds"
Set-Location -LiteralPath $ProjectRoot
& $Python $Protocol run-seeds *>> $Log
$ProtocolExitCode = $LASTEXITCODE
Write-ChainLog "fixed protocol exit code: $ProtocolExitCode"

if ($ProtocolExitCode -ne 0) {
    Write-ChainLog "fixed protocol failed; holdout validation skipped"
    exit $ProtocolExitCode
}

if (-not (Test-Path -LiteralPath $Seed3Checkpoint -PathType Leaf)) {
    Write-ChainLog "seed3 checkpoint missing; holdout validation skipped: $Seed3Checkpoint"
    exit 1
}

Write-ChainLog "starting holdout validation: $Holdout --run --seed3-checkpoint $Seed3Checkpoint"
& $Python $Holdout --run --seed3-checkpoint $Seed3Checkpoint *>> $Log
$HoldoutExitCode = $LASTEXITCODE
Write-ChainLog "holdout validation exit code: $HoldoutExitCode"
exit $HoldoutExitCode
