$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$env:PYTHONIOENCODING = "utf-8"

$Python = "C:\Users\hgcho\AppData\Local\Programs\Python\Python313\python.exe"
$Runner = "D:\project\unknown-contrastive\scripts\run_fcmae_adapter_residual_scale_screen.py"
$HoldoutGate = "D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_HOLDOUT_260725\fcmae_adapter_holdout_260725_gate.json"
$RunRoot = "E:\unknown-contrastive-runs\archives\fcmae_adapter_residual_scale_screen_260725"
$Output = Join-Path $RunRoot "residual_scale_screen.json"
$LockPath = "$RunRoot.lock"
$Log = "$RunRoot.log"

function Write-RunLog {
    param([string]$Message)
    Add-Content -LiteralPath $Log -Encoding UTF8 -Value (
        "{0} {1}" -f ([DateTime]::UtcNow.ToString("o")), $Message
    )
}

function Test-HoldoutGate {
    if (-not (Test-Path -LiteralPath $HoldoutGate -PathType Leaf)) {
        return $false
    }
    try {
        $value = Get-Content -LiteralPath $HoldoutGate -Raw -Encoding UTF8 | ConvertFrom-Json
        return (
            -not [string]::IsNullOrWhiteSpace([string]$value.protocol_id) -and
            $value.gate.accepted -is [bool]
        )
    }
    catch {
        return $false
    }
}

function Test-BlockedProcess {
    $needles = @(
        "fcmae_fixed_protocol.py",
        "run_fcmae_adapter_holdout_validation.py",
        "run_label_free_adaptation.py",
        "run_fcmae_adapter_residual_scale_screen.py"
    )
    foreach ($process in Get-CimInstance Win32_Process) {
        foreach ($needle in $needles) {
            if ([string]$process.CommandLine -like "*$needle*") {
                return $true
            }
        }
    }
    return $false
}

function Test-GpuReady {
    $row = @(
        nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits
    ) | Select-Object -First 1
    if ($null -eq $row) {
        return $false
    }
    $parts = [string]$row -split ","
    if ($parts.Count -lt 2) {
        return $false
    }
    return ([int]$parts[0].Trim() -lt 10 -and [int]$parts[1].Trim() -ge 12000)
}

function Get-FileSha256 {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-ResidualScaleResult {
    if (-not (Test-Path -LiteralPath $Output -PathType Leaf)) {
        return $false
    }
    try {
        $payload = Get-Content -LiteralPath $Output -Raw -Encoding UTF8 | ConvertFrom-Json
        $values = @($payload.gate.values)
        $rows = @($payload.rows)
        if ($values.Count -ne 4 -or $rows.Count -ne 10) {
            return $false
        }
        $alphas = @($values | ForEach-Object { [double]$_.alpha } | Sort-Object)
        $expected = @(0.25, 0.50, 0.75, 1.00)
        for ($index = 0; $index -lt $expected.Count; $index++) {
            if ($alphas[$index] -ne $expected[$index]) {
                return $false
            }
        }
        $source = [string]$payload.provenance.source_checkpoint
        if (
            [string]$payload.provenance.script_sha256 -cne [string](Get-FileSha256 $Runner) -or
            [string]$payload.provenance.source_checkpoint_sha256 -cne [string](Get-FileSha256 $source)
        ) {
            return $false
        }
        & $Python -u $Runner --validate-result *> $null
        return ([int]$LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

if (Test-ResidualScaleResult) {
    exit 0
}
if (-not (Test-HoldoutGate)) {
    exit 0
}
if (Test-BlockedProcess) {
    exit 0
}
if (-not (Test-GpuReady)) {
    exit 0
}

New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
$lock = $null
try {
    try {
        $lock = [System.IO.File]::Open(
            $LockPath,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
    }
    catch [System.IO.IOException] {
        exit 0
    }
    Write-RunLog "START"
    & $Python -u $Runner *>> $Log
    $code = [int]$LASTEXITCODE
    Write-RunLog ("END exit_code={0}" -f $code)
    exit $code
}
finally {
    if ($null -ne $lock) {
        $lock.Dispose()
    }
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}
