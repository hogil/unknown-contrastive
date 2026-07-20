param(
    [Parameter(Mandatory = $true)]
    [int]$MayQueuePid
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$mayLog = Join-Path $root "runs\may_new_tapt_removed_paired_2260\may_new_paired_queue.log"
$handoffLog = Join-Path $root "runs\three_seed_confirm_260720\handoff.log"
$threeSeed = Join-Path $root "_3seed_confirm.sh"
$bash = "C:\Program Files\Git\bin\bash.exe"

$handoffDir = Split-Path -Parent $handoffLog
New-Item -ItemType Directory -Path $handoffDir -Force | Out-Null

function Write-HandoffLog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $handoffLog -Value "[$stamp] $Message"
}

Write-HandoffLog "Waiting for May NEW paired queue PID $MayQueuePid."
while (Get-Process -Id $MayQueuePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 60
}

$completed = (Test-Path -LiteralPath $mayLog) -and
    ((Get-Content -LiteralPath $mayLog -Raw) -match "DONE all six May NEW paired runs\.")
if (-not $completed) {
    Write-HandoffLog "May queue exited without the all-six completion marker; 3-seed launch blocked."
    exit 2
}

$alreadyRunning = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "_3seed_confirm\.sh"
}
if ($alreadyRunning) {
    Write-HandoffLog "3-seed runner already active; duplicate launch skipped."
    exit 0
}

Write-HandoffLog "May queue completed; launching fixed-epoch 3-seed confirmation."
Push-Location $root
try {
    & $bash $threeSeed *>> $handoffLog
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
Write-HandoffLog "3-seed runner exited with code $exitCode."
exit $exitCode
