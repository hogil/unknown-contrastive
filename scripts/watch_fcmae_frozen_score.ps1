param([int]$PollSeconds = 60)

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$Embedding = Join-Path $Root "result_grouping\_unknown_mixed260710\embeddings\frozen_unknown_fcmae.npy"
$Log = Join-Path $Root "docs\paper\canonical_rescore_260713\unknown_strict_novel\fcmae_frozen_score_watcher.log"

function Write-WatcherLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $Log -Append
}

trap {
    Write-WatcherLog "FAIL: $($_.Exception.Message)"
    Write-WatcherLog "STACK: $($_.ScriptStackTrace)"
    exit 1
}

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null
Write-WatcherLog "waiting for FCMAE frozen embedding: $Embedding"
while (-not (Test-Path -LiteralPath $Embedding)) {
    Start-Sleep -Seconds $PollSeconds
}
Write-WatcherLog "FCMAE frozen embedding found; starting canonical rescore"
$env:CUDA_VISIBLE_DEVICES = ""
$env:PYTHONIOENCODING = "utf-8"
& python -u scripts\rescore_unknown_strict_novel.py *>> $Log
if ($LASTEXITCODE -ne 0) { throw "FCMAE canonical rescore failed with exit $LASTEXITCODE" }
& python -u scripts\build_robust_grouping_status.py *>> $Log
if ($LASTEXITCODE -ne 0) { throw "FCMAE status build failed with exit $LASTEXITCODE" }
& python -u scripts\plot_unknown_blend_sweep.py *>> $Log
if ($LASTEXITCODE -ne 0) { throw "FCMAE blend plot refresh failed with exit $LASTEXITCODE" }
Write-WatcherLog "FCMAE canonical scoring complete"
