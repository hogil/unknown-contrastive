param(
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$ResultsRoot = Join-Path $Root "runs\may37_original_reproduction"
$Summary = Join-Path $Root "scripts\summarize_may37_original_ablation.py"
$Log = Join-Path $ResultsRoot "may37_original_summary_watcher.log"

function Write-WatcherLog([string]$Message) {
    "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message |
        Tee-Object -FilePath $Log -Append
}

Write-WatcherLog "watching source-faithful May queue"
while ($true) {
    $queue = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq "powershell.exe" -and $_.CommandLine -match "run_may37_original_ablation_queue\.ps1" }
    if (-not $queue) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

$metrics = @(Get-ChildItem -Path $ResultsRoot -Filter metrics.json -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "canonical_eval" })
if ($metrics.Count -eq 0) {
    Write-WatcherLog "queue exited without completed canonical metrics; no summary written"
    exit 0
}

Write-WatcherLog "queue exited; summarizing $($metrics.Count) completed cells"
& python -u $Summary --results-root $ResultsRoot *>> $Log
if ($LASTEXITCODE -ne 0) {
    throw "May summary failed with exit $LASTEXITCODE"
}
Write-WatcherLog "summary complete"
