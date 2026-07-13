param(
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$Holdout = Join-Path $Root "data\images\unknown_holdout_100_260713"
$Embeddings = Join-Path $Root "result_grouping\_hard42_holdout_260713\embeddings"
$Scores = Join-Path $Root "docs\paper\canonical_rescore_260713\unknown_holdout_260713"
$Log = Join-Path $Scores "fcmae_holdout_queue.log"

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $Log -Append
}

function Wait-ForKnownCnn {
    while ($true) {
        $known = Get-CimInstance Win32_Process |
            Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "multilabel_synth\.run_wm38" }
        if (-not $known) { return }
        Write-QueueLog "waiting for user known-CNN job before CPU holdout extraction"
        Start-Sleep -Seconds $PollSeconds
    }
}

trap {
    Write-QueueLog "FAIL: $($_.Exception.Message)"
    Write-QueueLog "STACK: $($_.ScriptStackTrace)"
    exit 1
}

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $Embeddings, $Scores | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"

& python -u scripts\make_unknown_holdout_260713.py *>> $Log
if ($LASTEXITCODE -ne 0) { throw "holdout construction failed with exit $LASTEXITCODE" }

Wait-ForKnownCnn
$dino = Join-Path $Embeddings "dino_frozen_holdout.npy"
$fcmae = Join-Path $Embeddings "fcmae_frozen_holdout.npy"
if (-not (Test-Path $dino)) {
    Write-QueueLog "START DINOv3 frozen holdout extraction"
    & python -u _frozen_embed.py --eval-dir $Holdout --out $dino *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "DINOv3 holdout extraction failed with exit $LASTEXITCODE" }
}
if (-not (Test-Path $fcmae)) {
    Write-QueueLog "START FCMAE frozen holdout extraction"
    & python -u _frozen_embed.py --eval-dir $Holdout --out $fcmae --timm convnextv2_base.fcmae_ft_in22k_in1k_384 *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "FCMAE holdout extraction failed with exit $LASTEXITCODE" }
}

Write-QueueLog "START canonical holdout scoring"
& python -u scripts\rescore_unknown_strict_novel.py --embeddings $Embeddings --frozen $dino `
    --extra-frozen "fcmae_frozen=$fcmae" --pool $Holdout --output-dir $Scores --refresh *>> $Log
if ($LASTEXITCODE -ne 0) { throw "canonical holdout scoring failed with exit $LASTEXITCODE" }
Write-QueueLog "DONE FCMAE vs DINO hard-42 image-disjoint holdout validation"
