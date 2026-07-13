param(
    [int[]]$Seeds = @(1, 2),
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$MayTask = "unknown_contrastive_may37_source_repro"
$Train = Join-Path $Root "data\images\unknown_train_defectaware_260710"
$Eval = Join-Path $Root "data\images\unknown_eval100"
$Embeddings = Join-Path $Root "result_grouping\_unknown_mixed260710\embeddings"
$Log = Join-Path $Root "docs\paper\canonical_rescore_260713\unknown_strict_novel\hard42_seed_validation_queue.log"

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $Log -Append
}

trap {
    Write-QueueLog "FAIL: $($_.Exception.Message)"
    Write-QueueLog "STACK: $($_.ScriptStackTrace)"
    exit 1
}

function Wait-ForMayReproduction {
    while ($true) {
        $task = Get-ScheduledTask -TaskName $MayTask
        if ($task.State -eq "Running") {
            Write-QueueLog "waiting for May CNN/no-CNN source reproduction"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $info = Get-ScheduledTaskInfo -TaskName $MayTask
        if ($info.LastTaskResult -ne 0) {
            throw "May source reproduction ended with task result $($info.LastTaskResult)"
        }
        return
    }
}

function Wait-ForGpu {
    while ($true) {
        $busy = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and $_.CommandLine -match "multilabel_synth\.run_wm38|_ssl_methods\.py|run_contrastive\.py|run_may37_original_ablation\.py"
            }
        $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
        if ($line -match "(\d+)\s*%.*?(\d+)\s*MiB") {
            $util = [int]$Matches[1]
            $mem = [int]$Matches[2]
            $available = (-not $busy) -and $util -le 30 -and $mem -le 1200
            Write-QueueLog "GPU check util=$util% mem=${mem}MiB busy=$($null -ne $busy) available=$available"
            if ($available) { return }
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Log) | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUDA_VISIBLE_DEVICES = "0"

Write-QueueLog "hard-42 blend validation queue started; seeds=$($Seeds -join ',')"
Wait-ForMayReproduction
foreach ($seed in $Seeds) {
    Wait-ForGpu
    $tag = "unkda_nv050_s$seed"
    Write-QueueLog "START ${tag}: fixed NV=0.50, ep6 blend weight=0.86"
    & python -u _ssl_methods.py --method simclr --nv-filter 0.50 --epochs 10 --batch 8 `
        --train-dir $Train --eval-dir $Eval --out-dir $Embeddings --tag $tag --seed $seed `
        --palette-mode grade_only --ckpt-every 1000 --fresh *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "training failed for $tag exit=$LASTEXITCODE" }
    $trained = Join-Path $Embeddings "${tag}_ep6.npy"
    if (-not (Test-Path $trained)) { throw "missing ep6 embedding for ${tag}: $trained" }
    & python -u scripts\make_unknown_embedding_blends.py --trained $trained --output-dir $Embeddings `
        --weights 0.86 --tag-prefix $tag --epoch 6 *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "blend construction failed for $tag exit=$LASTEXITCODE" }
    $env:CUDA_VISIBLE_DEVICES = ""
    & python -u scripts\rescore_unknown_strict_novel.py *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "canonical rescore failed for $tag exit=$LASTEXITCODE" }
    & python -u scripts\build_robust_grouping_status.py *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "status report failed for $tag exit=$LASTEXITCODE" }
    $env:CUDA_VISIBLE_DEVICES = "0"
    Write-QueueLog "DONE $tag"
}
Write-QueueLog "DONE hard-42 blend validation queue"
