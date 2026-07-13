param(
    [int]$Seed = 0,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$MayTask = "unknown_contrastive_may37_source_repro"
$Train = Join-Path $Root "data\images\unknown_train_defectaware_260710"
$Eval = Join-Path $Root "data\images\unknown_eval100"
$Embeddings = Join-Path $Root "result_grouping\_unknown_mixed260710\embeddings"
$ScoreDir = Join-Path $Root "docs\paper\canonical_rescore_260713\unknown_strict_novel"
$Log = Join-Path $ScoreDir "hard42_method_ablation_queue.log"

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $Log -Append
}

function Wait-ForMaySource {
    while ($true) {
        $task = Get-ScheduledTask -TaskName $MayTask
        if ($task.State -eq "Running") {
            Write-QueueLog "waiting for May CNN/TAPT and no-CNN source reproduction"
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $info = Get-ScheduledTaskInfo -TaskName $MayTask
        Write-QueueLog "May source queue ended with result $($info.LastTaskResult); starting independent hard-42 ablations"
        return
    }
}

function Wait-ForGpu {
    while ($true) {
        $busy = Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and $_.CommandLine -match "multilabel_synth\.run_wm38|_ssl_methods\.py|run_may37_original_ablation\.py"
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

trap {
    Write-QueueLog "FAIL: $($_.Exception.Message)"
    Write-QueueLog "STACK: $($_.ScriptStackTrace)"
    exit 1
}

Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path $Embeddings, $ScoreDir | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:CUDA_VISIBLE_DEVICES = "0"

$configs = @(
    @{ tag = "unkda_q4k_s$Seed"; note = "queue4096 only"; args = @("--method", "simclr", "--use-queue", "--queue-size", "4096") },
    @{ tag = "unkda_local010_s$Seed"; note = "local loss 0.10 only"; args = @("--method", "simclr", "--local", "0.10") },
    @{ tag = "unkda_local030_s$Seed"; note = "local loss 0.30 only"; args = @("--method", "simclr", "--local", "0.30") },
    @{ tag = "unkda_adapter_s$Seed"; note = "zero-init adapter only"; args = @("--method", "simclr", "--head", "adapter", "--lr-bb", "0") },
    @{ tag = "unkda_neco010_s$Seed"; note = "NeCo 0.10 only"; args = @("--method", "simclr", "--neco", "0.10") },
    @{ tag = "unkda_nv050_noaffine_s$Seed"; note = "NV0.50 plus no affine invariance"; args = @("--method", "simclr", "--nv-filter", "0.50", "--wafer-rot-deg", "0", "--wafer-translate", "0", "--wafer-scale-min", "1.0") }
)

Write-QueueLog "hard-42 single-variable ablation queue started; seed=$Seed"
Wait-ForMaySource
foreach ($config in $configs) {
    $final = Join-Path $Embeddings "$($config.tag)_ep10.npy"
    if (Test-Path $final) {
        Write-QueueLog "SKIP $($config.tag): ep10 embedding already exists"
    } else {
        Wait-ForGpu
        Write-QueueLog "START $($config.tag): $($config.note)"
        & python -u _ssl_methods.py @($config.args) --epochs 10 --batch 8 --temp 0.05 `
            --train-dir $Train --eval-dir $Eval --out-dir $Embeddings --tag $config.tag `
            --seed $Seed --palette-mode grade_only --ckpt-every 1000 --fresh *>> $Log
        if ($LASTEXITCODE -ne 0) { throw "training failed for $($config.tag) exit=$LASTEXITCODE" }
    }
    $env:CUDA_VISIBLE_DEVICES = ""
    & python -u scripts\rescore_unknown_strict_novel.py *>> $Log
    if ($LASTEXITCODE -ne 0) { throw "canonical rescore failed for $($config.tag) exit=$LASTEXITCODE" }
    $env:CUDA_VISIBLE_DEVICES = "0"
    Write-QueueLog "DONE $($config.tag)"
}
Write-QueueLog "DONE hard-42 single-variable ablation queue"
