param()

$ErrorActionPreference = "Stop"
$Root = "D:\project\unknown-contrastive"
$Runner = Join-Path $Root "scripts\run_may37_original_ablation.py"
$Anchor = Join-Path $Root "data\images\anchor_avg30_repro"
$ControlId = "may37_protocol_control_current2260"
$ResultsRoot = Join-Path $Root "runs\may37_protocol_control_current2260"
$Log = Join-Path $ResultsRoot "may37_protocol_control_queue.log"
$Hard42StateDir = Join-Path $Root "docs\paper\canonical_rescore_260713\unknown_strict_novel"
$Hard42Completion = Join-Path $Hard42StateDir "hard42_seed_validation.complete"
$Hard42Failure = Join-Path $Hard42StateDir "hard42_seed_validation.failed"
$Hard42Deferred = Join-Path $Hard42StateDir "hard42_seed_validation.deferred"
$Hard42Task = "unknown_contrastive_hard42_blend_seed_validation"

function Write-QueueLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $Log -Append
}

trap {
    Write-QueueLog "FAIL: $($_.Exception.Message)"
    Write-QueueLog "STACK: $($_.ScriptStackTrace)"
    exit 1
}

function Wait-ForHard42SeedValidation {
    while ($true) {
        if (Test-Path -LiteralPath $Hard42Deferred) {
            Write-QueueLog "hard-42 seed validation deferred for May reproduction priority"
            return
        }
        $task = Get-ScheduledTask -TaskName $Hard42Task -ErrorAction SilentlyContinue
        if ($null -ne $task -and $task.State -eq "Running") {
            Write-QueueLog "waiting for running higher-priority hard-42 fixed-recipe seed validation"
            Start-Sleep -Seconds 60
            continue
        }
        if (Test-Path -LiteralPath $Hard42Completion) {
            Write-QueueLog "hard-42 fixed-recipe seed validation complete; starting May source reproduction"
            return
        }
        if (Test-Path -LiteralPath $Hard42Failure) {
            Write-QueueLog "hard-42 seed validation failed; preserving May reproduction by continuing"
            return
        }
        Write-QueueLog "waiting for higher-priority hard-42 fixed-recipe seed validation"
        Start-Sleep -Seconds 60
    }
}

function Wait-ForGpu {
    while ($true) {
        $known = Get-CimInstance Win32_Process |
            Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -match "multilabel_synth.run_wm38" }
        $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null
        if ($line -match "(\d+)\s*%.*?(\d+)\s*MiB") {
            $util = [int]$Matches[1]
            $mem = [int]$Matches[2]
            # WDDM desktop compositing holds the RTX 4060 Ti at roughly 15-20%
            # even when no compute job is running. The explicit known-CNN guard
            # and low memory ceiling still prevent a training collision.
            $available = (-not $known) -and $util -le 30 -and $mem -le 1200
            Write-QueueLog "GPU check util=$util% mem=${mem}MiB known_cnn=$($null -ne $known) available=$available"
            if ($available) { return }
        }
        Start-Sleep -Seconds 60
    }
}

function Test-Completed([string]$Backbone, [string]$Cell) {
    $suffix = "_{0}_{1}_{2}" -f $ControlId, $Backbone, $Cell.ToLower()
    $matches = @(Get-ChildItem -Path $ResultsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -like "*$suffix" -and
            (Test-Path (Join-Path $_.FullName "completion.json"))
        })
    return $matches.Count -gt 0
}

Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
Remove-Item Env:\PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "0"

New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
if (-not (Test-Path -LiteralPath $Anchor)) {
    throw "control anchor is unavailable: $Anchor"
}
Write-QueueLog "May source-protocol B0-B6 control started. source=b796ecbe5f70 anchor=$Anchor control=$ControlId"
Wait-ForHard42SeedValidation
Wait-ForGpu

$cells = @("FROZEN", "PCA128", "RANDOM128", "B0", "B1", "B2", "B3", "B4", "B5", "B6")
# Establish the historical TAPT control before changing exactly one variable:
# replace the TAPT checkpoint with the ImageNet FCMAE checkpoint.
foreach ($backbone in @("cnn_tapt", "nocnn")) {
    foreach ($cell in $cells) {
        if (Test-Completed $backbone $cell) {
            Write-QueueLog "SKIP completed backbone=$backbone cell=$cell"
            continue
        }
        Wait-ForGpu
        Write-QueueLog "START backbone=$backbone cell=$cell"
        & python -u $Runner --backbone $backbone --cell $cell --anchor $Anchor --control-id $ControlId --output-root $ResultsRoot *>> $Log
        if ($LASTEXITCODE -ne 0) {
            throw "failed backbone=$backbone cell=$cell exit=$LASTEXITCODE"
        }
        Write-QueueLog "DONE backbone=$backbone cell=$cell"
    }
}
& python -u (Join-Path $Root "scripts\summarize_may37_original_ablation.py") --results-root $ResultsRoot *>> $Log
if ($LASTEXITCODE -ne 0) {
    throw "failed to summarize May source-protocol B0-B6 control cells exit=$LASTEXITCODE"
}
Write-QueueLog "DONE all source-protocol May B0-B6 control cells."

$Hard42Runner = Join-Path $Root "scripts\run_hard42_headonly_coordinate_descent.py"
$Hard42Output = Join-Path $Root "runs\hard42_headonly_coordinate_descent"
Write-QueueLog "START hard-42 frozen-backbone head-only coordinate descent (cnn_tapt -> nocnn)."
& python -u $Hard42Runner --output-root $Hard42Output --backbones "cnn_tapt,nocnn" *>> $Log
if ($LASTEXITCODE -ne 0) {
    throw "failed hard-42 head-only coordinate descent exit=$LASTEXITCODE"
}
Write-QueueLog "DONE hard-42 frozen-backbone head-only coordinate descent."
