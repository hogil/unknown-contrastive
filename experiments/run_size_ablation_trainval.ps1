$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
New-Item -ItemType Directory -Force -Path "log" | Out-Null

$runs = @(
    @{ Size = 384;  Batch = 16; Tag = "sz384_n50"  },
    @{ Size = 512;  Batch = 8;  Tag = "sz512_n50"  },
    @{ Size = 1024; Batch = 2;  Tag = "sz1024_n50" }
)

foreach ($run in $runs) {
    # cnn_train.py 본체가 log/<run_dir>/run.log 자체 기록하므로 외부 Tee는 제거.
    # 콘솔 stdout은 그대로 보이고, 영구 기록은 run 폴더 안에 정리됨.
    # 3-way split (default 0.8/0.1/0.1) — best val 갱신 시 test도 자동 평가/저장.
    $pythonArgs = @(
        "-u",
        "cnn_train.py",
        "--epochs", "30",
        "--subset-config", "experiments/ablation_size_n50.yaml",
        "--img-size", "$($run.Size)",
        "--batch", "$($run.Batch)",
        "--model-tag", "$($run.Tag)"
    )

    Write-Host "==== start size=$($run.Size) batch=$($run.Batch) tag=$($run.Tag) $(Get-Date -Format o) ===="
    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "size=$($run.Size) failed with exit code $LASTEXITCODE"
    }
    Write-Host "==== done size=$($run.Size) $(Get-Date -Format o) ===="
}
