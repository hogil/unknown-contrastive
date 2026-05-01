# =============================================================================
# Runner: 입력 사이즈 ablation 순차 실행 (PowerShell)
# =============================================================================
# Intent:
#   `experiments/ablation_size_n50.yaml` (33 class × 50 sample) 으로 384 / 512 /
#   1024 BICUBIC 입력 사이즈를 순차 학습. PS1 한 번 실행으로 3 run 모두 돌리고
#   각 run 종료 후 폴더가 `log/sz{N}_n50_<TS>_<test_f1>_<val_f1>/` 으로 rename 됨.
#
# Hypothesis:
#   1024 BICUBIC > 512 > 384 (chip 격자 fidelity 비례)
#
# Why these batch sizes (4060 Ti 16GB 기준):
#   - 384 / batch 16: 표준. ConvNeXtV2 base + AMP bf16 에서 GPU mem ~10GB.
#   - 512 / batch 8:  pixel area ~1.78x, batch 절반. mem ~12GB.
#   - 1024 / batch 2: pixel area ~7.1x, batch 1/8. mem ~13GB. batch 4는 OOM.
#
# Run:
#   .\experiments\run_size_ablation_trainval.ps1
#
# Outputs (각 run 종료 후):
#   log/sz384_n50_<TS>_<test_f1>_<val_f1>/best_history.txt
#   log/sz512_n50_<TS>_<test_f1>_<val_f1>/best_history.txt
#   log/sz1024_n50_<TS>_<test_f1>_<val_f1>/best_history.txt
#   각 폴더 내 best_history.txt 의 [0] BEST OVERALL 줄을 비교하면 ablation 결론.
#
# 참고:
#   - cnn_train.py 본체가 log/<run_dir>/run.log 를 자체 기록하므로 외부 Tee
#     redirect 추가 금지 (top-level orphan log 방지).
#   - 3-way split (default 0.8/0.1/0.1) — best val 갱신 시 test도 자동 평가/저장.
#   - 한 run 실패 시 즉시 throw → 나머지 run 진행 안 함 ($ErrorActionPreference=Stop).
#   - 코드 수정/버그 fix 후 재돌릴 때는 model-tag 에 버전 식별자 붙여 새 폴더로
#     생성 (skip 우회용 log/ 삭제 절대 금지).
# =============================================================================

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
New-Item -ItemType Directory -Force -Path "log" | Out-Null

# 사이즈/배치/tag 매핑 — 새 ablation 추가 시 이 배열만 확장
$runs = @(
    @{ Size = 384;  Batch = 16; Tag = "sz384_n50"  },
    @{ Size = 512;  Batch = 8;  Tag = "sz512_n50"  },
    @{ Size = 1024; Batch = 2;  Tag = "sz1024_n50" }
)

foreach ($run in $runs) {
    $pythonArgs = @(
        "-u",                                                   # unbuffered stdout (실시간 진행)
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
