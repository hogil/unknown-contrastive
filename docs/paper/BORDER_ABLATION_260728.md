# palette 전처리 ablation — 칩경계를 남길까 지울까 (260728)

## 질문

grouping 할 때 wafer PNG 의 **칩경계 픽셀을 지우면 성능이 더 좋은가?**

palette PNG 는 index 0~7 이 결함 grade 이고 그 위는 배경·경계·invalid 계열이다.
경계는 모든 웨이퍼에 똑같이 있는 격자라 결함 구분에 정보가 없어 보이지만,
chip 크기·정렬을 알려주는 **구조 신호**이기도 하다. 지우는 게 이득인지는 재봐야 안다.

## 조건 4개 — **한 번에 하나씩만 벗긴다**

| | 이름 | `UC_PALETTE_MASK` / `UC_PALETTE_MODE` | 남기는 index | ①에서 뺀 것 |
|---|---|---|---|---|
| ① | 일반 이미지 (원본) | `0` / — | 전부 | — |
| ④ | 칩경계 통일 + 배경유지 | `1` / `grade_bg` | 0~7 + 경계10 + 배경8 | 컬러 마커 |
| ② | 칩경계 통일 + 배경삭제 | `1` / `grade_only` | 0~7 + 경계10 | + 배경 |
| ③ | 칩경계 삭제 + 배경삭제 | `1` / `grade_noborder` | 0~7 만 | + 경계 |

"통일"은 파랑·초록·노랑·자주 같은 **컬러 마커(11~23)가 전부 흰색이 되고 회색 격자(10)만
남는다**는 뜻이다. ①→④→②→③ 이 한 단계씩 더 벗기는 순서라 **이득의 출처를 분리**할 수 있다.

전처리 결과 비교: `runs/_border_abl/_compare/preproc_compare4.png`

## 설정

| 항목 | 값 |
|---|---|
| pool | 43 class × 20장 = **860장** (`data/pools/_border_ablation.json`) |
| backbone | `convnextv2_base.fcmae_ft_in22k_in1k_384` (frozen, head 없음) |
| 다이얼 | mcs ∈ {6, 10, 20, 40}, ms=3, leaf, eps=0.06 |
| reassign | `nearest_q90` |
| 채점 | `--offline-eval` (폴더명 = 정답 라벨) |
| 잡음폭 | noise ±2.28pp, ARI ±0.019, Hom ±0.005, Comp ±0.033 |

★ 네 조건이 **같은 pool·같은 backbone·같은 다이얼**을 쓴다. 다른 건 palette 전처리 하나뿐이다.
★ 캐시는 조건마다 따로 만들었다 — 아래 "캐시 키 버그" 참조.

## 결과 (mcs=10)

| 조건 | P1 capture | P2 noise | P3 Comp | P4 Hom | ARI | AMI | Sil | k | frag | seed_nz |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ① 원본 | 19/37 | 1.35 | **0.9897** | 0.8170 | 0.4879 | 0.8696 | 0.4500 | 30 | 0.81 | 11.74 |
| ④ 마커만 제거 | 25/37 | 2.30 | 0.9695 | 0.8980 | 0.7758 | 0.9130 | 0.4557 | 35 | 0.95 | 20.12 |
| **② +배경삭제** | **28/37** | 1.76 | 0.9564 | **0.9015** | **0.7766** | 0.9066 | 0.4580 | 37 | 1.00 | 17.79 |
| ③ +경계삭제 | 25/37 | 1.49 | 0.9767 | 0.8956 | 0.7645 | **0.9161** | **0.4728** | 34 | 0.92 | 14.88 |

## ★ 이득의 출처 — 한 단계씩 벗겨보면

| 단계 | mcs=6 | mcs=10 | mcs=20 |
|---|--:|--:|--:|
| **① → ④  컬러 마커만 제거** | **P1 +8  ARI +0.338★** | **P1 +6  ARI +0.288★** | **P1 +4  ARI +0.221★** |
| ④ → ②  배경도 제거 | P1 −1  ARI −0.005 | P1 +3  ARI +0.001 | P1 −1  ARI −0.071★ |
| ② → ③  경계까지 제거 | P1 −5  ARI −0.065★ | P1 −3  ARI −0.012 | P1 ±0  ARI +0.010 |

**이득 전부가 첫 단계에서 나온다.** 세 다이얼 모두에서 컬러 마커 제거만 잡음폭 밖이고,
배경·경계 제거는 잡음폭 안이거나 **마이너스**다.

## 다이얼을 흔들어도 유지되는가

한 측정 지점에서 결론 내지 말라는 규칙에 따라 4개 다이얼에서 확인했다.

| mcs | ① 원본 | ④ 마커만 제거 | ② +배경삭제 | ③ +경계삭제 |
|--:|--:|--:|--:|--:|
| 6 | 26/37 · 0.4814 | **34/37 · 0.8199** | 33/37 · 0.8149 | 28/37 · 0.7496 |
| 10 | 19/37 · 0.4879 | 25/37 · 0.7758 | **28/37 · 0.7766** | 25/37 · 0.7645 |
| 20 | 10/37 · 0.4086 | **14/37 · 0.6298** | 13/37 · 0.5587 | 13/37 · 0.5691 |
| 40 | 0/37 · 0.2542 | 0/37 · 0.3131 | 0/37 · 0.2790 | 0/37 · 0.3809 |

**mcs=40 은 네 arm 다 P1 0/37 로 붕괴한 구간이라 판정에서 뺀다.**
(860장을 mcs40 으로 자르면 클래스를 하나도 못 잡는다. 이 구간의 ARI 우열은 의미 없다.)

② vs ③ (경계를 남기나 지우나):

| mcs | P1 (②:③) | 승 | ARI Δ(②−③) |
|--:|--:|---|--:|
| 6 | 33 : 28 (+5) | ② | +0.0653 (잡음폭 밖) |
| 10 | 28 : 25 (+3) | ② | +0.0121 (잡음폭 안) |
| 20 | 13 : 13 (0) | 동점 | −0.0104 (잡음폭 안) |

④ vs ② (배경을 남기나 지우나) — **일관된 승자가 없다**:

| mcs | P1 (④:②) | 승 | ARI Δ(④−②) |
|--:|--:|---|--:|
| 6 | 34 : 33 (+1) | ④ | +0.0050 (잡음폭 안) |
| 10 | 25 : 28 (−3) | ② | −0.0008 (잡음폭 안) |
| 20 | 14 : 13 (+1) | ④ | +0.0711 (잡음폭 밖) |

## 결론

1. **범인은 컬러 마커 하나였다.** ①→④(마커만 제거)에서 ARI +0.22~+0.34, P1 +4~+8.
   세 다이얼 전부 잡음폭 밖. 파랑·초록·노랑·자주 마커가 clustering shortcut 으로
   작동하고 있었다. **이 한 단계가 전체 이득의 100% 다.**

2. **배경(idx 8)은 지우든 남기든 무의미하다.** ④ vs ② 는 P1 +1/−3/+1 로 승자가
   다이얼마다 바뀌고 ARI 도 3개 중 2개가 잡음폭 안이다. 판정 불가 = 아무거나.

3. **경계(idx 10)는 남기는 쪽이 낫다.** ② 가 ③ 을 P1 에서 +5/+3/0 으로 **한 번도 지지 않는다**.
   → 회색 격자는 shortcut 이 아니라 **쓸모 있는 구조 신호**다. 지우면 손해거나 본전이다.

4. **채택: `UC_PALETTE_MASK=1` + `UC_PALETTE_MODE=grade_only` (기본값 유지).**
   ④(`grade_bg`)와 성능이 사실상 같지만, 배경을 지우는 쪽이 입력이 단순하고 기본값이라
   바꿀 이유가 없다. `grade_noborder` 는 남겨두되 쓰지 않는다.

### ⚠ 이전 기록과 상충

anchor pool 기록에는 "경계 idx10 을 남기면 이득의 대부분이 사라진다 → 지워라"가 있었다.
이 pool(43 class 860장)에서는 **정반대**다. 두 pool 의 차이(클래스 구성·경계 밀도)를
가르지 못했으므로, **pool 마다 다시 재는 것**이 맞다 — step1 의 `frozen_masked` arm 이 그 용도다.

## ★★ 가장 중요한 발견 — 무라벨 지표가 반대를 가리켰다

| 순위 | seed_noise 최소 (무라벨) | 실제 P1 최대 (라벨) |
|---|---|---|
| 1 | **① 원본 (11.74)** | **② 경계통일 (28/37)** |
| 2 | ③ 경계삭제 (14.88) | ④ 마커만 제거 / ③ (25/37) |
| 3 | ② 경계통일 (17.79) | ① 원본 (19/37) |
| 4 | ④ 마커만 제거 (20.12) | — |

★ **seed_noise 가 가장 나쁜 ④(20.12)가 라벨로는 상위권**이다. 완전한 역상관이다.

**순위가 완전히 뒤집혔다.** 무라벨 `argmin(seed_noise)` 로 골랐다면 **가장 나쁜 ①을** 골랐다.

이유는 분명하다. 마스킹은 배경·마커를 흰색으로 지우므로 이미지가 서로 비슷해지고,
HDBSCAN 이 자신 없는 점을 noise 로 더 많이 뱉는다. seed_noise 는 올라가지만
**남은 클러스터의 품질은 훨씬 좋아진다**.

### 함의 — 무라벨 선택 규칙의 적용 범위

`argmin(seed_noise)` 는 **레시피/epoch 선택**(같은 입력, 같은 전처리, encoder 만 다름)에서
검증된 규칙이다. **입력 전처리를 바꾸는 축에는 쓰면 안 된다** — 이번처럼 정반대를 고른다.

- Rule C (epoch 선택), step3 셀 선택: 전처리 고정 → 계속 유효
- palette 마스킹 on/off 결정: **seed_noise 로 고르지 마라.**
  라벨이 있는 pool(여기)에서 정하고 그 설정을 사내에 이식하거나,
  사내에서 소량이라도 라벨을 확보해 정해야 한다.

step1 의 `frozen_masked` arm 은 seed_noise 만 보고하므로 **그 숫자로 마스킹을 판단하면 틀린다.**
(실제로 `runs/_invstep` 240장 pool 에서 frozen 23.33 vs frozen_masked 27.08 이 나와
마스킹이 나쁜 것처럼 보였지만, 라벨로 재면 마스킹이 훨씬 낫다.)

## 곁가지로 잡은 버그 — 캐시 키 충돌

`deploy/build_cache.py` 의 캐시 키가 `(파일목록, IMG, palette_mask 불린)` 뿐이었다.
`grade_only` 와 `grade_noborder` 는 **픽셀이 다른데 키가 같아** 같은 캐시를 공유했다.
이 실험을 그대로 돌렸으면 ②와 ③이 같은 결과로 나왔을 것이다.

→ 키에 `UC_PALETTE_MODE` 를 넣고, spawn 워커에도 모드를 **명시적으로** 전달하도록 고쳤다
(부모 env 상속에만 기대면 실행 방식에 따라 조용히 다른 모드로 캐시가 만들어진다).
검증: 4개 조합(mask off / grade_only / grade_noborder / grade_bg) 키가 전부 다르고,
네 캐시의 픽셀도 실제로 다르다:
raw↔unify 608,641 / raw↔noborder 1,322,749 / unify↔noborder 833,399 /
keepbg↔raw 158,709 / keepbg↔unify 455,359 / keepbg↔noborder 1,263,591.

## 한계

- **pool 1개, seed 1개.** 다이얼 3개에서 결론(마커 제거가 전부)이 유지된 건 확인했지만 pool 을 바꾸면
  달라질 수 있다 (실제로 anchor 기록과 상충한다).
- **arm 마다 다이얼을 다시 고르지 않았다.** 규칙상 각 arm 을 자기 최적점에서 비교해야
  하는데, 여기서는 공통 다이얼 4개에서 순위 유지를 확인하는 것으로 대신했다.
- 합성 데이터다. 사내 실데이터의 경계·마커 구성이 다르면 재측정해야 한다.

## 재현

```bash
# pool
python - <<'EOF'
import json, pathlib
files=[{"path": f"{d.name}/{p.name}", "label": d.name}
       for d in sorted(pathlib.Path("E:/data/images/unknown").iterdir()) if d.is_dir()
       for p in sorted(d.glob("*.png"))[:20]]
pathlib.Path("data/pools/_border_ablation.json").write_text(
    json.dumps({"root":"E:/data/images/unknown","files":files}), encoding="utf-8")
EOF

# 조건별 캐시 + 채점
for arm in "raw:0:grade_only" "keepbg:1:grade_bg" "unify:1:grade_only" "noborder:1:grade_noborder"; do
  NAME=${arm%%:*}; REST=${arm#*:}; MASK=${REST%%:*}; MODE=${REST#*:}
  UC_PALETTE_MASK=$MASK UC_PALETTE_MODE=$MODE \
    python deploy/build_cache.py --pool data/pools/_border_ablation.json \
      --cache-dir runs/_border_abl/_cache_$NAME
  UC_PALETTE_MASK=$MASK UC_PALETTE_MODE=$MODE \
    python grouping_deploy.py --pool data/pools/_border_ablation.json \
      --backbone weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \
      --cache runs/_border_abl/_cache_$NAME --out runs/_border_abl/$NAME \
      --mcs 10 --ms 3 --reassign nearest_q90 --offline-eval --no-composites
done
```

산출: `runs/_border_abl/` (조건 4개 × 다이얼 4개 + `_compare/preproc_compare4.png` + `_compare/metrics.csv` 16행)
