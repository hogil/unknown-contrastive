# palette 전처리 4조건 샘플

같은 웨이퍼 **한 장**(`Center_scratch/ADZ231_00P_10_...png`)을 4가지로 전처리한 것.

| 파일 | 내용 |
|---|---|
| `compare_full4.png` | 웨이퍼 전체 4패널 |
| `compare_crop4_1to1.png` | 칩 4×4 영역을 **1:1 원본 픽셀**로 4패널 (전처리 차이가 보이는 배율) |

| 조건 | `UC_PALETTE_MASK` / `UC_PALETTE_MODE` | 남기는 index | 흰색 비율 |
|---|---|---|--:|
| ① 원본 | `0` / — | 전부 | 55.4% |
| ④ 마커만 제거 (배경유지) | `1` / `grade_bg` | 0~7 + 경계10 + 배경8 | 55.5% |
| ② 경계통일 + 배경삭제 | `1` / `grade_only` | 0~7 + 경계10 | 74.2% |
| ③ 경계삭제 + 배경삭제 | `1` / `grade_noborder` | 0~7 만 | 75.7% |

①→④ 는 흰색 비율이 0.1%p 밖에 안 바뀐다(컬러 마커가 그만큼 작다). 그런데도
ARI 가 +0.22~+0.34 오른다 — **작은 픽셀 몇 개가 clustering shortcut 이었다.**

원본 해상도(6400×6400) 파일은 용량 때문에 repo 에 넣지 않았다:
`runs/_border_abl/_samples_full/` (`{조건}_full.png`, `{조건}_view.png`)
재생성은 `docs/paper/BORDER_ABLATION_260728.md` 의 재현 절차 참조.
