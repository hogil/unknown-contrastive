---
name: performance-research
description: arxiv 논문 + GitHub repo 검색해 SOTA 기법 정리. WebFetch 실패 시 추정 금지 — fail 명시 원칙.
---

# performance-research skill

`performance-research` agent 의 작업 표준.

## query 템플릿

| topic | arxiv query | github query |
|---|---|---|
| `feature_separation_contrastive` | `site:arxiv.org "feature separation" contrastive 2024..2026` | `site:github.com contrastive feature separation pytorch stars:>100` |
| `hdbscan_noise_reduction` | `site:arxiv.org HDBSCAN noise OR outlier reduction 2024..2026` | `site:github.com hdbscan noise reduction stars:>100` |
| `cluster_purity_self_supervised` | `site:arxiv.org cluster purity self-supervised 2024..2026` | `site:github.com cluster purity self-supervised stars:>100` |
| `hdbscan_min_cluster_size` | `site:arxiv.org HDBSCAN cluster size selection 2024..2026` | `site:github.com hdbscan min_cluster_size pytorch stars:>100` |
| `supcon_loss` | `site:arxiv.org supervised contrastive loss SupCon 2023..2026` | `site:github.com supcon loss pytorch stars:>100` |
| `convnextv2_finetune` | `site:arxiv.org ConvNeXtV2 fine-tune transfer 2024..2026` | `site:github.com convnextv2 fine-tune stars:>100` |
| `wafer_defect_clustering` | `site:arxiv.org wafer defect clustering self-supervised 2024..2026` | `site:github.com wafer defect detection clustering stars:>100` |

domain 별 도메인 추가:
- `arxiv` 만 — `site:arxiv.org` 강제
- `github` 만 — `site:github.com` 강제

## arxiv WebFetch 패턴

URL 형식: `https://arxiv.org/abs/<id>` (e.g., `https://arxiv.org/abs/2401.04565`)

추출 대상 (HTML):
- `<h1 class="title mathjax">` — title
- `<div class="authors">` — author list
- `<blockquote class="abstract mathjax">` — abstract (1 단락)
- `<div class="bibtex">` 또는 메타 — venue / submission date

WebFetch prompt 예: `"Extract the title, authors, abstract, and submission date from this arxiv abstract page. Format as JSON."`

## github README WebFetch 패턴

raw README URL: `https://raw.githubusercontent.com/<owner>/<repo>/<default_branch>/README.md`
또는 standard URL: `https://github.com/<owner>/<repo>` 의 메인 README

추출 대상:
- repo description (h1 / top blurb)
- "Results" / "Benchmark" / "Performance" 섹션 (있을 때)
- "Key Contributions" / "Features" 섹션
- last commit / stars (page sidebar)

WebFetch prompt 예: `"Extract from this GitHub README: (1) one-sentence project purpose, (2) key technical contributions, (3) reported metrics if any, (4) main code path for the contribution. Output as JSON."`

## 사실성 가드 (절대)

- WebFetch 가 `4xx` / timeout / empty content 반환 시:
  - hit 를 `skip` 처리
  - 보고에 `"fetch_failed: <url> (<error_summary>)"` 명시
  - **LLM training 지식 으로 paper/repo 내용 추정 절대 금지**
- WebSearch 0 hit:
  - `"no hits for topic <X>"` 명시 후 다음 topic
  - 결과 없는 topic 은 빈 papers/repos 로 보고
- 의심 도메인 차단: SEO 농장, sponsored 글, paper-mill 등 — 유효 publishing platform 만 채택
  (arxiv, ACM, IEEE, NeurIPS, ICML, ICLR, CVPR, ECCV, ICCV, AAAI, KDD, NeurIPS 등 OR github)

## 적용 가능성 점수 (★ 1-3)

| 점수 | 기준 |
|---|---|
| ★★★ | 본 repo 의 contrastive.py / wrapper 의 hparam 직접 변경으로 적용 가능 (예: LOSS_TEMP, HDBSCAN args) |
| ★★ | 약간의 wrapper 코드 추가 필요 (e.g., 새 augmentation, 새 loss term — wrapper layer 에서) |
| ★ | 큰 architectural 변경 필요 또는 도메인 mismatch (참고만) |

## 금지

- 외부 weights / pth / 모델 자동 다운로드 — URL 만 보고
- `contrastive.py` / `_*.py` 자동 patch
- 학습 dispatch / GPU 사용
- 새 `.py` 파일 생성 — agent 산출은 .md / .json 만

## 출력 markdown 섹션 (고정)

```
## Diagnosis
## Search queries used
## Papers (top N)
## Repositories (top N)
## Recommended action items
```
