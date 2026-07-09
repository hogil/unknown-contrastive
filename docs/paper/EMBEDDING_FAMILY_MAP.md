# Embedding Family Map

This note separates what was trained, what was saved as embedding, and what was clustered.

## 0. Key Diagnosis

Current WM quick runs are not using the learned projection head as the eval embedding.

```
train loss path:
  image -> DINOv3 backbone -> f_t -> MLP projection head -> z_t
                                      |
                                      +-- InfoNCE / local / queue loss is applied here

eval/save path in current _ssl_methods.py:
  eval image -> same trained DINOv3 backbone -> f_t -> normalize -> *.npy -> FINCH/HDBSCAN

missing path:
  eval image -> backbone -> f_t -> projection head -> z_t -> save/evaluate
```

So the previous WM tables mostly measured the trained backbone feature `f_t`, not the learned projection embedding `z_t`.
This is why projection-head-only improvements can be invisible in grouping and fragment ratio.

Relevant code:
- Current head default: `D:\project\unknown-contrastive\_ssl_methods.py:202`
- Current train uses `z`: `D:\project\unknown-contrastive\_ssl_methods.py:428`
- Current eval saves `f`: `D:\project\unknown-contrastive\_ssl_methods.py:483`

## 1. Text Visualization

```
Legend
  f0  = frozen backbone feature
  f_t = backbone feature after contrastive fine-tuning
  z_t = projection/adaptor embedding after contrastive head
  CL  = clustering layer, e.g. FINCH / Louvain / HDBSCAN


[A] Frozen baseline, no training

  eval image
      |
      v
  pretrained backbone
      |
      v
  f0  -------------------------------> CL -> metrics / groups

  meaning:
    pure representation baseline.
    no loss, no projection head, no adapter.


[B] Current WM quick runs, what actually happened

  train image pair
      |
      v
  DINOv3 backbone, lr_bb=2e-6
      |
      v
  f_t
      |
      v
  MLP projection head, lr_head=1e-3
      |
      v
  z_t  ------------------------------> InfoNCE / local / queue loss

  eval image
      |
      v
  trained DINOv3 backbone
      |
      v
  f_t  -------------------------------> saved *.npy -> CL -> metrics / groups

  problem:
    z_t was trained but not saved/evaluated.
    if the head learned better separation, the current table does not directly show it.


[C] Initial contrastive from GitHub history, contrastive_init

  train image pair
      |
      v
  TAPT/FCMAE ConvNeXtV2 backbone, frozen=True
      |
      v
  pooled feature f0
      |
      v
  128-dim projection head
      |
      v
  z_t  ------------------------------> Global InfoNCE
       \-----------------------------> Queue InfoNCE, K=16384
        \----------------------------> false-negative ignore, cos > 0.72
         \---------------------------> local patch InfoNCE, grid36, window=4

  eval unknown image
      |
      v
  backbone + projection
      |
      v
  z_t  -------------------------------> HDBSCAN -> cluster folders / medoids

  meaning:
    initial contrastive used the trained projection embedding for clustering.
    this is closer to what we should do again.


[D] Adapter family we should add now

  eval/train image
      |
      v
  backbone feature f
      |
      v
  residual adapter: f' = f + gamma * Adapter(f)
      |
      +------------------------------> save/evaluate f'      (adapter embedding)
      |
      v
  projection head
      |
      +------------------------------> save/evaluate z       (projection embedding)

  reason:
    adapter changes the feature that clustering sees.
    it is safer than only training a head and then accidentally evaluating raw f.
```

## 2. Family Table

| family | training target | saved/eval embedding | what it proves | current status |
|---|---|---|---|---|
| `00_no_train` frozen | none | `f0` | raw backbone baseline | exists |
| current `temp/local/queue` quick | loss on `z_t`; backbone tiny lr | `f_t` only | backbone drift effect, not true head effect | exists, incomplete for head claim |
| initial GitHub `contrastive_init` | loss on projection `z_t` | `z_t` | projection embedding can drive grouping | historical reference |
| `head=linear/ad/mlp` | loss on `z_t` | must save both `f_t` and `z_t` | whether projection head itself helps | not yet swept correctly |
| `head=adapter` | adapter changes `f`; loss also on `z` | save `f_adapter` and `z` | whether light feature adaptation lowers fragmentation | next univariate |
| `head=adapterN2/N3` | multi-stage residual adapter | save `f_adapterN` and `z` | whether deeper adapter has a sweet spot | next univariate |
| combo after single-variable wins | top adapter x top temp/local/queue | save all embedding modes | paper-style cumulative row | only after single-variable proof |

## 3. Initial Contrastive From GitHub History

Git remote:

```
D:\project\unknown-contrastive -> https://github.com/hogil/unknown-contrastive.git
```

Initial contrastive was introduced in commit:

```
aa50373 Add contrastive init baseline and comparison notes
```

File:

```
D:\project\unknown-contrastive\scripts\contrastive_init.py
```

It was a single-GPU self-supervised recipe:

```
TRAIN_DIR only, no labels
UNKNOWN_DIR embedded and clustered
backbone: ConvNeXtV2 FCMAE/TAPT checkpoint
backbone frozen: True
projection dim: 128
epochs: 20
batch: 256
temperature: 0.07
queue: on, size 16384
false-negative ignore: cos > 0.72
local InfoNCE: on, weight 0.5
local anchors: grid36_full
local window: 4
clusterer: HDBSCAN
outputs: cluster folders, medoids, summaries
```

The important difference:

```
initial contrastive:
  extract() calls model(x)
  model(x) returns normalized projection z
  HDBSCAN clusters z

current WM quick:
  train loss uses z
  eval saves f
  FINCH/HDBSCAN clusters f
```

Relevant code:
- Initial projection head: `D:\project\unknown-contrastive\scripts\contrastive_init.py:201`
- Initial backbone frozen flag: `D:\project\unknown-contrastive\scripts\contrastive_init.py:232`
- Initial queue and local losses: `D:\project\unknown-contrastive\scripts\contrastive_init.py:501`
- Initial extraction saves projection output: `D:\project\unknown-contrastive\scripts\contrastive_init.py:360`

## 4. Portfolio Unknown Model

Reference file:

```
D:\project\fbm_paper\recommendation\portfolio.md
```

The portfolio says production unknown grouping used:

```
TAPT ConvNeXtV2 backbone
  + Global InfoNCE
  + Queue, size 16384
  + Negative similarity filter, cos > 0.72 excluded
  + Local InfoNCE, grid36_full, window=4
  + HDBSCAN grouping
```

Production claim:

```
real production data:
  train: specific product real data, about 10k wafers
  eval/apply: held-out about 2k wafers
  output: 13 candidate groups
  field verification: 7 real failure groups confirmed
```

Development metric ladder in the portfolio:

| row | recipe | capture | noise | completeness | homogeneity | ARI | AMI | Sil |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Global InfoNCE only | 0.9337 | 15.78% | 0.9468 | 0.8348 | 0.5489 | 0.8855 | 0.582 |
| 2 | + Local DenseCL, LW=0.5 | 0.9361 | 13.87% | 0.9502 | 0.8111 | 0.5314 | 0.8734 | 0.514 |
| 3 | + MoCo Queue 4096 | 0.9356 | 9.45% | 0.9474 | 0.8368 | 0.5596 | 0.8870 | 0.573 |
| 4 | + NV-Retriever NEG 0.72 | 0.9250 | 8.23% | 0.9485 | 0.8291 | 0.5683 | 0.8831 | 0.611 |
| 5 | + NeCo 0.2, 5-tool full | 0.9559 | 6.66% | 0.9660 | 0.8208 | 0.5648 | 0.8861 | 0.6104 |
| 6 | final recipe, Local DenseCL excluded 4-tool | 0.9559 | 6.66% | 0.9660 | 0.8208 | 0.5648 | 0.8861 | 0.781 |
| 7 | final recipe + noise postprocess tau=0.5 | 0.9619 | 0.00% | 0.9679 | 0.8184 | 0.5489 | 0.8856 | 0.781 |

Interpretation:

```
portfolio unknown winner was not "MLP head trained but backbone feature clustered".
It was closer to:
  projection/contrastive embedding + queue/negative control/local-neighborhood logic + HDBSCAN/postprocess.
```

## 5. What To Fix Next

Minimum correction before more paper tables:

```
1. Add --embed-mode backbone|projection|both to _ssl_methods.py.
2. For every epoch, save:
     tag_epN_backbone.npy
     tag_epN_projection.npy
3. Add adapter runs:
     --head linear
     --head ad
     --head adapter
     --head adapterN2
     --head adapterN3
4. Score each embedding mode with the same FINCH p2 / Louvain / HDBSCAN table.
5. Only after single-variable head/adapter winners are confirmed, run combo:
     top adapter x top temp/local/queue
```

Paper-style row order should be:

```
0. DINOv3 frozen, no train, f0
1. SimCLR baseline, backbone embedding f_t
2. SimCLR baseline, projection embedding z_t
3. best temperature, z_t and f_t
4. best local weight, z_t and f_t
5. best queue size, z_t and f_t
6. best adapter, adapter feature and z_t
7. combo only after 1-variable improvements are verified
8. postprocess, separate row
```

