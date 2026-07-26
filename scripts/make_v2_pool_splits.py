#!/usr/bin/env python3
"""Build data/pools/v2/<dataset>/... — SHA-clean, class-disjoint pool splits.

team-lead directive (260726, codex 캠페인 계획 2절): the pre-existing pools
(data/pools/*.json) turned out to overlap (see scripts/audit_pool_leakage.py
and the audit JSON this script's output is verified against). This script
builds a *new* set of manifests from scratch, disjoint by construction:

  unknown        : Normal + known-train classes (one bucket) vs class-disjoint
                   novel validation classes vs class-disjoint novel test
                   classes. Class -> bucket assignment is a deterministic
                   SHA-256 hash of the class name (see hash01()), not a
                   physical copy — every entry is (root, relative path).
  unknown_multi  : single "stress_test" pool, not for training (multi-label
                   combos — team-lead: "학습에 쓰지 마라").
  mixedwm38 /
  hf_dtd / hf_flowers102 / hf_resisc45
                 : whole-class deterministic hash split, ~60/20/20.
  severstal      : leave-two-classes-out, 6 folds over {Class1..Class4}
                   (Normal fixed as background in every fold's train split).

Every manifest carries a "track" field ("strict_novel" or "stress_test") per
team-lead's two-track policy:
  - transductive : adapt on a pool and score on the *same* pool (existing
    pools such as data/pools/severstal_pilot260726.json already are this;
    this script does not touch them).
  - strict_novel : train/validation/test classes and files are disjoint;
    test labels are sealed ("sealed": true) until final campaign scoring.

Nothing here reads image bytes or copies/moves/links files — manifests only
reference (root, relative path) pairs already present in the allowlisted E:
data roots (see scripts/audit_pool_leakage.py::ALLOWLIST_ROOTS).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts._common import resolve_path  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
OUT_ROOT = REPO / "data" / "pools" / "v2"

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2  # remainder (~0.2) is test


def hash01(key: str, seed: str) -> float:
    """Deterministic pseudo-random value in [0, 1) for (seed, key). Same
    (seed, key) always maps to the same value — this is the whole point:
    reproducible splits without storing a random.choice() draw anywhere."""
    h = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def bucket_name(frac: float, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC) -> str:
    if frac < train_frac:
        return "train"
    if frac < train_frac + val_frac:
        return "val"
    return "test"


def list_class_files(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def write_manifest(out_path: Path, root: Path, entries: list[dict], extra: dict) -> None:
    manifest = {
        "root": str(root),
        "source_pool": None,
        "n_files": len(entries),
        "verify": "master-direct",
        "files": entries,
        **extra,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[make_v2_pool_splits] {len(entries)} files, {len(set(e['label'] for e in entries))} class(es) -> {out_path}")


# ===================== unknown =====================
UNKNOWN_MASTER = Path("E:/data/images/unknown")
UNKNOWN_UTILITY_DIRS = {"classification", "classification_chips"}
UNKNOWN_SEED = "v2splits_260726_unknown"


def build_unknown() -> dict:
    master = resolve_path(UNKNOWN_MASTER)
    class_dirs = sorted(
        d for d in master.iterdir()
        if d.is_dir() and d.name not in UNKNOWN_UTILITY_DIRS and d.name != "Normal"
    )
    class_names = [d.name for d in class_dirs]
    buckets = {name: bucket_name(hash01(name, UNKNOWN_SEED)) for name in class_names}

    known_train_classes = sorted(n for n, b in buckets.items() if b == "train")
    novel_val_classes = sorted(n for n, b in buckets.items() if b == "val")
    novel_test_classes = sorted(n for n, b in buckets.items() if b == "test")
    assert set(known_train_classes).isdisjoint(novel_val_classes)
    assert set(known_train_classes).isdisjoint(novel_test_classes)
    assert set(novel_val_classes).isdisjoint(novel_test_classes)

    def entries_for(classes: list[str]) -> list[dict]:
        out = []
        for name in classes:
            for f in list_class_files(master / name):
                out.append({"path": f.relative_to(master).as_posix(), "label": name})
        return out

    normal_entries = entries_for(["Normal"])
    train_entries = normal_entries + entries_for(known_train_classes)
    val_entries = entries_for(novel_val_classes)
    test_entries = entries_for(novel_test_classes)

    out_dir = OUT_ROOT / "unknown"
    common = {
        "seed": UNKNOWN_SEED,
        "split_algorithm": "hash01(class_name, seed) thresholded at train<0.6, val<0.8, else test; "
                            "Normal is always in the train bucket (not class-hash-split)",
        "known_train_classes": known_train_classes,
        "novel_val_classes": novel_val_classes,
        "novel_test_classes": novel_test_classes,
    }
    write_manifest(out_dir / "strict_novel_train.json", master, train_entries,
                    {"track": "strict_novel", "split": "train", "sealed": False, **common})
    write_manifest(out_dir / "strict_novel_val.json", master, val_entries,
                    {"track": "strict_novel", "split": "validation", "sealed": False, **common})
    write_manifest(out_dir / "strict_novel_test.json", master, test_entries,
                    {"track": "strict_novel", "split": "test", "sealed": True,
                     "sealed_note": "labels are present in this file for audit/reproducibility only. "
                                     "Do not read/score against them until final campaign scoring.",
                     **common})
    return {
        "train": {"n_files": len(train_entries), "n_classes": 1 + len(known_train_classes)},
        "val": {"n_files": len(val_entries), "n_classes": len(novel_val_classes)},
        "test": {"n_files": len(test_entries), "n_classes": len(novel_test_classes)},
    }


# ===================== unknown_multi =====================
UNKNOWN_MULTI_MASTER = Path("E:/data/images/unknown_multi")
MULTI_LABEL_RE = re.compile(r"__d-(?P<defects>.+?)__o-(?P<objs>.+)\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def build_unknown_multi() -> dict:
    master = resolve_path(UNKNOWN_MULTI_MASTER)
    files = sorted(p for p in master.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    entries = []
    unparsed = 0
    for f in files:
        m = MULTI_LABEL_RE.search(f.name)
        label = m.group("defects") if m else None
        if m is None:
            unparsed += 1
        entries.append({"path": f.relative_to(master).as_posix(), "label": label})
    out_dir = OUT_ROOT / "unknown_multi"
    write_manifest(out_dir / "stress_test_all.json", master, entries, {
        "track": "stress_test",
        "note": "complex multi-defect stress set — evaluation only, do NOT use for training "
                "(team-lead directive 260726).",
        "label_semantics": "label = '+' joined defect-combo parsed from filename '__d-<...>__o-<...>' "
                            "(informational; images are multi-label, not single-class).",
        "n_unparsed_filenames": unparsed,
    })
    return {"n_files": len(entries), "n_unparsed": unparsed}


# ===================== generic whole-class hash split =====================
def build_class_hash_split(dataset: str, master_dir: Path, out_dir_name: str | None = None) -> dict:
    master = resolve_path(master_dir)
    class_dirs = sorted(d for d in master.iterdir() if d.is_dir())
    class_names = [d.name for d in class_dirs]
    seed = f"v2splits_260726_{dataset}"
    buckets = {name: bucket_name(hash01(name, seed)) for name in class_names}
    train_classes = sorted(n for n, b in buckets.items() if b == "train")
    val_classes = sorted(n for n, b in buckets.items() if b == "val")
    test_classes = sorted(n for n, b in buckets.items() if b == "test")
    assert set(train_classes).isdisjoint(val_classes)
    assert set(train_classes).isdisjoint(test_classes)
    assert set(val_classes).isdisjoint(test_classes)

    def entries_for(classes: list[str]) -> list[dict]:
        out = []
        for name in classes:
            for f in list_class_files(master / name):
                out.append({"path": f.relative_to(master).as_posix(), "label": name})
        return out

    train_entries = entries_for(train_classes)
    val_entries = entries_for(val_classes)
    test_entries = entries_for(test_classes)

    out_dir = OUT_ROOT / (out_dir_name or dataset)
    common = {
        "seed": seed,
        "split_algorithm": "hash01(class_name, seed) thresholded at train<0.6, val<0.8, else test "
                            "(whole class -> single bucket, ~60/20/20 by class count)",
        "train_classes": train_classes,
        "val_classes": val_classes,
        "test_classes": test_classes,
    }
    write_manifest(out_dir / "strict_novel_train.json", master, train_entries,
                    {"track": "strict_novel", "split": "train", "sealed": False, **common})
    write_manifest(out_dir / "strict_novel_val.json", master, val_entries,
                    {"track": "strict_novel", "split": "validation", "sealed": False, **common})
    write_manifest(out_dir / "strict_novel_test.json", master, test_entries,
                    {"track": "strict_novel", "split": "test", "sealed": True,
                     "sealed_note": "labels are present in this file for audit/reproducibility only. "
                                     "Do not read/score against them until final campaign scoring.",
                     **common})
    return {
        "train": {"n_files": len(train_entries), "n_classes": len(train_classes)},
        "val": {"n_files": len(val_entries), "n_classes": len(val_classes)},
        "test": {"n_files": len(test_entries), "n_classes": len(test_classes)},
    }


# ===================== severstal: leave-two-classes-out 6-fold =====================
SEVERSTAL_MASTER = Path("E:/data/images/severstal")
SEVERSTAL_DEFECT_CLASSES = ["Class1", "Class2", "Class3", "Class4"]


def _severstal_single_label_index(master: Path) -> dict[str, list[str]]:
    """filename -> label, restricted to single-label images (multi-hot sum in {0,1}).
    sum==0 -> Normal, sum==1 -> Class<idx+1>. Mirrors the selection rule already
    used for data/pools/severstal_pilot260726.json (reproduced/verified against it)."""
    classid = json.loads((master / "classid.json").read_text(encoding="utf-8"))
    by_label: dict[str, list[str]] = {"Normal": [], "Class1": [], "Class2": [], "Class3": [], "Class4": []}
    for fname, vec in classid.items():
        s = sum(vec)
        if s == 0:
            by_label["Normal"].append(fname)
        elif s == 1:
            by_label[f"Class{vec.index(1) + 1}"].append(fname)
        # s > 1: multi-label, excluded (matches severstal_pilot260726.json convention)
    for label in by_label:
        by_label[label].sort()
    return by_label


def build_severstal_folds() -> dict:
    master = resolve_path(SEVERSTAL_MASTER)
    by_label = _severstal_single_label_index(master)

    def entries_for(labels: list[str]) -> list[dict]:
        out = []
        for label in labels:
            for fname in by_label[label]:
                out.append({"path": f"train_images/{fname}", "label": label})
        return out

    out_dir = OUT_ROOT / "severstal"
    summary = {}
    for i, (held_a, held_b) in enumerate(combinations(SEVERSTAL_DEFECT_CLASSES, 2), start=1):
        fold_tag = f"fold{i:02d}_{held_a}_{held_b}"
        train_classes = ["Normal"] + [c for c in SEVERSTAL_DEFECT_CLASSES if c not in (held_a, held_b)]
        test_classes = [held_a, held_b]
        train_entries = entries_for(train_classes)
        test_entries = entries_for(test_classes)
        common = {
            "fold": fold_tag,
            "held_out_novel_classes": test_classes,
            "single_label_selection": "classid.json multi-hot sum==1 -> Class<idx+1>, sum==0 -> Normal, "
                                       "sum>1 excluded (matches data/pools/severstal_pilot260726.json)",
        }
        write_manifest(out_dir / f"{fold_tag}_train.json", master, train_entries,
                        {"track": "strict_novel", "split": "train", "sealed": False, **common})
        write_manifest(out_dir / f"{fold_tag}_novel_test.json", master, test_entries,
                        {"track": "strict_novel", "split": "test", "sealed": True,
                         "sealed_note": "labels present for audit only — do not read/score until final "
                                         "campaign scoring.",
                         **common})
        summary[fold_tag] = {"train_n": len(train_entries), "test_n": len(test_entries)}
    return summary


DATASETS = {
    "unknown": build_unknown,
    "unknown_multi": build_unknown_multi,
    "mixedwm38": lambda: build_class_hash_split("mixedwm38", Path("E:/data/images/mixedwm38/rendered/all")),
    "hf_dtd": lambda: build_class_hash_split("hf_dtd", Path("E:/data/images/hf_dtd")),
    "hf_flowers102": lambda: build_class_hash_split("hf_flowers102", Path("E:/data/images/hf_flowers102")),
    "hf_resisc45": lambda: build_class_hash_split("hf_resisc45", Path("E:/data/images/hf_resisc45")),
    "severstal": build_severstal_folds,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["all"] + sorted(DATASETS), default="all")
    args = ap.parse_args()

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_summary = {}
    for name in targets:
        print(f"\n=== {name} ===")
        all_summary[name] = DATASETS[name]()

    summary_path = OUT_ROOT / "BUILD_SUMMARY.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] {OUT_ROOT}")
    print(f"[OUT] {summary_path}")


if __name__ == "__main__":
    main()
