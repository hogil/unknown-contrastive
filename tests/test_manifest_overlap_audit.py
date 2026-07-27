import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_manifest_overlap import audit, main


def _image(path: Path, mode: str, *, color=(0, 0, 0)) -> None:
    if mode == "solid":
        image = Image.new("RGB", (16, 16), color)
    else:
        image = Image.new("RGB", (16, 16))
        pixels = image.load()
        for y in range(16):
            for x in range(16):
                value = int((x if mode == "rising" else 15 - x) * 255 / 15)
                pixels[x, y] = (value, value, value)
    image.save(path)


def _manifest(path: Path, root: Path, rows) -> Path:
    path.write_text(json.dumps({"root": str(root), "files": rows}), encoding="utf-8")
    return path


def test_exact_duplicate_and_explicit_block_are_reported(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    _image(root / "train.png", "solid", color=(10, 20, 30))
    (root / "validation.png").write_bytes((root / "train.png").read_bytes())
    train = _manifest(tmp_path / "train.json", root, [{"path": "train.png", "block_id": "LOT-A"}])
    validation = _manifest(
        tmp_path / "validation.json", root, [{"path": "validation.png", "block_id": "LOT-A"}]
    )

    result = audit(train, validation, [root])

    assert result["status"] == "overlap_found"
    assert result["exact"]["content_pair_count"] == 1
    assert result["exact"]["same_resolved_path_count"] == 0
    assert result["provenance"]["method_counts"]["explicit:block_id"] == 2
    assert result["provenance"]["cross_split_block_overlap_examples"] == ["LOT-A"]


def test_near_candidate_excludes_exact_and_requires_review(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    _image(root / "a.png", "solid", color=(10, 10, 10))
    _image(root / "b.png", "solid", color=(40, 40, 40))
    train = _manifest(tmp_path / "train.json", root, ["a.png"])
    validation = _manifest(tmp_path / "validation.json", root, ["b.png"])

    result = audit(train, validation, [root], near_threshold=0)

    assert result["exact"]["content_pair_count"] == 0
    assert result["near"]["candidate_pair_count"] == 1
    assert result["near"]["examples"][0]["hamming_distance"] == 0
    assert result["status"] == "review_required"
    assert result["review_required"] is True
    assert result["provenance"]["method_counts"]["fallback:filename_token_v1"] == 2


def test_visually_different_pair_is_clean(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    _image(root / "rise.png", "rising")
    _image(root / "fall.png", "falling")
    train = _manifest(tmp_path / "train.json", root, ["rise.png"])
    validation = _manifest(tmp_path / "validation.json", root, ["fall.png"])

    result = audit(train, validation, [root], near_threshold=5)

    assert result["status"] == "clean"
    assert result["near"]["candidate_pair_count"] == 0


def test_result_order_is_deterministic(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for name, color in (("a_1.png", (1, 1, 1)), ("b_1.png", (2, 2, 2)), ("c_1.png", (3, 3, 3))):
        _image(root / name, "solid", color=color)
    train_a = _manifest(tmp_path / "train_a.json", root, ["b_1.png", "a_1.png"])
    train_b = _manifest(tmp_path / "train_b.json", root, ["a_1.png", "b_1.png"])
    validation = _manifest(tmp_path / "validation.json", root, ["c_1.png"])

    first = audit(train_a, validation, [root], near_threshold=0)
    second = audit(train_b, validation, [root], near_threshold=0)

    assert first["near"]["examples"] == second["near"]["examples"]
    assert first["near"]["distance_histogram"] == second["near"]["distance_histogram"]
    assert first["provenance"] == second["provenance"]


@pytest.mark.parametrize(
    "bad_entry",
    ["missing.png", "../outside.png"],
)
def test_missing_or_escape_fails_before_output_replace(tmp_path, bad_entry):
    root = tmp_path / "images"
    root.mkdir()
    outside = tmp_path / "outside.png"
    _image(outside, "solid")
    _image(root / "ok.png", "solid")
    train = _manifest(tmp_path / "train.json", root, [bad_entry])
    validation = _manifest(tmp_path / "validation.json", root, ["ok.png"])
    output = tmp_path / "audit.json"
    output.write_text("sentinel", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--train-manifest",
                str(train),
                "--validation-manifest",
                str(validation),
                "--allowed-root",
                str(root),
                "--out",
                str(output),
            ]
        )

    assert exc.value.code == 2
    assert output.read_text(encoding="utf-8") == "sentinel"


def test_corrupt_image_fails_closed(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "bad.png").write_bytes(b"not-an-image")
    _image(root / "ok.png", "solid")
    train = _manifest(tmp_path / "train.json", root, ["bad.png"])
    validation = _manifest(tmp_path / "validation.json", root, ["ok.png"])

    with pytest.raises(OSError):
        audit(train, validation, [root])


def test_cli_writes_atomic_report_and_returns_one_for_exact_overlap(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    _image(root / "same.png", "solid")
    train = _manifest(tmp_path / "train.json", root, ["same.png"])
    validation = _manifest(tmp_path / "validation.json", root, ["same.png"])
    output = tmp_path / "audit.json"

    code = main(
        [
            "--train-manifest",
            str(train),
            "--validation-manifest",
            str(validation),
            "--allowed-root",
            str(root),
            "--out",
            str(output),
        ]
    )

    assert code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "overlap_found"
