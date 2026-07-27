"""Create a path-only manifest without opening or copying image files."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    src = json.loads(a.input.read_text(encoding="utf-8"))
    if not isinstance(src.get("root"), str) or not isinstance(src.get("files"), list):
        raise SystemExit("invalid source manifest")
    paths = []
    for entry in src["files"]:
        value = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(value, str):
            raise SystemExit("manifest entry lacks path")
        paths.append(value)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"schema_version": "path_only_manifest.v1", "root": src["root"],
                                    "n_files": len(paths), "files": paths}, indent=2) + "\n", encoding="utf-8")
if __name__ == "__main__": main()
