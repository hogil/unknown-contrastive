#!/usr/bin/env python3
"""Run the six-cell May NEW paired queue and dispatch result analysis."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "runs" / "may_new_tapt_removed_paired_2260"
QUEUE_LOG = OUTPUT_ROOT / "may_new_paired_queue.log"
RUNNER = ROOT / "scripts" / "run_may_new_paired.py"
SUMMARIZER = ROOT / "scripts" / "summarize_may_new_paired.py"
ANALYZER = ROOT / "scripts" / "run_result_analysis_agent.py"
ORDER = (
    ("nocnn", 42),
    ("cnn_tapt", 42),
    ("nocnn", 1),
    ("cnn_tapt", 1),
    ("nocnn", 2),
    ("cnn_tapt", 2),
)


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def completed(backbone: str, seed: int) -> Path | None:
    for path in OUTPUT_ROOT.glob("*/completion.json"):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if event.get("cell") == "NEW_FIXED" and event.get("backbone") == backbone and int(event.get("seed", -1)) == seed:
            return path
    return None


def run_logged(command: list[str]) -> None:
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def dispatch_analysis(event_file: Path) -> None:
    analysis_log = event_file.parent / "result_analysis_agent.log"
    handle = analysis_log.open("a", encoding="utf-8")
    subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(ANALYZER),
            "--event-file",
            str(event_file),
            "--context",
            "may_source",
            "--max-attempts",
            "3",
        ],
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    handle.close()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("May NEW paired queue started: exact semantic recipe, TAPT checkpoint is the only paired variable.")
    for backbone, seed in ORDER:
        event = completed(backbone, seed)
        if event is not None:
            log(f"SKIP completed backbone={backbone} seed={seed} event={event}")
            dispatch_analysis(event)
            continue
        log(f"START backbone={backbone} seed={seed}")
        run_logged(
            [
                sys.executable,
                "-u",
                str(RUNNER),
                "--backbone",
                backbone,
                "--seed",
                str(seed),
                "--output-root",
                str(OUTPUT_ROOT),
            ]
        )
        event = completed(backbone, seed)
        if event is None:
            raise RuntimeError(f"completion event missing: backbone={backbone} seed={seed}")
        log(f"DONE backbone={backbone} seed={seed} event={event}")
        dispatch_analysis(event)
        run_logged([sys.executable, "-u", str(SUMMARIZER), "--results-root", str(OUTPUT_ROOT)])
    run_logged([sys.executable, "-u", str(SUMMARIZER), "--results-root", str(OUTPUT_ROOT)])
    log("DONE all six May NEW paired runs.")


if __name__ == "__main__":
    main()
