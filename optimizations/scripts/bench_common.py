"""Shared pieces for TTS engine benchmarks.

Every engine script (scripts/bench_<engine>.py) uses the same TEXTS, writes
samples to samples/<engine>/ for listening, and writes one JSON file to
results/<engine>.json with a list of rows in the ROW schema below.
scripts/make_results.py turns results/*.json into RESULTS.md.

Row schema (one per engine x voice x config x text):
    engine, model, voice, device, config, text_id, chars,
    audio_s, gen_s (median), gen_s_all, rtf (gen_s / audio_s),
    ttfa_s (median time to first audio chunk, None if not streaming),
    sample (path to the wav, relative to repo root)
"""
from __future__ import annotations

import json
import os
import platform
import resource
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SAMPLES_DIR = ROOT / "samples"

# Fixed English test texts. Keep them stable so results stay comparable.
# "long" is kept under ~24 s of speech so engines with a length cap still fit.
TEXTS = {
    "short": "Hello there. How can I help you today?",
    "medium": (
        "Good morning, and welcome back. The meeting starts at nine, "
        "and the coffee is already on the table."
    ),
    "long": (
        "The old lighthouse stood at the edge of the cliff for more than a century. "
        "Every night its keeper climbed the narrow stairs, lit the lamp, and watched "
        "the ships pass safely through the fog. When the storms came, the light never "
        "failed, and the sailors who saw it knew they were almost home."
    ),
    "numbers": "There are 24 students in room 12, and the class starts at 9:30.",
}


def median(xs):
    return statistics.median(xs) if xs else None


def timed(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t


def peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def system_info() -> dict:
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = None
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
    except Exception:
        pass
    return {
        "cpu": cpu,
        "affinity_cpus": len(os.sched_getaffinity(0)),
        "loadavg": os.getloadavg(),
        "gpu": gpu,
        "python": platform.python_version(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def sample_path(engine: str, name: str) -> Path:
    p = SAMPLES_DIR / engine / f"{name}.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_results(engine: str, meta: dict, rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{engine}.json"
    out.write_text(json.dumps({"engine": engine, "meta": meta, "rows": rows}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")
    return out
