"""Path setup shared by the fast-engine scripts (works in /root/tts/scripts/fast and the repo's
optimizations/scripts/fast). Importing it puts scripts/ and scripts/fast/ on sys.path."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../scripts/fast
SCRIPTS = HERE.parent                            # .../scripts (qwen_fused_predictor, bench_common, ...)
ROOT = HERE.parents[1]                           # results/ and samples/ live here
TUNED = HERE / "tuned"
RESULTS = ROOT / "results" / "fast"
SAMPLES = ROOT / "samples" / "fast"
for p in (str(SCRIPTS), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
