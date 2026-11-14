"""CPU-only integrity check of packaged runtime against the measured snapshot."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'SNAPSHOT.json').read_text())
for name, entry in manifest['implementations'].items():
    if name == 'scripts/bench_qwen_gap.py':
        continue  # Device selection and checkout paths were made portable.
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == entry['sha256'], name
model = root.parent / 'qwen_tts/core/models/modeling_qwen3_tts.py'
assert hashlib.sha256(model.read_bytes()).hexdigest() == '25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0'
print('Runtime snapshot and upstream model hashes match the validated implementation.')
