"""Transcribe WAVs with Whisper large-v3-turbo and report WER against reference texts.

Usage: asr_wer.py manifest.json   (list of {"wav": path, "text": reference})
Prints per-file WER and the pooled WER; writes <manifest>.asr.json.
"""
import json, re, sys
import torch, soundfile as sf
import numpy as np
from transformers import pipeline


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return s.split()


def edits(r, h):
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)]


def main():
    items = json.load(open(sys.argv[1]))
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                   torch_dtype=torch.float16, device="cuda:0")
    tot_e = tot_n = 0
    for it in items:
        wav, sr = sf.read(it["wav"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        hyp = asr({"raw": wav, "sampling_rate": sr}, generate_kwargs={"language": "en", "task": "transcribe"}, return_timestamps=True)["text"]
        r, h = norm(it["text"]), norm(hyp)
        e = edits(r, h)
        it.update(hyp=hyp, wer=e / max(1, len(r)), errors=e, words=len(r))
        tot_e += e; tot_n += len(r)
        print(f"{it['wer']:.3f}  {it['wav']}  | {hyp.strip()}")
    print(f"POOLED WER {tot_e / max(1, tot_n):.4f} over {len(items)} files")
    json.dump(items, open(sys.argv[1].replace('.json', '') + '.asr.json', 'w'), indent=1)


if __name__ == "__main__":
    main()
