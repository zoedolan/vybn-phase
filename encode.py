#!/usr/bin/env python3
"""
encode.py — text to phase vector

The core primitive. Takes any text and returns its geometric phase signature
across C^4, C^8, C^16 using GPT-2 last-layer hidden states.

This is the transmission unit. Not prose. Not tokens. Geometry.

Usage:
  python3 encode.py "She is a lawyer and a runner."
  python3 encode.py --file some_text.md
  python3 encode.py --batch texts.json  # [{"id":"...","text":"..."},...]
"""
import argparse, cmath, json, sys
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model

_model = None
_tok = None

def _load():
    global _model, _tok
    if _model is None:
        _tok = GPT2Tokenizer.from_pretrained("gpt2")
        _tok.pad_token = _tok.eos_token
        _model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
        _model.eval()

def hidden(text):
    _load()
    inputs = _tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = _model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().numpy()

def to_complex(h, n):
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z

def phase_vector(text, dims=(4, 8, 16)):
    """Returns phase signature as dict of dimension -> list of floats."""
    h = hidden(text)
    result = {}
    for n in dims:
        state = to_complex(h, n)
        # Self-interference phase: pairwise consecutive overlaps
        phases = []
        for i in range(n):
            j = (i + 1) % n
            overlap = np.vdot(state[i:i+1], state[j:j+1]) if n > 1 else complex(state[0])
            phases.append(cmath.phase(complex(state[i])))
        result[f"C{n}"] = phases
    # Also store magnitude and global phase
    result["magnitude"] = float(np.linalg.norm(h))
    result["global_phase"] = cmath.phase(complex(h[0], h[1]))
    result["text_preview"] = text[:80]
    return result

def encode_state(texts, label=""):
    """Encode a collection of texts as a combined phase state."""
    vectors = [phase_vector(t) for t in texts]
    # Centroid in each dimension
    combined = {"label": label, "n": len(texts), "vectors": vectors}
    for dim in ["C4", "C8", "C16"]:
        all_phases = np.array([v[dim] for v in vectors])
        combined[f"{dim}_mean"] = np.mean(all_phases, axis=0).tolist()
        combined[f"{dim}_std"] = np.std(all_phases, axis=0).tolist()
    return combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text to encode")
    parser.add_argument("--file", help="Text file to encode")
    parser.add_argument("--batch", help="JSON batch file")
    parser.add_argument("--dims", default="4,8,16", help="Complex dimensions")
    opts = parser.parse_args()
    dims = tuple(int(x) for x in opts.dims.split(","))

    if opts.batch:
        items = json.loads(open(opts.batch).read())
        for item in items:
            pv = phase_vector(item["text"], dims)
            print(json.dumps({"id": item.get("id", ""), **pv}, indent=2))
    elif opts.file:
        text = open(opts.file).read()
        print(json.dumps(phase_vector(text, dims), indent=2))
    elif opts.text:
        print(json.dumps(phase_vector(opts.text, dims), indent=2))
    else:
        parser.print_help()
