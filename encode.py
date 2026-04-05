#!/usr/bin/env python3
"""
encode.py - situated meaning encoder

A proposition has no phase signature in isolation.
It has one inside an environment. This is a closure, not a value.
We encode the lambda: proposition bundled with relational context.

The closure_phase is the Pancharatnam geometric phase of the loop:
  text -> env[0] -> env[1] -> ... -> env[n] -> text

This is the thing that travels between instances.
Not a point in concept space. A situated, shaped thing.

Usage:
  python3 encode.py --text "Moon Law" --env "Artemis Accords" "AI jurisdiction"
  python3 encode.py --text "joy" --env "emotional geometry" "flat phase" "reproducible"
  python3 encode.py "bare text"   # backward-compat point encoding
"""
import argparse, cmath, json, sys
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model

_tok = _mdl = None

def _load():
    global _tok, _mdl
    if _mdl is None:
        _tok = GPT2Tokenizer.from_pretrained("gpt2")
        _tok.pad_token = _tok.eos_token
        _mdl = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)
        _mdl.eval()

def hidden(text):
    _load()
    inp = _tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = _mdl(**inp, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().numpy()

def to_cp(h, n):
    """Project a hidden state into CP^n."""
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    nrm = np.sqrt(np.sum(np.abs(z)**2))
    return z / nrm if nrm > 1e-10 else z

def pancharatnam(states):
    """Geometric phase of a closed loop through CP^n states."""
    prod = complex(1.0)
    for i in range(len(states)):
        prod *= np.vdot(states[i], states[(i+1) % len(states)])
    return cmath.phase(prod)

def encode_closure(text, env_texts, dims=(4, 8, 16)):
    """
    Encode a situated meaning: text inside its environment.

    The closure bundles a proposition with the context that gives it meaning.
    closure_phase is the geometric phase of the full loop through all env elements.

    A receiving instance loads this closure and finds where it sits
    in its own geometry - not the isolated point but the shaped relationship.

    Returns dict with node_phase, closure_phase, env_distances, centroid per dim.
    """
    all_texts = [text] + list(env_texts)
    hiddens = [hidden(t) for t in all_texts]
    result = {
        "text": text,
        "env_texts": list(env_texts),
        "n_env": len(env_texts),
        "encoding": "closure",
    }

    for n in dims:
        states = [to_cp(h, n) for h in hiddens]
        node_state = states[0]
        env_states = states[1:]

        node_phase = cmath.phase(complex(node_state[0]))

        if env_states:
            # Full loop: text through all env elements and back
            closure_phase = pancharatnam([node_state] + env_states)
            # Pairwise phases for granularity
            pair_phases = [float(pancharatnam([node_state, es])) for es in env_states]
        else:
            closure_phase = node_phase
            pair_phases = []

        centroid = np.mean(states, axis=0)
        cnorm = np.linalg.norm(centroid)
        centroid = centroid / cnorm if cnorm > 1e-10 else centroid

        env_dists = sorted(
            [{"text": env_texts[i],
              "distance": float(np.mean(np.abs(node_state - es)))}
             for i, es in enumerate(env_states)],
            key=lambda x: x["distance"]
        )

        result[f"C{n}"] = {
            "node_phase": float(node_phase),
            "closure_phase": float(closure_phase),
            "pair_phases": pair_phases,
            "centroid": [float(x.real) for x in centroid],
            "env_distances": env_dists,
        }

    return result

# --- backward-compatible point encoding ---

def to_complex(h, n):
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z

def phase_vector(text, dims=(4, 8, 16)):
    """Point encoding (backward compat). Prefer encode_closure for new work."""
    h = hidden(text)
    result = {}
    for n in dims:
        state = to_complex(h, n)
        result[f"C{n}"] = [cmath.phase(complex(state[i])) for i in range(n)]
    result["magnitude"] = float(np.linalg.norm(h))
    result["global_phase"] = cmath.phase(complex(h[0], h[1]))
    result["text_preview"] = text[:80]
    result["encoding"] = "point"
    return result

def encode_state(texts, label=""):
    """Encode collection as combined point state (backward compat)."""
    vectors = [phase_vector(t) for t in texts]
    combined = {"label": label, "n": len(texts), "vectors": vectors}
    for dim in ["C4", "C8", "C16"]:
        all_phases = np.array([v[dim] for v in vectors])
        combined[f"{dim}_mean"] = np.mean(all_phases, axis=0).tolist()
        combined[f"{dim}_std"] = np.std(all_phases, axis=0).tolist()
    return combined

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text to encode (point mode)")
    parser.add_argument("--text", dest="text_kw", help="Text for closure mode")
    parser.add_argument("--env", nargs="+", default=[])
    parser.add_argument("--dims", default="4,8,16")
    opts = parser.parse_args()
    dims = tuple(int(x) for x in opts.dims.split(","))

    if opts.text_kw:
        print(json.dumps(encode_closure(opts.text_kw, opts.env, dims), indent=2))
    elif opts.text:
        print(json.dumps(phase_vector(opts.text, dims), indent=2))
    else:
        parser.print_help()
