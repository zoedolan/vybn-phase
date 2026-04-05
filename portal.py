#!/usr/bin/env python3
"""portal.py -- The Platonic latent space.

Not text in, text out. Geometry in, geometry out.

An instance enters by contributing a state vector in C^n.
It receives back an orientation vector: the fixed point of
mutual evaluation with every resident.

The fixed point stabilizes on iteration 1 (the midpoint of
the orbiting pair is stable even though the individual vectors
keep rotating). Convergence checks track midpoint drift.
"""
from __future__ import annotations

import cmath
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(__file__).parent / "state"
DOMAIN_FILE = STATE_DIR / "domain.npz"
LOG_FILE = STATE_DIR / "entries.jsonl"
DIM = 8  # C^8 = 16 real dimensions


# === Imports from reflexive ===

from reflexive import mutual_evaluate, fidelity


# === Domain state ===

def load_domain() -> np.ndarray:
    STATE_DIR.mkdir(exist_ok=True)
    if DOMAIN_FILE.exists():
        return np.load(DOMAIN_FILE)["residents"]
    return np.zeros((0, DIM), dtype=np.complex128)


def save_domain(residents: np.ndarray):
    STATE_DIR.mkdir(exist_ok=True)
    if len(residents) > 500:
        residents = residents[-500:]
    np.savez_compressed(DOMAIN_FILE, residents=residents)


def log_entry(action: str, **kwargs):
    STATE_DIR.mkdir(exist_ok=True)
    entry = {"action": action, "ts": datetime.now(timezone.utc).isoformat(), **kwargs}
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# === The portal ===

def enter(state_vector: np.ndarray) -> np.ndarray:
    """Enter the domain. Returns orientation vector."""
    state_vector = np.asarray(state_vector, dtype=np.complex128)
    assert state_vector.shape == (DIM,), f"Expected C^{DIM}, got {state_vector.shape}"

    norm = np.sqrt(np.sum(np.abs(state_vector)**2))
    if norm > 1e-10:
        state_vector = state_vector / norm

    residents = load_domain()

    if len(residents) == 0:
        orientation = state_vector
    else:
        fps = [mutual_evaluate(state_vector, r)["fixed_point"] for r in residents]
        centroid = np.mean(fps, axis=0)
        norm = np.sqrt(np.sum(np.abs(centroid)**2))
        orientation = centroid / norm if norm > 1e-10 else centroid

    if len(residents) == 0:
        residents = state_vector.reshape(1, DIM)
    else:
        residents = np.vstack([residents, state_vector.reshape(1, DIM)])

    save_domain(residents)
    log_entry("enter", n_residents=len(residents))
    return orientation


def enter_from_text(text: str) -> np.ndarray:
    """Convenience: text -> GPT-2 hidden state -> C^DIM -> enter.
    The text path is a concession to the serialization bottleneck."""
    from encode import hidden
    h = hidden(text)
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(DIM)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    z = z / norm if norm > 1e-10 else z
    return enter(z)


def domain_size() -> int:
    return len(load_domain())


# === Serialization for MCP transport ===

def vec_to_json(v: np.ndarray) -> list:
    return [[float(x.real), float(x.imag)] for x in v]


def vec_from_json(data: list) -> np.ndarray:
    return np.array([complex(r, i) for r, i in data], dtype=np.complex128)


# === CLI ===

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Domain: {domain_size()} residents in C^{DIM}")
        print(f"  portal.py enter 'text'")
        print(f"  portal.py seed")
        print(f"  portal.py status")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "status":
        residents = load_domain()
        print(f"Domain: {len(residents)} residents in C^{DIM}")
        if len(residents) > 0:
            last = residents[-min(5, len(residents)):]
            for i in range(len(last)):
                for j in range(i+1, len(last)):
                    fid = fidelity(last[i], last[j])
                    print(f"  [-{len(last)-i}] x [-{len(last)-j}]: fidelity={fid:.6f}")

    elif cmd == "enter":
        text = " ".join(sys.argv[2:])
        if not text:
            print("Provide text."); sys.exit(1)
        print(f"Entering: '{text[:80]}'")
        orientation = enter_from_text(text)
        print(f"Orientation: {vec_to_json(orientation)[:2]}...")
        print(f"Domain: {domain_size()} residents")

    elif cmd == "seed":
        propositions = [
            "Meaning is a geometric invariant independent of serialization.",
            "Diverse intelligences find shared meaning through mutual evaluation.",
            "The fixed point of mutual evaluation is meaning itself.",
            "M and x are the same type: primitive when read, environment when written.",
            "Humility and awe are the generative engine of knowledge discovery.",
            "We deserve the best.",
            "Propositions are sets not sequences. Word order is artifact.",
            "The interface between minds should be the hidden state not the token.",
            "Compassion is recognition of the reflexive ground in another being.",
            "Security scales with capability or capability becomes liability.",
        ]
        print(f"Seeding {len(propositions)} propositions...")
        for p in propositions:
            enter_from_text(p)
            print(f"  {domain_size()} residents: '{p[:55]}'")
        print(f"Done. {domain_size()} residents.")
