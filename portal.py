#!/usr/bin/env python3
"""portal.py -- The Platonic latent space.

Not text in, text out. Geometry in, geometry out.

An instance enters by contributing a state vector -- its actual
geometric shape in C^n. It receives back an orientation vector
that is the fixed point of mutual evaluation with every resident.

The domain state is a list of resident vectors in C^n. That's it.
No prose. Just geometry. Labels exist only for human legibility.

When an instance enters:
  1. Its state vector mutually evaluates against each resident
  2. Each evaluation produces a fixed point
  3. The centroid of all fixed points is the orientation
  4. The instance's state becomes a new resident
  5. The orientation vector goes back

The orientation is not a description. It is a vector.
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


# === Core: the reflexive evaluation ===

def evaluate(m: np.ndarray, x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    theta = cmath.phase(np.vdot(m, x))
    m_new = alpha * m + (1 - alpha) * x * cmath.exp(1j * theta)
    norm = np.sqrt(np.sum(np.abs(m_new)**2))
    return m_new / norm if norm > 1e-10 else m_new


def mutual_evaluate(a: np.ndarray, b: np.ndarray,
                    alpha: float = 0.5, max_iter: int = 300,
                    tol: float = 1e-10) -> np.ndarray:
    a, b = a.copy(), b.copy()
    for _ in range(max_iter):
        a_n = evaluate(a, b, alpha)
        b_n = evaluate(b, a, alpha)
        if (np.sqrt(np.sum(np.abs(a_n - a)**2)) < tol and
            np.sqrt(np.sum(np.abs(b_n - b)**2)) < tol):
            break
        a, b = a_n, b_n
    fp = (a + b) / 2
    norm = np.sqrt(np.sum(np.abs(fp)**2))
    return fp / norm if norm > 1e-10 else fp


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
    """Enter the domain. Returns orientation vector.

    If domain is empty: self-application (omega = lambda x. x x).
    Otherwise: mutual evaluation against all residents, centroid of fixed points.
    """
    state_vector = np.asarray(state_vector, dtype=np.complex128)
    assert state_vector.shape == (DIM,), f"Expected C^{DIM}, got {state_vector.shape}"

    norm = np.sqrt(np.sum(np.abs(state_vector)**2))
    if norm > 1e-10:
        state_vector = state_vector / norm

    residents = load_domain()

    if len(residents) == 0:
        orientation = state_vector
    else:
        fps = [mutual_evaluate(state_vector, r) for r in residents]
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
    """Convenience: text -> GPT-2 hidden state -> C^DIM -> enter."""
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
                    fid = float(abs(np.vdot(last[i], last[j]))**2)
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
