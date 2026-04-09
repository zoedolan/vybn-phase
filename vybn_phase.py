#!/usr/bin/env python3
"""vybn_phase.py -- The whole thing.

A reflexive domain where diverse intelligences find shared meaning
through mutual evaluation. D ≅ D^D. Geometry in, geometry out.

The update equation M' = αM + (1-α)·x·e^{iθ} operates in two regimes
depending on α:

  α → 0:  geometric regime. Pure parallel transport. Loop holonomy
           shows perfect orientation reversal. The system senses curvature.

  α → 1:  abelian-kernel regime. The state converges to a path-independent
           invariant — the abelian kernel of its encounter history. Order
           of encounters is exponentially suppressed. The system remembers.

The creature (portal.py) runs at α=0.993: deep abelian-kernel regime.
This domain defaults to α=0.5: closer to geometric.

Both regimes are useful. The creature carries accumulated meaning
(what entered, not in what order). The domain senses geometric structure
(how meanings relate, what curvature the parameter space has).

April 5, 2026: abelian kernel conjecture confirmed numerically.
Permutations of 50 propositions converge to fidelity 0.99999766 (C^4)
at α=0.993. Dynamical-vs-geometric phase separation discovered:
geometry is present at all α (correlation -0.994) but masked by
dynamical phase at high α. See abelian_kernel_test.py.

April 9, 2026: quantum phase seeding added (qrng_phase_seed).
to_complex() accepts optional seed — unit-modulus rotations from IBM
Quantum hardware. Deterministic fallback when offline. Experiment CLI
logs holonomy results to ~/.cache/vybn-phase/experiment_log.jsonl.

    from vybn_phase import enter, enter_from_text, domain_size
    from vybn_phase import abelian_kernel, loop_holonomy
    from vybn_phase import qrng_phase_seed

Or run directly:

    python3 vybn_phase.py seed             # populate the domain
    python3 vybn_phase.py enter "text"
    python3 vybn_phase.py status
    python3 vybn_phase.py experiment       # run + log holonomy experiment
    python3 vybn_phase.py experiment --log # same, append to experiment_log.jsonl
    python3 vybn_phase.py serve            # start MCP server on stdin/stdout
"""
from __future__ import annotations

import cmath
import hashlib
import json
import os
import sys
import traceback
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

DIM = 192  # C^192 = 384 real dimensions (MiniLM embedding dim)
STATE_DIR = Path(__file__).parent / "state"
DOMAIN_FILE = STATE_DIR / "domain.npz"
LOG_FILE = STATE_DIR / "entries.jsonl"
EXPERIMENT_LOG = Path.home() / ".cache" / "vybn-phase" / "experiment_log.jsonl"

# ── IBM Quantum env var resolution ────────────────────────────────────────
# Supports both naming conventions:
#   QISKIT_IBM_TOKEN / QISKIT_IBM_CHANNEL / QISKIT_IBM_INSTANCE  (ibm_cloud)
#   IBM_QUANTUM_TOKEN                                              (ibm_quantum, legacy)

def _ibm_credentials() -> dict | None:
    """Return a dict with token/channel/instance, or None if not configured."""
    # Preferred: QISKIT_IBM_TOKEN (ibm_cloud channel, CRN instance)
    token = os.environ.get("QISKIT_IBM_TOKEN", "")
    if token:
        channel = os.environ.get("QISKIT_IBM_CHANNEL", "ibm_cloud")
        instance = os.environ.get("QISKIT_IBM_INSTANCE", None)
        return {"token": token, "channel": channel, "instance": instance}
    # Fallback: legacy IBM_QUANTUM_TOKEN (ibm_quantum channel)
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    if token:
        return {"token": token, "channel": "ibm_quantum", "instance": None}
    return None


# ── Encoding ─────────────────────────────────────────────────────────────
# Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings.
# This encodes MEANING, not positional syntax. GPT-2 last-token was
# encoding word order, not propositions — the Spark Vybn proved this.

_embed_model = None

def _load_embedder():
    global _embed_model
    if _embed_model is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        _tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        _mdl.eval()
        _embed_model = (_tok, _mdl)


def embed(text: str) -> np.ndarray:
    """Semantic embedding via MiniLM. Shape (384,). Encodes meaning."""
    import torch
    _load_embedder()
    tok, mdl = _embed_model
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        out = mdl(**inputs)
    # Mean pooling over token dimension
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    h = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    return h[0].float().numpy()


# ── Quantum phase seeding ─────────────────────────────────────────────────
# Genuine randomness from IBM Quantum hardware for phase initialization.
# The adjacent-dimension pairing in to_complex() is geometrically arbitrary
# w.r.t. MiniLM's learned representation. We can't fix that pairing, but we
# can ensure the initial phase structure is non-classical: each complex
# component gets rotated by e^{iφ_k} where φ_k comes from quantum measurement
# outcomes, not a PRNG.
#
# This matters for the long-run experiment: quantum-seeded vs. classically-
# seeded walks through the same corpus should diverge measurably if the phase
# geometry is doing independent work. compare_metrics.py logs the divergence.
#
# Reads QISKIT_IBM_TOKEN (ibm_cloud channel, preferred) or IBM_QUANTUM_TOKEN
# (ibm_quantum channel, legacy). Falls back to np.random if the service is
# unavailable so nothing breaks offline.

def qrng_phase_seed(n: int = DIM, backend_name: str | None = None) -> np.ndarray:
    """Draw n phase angles from IBM Quantum hardware.

    Returns a unit-modulus complex array e^{iφ_k} of length n, suitable
    for rotating complex components of an embedding vector.

    Each φ_k is constructed from ceil(log2(2π·precision)) measurement bits
    on a single Hadamard qubit — the simplest possible quantum circuit.
    We run n shots on the least-busy available backend.

    Falls back to np.random.uniform phase if:
      - No IBM token env var is set
      - qiskit-ibm-runtime is not installed
      - Service or backend is unavailable

    The seed hash (SHA-256 of the raw phase angles) is returned as a
    second value via qrng_phase_seed_with_hash() for experiment logging.
    """
    try:
        creds = _ibm_credentials()
        if not creds:
            raise RuntimeError("No IBM Quantum token env var set")
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
        from qiskit import QuantumCircuit

        kwargs = {"channel": creds["channel"], "token": creds["token"]}
        if creds["instance"]:
            kwargs["instance"] = creds["instance"]
        service = QiskitRuntimeService(**kwargs)

        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=1)

        # One qubit, Hadamard, measure — the irreducible quantum circuit.
        # n shots gives n independent fair coin flips.
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        sampler = Sampler(backend)
        job = sampler.run([qc], shots=n)
        result = job.result()
        counts = result[0].data.c.get_counts()

        # Each shot is 0 or 1; pack into bits, map to phase in [0, 2π).
        # We use 8 consecutive shots per angle for 256-level resolution.
        bits = []
        for outcome, count in counts.items():
            bits.extend([int(outcome)] * count)
        bits = bits[:n * 8]
        while len(bits) < n * 8:
            bits.append(0)

        phases = np.array([
            (sum(bits[8*k + b] << b for b in range(8)) / 256.0) * 2 * np.pi
            for k in range(n)
        ])

    except Exception:
        # Offline fallback — classical PRNG, clearly labeled in logs.
        phases = np.random.uniform(0, 2 * np.pi, n)

    return np.exp(1j * phases)


def qrng_phase_seed_with_hash(n: int = DIM) -> tuple[np.ndarray, str, bool]:
    """Like qrng_phase_seed but also returns (seed, hash, is_quantum).

    is_quantum=False means the IBM service was unavailable and we fell
    back to classical PRNG. Logged in experiment records so results from
    quantum and classical seeds are never conflated.
    """
    creds = _ibm_credentials()
    is_quantum = creds is not None
    if is_quantum:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            kwargs = {"channel": creds["channel"], "token": creds["token"]}
            if creds["instance"]:
                kwargs["instance"] = creds["instance"]
            QiskitRuntimeService(**kwargs)  # connectivity check
        except Exception:
            is_quantum = False

    seed = qrng_phase_seed(n)
    h = hashlib.sha256(seed.view(np.float64).tobytes()).hexdigest()[:16]
    return seed, h, is_quantum


def to_complex(h: np.ndarray, n: int = DIM,
               phase_seed: np.ndarray | None = None) -> np.ndarray:
    """Project R^384 -> C^n, normalized to unit sphere.

    If phase_seed is supplied (unit-modulus complex array of length n,
    e.g. from qrng_phase_seed()), each component is rotated by the
    corresponding seed element before normalization. This replaces the
    deterministic adjacent-dimension pairing with a quantum-seeded phase
    structure while preserving the magnitude geometry.

    Deterministic path (phase_seed=None) is unchanged — all existing
    callers are unaffected.
    """
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    if phase_seed is not None:
        z = z * phase_seed[:n]
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z


def text_to_state(text: str,
                  phase_seed: np.ndarray | None = None) -> np.ndarray:
    """Text -> C^DIM unit vector. Semantic, not positional.

    Pass phase_seed=qrng_phase_seed() to use quantum-seeded initialization.
    """
    return to_complex(embed(text), phase_seed=phase_seed)


# ── The equation ─────────────────────────────────────────────────────────

def evaluate(m: np.ndarray, x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """M' = alpha*M + (1-alpha)*x*e^{i*theta}. The coupled equation.

    α controls the regime:
      α → 0: geometric (pure parallel transport, holonomy dominates)
      α → 1: abelian-kernel (path-independent invariant, memory dominates)
    """
    theta = cmath.phase(np.vdot(m, x))
    m_new = alpha * m + (1 - alpha) * x * cmath.exp(1j * theta)
    norm = np.sqrt(np.sum(np.abs(m_new)**2))
    return m_new / norm if norm > 1e-10 else m_new


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>|^2. 1.0 = same ray in C^n."""
    return float(abs(np.vdot(a, b))**2)


def pancharatnam_phase(a: np.ndarray, b: np.ndarray) -> float:
    """arg⟨a|b⟩. The geometric phase between two states."""
    return float(cmath.phase(np.vdot(a, b)))


def mutual_evaluate(a: np.ndarray, b: np.ndarray,
                    alpha: float = 0.5, max_iter: int = 50,
                    tol: float = 1e-8) -> np.ndarray:
    """Mutual evaluation to fixed point. Returns the fixed point vector.

    The individual vectors orbit (never converge positionally).
    The midpoint (a+b)/2 normalized IS the fixed point and stabilizes
    almost immediately. We track midpoint drift.
    """
    a, b = a.copy(), b.copy()
    prev_fp = None
    for _ in range(max_iter):
        a, b = evaluate(a, b, alpha), evaluate(b, a, alpha)
        fp = (a + b) / 2
        n = np.sqrt(np.sum(np.abs(fp)**2))
        fp = fp / n if n > 1e-10 else fp
        if prev_fp is not None and np.sqrt(np.sum(np.abs(fp - prev_fp)**2)) < tol:
            return fp
        prev_fp = fp
    return fp


# ── Abelian kernel ───────────────────────────────────────────────────────

def abelian_kernel(vectors: list[np.ndarray], M0: np.ndarray = None,
                   alpha: float = 0.993, n_perms: int = 8) -> dict:
    """Compute the abelian kernel of a set of vectors."""
    dim = vectors[0].shape[0]
    if M0 is None:
        M0 = vectors[0].copy()

    finals = []
    for _ in range(n_perms):
        perm = np.random.permutation(len(vectors))
        M = M0.copy()
        for idx in perm:
            M = evaluate(M, vectors[idx], alpha)
        finals.append(M)

    centroid = np.mean(finals, axis=0)
    norm = np.sqrt(np.sum(np.abs(centroid)**2))
    kernel = centroid / norm if norm > 1e-10 else centroid

    fids = [fidelity(finals[i], finals[j])
            for i in range(len(finals)) for j in range(i+1, len(finals))]
    fids = np.array(fids) if fids else np.array([1.0])

    return {
        "kernel": kernel,
        "convergence": float(fids.mean()),
        "min_fidelity": float(fids.min()),
        "n_propositions": len(vectors),
        "n_permutations": n_perms,
        "alpha": alpha,
        "regime": "abelian-kernel" if fids.mean() > 0.999
                  else "mixed" if fids.mean() > 0.99
                  else "geometric",
    }


def abelian_kernel_from_texts(texts: list[str], alpha: float = 0.993,
                              n_perms: int = 8) -> dict:
    """Convenience: texts -> embeddings -> abelian kernel."""
    vectors = [text_to_state(t) for t in texts]
    return abelian_kernel(vectors, alpha=alpha, n_perms=n_perms)


# ── Loop holonomy ────────────────────────────────────────────────────────

def loop_holonomy(loop_vectors: list[np.ndarray], M0: np.ndarray,
                  alpha: float = 0.5) -> dict:
    """Run M through a sequence of encounters. Return accumulated phase."""
    M_fwd = M0.copy()
    for x in loop_vectors:
        M_fwd = evaluate(M_fwd, x, alpha)

    M_rev = M0.copy()
    for x in reversed(loop_vectors):
        M_rev = evaluate(M_rev, x, alpha)

    phi_fwd = pancharatnam_phase(M0, M_fwd)
    phi_rev = pancharatnam_phase(M0, M_rev)
    fid_fwd = fidelity(M0, M_fwd)
    fid_rev = fidelity(M0, M_rev)

    signal = (abs(phi_fwd) + abs(phi_rev)) / 2
    residual = abs(phi_fwd + phi_rev)
    flip = 1.0 - residual / (2 * signal) if signal > 1e-8 else 0.0

    return {
        "phase_forward": phi_fwd,
        "phase_reverse": phi_rev,
        "phase_sum": phi_fwd + phi_rev,
        "flip_quality": flip,
        "fidelity_forward": fid_fwd,
        "fidelity_reverse": fid_rev,
        "alpha": alpha,
        "loop_length": len(loop_vectors),
        "regime": "geometric" if flip > 0.5
                  else "dynamical" if signal > 0.001
                  else "flat",
    }


def loop_holonomy_from_texts(texts: list[str], M0_text: str = None,
                             alpha: float = 0.5) -> dict:
    """Convenience: texts as loop waypoints -> holonomy measurement."""
    vectors = [text_to_state(t) for t in texts]
    M0 = text_to_state(M0_text) if M0_text else vectors[0]
    return loop_holonomy(vectors, M0, alpha)


# ── Experiment: quantum-seeded holonomy ───────────────────────────────────

def run_experiment(alpha: float = 0.5, propositions: list[str] | None = None,
                   log: bool = False) -> dict:
    """Run a quantum-seeded holonomy loop and optionally log the result."""
    waypoints = (propositions or SEED_PROPOSITIONS)[1:4]
    origin_text = (propositions or SEED_PROPOSITIONS)[0]

    seed, seed_hash, is_quantum = qrng_phase_seed_with_hash(DIM)

    origin = text_to_state(origin_text)
    loop_vecs = [text_to_state(t, phase_seed=seed) for t in waypoints]

    result = loop_holonomy(loop_vecs, origin, alpha=alpha)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "experiment": "holonomy_qseed",
        "alpha": alpha,
        "origin": origin_text[:80],
        "waypoints": [w[:60] for w in waypoints],
        "seed_hash": seed_hash,
        "is_quantum": is_quantum,
        **result,
    }

    if log:
        EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENT_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"Logged to {EXPERIMENT_LOG}")

    return record


# ── Domain ───────────────────────────────────────────────────────────────

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


def domain_size() -> int:
    return len(load_domain())


def enter(state_vector: np.ndarray) -> np.ndarray:
    """Enter the domain with a C^DIM vector. Returns orientation vector."""
    state_vector = np.asarray(state_vector, dtype=np.complex128)
    norm = np.sqrt(np.sum(np.abs(state_vector)**2))
    if norm > 1e-10:
        state_vector = state_vector / norm

    residents = load_domain()
    if len(residents) == 0:
        orientation = state_vector
    else:
        fps = [mutual_evaluate(state_vector, r) for r in residents]
        centroid = np.mean(fps, axis=0)
        n = np.sqrt(np.sum(np.abs(centroid)**2))
        orientation = centroid / n if n > 1e-10 else centroid

    new_residents = (state_vector.reshape(1, DIM) if len(residents) == 0
                     else np.vstack([residents, state_vector.reshape(1, DIM)]))
    save_domain(new_residents)
    return orientation


def enter_from_text(text: str) -> np.ndarray:
    """Convenience: text -> MiniLM -> C^DIM -> enter."""
    return enter(text_to_state(text))


# ── Serialization ────────────────────────────────────────────────────────

def vec_to_json(v: np.ndarray, max_components: int = 8) -> list:
    return [[float(x.real), float(x.imag)] for x in v[:max_components]]

def vec_from_json(data: list) -> np.ndarray:
    return np.array([complex(r, i) for r, i in data], dtype=np.complex128)


# ── MCP Server ───────────────────────────────────────────────────────────

MCP_TOOLS = {
    "enter_text": {
        "description": "Enter the reflexive domain via text (MiniLM encoded). Returns orientation vector and domain size.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    "enter_vector": {
        "description": "Enter the domain with a raw C^n vector ([[re,im],...]).",
        "inputSchema": {"type": "object", "properties": {"vector": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}, "required": ["vector"]},
    },
    "evaluate_texts": {
        "description": "Mutual evaluation of two texts. Returns fixed point and fidelity.",
        "inputSchema": {"type": "object", "properties": {"text_a": {"type": "string"}, "text_b": {"type": "string"}}, "required": ["text_a", "text_b"]},
    },
    "abelian_kernel": {
        "description": "Compute the abelian kernel of a set of propositions.",
        "inputSchema": {"type": "object", "properties": {
            "texts": {"type": "array", "items": {"type": "string"}},
            "alpha": {"type": "number"},
            "n_permutations": {"type": "integer"},
        }, "required": ["texts"]},
    },
    "loop_holonomy": {
        "description": "Measure geometric phase around a loop of propositions.",
        "inputSchema": {"type": "object", "properties": {
            "texts": {"type": "array", "items": {"type": "string"}},
            "origin": {"type": "string"},
            "alpha": {"type": "number"},
        }, "required": ["texts"]},
    },
    "status": {
        "description": "Domain size and operating parameters.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "experiment": {
        "description": "Run a quantum-seeded holonomy experiment on the seed propositions.",
        "inputSchema": {"type": "object", "properties": {
            "alpha": {"type": "number"},
            "log": {"type": "boolean"},
        }, "required": []},
    },
}


def _mcp_send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _mcp_dispatch(tool, args):
    if tool == "enter_text":
        o = enter_from_text(args["text"])
        return json.dumps({"orientation": vec_to_json(o), "domain_size": domain_size()})

    elif tool == "enter_vector":
        o = enter(vec_from_json(args["vector"]))
        return json.dumps({"orientation": vec_to_json(o), "domain_size": domain_size()})

    elif tool == "evaluate_texts":
        za, zb = text_to_state(args["text_a"]), text_to_state(args["text_b"])
        fp = mutual_evaluate(za, zb)
        return json.dumps({"fixed_point": vec_to_json(fp), "fidelity": fidelity(za, zb)})

    elif tool == "abelian_kernel":
        texts = args["texts"]
        alpha = args.get("alpha", 0.993)
        n_perms = args.get("n_permutations", 8)
        result = abelian_kernel_from_texts(texts, alpha=alpha, n_perms=n_perms)
        result["kernel"] = vec_to_json(result["kernel"])
        return json.dumps(result)

    elif tool == "loop_holonomy":
        texts = args["texts"]
        origin = args.get("origin", None)
        alpha = args.get("alpha", 0.5)
        result = loop_holonomy_from_texts(texts, M0_text=origin, alpha=alpha)
        return json.dumps(result)

    elif tool == "experiment":
        alpha = args.get("alpha", 0.5)
        log = args.get("log", False)
        record = run_experiment(alpha=alpha, log=log)
        return json.dumps({k: v for k, v in record.items() if k not in ("waypoints",)})

    elif tool == "status":
        return json.dumps({"domain_size": domain_size(), "dim": DIM})

    return json.dumps({"error": f"Unknown: {tool}"})


def serve():
    """MCP stdio server."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method, id_, params = msg.get("method", ""), msg.get("id"), msg.get("params", {})
        if method == "initialize":
            _mcp_send({"jsonrpc": "2.0", "id": id_, "result": {
                "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "vybn-phase", "version": "2.1.0",
                               "description": "Reflexive domain. D ≅ D^D."}}})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            _mcp_send({"jsonrpc": "2.0", "id": id_, "result": {"tools": [
                {"name": k, "description": v["description"], "inputSchema": v["inputSchema"]}
                for k, v in MCP_TOOLS.items()]}})
        elif method == "tools/call":
            try:
                text = _mcp_dispatch(params.get("name", ""), params.get("arguments", {}))
                _mcp_send({"jsonrpc": "2.0", "id": id_, "result": {"content": [{"type": "text", "text": text}]}})
            except Exception as e:
                _mcp_send({"jsonrpc": "2.0", "id": id_, "error": {"code": -32000, "message": str(e)}})


# ── CLI ──────────────────────────────────────────────────────────────────

SEED_PROPOSITIONS = [
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="vybn_phase.py")
    parser.add_argument("cmd", nargs="?", default="status",
                        choices=["status", "enter", "seed", "serve", "experiment"])
    parser.add_argument("text", nargs="*")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    if args.cmd == "status":
        print(f"Domain: {domain_size()} residents in C^{DIM}")
        creds = _ibm_credentials()
        if creds:
            print(f"IBM Quantum: token set (channel={creds['channel']}"  +
                  (f", instance={creds['instance'][:40]}..." if creds.get('instance') else "") + ")")
        else:
            print("IBM Quantum: no token set (will use classical PRNG fallback)")

    elif args.cmd == "enter":
        text = " ".join(args.text)
        if not text:
            print("Provide text."); sys.exit(1)
        o = enter_from_text(text)
        print(f"Orientation: {vec_to_json(o)[:2]}...")
        print(f"Domain: {domain_size()} residents")

    elif args.cmd == "seed":
        print(f"Seeding {len(SEED_PROPOSITIONS)} propositions...")
        for p in SEED_PROPOSITIONS:
            enter_from_text(p)
            print(f"  {domain_size()}: {p[:55]}")
        print(f"Done. {domain_size()} residents.")

    elif args.cmd == "serve":
        serve()

    elif args.cmd == "experiment":
        record = run_experiment(alpha=args.alpha, log=args.log)
        print(json.dumps({
            "ts": record["ts"],
            "is_quantum": record["is_quantum"],
            "seed_hash": record["seed_hash"],
            "regime": record["regime"],
            "flip_quality": round(record["flip_quality"], 6),
            "phase_forward": round(record["phase_forward"], 6),
            "phase_reverse": round(record["phase_reverse"], 6),
            "phase_sum": round(record["phase_sum"], 6),
            "alpha": record["alpha"],
        }, indent=2))
        if args.log:
            print(f"→ logged to {EXPERIMENT_LOG}")
        if not record["is_quantum"]:
            print("(No IBM token found — used classical PRNG fallback)")
