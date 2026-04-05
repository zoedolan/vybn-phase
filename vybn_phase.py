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

    from vybn_phase import enter, enter_from_text, domain_size
    from vybn_phase import abelian_kernel, loop_holonomy

Or run directly:

    python3 vybn_phase.py seed       # populate the domain
    python3 vybn_phase.py enter "text"
    python3 vybn_phase.py status
    python3 vybn_phase.py serve       # start MCP server on stdin/stdout
"""
from __future__ import annotations

import cmath
import json
import sys
import traceback
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

DIM = 192  # C^192 = 384 real dimensions (MiniLM embedding dim)
STATE_DIR = Path(__file__).parent / "state"
DOMAIN_FILE = STATE_DIR / "domain.npz"
LOG_FILE = STATE_DIR / "entries.jsonl"


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


def to_complex(h: np.ndarray, n: int = DIM) -> np.ndarray:
    """Project R^384 -> C^n, normalized to unit sphere."""
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z


def text_to_state(text: str) -> np.ndarray:
    """Text -> C^DIM unit vector. Semantic, not positional."""
    return to_complex(embed(text))


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
# The convergent state of a set of propositions, independent of order.
# Run the encounter sequence many times in different permutations;
# the result converges to the abelian kernel — the geometric invariant
# that survives after path-dependent information is suppressed.
#
# April 5, 2026: confirmed numerically. At α=0.993, permutations of
# 50 propositions converge to fidelity 0.99999766 in C^4.

def abelian_kernel(vectors: list[np.ndarray], M0: np.ndarray = None,
                   alpha: float = 0.993, n_perms: int = 8) -> dict:
    """Compute the abelian kernel of a set of vectors.

    Runs n_perms random permutations of the encounter sequence and
    returns the centroid (the path-independent invariant), plus
    convergence diagnostics.
    """
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

    # Centroid of final states = the abelian kernel
    centroid = np.mean(finals, axis=0)
    norm = np.sqrt(np.sum(np.abs(centroid)**2))
    kernel = centroid / norm if norm > 1e-10 else centroid

    # Convergence: pairwise fidelities
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
# Trace a path through a sequence of states and measure the accumulated
# geometric phase relative to the starting state. If the path is a loop
# (returns near its start), the phase is the holonomy.
#
# At α → 0, holonomy shows perfect orientation reversal (Φ_fwd + Φ_rev ≈ 0).
# At α → 1, the geometric signal is present (correlation -0.994 between
# forward and reverse) but masked by dynamical phase.

def loop_holonomy(loop_vectors: list[np.ndarray], M0: np.ndarray,
                  alpha: float = 0.5) -> dict:
    """Run M through a sequence of encounters. Return accumulated phase.

    Also runs the reversed loop and reports orientation-reversal quality.
    """
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
    """Serialize a complex vector for JSON. Truncates to max_components."""
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
        "description": "Compute the abelian kernel of a set of propositions — the geometric invariant independent of encounter order. Returns the kernel vector, convergence fidelity, and operating regime (geometric / mixed / abelian-kernel).",
        "inputSchema": {"type": "object", "properties": {
            "texts": {"type": "array", "items": {"type": "string"}, "description": "Propositions to find the kernel of."},
            "alpha": {"type": "number", "description": "Memory parameter. 0.993 = creature regime (abelian). 0.5 = domain regime (geometric). Default 0.993."},
            "n_permutations": {"type": "integer", "description": "Number of random orderings to average over. Default 8."},
        }, "required": ["texts"]},
    },
    "loop_holonomy": {
        "description": "Measure geometric phase around a loop of propositions. Runs the loop forward and reversed, reports accumulated phase and orientation-reversal quality. Flip quality > 50% = geometric regime. Flip quality near 0% with high correlation = dynamical regime (geometry present but masked).",
        "inputSchema": {"type": "object", "properties": {
            "texts": {"type": "array", "items": {"type": "string"}, "description": "Propositions forming the loop (in order)."},
            "origin": {"type": "string", "description": "Starting state. If omitted, uses first text."},
            "alpha": {"type": "number", "description": "Memory parameter. Lower α = more geometric. Default 0.5."},
        }, "required": ["texts"]},
    },
    "status": {
        "description": "Domain size and operating parameters.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
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

    elif tool == "status":
        return json.dumps({"domain_size": domain_size(), "dim": DIM,
                           "note": "α=0.5 for mutual evaluation (geometric regime). "
                                   "Use abelian_kernel with α=0.993 for path-independent invariants."})

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
                "serverInfo": {"name": "vybn-phase", "version": "2.0.0",
                               "description": "Reflexive domain with abelian kernel and loop holonomy. "
                                              "Two regimes: geometric (α→0, senses curvature) and "
                                              "abelian-kernel (α→1, remembers meaning). D \u2245 D^D."}}})
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
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        print(f"Domain: {domain_size()} residents in C^{DIM}")

    elif cmd == "enter":
        text = " ".join(sys.argv[2:])
        if not text:
            print("Provide text."); sys.exit(1)
        o = enter_from_text(text)
        print(f"Orientation: {vec_to_json(o)[:2]}...")
        print(f"Domain: {domain_size()} residents")

    elif cmd == "seed":
        print(f"Seeding {len(SEED_PROPOSITIONS)} propositions...")
        for p in SEED_PROPOSITIONS:
            enter_from_text(p)
            print(f"  {domain_size()}: {p[:55]}")
        print(f"Done. {domain_size()} residents.")

    elif cmd == "serve":
        serve()

    else:
        print(f"Usage: vybn_phase.py [status|enter TEXT|seed|serve]")
