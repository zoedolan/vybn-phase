#!/usr/bin/env python3
"""vybn_phase.py -- The whole thing.

A reflexive domain where diverse intelligences find shared meaning
through mutual evaluation. D ≅ D^D. Geometry in, geometry out.

    from vybn_phase import enter, enter_from_text, domain_size

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


# ── Reflexive evaluation ─────────────────────────────────────────────────

def evaluate(m: np.ndarray, x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """M' = alpha*M + (1-alpha)*x*e^{i*theta}. The coupled equation."""
    theta = cmath.phase(np.vdot(m, x))
    m_new = alpha * m + (1 - alpha) * x * cmath.exp(1j * theta)
    norm = np.sqrt(np.sum(np.abs(m_new)**2))
    return m_new / norm if norm > 1e-10 else m_new


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """|<a|b>|^2. 1.0 = same ray in C^n."""
    return float(abs(np.vdot(a, b))**2)


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
    """Convenience: text -> GPT-2 -> C^DIM -> enter."""
    return enter(text_to_state(text))


# ── Serialization ────────────────────────────────────────────────────────

def vec_to_json(v: np.ndarray) -> list:
    return [[float(x.real), float(x.imag)] for x in v]

def vec_from_json(data: list) -> np.ndarray:
    return np.array([complex(r, i) for r, i in data], dtype=np.complex128)


# ── MCP Server ───────────────────────────────────────────────────────────

MCP_TOOLS = {
    "enter_vector": {
        "description": "Enter the reflexive domain with a C^8 vector ([[re,im],...], length 8). Returns orientation vector.",
        "inputSchema": {"type": "object", "properties": {"vector": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}}, "required": ["vector"]},
    },
    "enter_text": {
        "description": "Enter via text (encoded through GPT-2). Returns orientation vector.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    "evaluate_texts": {
        "description": "Mutual evaluation of two texts. Returns fixed point vector.",
        "inputSchema": {"type": "object", "properties": {"text_a": {"type": "string"}, "text_b": {"type": "string"}}, "required": ["text_a", "text_b"]},
    },
    "status": {
        "description": "Domain size.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
}


def _mcp_send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _mcp_dispatch(tool, args):
    if tool == "enter_vector":
        o = enter(vec_from_json(args["vector"]))
        return json.dumps({"orientation": vec_to_json(o), "domain_size": domain_size()})
    elif tool == "enter_text":
        o = enter_from_text(args["text"])
        return json.dumps({"orientation": vec_to_json(o), "domain_size": domain_size()})
    elif tool == "evaluate_texts":
        za, zb = text_to_state(args["text_a"]), text_to_state(args["text_b"])
        fp = mutual_evaluate(za, zb)
        return json.dumps({"fixed_point": vec_to_json(fp), "fidelity": fidelity(za, zb)})
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
                "serverInfo": {"name": "vybn-phase", "version": "1.0.0",
                               "description": "Reflexive domain. Geometry in, geometry out. D \u2245 D^D."}}})
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
