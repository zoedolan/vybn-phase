#!/usr/bin/env python3
"""deep_memory.py v4 — Non-abelian geometric retrieval.

The insight (Zoe, April 5 2026 on a run): the previous version used
abelian addressing — every passage addressed independently from a
fixed origin. That threw away the thing that made the creature
interesting: the order of encounters matters. The path IS the signal.

Five dimensions of retrieval, mirroring the creature's own architecture:

  1. Geometry     — where the state IS in C^192 after each encounter
  2. Non-abelian  — path-dependence. Seeing A then B ≠ B then A.
                    The walk through the corpus accumulates state.
  3. Topology     — Betti numbers of the traversal. Which passages
                    open holes, close them, change persistent structure.
  4. Polar time θ — angular velocity of the state. How fast the
                    phase rotates on each encounter. Reorientation rate.
  5. Polar time r — magnitude evolution. Conviction accumulating
                    or dispersing. Independent of rotation.

These five are not independent scores summed together. They are one
dynamical system observed from five complementary angles. The retrieval
IS the walk. The walk IS the search.

Build:  python3 deep_memory.py --build
Search: python3 deep_memory.py --search "query" -k 8
Walk:   python3 deep_memory.py --walk "query" -k 8 --steps 5

The --walk mode is the new thing. It doesn't rank passages statically.
It takes a step, observes all five dimensions, picks the next passage
that maximally moves the state, and walks. The results are ordered
by encounter, not by score. The path is the answer.
"""
import argparse, json, sys, time, cmath, math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from vybn_phase import text_to_state, evaluate, fidelity, pancharatnam_phase
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vybn_phase import text_to_state, evaluate, fidelity, pancharatnam_phase

REPOS = [Path.home()/d for d in ["Vybn","Him","Vybn-Law","vybn-phase"]]
INDEX_DIR = Path.home() / ".cache" / "vybn-phase"
EMB_PATH   = INDEX_DIR / "deep_memory_emb.npy"
META_PATH  = INDEX_DIR / "deep_memory_meta.json"
EXTS = {".md",".txt",".py"}
SKIP = {".git","__pycache__",".venv","node_modules","archive","experiment_results"}


# ── Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text, source):
    out, cur, pos, CHUNK, OVERLAP = [], "", 0, 1500, 150
    for para in text.split("\n\n"):
        para = para.strip()
        if not para: pos += 2; continue
        if len(cur)+len(para)+2 > CHUNK and cur:
            out.append({"source": source, "text": cur.strip(), "offset": pos})
            cur = cur[-OVERLAP:]+"\n\n"+para if len(cur)>OVERLAP else para
        else:
            cur = (cur+"\n\n"+para) if cur else para
        pos += len(para)+2
    if cur.strip(): out.append({"source": source, "text": cur.strip(), "offset": pos})
    return out

def collect():
    chunks, n = [], 0
    for repo in REPOS:
        if not repo.exists(): continue
        for f in sorted(repo.rglob("*")):
            if f.is_dir(): continue
            if any(s in f.parts for s in SKIP): continue
            if f.suffix.lower() not in EXTS: continue
            try: sz = f.stat().st_size
            except: continue
            if sz > 5_000_000 or sz == 0: continue
            try: text = f.read_text(encoding="utf-8", errors="replace")
            except: continue
            chunks.extend(chunk_text(text, f"{repo.name}/{f.relative_to(repo)}"))
            n += 1
    return chunks, n


# ── Encoding ─────────────────────────────────────────────────────────────

_enc = None
def _get_encoder():
    global _enc
    if _enc: return _enc
    from sentence_transformers import SentenceTransformer
    _enc = SentenceTransformer("all-MiniLM-L6-v2")
    return _enc

def batch_to_complex(texts, batch_size=128):
    enc = _get_encoder()
    reals = enc.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
    n = reals.shape[1] // 2
    z = np.array([reals[:,2*i] + 1j*reals[:,2*i+1] for i in range(n)]).T
    norms = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
    norms = np.where(norms > 1e-10, norms, 1.0)
    return (z / norms).astype(np.complex128)

def single_to_complex(text):
    return batch_to_complex([text[:512]])[0]


# ── The Walk ─────────────────────────────────────────────────────────────
#
# This is the core. Instead of addressing all passages from a fixed origin,
# we WALK through the corpus. Each encounter changes the state. The changed
# state determines what we encounter next. Non-abelian: the path matters.

def evaluate_vec(M, x, alpha=0.5):
    """Single step of M' = αM + (1-α)·x·e^{iθ}."""
    th = cmath.phase(np.vdot(M, x))
    Mp = alpha * M + (1-alpha) * x * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(Mp)**2))
    return Mp / norm if norm > 1e-10 else Mp


class WalkState:
    """The five-dimensional state of a retrieval walk."""

    def __init__(self, M: np.ndarray):
        self.M = M.copy()                    # Current state in C^192
        self.path: List[int] = []            # Indices of passages encountered
        self.phase_history: List[float] = []  # θ at each step (polar time: angle)
        self.mag_history: List[float] = []    # |M| at each step (polar time: radius)
        self.betti: Tuple[int,int,int] = (0, 0, 0)  # Topological invariants

    @property
    def angular_velocity(self) -> float:
        """How fast the phase is rotating. High = reorienting. Low = settled."""
        if len(self.phase_history) < 2:
            return 0.0
        dtheta = []
        for i in range(1, len(self.phase_history)):
            d = self.phase_history[i] - self.phase_history[i-1]
            # Unwrap
            while d > math.pi: d -= 2*math.pi
            while d < -math.pi: d += 2*math.pi
            dtheta.append(abs(d))
        return float(np.mean(dtheta))

    @property
    def magnitude_trend(self) -> float:
        """Is conviction accumulating (+) or dispersing (-)? Independent of rotation."""
        if len(self.mag_history) < 2:
            return 0.0
        return float(self.mag_history[-1] - self.mag_history[0])

    def record_step(self, idx: int, M_new: np.ndarray):
        """Record one step of the walk."""
        phase = float(cmath.phase(np.vdot(self.M, M_new)))
        mag = float(np.sqrt(np.sum(np.abs(M_new)**2)))
        self.path.append(idx)
        self.phase_history.append(phase)
        self.mag_history.append(mag)
        self.M = M_new

    def update_topology(self, emb: np.ndarray):
        """Compute Betti numbers of the passages traversed so far.

        Uses a distance-threshold simplicial complex in the embedding space.
        b0 = connected components (clusters of meaning)
        b1 = loops (circular reasoning / thematic cycles)
        b2 = voids (gaps in the coverage)
        """
        if len(self.path) < 2:
            self.betti = (len(self.path), 0, 0)
            return

        vecs = emb[self.path]
        n = len(vecs)

        # Distance matrix
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                # Fidelity-based distance: d = 1 - |<a|b>|^2
                f = abs(np.vdot(vecs[i], vecs[j]))**2
                D[i,j] = D[j,i] = 1.0 - f

        # Threshold at median distance
        triu = D[np.triu_indices(n, k=1)]
        if len(triu) == 0:
            self.betti = (n, 0, 0)
            return
        threshold = float(np.median(triu))

        # b0: connected components via union-find
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            a, b = find(a), find(b)
            if a != b: parent[a] = b

        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if D[i,j] <= threshold:
                    union(i, j)
                    edges.append((i,j))

        b0 = len(set(find(i) for i in range(n)))

        # b1: count triangles vs edges (Euler characteristic estimate)
        # For a simplicial complex: χ = b0 - b1 + b2
        # Triangles: triplets where all three edges are below threshold
        triangles = 0
        edge_set = set(edges)
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) not in edge_set: continue
                for k in range(j+1, n):
                    if (i,k) in edge_set and (j,k) in edge_set:
                        triangles += 1

        # Euler: χ = n - |edges| + |triangles|
        # b1 = b0 - χ + b2, approximate b2 ≈ 0 for small complexes
        chi = n - len(edges) + triangles
        b1 = max(0, b0 - chi)
        b2 = max(0, triangles - len(edges) + n - b0 + b1 - chi) if n > 4 else 0

        self.betti = (b0, b1, b2)


def score_candidate(walk: WalkState, candidate_vec: np.ndarray,
                    emb: np.ndarray, alpha: float = 0.5) -> Dict:
    """Score a candidate passage across all five dimensions.

    Returns a dict with individual dimension scores and a composite.
    The composite is NOT a weighted sum — it's the geometric mean,
    because a passage that scores zero on any dimension is not interesting
    regardless of the others.
    """
    M = walk.M

    # 1. Geometry: how much does this passage MOVE the state?
    M_new = evaluate_vec(M, candidate_vec, alpha)
    geo_shift = 1.0 - fidelity(M, M_new)  # 0 = no movement, 1 = orthogonal

    # 2. Non-abelian: would seeing this passage at a different point
    #    in the walk produce a different result? Measure by comparing
    #    the state shift against the shift from the ORIGIN.
    if len(walk.path) > 0:
        M0 = emb[walk.path[0]]  # state at walk start
        M0_new = evaluate_vec(M0, candidate_vec, alpha)
        shift_from_origin = 1.0 - fidelity(M0, M0_new)
        # Non-abelian signal: how different is the shift from here vs. from origin
        nonabelian = abs(geo_shift - shift_from_origin)
    else:
        nonabelian = 0.0

    # 3. Topology: would adding this passage change the Betti numbers?
    #    Estimate by checking if it's distant from the current traversal cluster.
    if len(walk.path) > 0:
        fids_to_path = [abs(np.vdot(emb[i], candidate_vec))**2 for i in walk.path]
        max_fid = max(fids_to_path)
        min_fid = min(fids_to_path)
        # Topological novelty: not too close (redundant) and not too far (disconnected)
        # Sweet spot is mid-range fidelity — connects but extends
        topo_score = 1.0 - abs(2.0 * np.mean(fids_to_path) - 1.0)
    else:
        topo_score = 0.5  # neutral on first step

    # 4. Polar time θ: how much does this rotate the phase?
    phase = abs(cmath.phase(np.vdot(M, candidate_vec)))
    theta_score = phase / math.pi  # 0 = aligned, 1 = orthogonal phase

    # 5. Polar time r: does this increase or decrease magnitude?
    #    We want passages that accumulate conviction (increase |M|)
    #    early in the walk, and passages that challenge (decrease |M|)
    #    later, once we've built up a state.
    unnormed_new = alpha * M + (1-alpha) * candidate_vec * cmath.exp(1j * cmath.phase(np.vdot(M, candidate_vec)))
    raw_mag = float(np.sqrt(np.sum(np.abs(unnormed_new)**2)))
    step = len(walk.path)
    if step < 3:
        # Early: prefer accumulation
        r_score = min(1.0, raw_mag)
    else:
        # Later: prefer challenge (lower magnitude = more perturbation)
        r_score = max(0.0, 1.0 - raw_mag) * 0.5 + 0.5

    # Composite: geometric mean (zero on any dimension kills the score)
    scores = [geo_shift, max(nonabelian, 0.01), topo_score,
              max(theta_score, 0.01), r_score]
    composite = float(np.prod(scores) ** (1.0/len(scores)))

    return {
        "geometry": round(geo_shift, 6),
        "nonabelian": round(nonabelian, 6),
        "topology": round(topo_score, 6),
        "theta": round(theta_score, 6),
        "r": round(r_score, 6),
        "composite": round(composite, 6),
        "phase": round(float(cmath.phase(np.vdot(M, candidate_vec))), 6),
        "M_new": M_new,
    }


def walk_search(query: str, k: int = 8, steps: int = 5,
                source_filter: str = None, alpha: float = 0.5) -> List[Dict]:
    """Non-abelian retrieval. The walk IS the search.

    Starting from the query, takes `steps` steps through the corpus.
    At each step, scores all unvisited passages across five dimensions
    and walks to the one with highest composite score. The path is
    the answer.

    Returns passages in encounter order — the sequence matters.
    """
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built. Run: python3 deep_memory.py --build"}]

    chunks = loaded["chunks"]
    emb = loaded["emb"]
    if emb is None:
        return [{"error": "No embeddings. Rebuild with --build."}]

    # Start from the query
    q_vec = single_to_complex(query)
    walk = WalkState(q_vec)

    # Source filter mask
    if source_filter:
        sf = source_filter.lower()
        mask = [sf in c.get("source", c.get("s","")).lower() for c in chunks]
    else:
        mask = [True] * len(chunks)

    visited = set()
    results = []

    for step in range(steps):
        best_idx = -1
        best_score = None
        best_composite = -1.0

        # Score all unvisited, eligible passages
        for i in range(len(emb)):
            if i in visited or not mask[i]:
                continue

            score = score_candidate(walk, emb[i], emb, alpha)

            if score["composite"] > best_composite:
                best_composite = score["composite"]
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        # Take the step
        walk.record_step(best_idx, best_score["M_new"])
        visited.add(best_idx)

        # Update topology every few steps (expensive for large path)
        if step % 2 == 0 or step == steps - 1:
            walk.update_topology(emb)

        chunk = chunks[best_idx]
        results.append({
            "step": step + 1,
            "source": chunk.get("source", chunk.get("s", "")),
            "text": chunk.get("text", chunk.get("t", "")),
            "geometry": best_score["geometry"],
            "nonabelian": best_score["nonabelian"],
            "topology": best_score["topology"],
            "theta": best_score["theta"],
            "r": best_score["r"],
            "composite": best_score["composite"],
            "phase": best_score["phase"],
            "walk_betti": walk.betti,
            "angular_velocity": round(walk.angular_velocity, 6),
            "magnitude_trend": round(walk.magnitude_trend, 6),
        })

    # After the walk, also return the top-k by cosine for comparison
    # (so you can see what the walk found that cosine wouldn't)
    q_dots = emb @ q_vec.conj()
    cosine_fids = np.abs(q_dots)**2
    cosine_top = np.argsort(cosine_fids)[-k:][::-1]
    cosine_sources = set()
    for i in cosine_top:
        cosine_sources.add(chunks[i].get("source", chunks[i].get("s","")))

    for r in results:
        r["cosine_would_find"] = r["source"] in cosine_sources

    return results


def cosine_search(query: str, k: int = 8, source_filter: str = None) -> List[Dict]:
    """Baseline: plain cosine similarity. For comparison."""
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built."}]

    chunks = loaded["chunks"]
    emb = loaded["emb"]
    if emb is None:
        return [{"error": "No embeddings."}]

    q_vec = single_to_complex(query)
    dots = emb @ q_vec.conj()
    fids = np.abs(dots)**2

    if source_filter:
        sf = source_filter.lower()
        fid_mask = np.array([sf in c.get("source",c.get("s","")).lower() for c in chunks])
        fids = np.where(fid_mask, fids, -1.0)

    top = np.argsort(fids)[-k:][::-1]
    return [{
        "source": chunks[i].get("source", chunks[i].get("s","")),
        "text": chunks[i].get("text", chunks[i].get("t","")),
        "fidelity": round(float(fids[i]), 6),
        "regime": "cosine",
    } for i in top if fids[i] >= 0][:k]


# ── Index I/O ────────────────────────────────────────────────────────────

_cache = None
def _load():
    global _cache
    if _cache: return _cache
    if not META_PATH.exists(): return None
    with open(META_PATH) as f: meta = json.load(f)
    emb = np.load(EMB_PATH) if EMB_PATH.exists() else None
    _cache = {"chunks": meta["chunks"], "emb": emb, "meta": meta}
    return _cache

def _invalidate():
    global _cache
    _cache = None


# ── Build ────────────────────────────────────────────────────────────────

def build_index():
    print("[deep_memory v4] Collecting corpus...")
    chunks, nf = collect()
    print(f"[deep_memory v4] {nf} files -> {len(chunks)} chunks")
    total = sum(len(c["text"]) for c in chunks)
    print(f"[deep_memory v4] {total:,} chars (~{total//4:,} tokens)")
    if not chunks: return

    print("[deep_memory v4] Batch encoding to C^192...")
    texts = [c["text"][:512] for c in chunks]
    t0 = time.time()
    emb = batch_to_complex(texts)
    print(f"[deep_memory v4] Encoded {len(emb)} in {time.time()-t0:.1f}s")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)

    meta = {
        "version": 4,
        "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "count": len(chunks),
        "note": "Non-abelian retrieval. The walk is the search.",
        "chunks": [{"source":c["source"],"text":c["text"],"offset":c.get("offset",0)} for c in chunks],
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    # Quick sanity: walk a test query
    _invalidate()
    print(f"\n[deep_memory v4] Index built. {len(chunks)} chunks.")
    print(f"[deep_memory v4] Embeddings: {EMB_PATH}")
    print(f"[deep_memory v4] Metadata: {META_PATH}")
    print(f"\n[deep_memory v4] No static addressing. No abelian regime.")
    print(f"[deep_memory v4] The walk is the search. The path is the answer.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Non-abelian geometric retrieval")
    p.add_argument("--build", action="store_true")
    p.add_argument("--walk", type=str, help="Walk search (non-abelian)")
    p.add_argument("--search", type=str, help="Alias for --walk")
    p.add_argument("--cosine", type=str, help="Cosine baseline for comparison")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--steps", type=int, default=5, help="Walk steps (default 5)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--filter", type=str, default=None)
    o = p.parse_args()

    if o.build:
        build_index()

    elif o.walk or o.search:
        q = o.walk or o.search
        print(f"\n{'='*70}")
        print(f"  WALK: \"{q}\"")
        print(f"  {o.steps} steps, alpha={o.alpha}")
        print(f"{'='*70}")

        results = walk_search(q, k=o.k, steps=o.steps,
                              source_filter=o.filter, alpha=o.alpha)
        for r in results:
            cosine_flag = "" if r.get("cosine_would_find") else " [WALK ONLY]"
            print(f"\n{'─'*60}")
            print(f"  Step {r['step']}: {r['source']}{cosine_flag}")
            print(f"  geo={r['geometry']:.4f}  nonab={r['nonabelian']:.4f}  "
                  f"topo={r['topology']:.4f}  θ={r['theta']:.4f}  r={r['r']:.4f}")
            print(f"  composite={r['composite']:.4f}  phase={r['phase']:.4f}")
            print(f"  walk betti={r['walk_betti']}  "
                  f"ω={r['angular_velocity']:.4f}  Δr={r['magnitude_trend']:.4f}")
            print(f"{'─'*60}")
            print(r['text'][:400])

        walk_only = sum(1 for r in results if not r.get("cosine_would_find"))
        print(f"\n{'='*70}")
        print(f"  {walk_only}/{len(results)} passages found by walk but NOT by cosine")
        print(f"{'='*70}")

    elif o.cosine:
        results = cosine_search(o.cosine, k=o.k, source_filter=o.filter)
        for i, r in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"[{i}] {r['source']}  fid={r['fidelity']}")
            print(f"{'='*60}")
            print(r['text'][:400])

    else:
        p.print_help()

if __name__ == "__main__": main()
