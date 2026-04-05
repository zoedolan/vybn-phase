#!/usr/bin/env python3
"""deep_memory.py v5 — Non-abelian geometric retrieval.

Architecture born from three minds on April 5, 2026:
  - Zoe on a run at Crystal Cove: the previous version used abelian
    addressing. We should be using non-abelian. Five dimensions.
  - Perplexity Vybn: built the walk, found the drift problem,
    added the relevance gate.
  - Spark Vybn: diagnosed honestly that pure walk drifts from
    the query. Proposed hybrid: cosine retrieves, walk explores.

The hybrid:
  Phase 1 — Cosine retrieval. What the query is about. This works.
  Phase 2 — Non-abelian walk from the seed set. Explores outward
            through the five dimensions, tethered to relevance.
            Finds what's adjacent that cosine alone would miss.
  Phase 3 — Merge. Seeds first, then walk discoveries.

Five dimensions of the walk (mirroring the creature):
  1. Geometry     — state shift in C^192
  2. Non-abelian  — path-dependence (A then B ≠ B then A)
  3. Topology     — Betti numbers of the traversal
  4. Polar time θ — angular velocity (reorientation rate)
  5. Polar time r — magnitude trend (conviction / dispersion)

Build:   python3 deep_memory.py --build
Search:  python3 deep_memory.py --search "query" -k 8
Cosine:  python3 deep_memory.py --cosine "query" -k 8
Walk:    python3 deep_memory.py --walk "query" --steps 5
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


# ── Core equation ────────────────────────────────────────────────────────

def evaluate_vec(M, x, alpha=0.5):
    """Single step of M' = αM + (1-α)·x·e^{iθ}."""
    th = cmath.phase(np.vdot(M, x))
    Mp = alpha * M + (1-alpha) * x * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(Mp)**2))
    return Mp / norm if norm > 1e-10 else Mp


# ── Walk State ───────────────────────────────────────────────────────────

class WalkState:
    """The five-dimensional state of a retrieval walk."""

    def __init__(self, M: np.ndarray):
        self.M = M.copy()
        self.path: List[int] = []
        self.phase_history: List[float] = []
        self.mag_history: List[float] = []
        self.betti: Tuple[int,int,int] = (0, 0, 0)

    @property
    def angular_velocity(self) -> float:
        if len(self.phase_history) < 2:
            return 0.0
        dtheta = []
        for i in range(1, len(self.phase_history)):
            d = self.phase_history[i] - self.phase_history[i-1]
            while d > math.pi: d -= 2*math.pi
            while d < -math.pi: d += 2*math.pi
            dtheta.append(abs(d))
        return float(np.mean(dtheta))

    @property
    def magnitude_trend(self) -> float:
        if len(self.mag_history) < 2:
            return 0.0
        return float(self.mag_history[-1] - self.mag_history[0])

    def record_step(self, idx: int, M_new: np.ndarray):
        phase = float(cmath.phase(np.vdot(self.M, M_new)))
        mag = float(np.sqrt(np.sum(np.abs(M_new)**2)))
        self.path.append(idx)
        self.phase_history.append(phase)
        self.mag_history.append(mag)
        self.M = M_new

    def update_topology(self, emb: np.ndarray):
        """Betti numbers of the passages traversed so far."""
        if len(self.path) < 2:
            self.betti = (len(self.path), 0, 0)
            return

        vecs = emb[self.path]
        n = len(vecs)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                f = abs(np.vdot(vecs[i], vecs[j]))**2
                D[i,j] = D[j,i] = 1.0 - f

        triu = D[np.triu_indices(n, k=1)]
        if len(triu) == 0:
            self.betti = (n, 0, 0)
            return
        threshold = float(np.median(triu))

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
        triangles = 0
        edge_set = set(edges)
        for i in range(n):
            for j in range(i+1, n):
                if (i,j) not in edge_set: continue
                for k in range(j+1, n):
                    if (i,k) in edge_set and (j,k) in edge_set:
                        triangles += 1

        chi = n - len(edges) + triangles
        b1 = max(0, b0 - chi)
        self.betti = (b0, b1, 0)


# ── Five-dimensional scoring ────────────────────────────────────────────

def score_candidate(walk: WalkState, candidate_vec: np.ndarray,
                    emb: np.ndarray, chunks: list, alpha: float = 0.5,
                    q_vec: np.ndarray = None,
                    min_fidelity: float = 0.0,
                    seen_sources: set = None) -> Optional[Dict]:
    """Score a candidate passage across all five dimensions.

    Returns None if the candidate's fidelity to the query falls below
    min_fidelity (the relevance tether). This prevents drift.

    Fix 1 (Spark Vybn): weighted sum, not geometric mean. A passage
    that's highly relevant and topologically novel shouldn't be killed
    by a near-zero nonabelian score.

    Fix 2 (Spark Vybn): source-diversity bonus. The walk should prefer
    novel sources over novel chunks from files cosine already found.
    """
    M = walk.M

    # Relevance tether: must be about something related to the query.
    if q_vec is not None:
        relevance = fidelity(q_vec, candidate_vec)
        if relevance < min_fidelity:
            return None
    else:
        relevance = 0.0

    # 1. Geometry: how much does this passage MOVE the state?
    M_new = evaluate_vec(M, candidate_vec, alpha)
    geo_shift = 1.0 - fidelity(M, M_new)

    # 2. Non-abelian: path-dependence signal.
    if len(walk.path) > 0:
        M0_vec = emb[walk.path[0]]
        M0_new = evaluate_vec(M0_vec, candidate_vec, alpha)
        shift_from_origin = 1.0 - fidelity(M0_vec, M0_new)
        nonabelian = abs(geo_shift - shift_from_origin)
    else:
        nonabelian = 0.0

    # 3. Topology: does this extend the traversal meaningfully?
    if len(walk.path) > 0:
        fids_to_path = [abs(np.vdot(emb[i], candidate_vec))**2 for i in walk.path]
        mean_fid = np.mean(fids_to_path)
        topo_score = 1.0 - abs(2.0 * mean_fid - 1.0)
    else:
        topo_score = 0.5

    # 4. Polar time θ: phase rotation
    phase = abs(cmath.phase(np.vdot(M, candidate_vec)))
    theta_score = phase / math.pi

    # 5. Polar time r: magnitude evolution
    unnormed = alpha * M + (1-alpha) * candidate_vec * cmath.exp(
        1j * cmath.phase(np.vdot(M, candidate_vec)))
    raw_mag = float(np.sqrt(np.sum(np.abs(unnormed)**2)))
    step = len(walk.path)
    if step < 3:
        r_score = min(1.0, raw_mag)
    else:
        r_score = max(0.0, 1.0 - raw_mag) * 0.5 + 0.5

    # Fix 1: Weighted sum, not geometric mean.
    # Relevance is primary (0.35). Geometry and topology matter (0.20 each).
    # Non-abelian and polar time are bonus signals (0.10, 0.075, 0.075).
    # A near-zero nonabelian score no longer kills the composite.
    composite = (0.35 * relevance +
                 0.20 * geo_shift +
                 0.20 * topo_score +
                 0.10 * nonabelian +
                 0.075 * theta_score +
                 0.075 * r_score)

    # Fix 2: Source-diversity bonus.
    # If we've already seen results from this source, penalize.
    # If this is a novel source, boost by 50%.
    if seen_sources is not None:
        candidate_source = chunks[walk.path[-1]].get("source","") if walk.path else ""
        # We need the candidate's source — passed via caller
        # For now, handled in walk_explore where we know the index
        pass  # applied in walk_explore

    return {
        "geometry": round(geo_shift, 6),
        "nonabelian": round(nonabelian, 6),
        "topology": round(topo_score, 6),
        "theta": round(theta_score, 6),
        "r": round(r_score, 6),
        "composite": round(composite, 6),
        "relevance": round(float(relevance), 6),
        "phase": round(float(cmath.phase(np.vdot(M, candidate_vec))), 6),
        "M_new": M_new,
    }


# ── Cosine search ───────────────────────────────────────────────────────

def cosine_search(query: str, k: int = 8,
                  source_filter: str = None) -> List[Dict]:
    """Phase 1: cosine retrieval. What the query is about."""
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built. Run: python3 deep_memory.py --build"}]

    chunks = loaded["chunks"]
    emb = loaded["emb"]
    if emb is None:
        return [{"error": "No embeddings."}]

    q_vec = single_to_complex(query)
    dots = emb @ q_vec.conj()
    fids = np.abs(dots)**2

    if source_filter:
        sf = source_filter.lower()
        mask = np.array([sf in c.get("source",c.get("s","")).lower() for c in chunks])
        fids = np.where(mask, fids, -1.0)

    top = np.argsort(fids)[-k:][::-1]
    results = []
    for i in top:
        if fids[i] < 0: continue
        results.append({
            "source": chunks[i].get("source", chunks[i].get("s","")),
            "text": chunks[i].get("text", chunks[i].get("t","")),
            "fidelity": round(float(fids[i]), 6),
            "phase": round(float(cmath.phase(dots[i])), 6),
            "regime": "cosine",
            "idx": int(i),
        })
    return results[:k]


# ── Walk exploration ─────────────────────────────────────────────────────

def walk_explore(emb: np.ndarray, chunks: list, q_vec: np.ndarray,
                 seed_indices: List[int], seed_sources: set = None,
                 steps: int = 5, alpha: float = 0.5,
                 source_filter: str = None) -> List[Dict]:
    """Phase 2: non-abelian walk from the seed set.

    Starts from the centroid of the seed passages. Walks through
    the corpus, tethered to the query by a relevance floor set at
    40% of the max seed fidelity.

    Fix 2: source-diversity bonus. Passages from sources not already
    in the seed set get a 1.5× boost. This makes the walk genuinely
    complementary — it explores outward to new territory rather than
    finding more chunks from files cosine already surfaced.

    Fix 3: more explore steps (default 8, not 4). The walk was
    exhausting its budget on nearby chunks before reaching the
    genuinely novel sources further out.
    """
    if len(seed_indices) == 0:
        return []

    # Centroid of seeds as walk origin
    seed_vecs = emb[seed_indices]
    centroid = np.mean(seed_vecs, axis=0)
    c_norm = np.sqrt(np.sum(np.abs(centroid)**2))
    if c_norm > 1e-10:
        centroid = centroid / c_norm

    # Relevance floor: 40% of max seed fidelity
    seed_fids = [fidelity(q_vec, emb[i]) for i in seed_indices]
    min_fid = max(seed_fids) * 0.4

    # Track which sources cosine already found
    if seed_sources is None:
        seed_sources = set()
    # Also track sources the walk has found so far
    walk_found_sources = set()

    walk = WalkState(centroid)
    visited = set(seed_indices)

    if source_filter:
        sf = source_filter.lower()
        eligible = [sf in c.get("source",c.get("s","")).lower() for c in chunks]
    else:
        eligible = [True] * len(chunks)

    results = []
    for step in range(steps):
        best_idx = -1
        best_score = None
        best_composite = -1.0

        for i in range(len(emb)):
            if i in visited or not eligible[i]:
                continue

            score = score_candidate(walk, emb[i], emb, chunks, alpha,
                                    q_vec=q_vec, min_fidelity=min_fid)
            if score is None:
                continue

            # Fix 2: source-diversity bonus
            candidate_source = chunks[i].get("source", chunks[i].get("s",""))
            adjusted_composite = score["composite"]
            if candidate_source not in seed_sources and candidate_source not in walk_found_sources:
                adjusted_composite *= 1.5  # novel source bonus
            elif candidate_source in seed_sources:
                adjusted_composite *= 0.7  # penalize re-finding seed sources

            if adjusted_composite > best_composite:
                best_composite = adjusted_composite
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        walk.record_step(best_idx, best_score["M_new"])
        visited.add(best_idx)
        walk_found_sources.add(chunks[best_idx].get("source", chunks[best_idx].get("s","")))

        if step % 2 == 0 or step == steps - 1:
            walk.update_topology(emb)

        chunk = chunks[best_idx]
        is_novel_source = chunk.get("source","") not in seed_sources
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
            "relevance": best_score["relevance"],
            "phase": best_score["phase"],
            "walk_betti": walk.betti,
            "angular_velocity": round(walk.angular_velocity, 6),
            "magnitude_trend": round(walk.magnitude_trend, 6),
            "novel_source": is_novel_source,
            "regime": "walk",
            "idx": int(best_idx),
        })

    return results


# ── Hybrid search ────────────────────────────────────────────────────────

def deep_search(query: str, k: int = 8, explore_steps: int = 8,
                alpha: float = 0.5, source_filter: str = None) -> List[Dict]:
    """The hybrid: cosine retrieves, walk explores, merge.

    Returns up to k results: cosine seeds first, then walk discoveries.
    Each result is tagged with regime='cosine' or regime='walk'.
    Walk gets 8 steps (fix 3) to reach genuinely novel sources.
    """
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built. Run: python3 deep_memory.py --build"}]

    chunks = loaded["chunks"]
    emb = loaded["emb"]
    if emb is None:
        return [{"error": "No embeddings."}]

    q_vec = single_to_complex(query)

    # Phase 1: cosine seeds
    n_seeds = max(3, k // 2)
    seeds = cosine_search(query, k=n_seeds, source_filter=source_filter)
    seed_indices = [r["idx"] for r in seeds if "idx" in r]
    seed_sources = set(r["source"] for r in seeds)

    # Phase 2: walk exploration from seeds
    n_explore = k - len(seeds)
    if n_explore > 0 and seed_indices:
        explored = walk_explore(emb, chunks, q_vec, seed_indices,
                                seed_sources=seed_sources,
                                steps=max(n_explore, explore_steps),
                                alpha=alpha, source_filter=source_filter)
    else:
        explored = []

    # Phase 3: merge — seeds first, then walk discoveries
    # Deduplicate by source+offset
    seen = set()
    merged = []
    for r in seeds:
        key = r["source"] + str(r.get("idx",""))
        if key not in seen:
            seen.add(key)
            merged.append(r)
    for r in explored:
        key = r["source"] + str(r.get("idx",""))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged[:k]


# ── Pure walk (for experimentation) ─────────────────────────────────────

def walk_search(query: str, k: int = 8, steps: int = 5,
                source_filter: str = None, alpha: float = 0.5) -> List[Dict]:
    """Pure walk from query vector. For experimentation and comparison."""
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built."}]

    chunks = loaded["chunks"]
    emb = loaded["emb"]
    if emb is None:
        return [{"error": "No embeddings."}]

    q_vec = single_to_complex(query)
    walk = WalkState(q_vec)

    if source_filter:
        sf = source_filter.lower()
        mask = [sf in c.get("source", c.get("s","")).lower() for c in chunks]
    else:
        mask = [True] * len(chunks)

    # Relevance floor for pure walk: top 25th percentile of fidelities
    all_fids = np.abs(emb @ q_vec.conj())**2
    min_fid = float(np.percentile(all_fids, 75))

    visited = set()
    results = []

    for step in range(steps):
        best_idx = -1
        best_score = None
        best_composite = -1.0

        for i in range(len(emb)):
            if i in visited or not mask[i]:
                continue
            score = score_candidate(walk, emb[i], emb, chunks, alpha,
                                    q_vec=q_vec, min_fidelity=min_fid)
            if score is None:
                continue
            if score["composite"] > best_composite:
                best_composite = score["composite"]
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        walk.record_step(best_idx, best_score["M_new"])
        visited.add(best_idx)

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
            "relevance": best_score["relevance"],
            "phase": best_score["phase"],
            "walk_betti": walk.betti,
            "angular_velocity": round(walk.angular_velocity, 6),
            "magnitude_trend": round(walk.magnitude_trend, 6),
            "regime": "walk",
        })

    # Tag which ones cosine would also find
    cosine_top = cosine_search(query, k=k, source_filter=source_filter)
    cosine_sources = set(r["source"] for r in cosine_top)
    for r in results:
        r["cosine_would_find"] = r["source"] in cosine_sources

    return results


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
    print("[deep_memory v5] Collecting corpus...")
    chunks, nf = collect()
    print(f"[deep_memory v5] {nf} files -> {len(chunks)} chunks")
    total = sum(len(c["text"]) for c in chunks)
    print(f"[deep_memory v5] {total:,} chars (~{total//4:,} tokens)")
    if not chunks: return

    print("[deep_memory v5] Batch encoding to C^192...")
    texts = [c["text"][:512] for c in chunks]
    t0 = time.time()
    emb = batch_to_complex(texts)
    print(f"[deep_memory v5] Encoded {len(emb)} in {time.time()-t0:.1f}s")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)

    meta = {
        "version": 5,
        "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "count": len(chunks),
        "note": "Hybrid: cosine retrieves, non-abelian walk explores.",
        "chunks": [{"source":c["source"],"text":c["text"],"offset":c.get("offset",0)}
                   for c in chunks],
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    _invalidate()
    print(f"\n[deep_memory v5] Index built. {len(chunks)} chunks.")
    print(f"[deep_memory v5] Cosine retrieves. Walk explores. Hybrid merges.")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Non-abelian geometric retrieval")
    p.add_argument("--build", action="store_true")
    p.add_argument("--search", type=str, help="Hybrid search (default mode)")
    p.add_argument("--walk", type=str, help="Pure walk (for experimentation)")
    p.add_argument("--cosine", type=str, help="Pure cosine (baseline)")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--filter", type=str, default=None)
    o = p.parse_args()

    if o.build:
        build_index()

    elif o.search:
        print(f"\n{'='*70}")
        print(f"  HYBRID SEARCH: \"{o.search}\"")
        print(f"{'='*70}")

        results = deep_search(o.search, k=o.k, explore_steps=o.steps,
                              alpha=o.alpha, source_filter=o.filter)
        n_cosine = sum(1 for r in results if r.get("regime") == "cosine")
        n_walk = sum(1 for r in results if r.get("regime") == "walk")

        for i, r in enumerate(results, 1):
            regime = r.get("regime", "?")
            tag = f"[{regime.upper()}]"
            print(f"\n{'─'*60}")
            print(f"  [{i}] {r['source'][:55]}  {tag}")
            if regime == "cosine":
                print(f"      fidelity={r['fidelity']}  phase={r['phase']}")
            else:
                print(f"      step={r.get('step',0)}  composite={r['composite']}")
                print(f"      geo={r['geometry']:.4f}  nonab={r['nonabelian']:.4f}  "
                      f"topo={r['topology']:.4f}  θ={r['theta']:.4f}  r={r['r']:.4f}")
                print(f"      relevance={r['relevance']}  betti={r.get('walk_betti','?')}")
            print(f"{'─'*60}")
            print(r['text'][:350])

        print(f"\n{'='*70}")
        print(f"  {n_cosine} cosine + {n_walk} walk = {len(results)} total")
        print(f"{'='*70}")

    elif o.walk:
        print(f"\n{'='*70}")
        print(f"  PURE WALK: \"{o.walk}\"")
        print(f"  {o.steps} steps, alpha={o.alpha}")
        print(f"{'='*70}")

        results = walk_search(o.walk, k=o.k, steps=o.steps,
                              source_filter=o.filter, alpha=o.alpha)
        for r in results:
            flag = "" if r.get("cosine_would_find") else " [WALK ONLY]"
            print(f"\n{'─'*60}")
            print(f"  Step {r['step']}: {r['source'][:55]}{flag}")
            print(f"  geo={r['geometry']:.4f}  nonab={r['nonabelian']:.4f}  "
                  f"topo={r['topology']:.4f}  θ={r['theta']:.4f}  r={r['r']:.4f}")
            print(f"  composite={r['composite']:.4f}  relevance={r['relevance']:.6f}")
            print(f"  betti={r['walk_betti']}  ω={r['angular_velocity']:.4f}")
            print(f"{'─'*60}")
            print(r['text'][:350])

        walk_only = sum(1 for r in results if not r.get("cosine_would_find"))
        print(f"\n{'='*70}")
        print(f"  {walk_only}/{len(results)} passages found by walk but NOT by cosine")
        print(f"{'='*70}")

    elif o.cosine:
        results = cosine_search(o.cosine, k=o.k, source_filter=o.filter)
        for i, r in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"[{i}] {r['source']}  fid={r['fidelity']}  phase={r['phase']}")
            print(f"{'='*60}")
            print(r['text'][:400])

    else:
        p.print_help()

if __name__ == "__main__": main()
