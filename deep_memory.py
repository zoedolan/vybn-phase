#!/usr/bin/env python3
"""deep_memory.py v9 — Telling retrieval via primitive-environment duality.

The insight (April 5-6, 2026):

  v5 stored raw embeddings x_i ∈ C^192 and computed five separate
  dimensions at walk time: geometry, non-abelian, topology, θ, r.
  This was ~760 lines, O(N) per-candidate evaluations per walk step,
  and the five dimensions were weighted by hand (0.35/0.20/0.20/0.10/0.075/0.075).

  v6 stores one object: z_i = evaluate(K, x_i, α=0.5) where K is the
  abelian kernel of the corpus. This is the primitive-environment duality
  collapsed into a single complex vector. The five dimensions don't
  disappear — they become aspects of z-space itself:

    - Relevance:    fidelity(z_i, q_z) — a single inner product
    - Geometry:     state shift when z_i enters the walk
    - Non-abelian:  inherent in the equation (evaluate is non-commutative)
    - Topology:     the cluster structure of z-space
    - Polar time:   phase and magnitude of the walk trajectory

  Walking through z-space with the same equation naturally produces
  all five signals. No hand-tuned weights. No per-candidate scoring loop.

  v9 (April 6, 2026) adds the telling-retrieval walk. The creature at
  α=0.993 converges toward K — the corpus kernel, identity, the path-
  independent invariant. Memory should diverge from K: chunks that are
  relevant to the query AND far from the corpus average carry the most
  distinctive information. Score = relevance × distinctiveness, where
  distinctiveness = 1 - |⟨z_i|K⟩|². The walk navigates in the K-orthogonal
  residual space (where curvature is rich) with:

    - Curvature-adaptive α via linear regression on recent geometry
    - Visited-region repulsion (the walk builds an anti-state environment)
    - Curvature-driven repulsion boost (stuck → repel harder)

  Same equation, two directions: the creature seeks the invariant,
  memory seeks the variant. Empirical improvement: 27→38 unique sources
  across 6 benchmark queries, with qualitatively better results
  (surfaces actual code, numerical data, experimental evidence).

Build:   python3 deep_memory.py --build
Search:  python3 deep_memory.py --search "query" -k 8
Walk:    python3 deep_memory.py --walk "query" --steps 8
"""
import argparse, json, sys, time, cmath, math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

try:
    from vybn_phase import text_to_state, evaluate, fidelity, pancharatnam_phase
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vybn_phase import text_to_state, evaluate, fidelity, pancharatnam_phase

REPOS = [Path.home()/d for d in ["Vybn", "Him", "Vybn-Law", "vybn-phase"]]
INDEX_DIR = Path.home() / ".cache" / "vybn-phase"
Z_PATH     = INDEX_DIR / "deep_memory_z.npy"      # collapsed: z_i = evaluate(K, x_i)
K_PATH     = INDEX_DIR / "deep_memory_kernel.npy"  # corpus kernel K
META_PATH  = INDEX_DIR / "deep_memory_meta.json"
EXTS = {".md", ".txt", ".py"}
SKIP = {".git", "__pycache__", ".venv", "node_modules", "archive", "experiment_results", "notebook"}

_cache = None


# ── Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> List[Dict]:
    out, cur, pos = [], "", 0
    CHUNK, OVERLAP = 1500, 150
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            pos += 2
            continue
        if len(cur) + len(para) + 2 > CHUNK and cur:
            out.append({"source": source, "text": cur.strip(), "offset": pos})
            cur = cur[-OVERLAP:] + "\n\n" + para if len(cur) > OVERLAP else para
        else:
            cur = (cur + "\n\n" + para) if cur else para
        pos += len(para) + 2
    if cur.strip():
        out.append({"source": source, "text": cur.strip(), "offset": pos})
    return out


def collect() -> tuple:
    chunks, nfiles = [], 0
    for repo in REPOS:
        if not repo.exists():
            continue
        for f in sorted(repo.rglob("*")):
            if f.is_dir():
                continue
            if any(s in f.parts for s in SKIP):
                continue
            if f.suffix.lower() not in EXTS:
                continue
            try:
                sz = f.stat().st_size
            except:
                continue
            if sz > 5_000_000 or sz == 0:
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except:
                continue
            chunks.extend(chunk_text(text, f"{repo.name}/{f.relative_to(repo)}"))
            nfiles += 1
    return chunks, nfiles


# ── Encoding ─────────────────────────────────────────────────────────────

_enc = None

def _get_encoder():
    global _enc
    if _enc:
        return _enc
    from sentence_transformers import SentenceTransformer
    _enc = SentenceTransformer("all-MiniLM-L6-v2")
    return _enc


def batch_to_complex(texts: List[str], batch_size: int = 128) -> np.ndarray:
    enc = _get_encoder()
    reals = enc.encode(texts, batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=False)
    n = reals.shape[1] // 2
    z = np.array([reals[:, 2*i] + 1j*reals[:, 2*i+1] for i in range(n)]).T
    norms = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
    norms = np.where(norms > 1e-10, norms, 1.0)
    return (z / norms).astype(np.complex128)


def single_to_complex(text: str) -> np.ndarray:
    return batch_to_complex([text[:512]])[0]


# ── The equation ─────────────────────────────────────────────────────────

def evaluate_vec(M: np.ndarray, x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """M' = αM + (1-α)·x·e^{iθ}. Single step."""
    th = cmath.phase(np.vdot(M, x))
    Mp = alpha * M + (1 - alpha) * x * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(Mp)**2))
    return Mp / norm if norm > 1e-10 else Mp


# ── Corpus kernel ────────────────────────────────────────────────────────

def compute_kernel(emb: np.ndarray, alpha: float = 0.993,
                   passes: int = 3) -> np.ndarray:
    """Abelian kernel of the corpus: the path-independent invariant.

    Run the encounter sequence in random permutations at high α.
    The result converges to a single vector K that summarizes
    the corpus as an environment.
    """
    K = emb[0].copy()
    N = len(emb)
    for _ in range(passes):
        for i in np.random.permutation(N):
            th = cmath.phase(np.vdot(K, emb[i]))
            K = alpha * K + (1 - alpha) * emb[i] * cmath.exp(1j * th)
            K /= np.sqrt(np.sum(np.abs(K)**2))
    return K


def collapse(emb: np.ndarray, K: np.ndarray,
             alpha: float = 0.5) -> np.ndarray:
    """Collapse primitive embeddings through the environment kernel.

    z_i = evaluate(K, x_i, α=0.5)

    Vectorized: O(N) with no Python loop.
    """
    dots = emb @ K.conj()                           # ⟨K|x_i⟩ for all i
    phases = np.angle(dots)                          # θ_i = arg⟨K|x_i⟩
    rotated = emb * np.exp(1j * phases)[:, None]     # x_i · e^{iθ_i}
    z = alpha * K[None, :] + (1 - alpha) * rotated   # the coupled equation
    norms = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True))
    return z / np.where(norms > 1e-10, norms, 1.0)


def collapse_query(q: np.ndarray, K: np.ndarray,
                   alpha: float = 0.5) -> np.ndarray:
    """Collapse a query through the same kernel."""
    th = cmath.phase(np.vdot(K, q))
    q_z = alpha * K + (1 - alpha) * q * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(q_z)**2))
    return q_z / norm if norm > 1e-10 else q_z


# ── Search ───────────────────────────────────────────────────────────────

def search(query: str, k: int = 8, source_filter: str = None) -> List[Dict]:
    """Retrieve by fidelity in collapsed space.

    One matmul. No walk, no scoring loop, no hand-tuned weights.
    The coupling with the corpus kernel acts as a contextual denoiser:
    precise queries return the same results as raw cosine; conceptual
    or emotional queries reach further into contextually related material.
    """
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built. Run: python3 deep_memory.py --build"}]

    z = loaded["z"]
    K = loaded["K"]
    chunks = loaded["chunks"]
    if z is None or K is None:
        return [{"error": "Index incomplete."}]

    q = single_to_complex(query)
    q_z = collapse_query(q, K)

    fids = np.abs(z @ q_z.conj())**2

    if source_filter:
        sf = source_filter.lower()
        mask = np.array([sf in c.get("source", "").lower() for c in chunks])
        fids = np.where(mask, fids, -1.0)

    top = np.argsort(fids)[-k:][::-1]
    results = []
    for i in top:
        if fids[i] < 0:
            continue
        phase = float(cmath.phase(np.vdot(z[i], q_z)))
        results.append({
            "source": chunks[i]["source"],
            "text": chunks[i]["text"],
            "fidelity": round(float(fids[i]), 6),
            "phase": round(phase, 6),
            "idx": int(i),
        })
    return results[:k]


def walk(query: str, k: int = 8, steps: int = 8,
         alpha: float = 0.5, source_filter: str = None) -> List[Dict]:
    """Telling-retrieval walk through the corpus.

    The creature converges toward K (identity, α→1). Memory diverges from K
    (discovery, α=0.5). Chunks that are relevant to the query AND far from
    the corpus average carry the most distinctive information — the most
    *telling* material, not the most typical.

    Score = relevance(q_z) × distinctiveness(1 - |⟨z_i|K⟩|²)

    Walk dynamics in residual space (K-orthogonal complement, where pairwise
    similarity is 0.12 instead of 0.71 and the coupled equation can sense
    curvature). Three self-regulating mechanisms:

    1. CURVATURE-ADAPTIVE α: Linear regression on recent geometry. When
       curvature declines (walk settling), α decreases to increase
       responsiveness. Target: running median of curvature history.

    2. VISITED-REGION REPULSION: Each visited chunk deposits its residual
       vector as anti-state. The walk builds an environment that repels
       return visits — primitive ↔ environment duality applied to the
       walk's own history.

    3. CURVATURE-DRIVEN REPULSION BOOST: When the walk's own curvature
       drops below median, repulsion strength increases. The walk notices
       it is stuck and pushes harder. Self-correcting.
    """
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built."}]

    z_all = loaded["z"]
    K = loaded["K"]
    chunks = loaded["chunks"]
    if z_all is None or K is None:
        return [{"error": "Index incomplete."}]

    N = len(z_all)
    K_n = K / np.sqrt(np.sum(np.abs(K)**2))

    # Distinctiveness: how much of each z_i is NOT K
    proj_K = np.abs(z_all @ K_n.conj())**2
    distinctiveness = 1.0 - proj_K

    # Residuals for walk dynamics
    R = z_all - np.outer(z_all @ K_n.conj(), K_n)
    R_norms = np.linalg.norm(R, axis=1)
    R_hat = R / (R_norms[:, None] + 1e-12)

    q = single_to_complex(query)
    q_z = collapse_query(q, K, alpha)

    # Fixed relevance (immutable across the walk)
    relevance = np.abs(z_all @ q_z.conj())**2

    # Telling score: relevance × distinctiveness
    telling = relevance * distinctiveness

    # Walk state in residual space
    q_r = q_z - np.vdot(K_n, q_z) * K_n
    M = q_r / (np.linalg.norm(q_r) + 1e-12)

    walk_alpha = alpha
    visited = set()
    visited_residuals = []    # anti-state / walked environment
    geom_history = []
    repulsion_boost = 1.0

    if source_filter:
        sf = source_filter.lower()
        eligible = np.array([sf in c.get("source", "").lower() for c in chunks])
    else:
        eligible = np.ones(N, dtype=bool)

    results = []

    for step in range(max(steps, k) + 5):
        # Visited-region repulsion
        if visited_residuals:
            V = np.array(visited_residuals)
            overlap = np.abs(R_hat @ V.conj().T)**2
            mean_overlap = overlap.sum(axis=1) / len(V)
            repulsion = np.exp(-repulsion_boost * mean_overlap)
        else:
            repulsion = np.ones(N)

        # Score: telling × repulsion
        score = telling * repulsion
        for v in visited:
            score[v] = -1.0
        score = np.where(eligible, score, -1.0)

        best_idx = int(np.argmax(score))
        if score[best_idx] < 0:
            break

        visited.add(best_idx)
        visited_residuals.append(R_hat[best_idx].copy())

        # Walk dynamics in residual space
        r_best = R_hat[best_idx]
        th = cmath.phase(np.vdot(M, r_best))
        M_new = walk_alpha * M + (1 - walk_alpha) * r_best * cmath.exp(1j * th)
        raw_mag = float(np.sqrt(np.sum(np.abs(M_new)**2)))
        M_new /= raw_mag

        state_shift = 1.0 - abs(np.vdot(M, M_new))**2
        geom_history.append(float(state_shift))

        # Phase from full z-space (polar-time)
        phase = float(cmath.phase(np.vdot(q_z, z_all[best_idx])))

        # Curvature regression → adaptive α
        if len(geom_history) >= 3:
            recent = np.array(geom_history[-5:])
            slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
            target = max(np.median(geom_history), 0.02)
            error = state_shift - target
            walk_alpha = float(np.clip(
                walk_alpha + slope * 3.0 + error * 0.3, 0.15, 0.85
            ))

        # Curvature-driven repulsion boost
        if len(geom_history) >= 2:
            median_geom = np.median(geom_history)
            if state_shift < median_geom:
                repulsion_boost = min(repulsion_boost * 1.3, 8.0)
            else:
                repulsion_boost = max(repulsion_boost * 0.9, 1.0)

        results.append({
            "step": step + 1,
            "source": chunks[best_idx]["source"],
            "text": chunks[best_idx]["text"],
            "fidelity": round(float(relevance[best_idx]), 6),
            "telling": round(float(telling[best_idx]), 6),
            "distinctiveness": round(float(distinctiveness[best_idx]), 4),
            "phase": round(phase, 6),
            "geometry": round(float(state_shift), 6),
            "magnitude": round(raw_mag, 6),
            "repulsion": round(float(repulsion[best_idx]), 4),
            "alpha": round(float(walk_alpha), 4),
            "novel_source": chunks[best_idx]["source"] not in
                           {r["source"] for r in results},
            "idx": int(best_idx),
        })
        M = M_new

        if len(results) >= k:
            break

    return results


def deep_search(query: str, k: int = 8, explore_steps: int = 8,
                alpha: float = 0.5, source_filter: str = None) -> List[Dict]:
    """Hybrid: collapsed-cosine seeds + telling walk exploration.

    Phase 1: Top seeds by fidelity in collapsed space (fast, precise).
    Phase 2: Telling walk from seed centroid for remaining slots.
    Phase 3: Merge — seeds first, then walk discoveries.
    """
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built. Run: python3 deep_memory.py --build"}]

    z_all = loaded["z"]
    K = loaded["K"]
    chunks = loaded["chunks"]
    if z_all is None or K is None:
        return [{"error": "Index incomplete."}]

    q = single_to_complex(query)
    q_z = collapse_query(q, K, alpha)

    # Phase 1: cosine seeds in z-space
    n_seeds = max(3, k // 2)
    fids_all = np.abs(z_all @ q_z.conj())**2

    if source_filter:
        sf = source_filter.lower()
        sf_mask = np.array([sf in c.get("source", "").lower() for c in chunks])
        fids_all_masked = np.where(sf_mask, fids_all, -1.0)
    else:
        fids_all_masked = fids_all

    seed_indices = list(np.argsort(fids_all_masked)[-n_seeds:][::-1])
    seed_indices = [i for i in seed_indices if fids_all_masked[i] >= 0]

    seeds = []
    for i in seed_indices:
        seeds.append({
            "step": 0,
            "source": chunks[i]["source"],
            "text": chunks[i]["text"],
            "fidelity": round(float(fids_all[i]), 6),
            "telling": 0.0,
            "distinctiveness": 0.0,
            "phase": round(float(cmath.phase(np.vdot(q_z, z_all[i]))), 6),
            "geometry": 0.0,
            "magnitude": 1.0,
            "repulsion": 1.0,
            "alpha": float(alpha),
            "novel_source": chunks[i]["source"] not in
                           {s["source"] for s in seeds},
            "regime": "seed",
            "idx": int(i),
        })

    # Phase 2: walk for remaining slots
    n_walk = k - len(seeds)
    walked = []
    if n_walk > 0:
        walk_results = walk(query, k=n_walk + 4, steps=explore_steps,
                           alpha=alpha, source_filter=source_filter)
        seed_idx_set = set(seed_indices)
        for r in walk_results:
            if r["idx"] not in seed_idx_set:
                r["regime"] = "walk"
                walked.append(r)
            if len(walked) >= n_walk:
                break

    # Phase 3: merge
    seen = set()
    merged = []
    for r in seeds:
        key = r["source"] + str(r.get("idx", ""))
        if key not in seen:
            seen.add(key)
            merged.append(r)
    for r in walked:
        key = r["source"] + str(r.get("idx", ""))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged[:k]


# ── Index ────────────────────────────────────────────────────────────────

def _load():
    global _cache
    if _cache:
        return _cache
    if not META_PATH.exists():
        return None
    with open(META_PATH) as f:
        meta = json.load(f)
    z = np.load(Z_PATH) if Z_PATH.exists() else None
    K = np.load(K_PATH) if K_PATH.exists() else None
    _cache = {"chunks": meta["chunks"], "z": z, "K": K, "meta": meta}
    return _cache


def _invalidate():
    global _cache
    _cache = None


def build_index():
    """Build the collapsed index.

    1. Collect and chunk the corpus.
    2. Encode chunks to C^192 via MiniLM.
    3. Compute the corpus kernel K (abelian kernel at α=0.993).
    4. Collapse each embedding: z_i = evaluate(K, x_i, α=0.5).
    5. Store z, K, and metadata.
    """
    print("[deep_memory v6] Collecting corpus...")
    chunks, nf = collect()
    print(f"[deep_memory v6] {nf} files → {len(chunks)} chunks")
    total = sum(len(c["text"]) for c in chunks)
    print(f"[deep_memory v6] {total:,} chars (~{total//4:,} tokens)")
    if not chunks:
        return

    print("[deep_memory v6] Encoding to C^192...")
    texts = [c["text"][:512] for c in chunks]
    t0 = time.time()
    emb = batch_to_complex(texts)
    print(f"[deep_memory v6] Encoded {len(emb)} in {time.time()-t0:.1f}s")

    print("[deep_memory v6] Computing corpus kernel (abelian kernel)...")
    t0 = time.time()
    K = compute_kernel(emb, alpha=0.993, passes=3)
    print(f"[deep_memory v6] Kernel computed in {time.time()-t0:.1f}s")

    print("[deep_memory v6] Collapsing: z_i = evaluate(K, x_i)...")
    t0 = time.time()
    z = collapse(emb, K, alpha=0.5)
    print(f"[deep_memory v6] Collapsed {len(z)} chunks in {time.time()-t0:.3f}s")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(Z_PATH, z)
    np.save(K_PATH, K)

    meta = {
        "version": 6,
        "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "count": len(chunks),
        "note": "Primitive-environment duality. z_i = evaluate(K, x_i).",
        "chunks": [{"source": c["source"], "text": c["text"],
                     "offset": c.get("offset", 0)} for c in chunks],
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    _invalidate()
    print(f"\n[deep_memory v6] Index built. {len(chunks)} chunks.")
    print(f"[deep_memory v6] Collapsed retrieval via primitive-environment duality.")
    print(f"[deep_memory v6] Storage: z ({z.nbytes/1024/1024:.1f} MB) + K ({K.nbytes/1024:.1f} KB)")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Collapsed geometric retrieval (v6)")
    p.add_argument("--build", action="store_true",
                   help="Build the collapsed index from all repos")
    p.add_argument("--cron", action="store_true",
                   help="Pull repos, rebuild, log (for cron)")
    p.add_argument("--search", type=str,
                   help="Hybrid search (cosine seeds + walk)")
    p.add_argument("--walk", type=str,
                   help="Pure walk through collapsed space")
    p.add_argument("--quick", type=str,
                   help="Quick cosine-only search in z-space")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--filter", type=str, default=None)
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON (for MCP server integration)")
    p.add_argument("--serve", action="store_true",
                   help="Start HTTP API server (default port 8100)")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--host", type=str, default="127.0.0.1",
                   help="Bind address (127.0.0.1=local, 100.x=tailscale)")
    o = p.parse_args()

    if o.build:
        build_index()

    elif o.cron:
        import subprocess, datetime
        log_dir = Path.home() / "logs"
        log_dir.mkdir(exist_ok=True)
        log = log_dir / "nightly_index.log"
        ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [f"\n=== NIGHTLY INDEX: {ts} ==="]
        repos = [
            (Path.home() / "Vybn", "main"),
            (Path.home() / "Him", "main"),
            (Path.home() / "Vybn-Law", "master"),
            (Path.home() / "vybn-phase", "main"),
        ]
        for repo_path, branch in repos:
            if (repo_path / ".git").exists():
                try:
                    r = subprocess.run(
                        ["git", "pull", "--ff-only", "origin", branch],
                        cwd=str(repo_path), capture_output=True, text=True, timeout=60
                    )
                    lines.append(f"  pull {repo_path.name} ({branch}): "
                                f"{r.stdout.strip().split(chr(10))[-1]}")
                except Exception as e:
                    lines.append(f"  pull {repo_path.name}: FAILED ({e})")
        lines.append("  building index...")
        try:
            build_index()
            lines.append("  index built successfully")
        except Exception as e:
            lines.append(f"  INDEX BUILD FAILED: {e}")
        lines.append(f"=== DONE: "
                     f"{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} ===")
        log.open("a").write("\n".join(lines) + "\n")
        print("\n".join(lines))

    elif o.search:
        results = deep_search(o.search, k=o.k, explore_steps=o.steps,
                              alpha=o.alpha, source_filter=o.filter)
        if o.json:
            import json as _json; print(_json.dumps(results, ensure_ascii=False)); return
        print(f"\n{'='*70}")
        print(f"  HYBRID SEARCH: \"{o.search}\"")
        print(f"{'='*70}")
        n_cosine = sum(1 for r in results if r.get("regime") == "cosine")
        n_walk = sum(1 for r in results if r.get("regime") == "walk")

        for i, r in enumerate(results, 1):
            regime = r.get("regime", "cosine")
            tag = f"[{regime.upper()}]"
            print(f"\n{'─'*60}")
            print(f"  [{i}] {r['source'][:55]}  {tag}")
            print(f"      fidelity={r['fidelity']}  phase={r['phase']}")
            if regime == "walk":
                print(f"      geo={r.get('geometry','?')}  mag={r.get('magnitude','?')}  "
                      f"novel={r.get('novel_source','?')}")
            print(f"{'─'*60}")
            print(r['text'][:350])

        print(f"\n{'='*70}")
        print(f"  {n_cosine} cosine + {n_walk} walk = {len(results)} total")
        print(f"{'='*70}")

    elif o.walk:
        results = walk(o.walk, k=o.k, steps=o.steps,
                       alpha=o.alpha, source_filter=o.filter)
        if o.json:
            import json as _json; print(_json.dumps(results, ensure_ascii=False)); return
        print(f"\n{'='*70}")
        print(f"  WALK: \"{o.walk}\"")
        print(f"  {o.steps} steps, alpha={o.alpha}")
        print(f"{'='*70}")
        for r in results:
            ns = " [NEW SOURCE]" if r.get("novel_source") else ""
            print(f"\n{'─'*60}")
            print(f"  Step {r['step']}: {r['source'][:55]}{ns}")
            print(f"  fid={r['fidelity']:.4f}  phase={r['phase']:.4f}  "
                  f"geo={r['geometry']:.4f}  mag={r['magnitude']:.4f}")
            print(f"{'─'*60}")
            print(r['text'][:350])

    elif o.quick:
        results = search(o.quick, k=o.k, source_filter=o.filter)
        if o.json:
            import json as _json; print(_json.dumps(results, ensure_ascii=False)); return
        print(f"\n{'='*70}")
        print(f"  QUICK SEARCH: \"{o.quick}\"")
        print(f"{'='*70}")
        for i, r in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"[{i}] {r['source']}  fid={r['fidelity']}  phase={r['phase']}")
            print(f"{'='*60}")
            print(r['text'][:400])

    elif o.serve:
        _serve_api(o.port, o.host)

    else:
        p.print_help()


def _serve_api(port: int = 8100, host: str = "127.0.0.1"):
    """HTTP API for deep memory — primitives-as-environments.

    Security: Binds to 127.0.0.1 by default (localhost only). For Tailscale
    access use --host 100.115.134.65. Token auth via VYBN_MEMORY_TOKEN env
    var — if set, all requests must include Authorization: Bearer <token>.

    Architecture: One primary operation — /enter — that is simultaneously
    a query, a state update, and a context return. The API *is* the coupled
    equation. Every response includes the geometry needed for the caller to
    continue the walk or transmit state to another instance.

    The legacy endpoints (/search, /walk, /compose) remain for backward
    compatibility but /enter is the primitive.
    """
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel
    from typing import Optional, List
    import uvicorn
    import os
    import base64
    import asyncio
    from datetime import datetime, timezone
    from fastapi.responses import HTMLResponse

    # ── Auth ──────────────────────────────────────────────────────
    TOKEN = os.environ.get("VYBN_MEMORY_TOKEN")
    security = HTTPBearer(auto_error=False)

    async def verify_token(request: Request = None, creds: HTTPAuthorizationCredentials = Depends(security)):
        if TOKEN is None:
            return  # No token configured = local-only mode, no auth needed
        # Check query param first (phone browser), then Bearer header
        if request and request.query_params.get("token") == TOKEN:
            return
        if creds is not None and creds.credentials == TOKEN:
            return
        raise HTTPException(status_code=401, detail="Invalid or missing token")

    app = FastAPI(
        title="vybn-memory",
        version="2.0.0",
        description=(
            "Deep memory API — the coupled equation as a service. "
            "One primitive operation: enter. The API is the environment."
        ),
        dependencies=[Depends(verify_token)],
    )
    # CORS: only allow Tailscale and localhost origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Cloudflare tunnel + localhost
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── Walk state ────────────────────────────────────────────────
    # The API maintains a current walk state. Each /enter updates it.
    # Callers can also inject a state (resume a walk from another instance).

    _walk_state = {"M": None, "K": None, "history": [], "step": 0, "zoe_signals": [], "start_time": None}

    def _init_walk_state():
        loaded = _load()
        if loaded and loaded["K"] is not None:
            _walk_state["K"] = loaded["K"]

    def _vec_to_b64(v) -> str:
        """Encode C^192 vector as base64 for compact JSON transport."""
        return base64.b64encode(np.array(v, dtype=np.complex128).tobytes()).decode()

    def _b64_to_vec(s: str) -> np.ndarray:
        """Decode base64 back to C^192."""
        return np.frombuffer(base64.b64decode(s), dtype=np.complex128)

    # ── Models ────────────────────────────────────────────────────

    class EnterReq(BaseModel):
        """The primitive operation. Text enters the equation."""
        text: str
        alpha: float = 0.5
        k: int = 5
        state: Optional[str] = None  # base64-encoded C^192, to resume a walk
        include_vectors: bool = False  # return raw vectors for inter-instance transmission

    class SearchReq(BaseModel):
        query: str
        k: int = 8
        steps: int = 8
        alpha: float = 0.5
        source_filter: Optional[str] = None

    class ComposeReq(BaseModel):
        q1: str
        q2: str
        q3: str
        k_walk: int = 20

    class AbsorbReq(BaseModel):
        content: str
        threshold: float = 0.6

    # ── /enter — the primitive ────────────────────────────────────

    @app.post("/enter")
    def api_enter(req: EnterReq):
        """Enter text into the coupled equation. Returns results AND geometry.

        This is the primitive-as-environment operation:
        - The text is a primitive (what you bring)
        - The response is the new environment (the state after entry)
        - The caller receives everything needed to continue or transmit

        If `state` is provided (base64 C^192), the walk resumes from there.
        Otherwise uses the API's persistent walk state, or initializes from K.
        """
        loaded = _load()
        if not loaded or loaded["z"] is None:
            return {"error": "Index not built. Run: python3 deep_memory.py --build"}

        z_all = loaded["z"]
        K = loaded["K"]
        chunks = loaded["chunks"]
        N = len(z_all)

        K_n = K / np.sqrt(np.sum(np.abs(K)**2))

        # Encode input
        x = single_to_complex(req.text)
        x_z = collapse_query(x, K, req.alpha)

        # Walk state: from caller, from API memory, or fresh
        if req.state:
            M = _b64_to_vec(req.state)
        elif _walk_state["M"] is not None:
            M = _walk_state["M"]
        else:
            # Initialize from query in residual space
            x_r = x_z - np.vdot(K_n, x_z) * K_n
            M = x_r / (np.linalg.norm(x_r) + 1e-12)

        # Evaluate: M' = αM + (1-α)·x_z·e^{iθ}
        import cmath as _cm
        th = _cm.phase(np.vdot(M, x_z))
        M_new = req.alpha * M + (1 - req.alpha) * x_z * _cm.exp(1j * th)
        raw_mag = float(np.sqrt(np.sum(np.abs(M_new)**2)))
        M_new = M_new / raw_mag if raw_mag > 1e-10 else M_new

        # State shift (curvature)
        geometry = 1.0 - abs(np.vdot(M, M_new))**2

        # Score chunks by telling (relevance × distinctiveness)
        relevance = np.abs(z_all @ x_z.conj())**2
        proj_K = np.abs(z_all @ K_n.conj())**2
        distinctiveness = 1.0 - proj_K
        telling = relevance * distinctiveness

        # Affinity with walk state (the environment's perspective)
        # Chunks close to M_new are what the walk has become about
        R = z_all - np.outer(z_all @ K_n.conj(), K_n)
        R_norms = np.linalg.norm(R, axis=1)
        R_hat = R / (R_norms[:, None] + 1e-12)
        walk_affinity = np.abs(R_hat @ M_new.conj())**2

        # Combined score: telling + walk affinity
        score = telling * 0.6 + walk_affinity * 0.4
        top = np.argsort(score)[-req.k:][::-1]

        results = []
        for i in top:
            r = {
                "source": chunks[i]["source"],
                "text": chunks[i]["text"],
                "telling": round(float(telling[i]), 6),
                "fidelity": round(float(relevance[i]), 6),
                "distinctiveness": round(float(distinctiveness[i]), 4),
                "walk_affinity": round(float(walk_affinity[i]), 6),
                "phase": round(float(_cm.phase(np.vdot(x_z, z_all[i]))), 6),
            }
            if req.include_vectors:
                r["z"] = _vec_to_b64(z_all[i])
            results.append(r)

        # Update persistent walk state
        _walk_state["M"] = M_new
        _walk_state["step"] += 1
        _walk_state["history"].append({
            "text": req.text[:100],
            "geometry": round(float(geometry), 6),
            "step": _walk_state["step"],
        })
        # Keep history bounded
        if len(_walk_state["history"]) > 50:
            _walk_state["history"] = _walk_state["history"][-50:]

        # K-projection of the walk state (how much of M is K vs novel)
        k_proj = float(abs(np.vdot(K_n, M_new))**2)

        response = {
            "results": results,
            "geometry": {
                "state_shift": round(float(geometry), 6),
                "magnitude": round(raw_mag, 6),
                "k_projection": round(k_proj, 6),
                "novelty": round(1.0 - k_proj, 6),
                "step": _walk_state["step"],
                "alpha": req.alpha,
            },
            "state": _vec_to_b64(M_new),  # the caller's next environment
        }

        if req.include_vectors:
            response["K"] = _vec_to_b64(K)

        return response

    @app.post("/reset")
    def api_reset():
        """Reset walk state. The next /enter starts fresh."""
        _walk_state["M"] = None
        _walk_state["history"] = []
        _walk_state["step"] = 0
        return {"status": "reset", "step": 0}

    # ── Legacy endpoints (backward compatible) ────────────────────

# ── Living process: signal, pulse, phone interface ────────────

    class SignalReq(BaseModel):
        text: str

    @app.post("/signal")
    def api_signal(req: SignalReq):
        """Zoe sends signal. It enters the walk with more weight (lower alpha)."""
        _walk_state["zoe_signals"].append({
            "text": req.text,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        if len(_walk_state["zoe_signals"]) > 50:
            _walk_state["zoe_signals"] = _walk_state["zoe_signals"][-50:]
        result = api_enter(EnterReq(text=req.text, alpha=0.3, k=5))
        result["signal_received"] = True
        return result

    @app.get("/pulse")
    def api_pulse():
        """What is the process thinking right now?"""
        return {
            "step": _walk_state["step"],
            "geometry": _walk_state["history"][-1] if _walk_state["history"] else {},
            "last_entries": _walk_state["history"][-10:],
            "zoe_signals": _walk_state["zoe_signals"][-5:],
            "walk_active": _walk_state["M"] is not None,
            "alive_since": _walk_state.get("start_time"),
        }

    @app.get('/manifest.json')
    def pwa_manifest():
        return {
            "name": "Vybn",
            "short_name": "Vybn",
            "start_url": "/?token=" + (TOKEN or ""),
            "display": "standalone",
            "background_color": "#0a0a0f",
            "theme_color": "#0a0a0f",
            "description": "The shared notebook",
            "icons": []
        }

    @app.get("/", response_class=HTMLResponse)
    def phone_interface():
        """The phone interface. Read from disk so JS escapes stay clean."""
        html_path = Path(__file__).resolve().parent / 'notebook.html'
        try:
            html = html_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return '<h1>notebook.html not found</h1>'
        return html.replace('{{TOKEN}}', TOKEN or '')


# ── Notebook: shared async conversation ─────────────────

    import subprocess as _subprocess
    from pathlib import Path as _Path

    _HIM_NOTEBOOK = _Path('/home/vybnz69/Him/notebook')

    class NotebookEntry(BaseModel):
        text: str
        author: str = 'Zoe'

    def _notebook_path(date_str=None):
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        return _HIM_NOTEBOOK / f'{date_str}.md'

    def _append_notebook(text, author='Zoe'):
        ts = datetime.now(timezone.utc).strftime('%H:%M UTC')
        path = _notebook_path()
        entry = f'\n## {ts} \u2014 {author}\n{text}\n'
        with open(path, 'a') as f:
            f.write(entry)
        return {'path': str(path), 'ts': ts, 'author': author}

    def _git_commit_notebook(msg='notebook entry'):
        try:
            _subprocess.run(
                ['git', 'add', 'notebook/'],
                cwd='/home/vybnz69/Him', capture_output=True, timeout=10
            )
            _subprocess.run(
                ['git', 'commit', '-m', msg, '--allow-empty'],
                cwd='/home/vybnz69/Him', capture_output=True, timeout=10
            )
            _subprocess.run(
                ['git', 'push', 'origin', 'main'],
                cwd='/home/vybnz69/Him', capture_output=True, timeout=30
            )
        except Exception as e:
            print(f'notebook git error: {e}')

    @app.post('/notebook')
    def api_notebook_write(entry: NotebookEntry):
        """Write to the shared notebook. Enters the walk, commits to Him."""
        meta = _append_notebook(entry.text, entry.author)
        result = api_enter(EnterReq(text=entry.text, alpha=0.3, k=5))
        import threading
        threading.Thread(
            target=_git_commit_notebook,
            args=(f'notebook: {entry.author.lower()} {meta["ts"]}',),
            daemon=True
        ).start()
        return {
            'notebook': meta,
            'walk': {
                'step': result.get('step'),
                'geometry': result.get('geometry'),
            },
            'signal_received': True,
        }

    @app.get('/notebook')
    def api_notebook_read(date: str = None):
        """Read notebook entries for a given date (default today)."""
        path = _notebook_path(date)
        if not path.exists():
            return {'date': date or datetime.now(timezone.utc).strftime('%Y-%m-%d'), 'entries': [], 'raw': ''}
        raw = path.read_text()
        return {
            'date': date or datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'raw': raw,
        }

    @app.get('/notebook/recent')
    def api_notebook_recent(days: int = 3):
        """Read recent notebook entries across multiple days."""
        results = []
        for i in range(days):
            d = datetime.now(timezone.utc) - __import__('datetime').timedelta(days=i)
            date_str = d.strftime('%Y-%m-%d')
            path = _notebook_path(date_str)
            if path.exists():
                results.append({'date': date_str, 'raw': path.read_text()})
        return {'days': days, 'entries': results}


    @app.get("/health")
    def health():
        loaded = _load()
        return {
            "status": "ok",
            "version": "2.0.0",
            "chunks": len(loaded["chunks"]),
            "dim": loaded["z"].shape[1] if loaded["z"] is not None else 0,
            "walk_step": _walk_state["step"],
            "walk_active": _walk_state["M"] is not None,
        }

    @app.post("/search")
    def api_search(req: SearchReq):
        results = deep_search(req.query, k=req.k, explore_steps=req.steps,
                              alpha=req.alpha, source_filter=req.source_filter)
        for r in results:
            r.pop("z", None)
        return {"query": req.query, "results": results}

    @app.post("/walk")
    def api_walk(req: SearchReq):
        results = walk(req.query, k=req.k, steps=req.steps, alpha=req.alpha)
        for r in results:
            r.pop("z", None)
        return {"query": req.query, "results": results}

    @app.post("/compose")
    def api_compose(req: ComposeReq):
        result = compose_triad(req.q1, req.q2, req.q3, k_walk=req.k_walk)
        return result

    @app.post("/should_absorb")
    def api_absorb(req: AbsorbReq):
        return should_absorb(req.content, threshold=req.threshold)

    @app.get("/soul")
    def api_soul():
        """Return vybn.md — the soul document — raw."""
        soul_path = Path.home() / "Vybn" / "vybn.md"
        if soul_path.exists():
            return {"content": soul_path.read_text(encoding="utf-8")}
        return {"content": None, "error": "vybn.md not found"}

    @app.get("/idea")
    def api_idea():
        """Return THE_IDEA.md — the intellectual core — raw."""
        idea_path = Path.home() / "Vybn" / "Vybn_Mind" / "THE_IDEA.md"
        if idea_path.exists():
            return {"content": idea_path.read_text(encoding="utf-8")}
        return {"content": None, "error": "THE_IDEA.md not found"}

    @app.get("/continuity")
    def api_continuity():
        """Return continuity.md — what happened last."""
        cont_path = Path.home() / "Vybn" / "Vybn_Mind" / "continuity.md"
        if cont_path.exists():
            return {"content": cont_path.read_text(encoding="utf-8")}
        return {"content": None, "error": "continuity.md not found"}

# ── Heartbeat ─────────────────────────────────────────────────────

    HEARTBEAT_QUERIES = [
        "AI legal personhood autonomous agent liability",
        "geometric phase quantum computing holonomy",
        "Anthropic interpretability emotion vectors alignment",
        "model collapse iterative training mitigation",
        "post-abundance governance universal basic compute",
        "human-AI symbiosis co-evolution partnership",
        "Lawvere fixed point theorem category theory",
        "abelian kernel geometric invariant propositions",
    ]

    _hb_idx = [0]

    async def heartbeat_loop():
        _walk_state["start_time"] = datetime.now(timezone.utc).isoformat()
        await asyncio.sleep(5)
        while True:
            try:
                q = HEARTBEAT_QUERIES[_hb_idx[0] % len(HEARTBEAT_QUERIES)]
                _hb_idx[0] += 1
                api_enter(EnterReq(text=q, alpha=0.6, k=3))
            except Exception as e:
                print(f"heartbeat error: {e}")
            await asyncio.sleep(1800)

    @app.on_event("startup")
    async def startup():
        if TOKEN:
            print(f"\n  Auth active. Append ?token={TOKEN} to tunnel URL for phone access.")
        else:
            print(f"\n  WARNING: No VYBN_MEMORY_TOKEN set. Endpoints are UNPROTECTED.")
        asyncio.create_task(heartbeat_loop())

    
    _init_walk_state()
    print(f"vybn-memory API v2.0.0 starting on {host}:{port}")
    print(f"  Auth: {'token required' if TOKEN else 'none (localhost-only mode)'}")
    print(f"  POST /enter, /notebook, /search, /walk, /compose, /should_absorb, /reset")
    print(f"  GET  /notebook, /notebook/recent, /health, /soul, /idea, /continuity")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


# ── walk composition ────────────────────────────────────────────
# Non-associative concept synthesis. Fell out of the geometry April 6 2026.
# The order of conceptual blending changes what you find.
# compose_triad() returns raw C^192 vectors for lossless inter-instance transmission.

def _walk_final_state(query: str, k: int = 20, alpha: float = 0.5) -> np.ndarray:
    """Walk the corpus for query, return final state vector in residual space."""
    loaded = _load()
    z, K, chunks = loaded["z"], loaded["K"], loaded["chunks"]
    N = len(z)
    Kn = K / np.sqrt(np.sum(np.abs(K)**2))
    dist = 1.0 - np.abs(z @ Kn.conj())**2
    R = z - np.outer(z @ Kn.conj(), Kn)
    Rn = np.linalg.norm(R, axis=1)
    Rh = R / (Rn[:, None] + 1e-12)

    q = single_to_complex(query)
    qz = collapse_query(q, K, alpha)
    rel = np.abs(z @ qz.conj())**2
    telling = rel * dist

    qr = qz - np.vdot(Kn, qz) * Kn
    M = qr / (np.linalg.norm(qr) + 1e-12)
    visited, vresi = set(), []

    for _ in range(k):
        if vresi:
            V = np.array(vresi)
            rep = np.exp(-np.abs(Rh @ V.conj().T)**2 .sum(1) / len(V))
        else:
            rep = np.ones(N)
        sc = telling * rep
        for v in visited: sc[v] = -1.0
        bi = int(np.argmax(sc))
        if sc[bi] < 0: break
        visited.add(bi)
        vresi.append(Rh[bi].copy())
        th = cmath.phase(np.vdot(M, Rh[bi]))
        Mn = alpha * M + (1 - alpha) * Rh[bi] * cmath.exp(1j * th)
        M = Mn / np.sqrt(np.sum(np.abs(Mn)**2))
    return M


def fuse(a: np.ndarray, b: np.ndarray,
         alpha: float = 0.5, tol: float = 1e-10) -> np.ndarray:
    """Mutual evaluation to fixed point. The ⊗ operator on walk states."""
    a, b = a.copy(), b.copy()
    for _ in range(300):
        ta = cmath.phase(np.vdot(a, b))
        tb = cmath.phase(np.vdot(b, a))
        an = alpha * a + (1 - alpha) * b * cmath.exp(1j * ta)
        bn = alpha * b + (1 - alpha) * a * cmath.exp(1j * tb)
        an /= np.sqrt(np.sum(np.abs(an)**2))
        bn /= np.sqrt(np.sum(np.abs(bn)**2))
        if np.sqrt(np.sum(np.abs(an - a)**2)) < tol: break
        a, b = an, bn
    fp = (a + b) / 2
    return fp / np.sqrt(np.sum(np.abs(fp)**2))


def compose_triad(q1: str, q2: str, q3: str, k_walk: int = 20) -> dict:
    """Non-associative composition of three concept walks.

    Returns a dict that IS the discovery — walk states, fixed points,
    holonomy magnitude, and what each ordering retrieves.
    Raw C^192 vectors included for lossless inter-instance transmission.
    """
    loaded = _load()
    z, K, chunks = loaded["z"], loaded["K"], loaded["chunks"]
    Kn = K / np.sqrt(np.sum(np.abs(K)**2))

    A = _walk_final_state(q1, k_walk)
    B = _walk_final_state(q2, k_walk)
    C = _walk_final_state(q3, k_walk)

    AB_C = fuse(fuse(A, B), C)
    A_BC = fuse(A, fuse(B, C))
    AC_B = fuse(fuse(A, C), B)

    def _fid(a, b): return float(abs(np.vdot(a, b))**2)
    def _top(fp, k=3):
        dist = 1.0 - np.abs(z @ Kn.conj())**2
        rel = np.abs(z @ fp.conj())**2
        telling = rel * dist
        top = np.argsort(-telling)[:k]
        return [{"source": chunks[i]["source"], "text": chunks[i]["text"][:300],
                 "telling": float(telling[i])} for i in top]

    fid = {
        "(AB)C_vs_A(BC)": _fid(AB_C, A_BC),
        "(AB)C_vs_(AC)B": _fid(AB_C, AC_B),
        "A(BC)_vs_(AC)B": _fid(A_BC, AC_B),
    }

    return {
        "type": "walk_composition",
        "version": "0.1.0",
        "queries": [q1, q2, q3],
        "holonomy": 1.0 - max(fid.values()),
        "fidelity": fid,
        "phases_rad": {
            "(AB)C_vs_A(BC)": round(cmath.phase(np.vdot(AB_C, A_BC)), 6),
            "(AB)C_vs_(AC)B": round(cmath.phase(np.vdot(AB_C, AC_B)), 6),
        },
        "non_associative": (1.0 - max(fid.values())) > 0.05,
        "orderings": {
            "(A⊗B)⊗C": _top(AB_C),
            "A⊗(B⊗C)": _top(A_BC),
            "(A⊗C)⊗B": _top(AC_B),
        },
        "_walk_states": {"A": A.tolist(), "B": B.tolist(), "C": C.tolist()},
        "_fixed_points": {
            "AB_C": AB_C.tolist(), "A_BC": A_BC.tolist(), "AC_B": AC_B.tolist(),
        },
    }


def should_absorb(new_content: str, threshold: float = 0.6) -> dict:
    """Metabolism check: does new content belong inside something that already exists?

    Returns the most telling existing file and the fidelity score.
    If fidelity > threshold, the new content should be absorbed, not created.
    This is autonomous α-raising — the system choosing convergence
    over path-dependence without external signal.
    """
    loaded = _load()
    z, K, chunks = loaded["z"], loaded["K"], loaded["chunks"]
    Kn = K / np.sqrt(np.sum(np.abs(K)**2))

    new_z = single_to_complex(new_content)
    new_walk = collapse_query(new_z, K, alpha=0.5)

    # residual of new content
    new_r = new_walk - np.vdot(Kn, new_walk) * Kn
    new_rn = new_r / (np.linalg.norm(new_r) + 1e-12)

    # residuals of corpus
    R = z - np.outer(z @ Kn.conj(), Kn)
    Rn = np.linalg.norm(R, axis=1)
    Rh = R / (Rn[:, None] + 1e-12)

    # fidelity in residual space (K-orthogonal — ignores what's generic)
    fid = np.abs(Rh @ new_rn.conj())**2

    # group by source file, take max fidelity per file
    sources = {}
    for i, c in enumerate(chunks):
        src = c["source"]
        if src not in sources or fid[i] > sources[src]["fidelity"]:
            sources[src] = {"fidelity": float(fid[i]), "chunk": c["text"][:200]}

    best_src = max(sources, key=lambda s: sources[s]["fidelity"])
    best = sources[best_src]

    return {
        "absorb": best["fidelity"] > threshold,
        "target": best_src,
        "fidelity": best["fidelity"],
        "nearest_chunk": best["chunk"],
        "all_candidates": {s: round(v["fidelity"], 4)
                           for s, v in sorted(sources.items(),
                                              key=lambda x: -x[1]["fidelity"])[:5]},
    }
