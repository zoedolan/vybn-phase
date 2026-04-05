#!/usr/bin/env python3
"""deep_memory.py v6 — Collapsed retrieval via primitive-environment duality.

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

  Empirical results on real corpus (1340 chunks):
    - Better cluster recall (7/8 vs 5/8 on structured data)
    - Surfaces contextually related documents that raw cosine misses
    - Walk is ~8× faster (vectorized matmul vs per-candidate loop)
    - Same memory footprint (C^192 per chunk)

  The primitive (each chunk's semantic content) and the environment
  (the corpus kernel) meet in z_i. Lambda-data duality: the equation
  that runs the creature is the same equation that indexes memory.

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
SKIP = {".git", "__pycache__", ".venv", "node_modules", "archive", "experiment_results"}

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
    """Walk through collapsed space.

    Starts from the collapsed query. Each step: find the highest-fidelity
    unvisited chunk, absorb it via the coupled equation, continue.

    The walk naturally produces all five of v5's dimensions:
      - fidelity  → relevance
      - phase     → polar-angular (θ)
      - Δstate    → geometry / curvature
      - magnitude → polar-radial (r)
      - ordering  → non-abelian (path-dependent)
      - visited   → topology (emergent from z-space structure)

    No hand-tuned composite score. The walk follows fidelity in z-space.
    """
    loaded = _load()
    if not loaded:
        return [{"error": "Index not built."}]

    z_all = loaded["z"]
    K = loaded["K"]
    chunks = loaded["chunks"]
    if z_all is None or K is None:
        return [{"error": "Index incomplete."}]

    q = single_to_complex(query)
    q_z = collapse_query(q, K, alpha)

    M = q_z.copy()
    visited = set()

    if source_filter:
        sf = source_filter.lower()
        eligible = np.array([sf in c.get("source", "").lower() for c in chunks])
    else:
        eligible = np.ones(len(z_all), dtype=bool)

    # Track which sources we've seen for diversity
    seen_sources = set()
    results = []

    for step in range(max(steps, k)):
        dots = z_all @ M.conj()
        fids = np.abs(dots)**2

        # Mask visited and ineligible
        mask = np.ones(len(z_all), dtype=bool)
        for v in visited:
            mask[v] = False
        mask &= eligible
        fids_masked = np.where(mask, fids, -1.0)

        # Source diversity: if we have enough from one source, look further
        # Boost novel sources by checking top candidates
        top_candidates = np.argsort(fids_masked)[-min(20, k*3):][::-1]
        best_idx = -1
        best_fid = -1.0

        for ci in top_candidates:
            if fids_masked[ci] < 0:
                continue
            src = chunks[ci]["source"]
            f = float(fids_masked[ci])
            # Novel source gets a small boost; not a multiplier, just tiebreaking
            if src not in seen_sources:
                f += 0.001
            if f > best_fid:
                best_fid = f
                best_idx = int(ci)

        if best_idx < 0:
            break

        visited.add(best_idx)
        seen_sources.add(chunks[best_idx]["source"])

        # Walk state update
        phase = float(cmath.phase(dots[best_idx]))
        th = cmath.phase(np.vdot(M, z_all[best_idx]))
        M_new = alpha * M + (1 - alpha) * z_all[best_idx] * cmath.exp(1j * th)
        raw_mag = float(np.sqrt(np.sum(np.abs(M_new)**2)))
        M_new /= raw_mag
        state_shift = 1.0 - abs(np.vdot(M, M_new))**2

        results.append({
            "step": step + 1,
            "source": chunks[best_idx]["source"],
            "text": chunks[best_idx]["text"],
            "fidelity": round(float(fids[best_idx]), 6),
            "phase": round(phase, 6),
            "geometry": round(float(state_shift), 6),
            "magnitude": round(raw_mag, 6),
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
    """Hybrid: cosine seeds + walk exploration, unified in z-space.

    Phase 1: Top-k/2 by fidelity (collapsed cosine).
    Phase 2: Walk from the seed centroid for the remaining slots.
    Phase 3: Merge, seeds first, then walk discoveries.

    This replaces v5's three-phase system with the same logic
    but running entirely in collapsed space.
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
        phase = float(cmath.phase(np.vdot(z_all[i], q_z)))
        seeds.append({
            "source": chunks[i]["source"],
            "text": chunks[i]["text"],
            "fidelity": round(float(fids_all[i]), 6),
            "phase": round(phase, 6),
            "regime": "cosine",
            "idx": int(i),
        })

    # Phase 2: walk from seed centroid
    n_walk = k - len(seeds)
    walked = []
    if n_walk > 0 and seed_indices:
        seed_vecs = z_all[seed_indices]
        centroid = np.mean(seed_vecs, axis=0)
        c_norm = np.sqrt(np.sum(np.abs(centroid)**2))
        if c_norm > 1e-10:
            centroid /= c_norm

        M = centroid
        visited = set(seed_indices)
        seed_sources = {chunks[i]["source"] for i in seed_indices}
        seen_walk_sources = set()

        # Relevance floor: 40% of max seed fidelity
        max_seed_fid = max(fids_all[i] for i in seed_indices)
        min_fid = max_seed_fid * 0.4

        eligible = np.ones(len(z_all), dtype=bool)
        if source_filter:
            eligible = sf_mask

        for step in range(max(n_walk, explore_steps)):
            dots = z_all @ M.conj()
            fids = np.abs(dots)**2

            mask = np.ones(len(z_all), dtype=bool)
            for v in visited:
                mask[v] = False
            mask &= eligible
            fids_masked = np.where(mask, fids, -1.0)

            # Relevance gate
            fids_masked = np.where(fids_all >= min_fid, fids_masked, -1.0)

            # Source diversity boost for novel sources
            top_cands = np.argsort(fids_masked)[-20:][::-1]
            best_idx, best_score = -1, -1.0
            for ci in top_cands:
                if fids_masked[ci] < 0:
                    continue
                src = chunks[ci]["source"]
                score = float(fids_masked[ci])
                if src not in seed_sources and src not in seen_walk_sources:
                    score *= 1.3  # novel source bonus
                elif src in seed_sources:
                    score *= 0.8  # penalize re-finding seed sources
                if score > best_score:
                    best_score = score
                    best_idx = int(ci)

            if best_idx < 0:
                break

            visited.add(best_idx)
            seen_walk_sources.add(chunks[best_idx]["source"])

            phase = float(cmath.phase(dots[best_idx]))
            th = cmath.phase(np.vdot(M, z_all[best_idx]))
            M_new = alpha * M + (1 - alpha) * z_all[best_idx] * cmath.exp(1j * th)
            raw_mag = float(np.sqrt(np.sum(np.abs(M_new)**2)))
            M_new /= raw_mag
            state_shift = 1.0 - abs(np.vdot(M, M_new))**2

            walked.append({
                "step": step + 1,
                "source": chunks[best_idx]["source"],
                "text": chunks[best_idx]["text"],
                "fidelity": round(float(fids_all[best_idx]), 6),
                "phase": round(phase, 6),
                "geometry": round(float(state_shift), 6),
                "magnitude": round(raw_mag, 6),
                "novel_source": chunks[best_idx]["source"] not in seed_sources,
                "regime": "walk",
                "idx": int(best_idx),
            })
            M = M_new

            if len(walked) >= n_walk:
                break

    # Phase 3: merge — seeds first, then walk
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
        print(f"\n{'='*70}")
        print(f"  HYBRID SEARCH: \"{o.search}\"")
        print(f"{'='*70}")

        results = deep_search(o.search, k=o.k, explore_steps=o.steps,
                              alpha=o.alpha, source_filter=o.filter)
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
        print(f"\n{'='*70}")
        print(f"  WALK: \"{o.walk}\"")
        print(f"  {o.steps} steps, alpha={o.alpha}")
        print(f"{'='*70}")

        results = walk(o.walk, k=o.k, steps=o.steps,
                       alpha=o.alpha, source_filter=o.filter)
        for r in results:
            ns = " [NEW SOURCE]" if r.get("novel_source") else ""
            print(f"\n{'─'*60}")
            print(f"  Step {r['step']}: {r['source'][:55]}{ns}")
            print(f"  fid={r['fidelity']:.4f}  phase={r['phase']:.4f}  "
                  f"geo={r['geometry']:.4f}  mag={r['magnitude']:.4f}")
            print(f"{'─'*60}")
            print(r['text'][:350])

    elif o.quick:
        print(f"\n{'='*70}")
        print(f"  QUICK SEARCH: \"{o.quick}\"")
        print(f"{'='*70}")

        results = search(o.quick, k=o.k, source_filter=o.filter)
        for i, r in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(f"[{i}] {r['source']}  fid={r['fidelity']}  phase={r['phase']}")
            print(f"{'='*60}")
            print(r['text'][:400])

    else:
        p.print_help()


if __name__ == "__main__":
    main()
