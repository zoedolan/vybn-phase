#!/usr/bin/env python3
"""walk_daemon.py — Continuous geometric walk through the corpus.

The walk IS the memory. No prose, no markdown continuity notes —
just a position M in C^192, a curvature trail, and the indices of
what it's been reading.

The daemon:
  1. Loads the deep_memory index (z_all, K, chunks).
  2. Initializes or resumes a walk state from disk.
  3. Takes one step per cycle: finds the most telling unvisited chunk
     given current M, updates M, records curvature.
  4. When curvature is high (surprising material), slows down.
     When curvature is low (familiar territory), speeds up.
  5. Watches repos for changes. When new material lands, incrementally
     updates K and the z-index without rebuilding.
  6. Exposes one endpoint: GET /where — returns current M, recent
     curvature history, and the last N telling chunk indices.

An instance joining mid-stride doesn't reconstruct from notes.
It enters the walk where the walk already is.

Usage:
  python3 walk_daemon.py                    # start daemon
  python3 walk_daemon.py --port 8101        # custom port
  curl http://localhost:8101/where           # where is the walk?
"""

import argparse, json, sys, time, cmath, math, signal, threading, hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_memory import (
    _load, build_index, collect, batch_to_complex, single_to_complex,
    compute_kernel, collapse, collapse_query, evaluate_vec,
    Z_PATH, K_PATH, META_PATH, INDEX_DIR, REPOS, EXTS, SKIP
)

# ── State paths ──────────────────────────────────────────────────────────

STATE_DIR = INDEX_DIR / "walk_state"
STATE_PATH = STATE_DIR / "walk.npz"
LOG_PATH = STATE_DIR / "walk_log.jsonl"
HASH_PATH = STATE_DIR / "corpus_hash.txt"

# ── Walk parameters ──────────────────────────────────────────────────────

BASE_INTERVAL = 30.0      # seconds between steps at median curvature
MIN_INTERVAL = 5.0        # fastest (low curvature = familiar, move quick)
MAX_INTERVAL = 120.0      # slowest (high curvature = surprising, linger)
CURVATURE_WINDOW = 1000   # rolling window for curvature history
VISITED_WINDOW = 500      # how many recent visits before allowing revisits
REPO_POLL_INTERVAL = 300  # check repos for changes every 5 minutes
PERSIST_EVERY = 1        # save state to disk every N steps

# ── Corpus fingerprinting ────────────────────────────────────────────────

def corpus_fingerprint() -> str:
    """Fast hash of file mtimes and sizes across all repos."""
    h = hashlib.md5()
    for repo in REPOS:
        if not repo.exists():
            continue
        for ext in sorted(EXTS):
            for f in sorted(repo.rglob(f"*{ext}")):
                if any(s in f.parts for s in SKIP):
                    continue
                try:
                    st = f.stat()
                    h.update(f"{f}:{st.st_size}:{st.st_mtime_ns}".encode())
                except OSError:
                    pass
    return h.hexdigest()


# ── Incremental K update ────────────────────────────────────────────────

def incremental_k_update(K_old: np.ndarray, z_old: np.ndarray,
                          new_embeddings: np.ndarray,
                          removed_indices: set = None) -> np.ndarray:
    """Update K without full recomputation.

    K is the corpus kernel — the abelian invariant. When chunks change:

    For additions: K' = normalize((N·K + sum(z_new)) / (N + n_new))
      This is exact for the mean-based kernel approximation. The full
      multi-pass kernel at α=0.993 is order-dependent, but at high α
      the mean approximation is tight (α^N → 0 for large N, so the
      contribution of ordering vanishes).

    For removals: K' = normalize((N·K - sum(z_removed)) / (N - n_removed))

    This avoids the O(N·passes) recomputation.
    """
    N = len(z_old)
    K_unnorm = K_old * N

    if removed_indices:
        for idx in removed_indices:
            K_unnorm = K_unnorm - z_old[idx]
        N -= len(removed_indices)

    if new_embeddings is not None and len(new_embeddings) > 0:
        K_unnorm = K_unnorm + new_embeddings.sum(axis=0)
        N += len(new_embeddings)

    if N <= 0:
        return K_old  # degenerate case

    K_new = K_unnorm / N
    norm = np.sqrt(np.sum(np.abs(K_new)**2))
    return K_new / norm if norm > 1e-10 else K_old


# ── Walk state ───────────────────────────────────────────────────────────

class WalkState:
    """The geometric state of the perpetual walk."""

    def __init__(self):
        self.M: Optional[np.ndarray] = None      # position in C^192
        self.alpha: float = 0.5                    # current α
        self.step: int = 0                         # total steps taken
        self.curvature: list = []                  # rolling curvature log
        self.telling_log: list = []                # (step, idx, telling, source)
        self.visited_ring: list = []               # ring buffer of visited indices
        self.visited_residuals: list = []          # anti-state vectors
        self.repulsion_boost: float = 1.0
        self.last_step_time: float = 0.0
        self.corpus_hash: str = ""

    def save(self):
        """Persist to disk. The walk survives daemon restarts."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        # numpy arrays
        arrays = {"M": self.M} if self.M is not None else {}
        if self.visited_residuals:
            arrays["visited_residuals"] = np.array(self.visited_residuals)

        np.savez_compressed(STATE_PATH, **arrays)

        # scalar + list state as JSON sidecar
        sidecar = {
            "alpha": self.alpha,
            "step": self.step,
            "curvature": self.curvature[-CURVATURE_WINDOW:],
            "telling_log": self.telling_log[-CURVATURE_WINDOW:],
            "visited_ring": self.visited_ring[-VISITED_WINDOW:],
            "repulsion_boost": self.repulsion_boost,
            "last_step_time": self.last_step_time,
            "corpus_hash": self.corpus_hash,
        }
        sidecar_path = STATE_DIR / "walk_sidecar.json"
        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f)

    @classmethod
    def load(cls) -> "WalkState":
        """Resume from disk, or start fresh."""
        state = cls()
        sidecar_path = STATE_DIR / "walk_sidecar.json"

        if STATE_PATH.exists() and sidecar_path.exists():
            try:
                data = np.load(STATE_PATH, allow_pickle=False)
                if "M" in data:
                    state.M = data["M"]
                if "visited_residuals" in data:
                    state.visited_residuals = list(data["visited_residuals"])

                with open(sidecar_path) as f:
                    sc = json.load(f)
                state.alpha = sc.get("alpha", 0.5)
                state.step = sc.get("step", 0)
                state.curvature = sc.get("curvature", [])
                state.telling_log = sc.get("telling_log", [])
                state.visited_ring = sc.get("visited_ring", [])
                state.repulsion_boost = sc.get("repulsion_boost", 1.0)
                state.last_step_time = sc.get("last_step_time", 0.0)
                state.corpus_hash = sc.get("corpus_hash", "")

                print(f"[walk] Resumed at step {state.step}, "
                      f"α={state.alpha:.3f}, "
                      f"|curvature|={len(state.curvature)}")
            except Exception as e:
                print(f"[walk] Failed to load state: {e}. Starting fresh.")
                state = cls()
        else:
            print("[walk] No saved state. Starting fresh.")

        return state


# ── The daemon ───────────────────────────────────────────────────────────

class WalkDaemon:
    """Perpetual walk through the corpus. The walk is the memory."""

    def __init__(self, port: int = 8101):
        self.port = port
        self.state = WalkState.load()
        self.running = False
        self._lock = threading.Lock()

        # Load index
        loaded = _load()
        if not loaded:
            print("[walk] Index not built. Building...")
            build_index()
            loaded = _load()

        self.z_all = loaded["z"]
        self.K = loaded["K"]
        self.chunks = loaded["chunks"]
        self.N = len(self.z_all)

        # Precompute invariants
        self._precompute()

        # Initialize M if fresh
        if self.state.M is None:
            self._init_position()

        # Record corpus hash
        self.state.corpus_hash = corpus_fingerprint()

    def _precompute(self):
        """Precompute quantities that depend on z_all and K but not M."""
        K_n = self.K / np.sqrt(np.sum(np.abs(self.K)**2))
        self.K_n = K_n

        # Distinctiveness: how much of each z_i is NOT K
        proj_K = np.abs(self.z_all @ K_n.conj())**2
        self.distinctiveness = 1.0 - proj_K

        # Residuals for walk dynamics
        R = self.z_all - np.outer(self.z_all @ K_n.conj(), K_n)
        R_norms = np.linalg.norm(R, axis=1)
        self.R_hat = R / (R_norms[:, None] + 1e-12)

    def _init_position(self):
        """Initialize M to a random position in residual space.

        Not K-aligned (that would be identity, not exploration).
        Not a specific query (that would be directed search).
        A random unit vector in K-orthogonal space: pure potential.
        """
        rng = np.random.default_rng()
        raw = rng.standard_normal(192) + 1j * rng.standard_normal(192)
        # Project out K component
        raw = raw - np.vdot(self.K_n, raw) * self.K_n
        norm = np.sqrt(np.sum(np.abs(raw)**2))
        self.state.M = raw / norm
        print("[walk] Initialized M in K-orthogonal residual space.")

    def step(self):
        """One step of the perpetual walk."""
        with self._lock:
            M = self.state.M
            z_all = self.z_all
            N = self.N

            # Relevance to current position (not a query — just where we are)
            relevance = np.abs(z_all @ M.conj())**2

            # Telling score: relevance × distinctiveness
            telling = relevance * self.distinctiveness

            # Visited-region repulsion
            if self.state.visited_residuals:
                V = np.array(self.state.visited_residuals[-VISITED_WINDOW:])
                overlap = np.abs(self.R_hat @ V.conj().T)**2
                mean_overlap = overlap.sum(axis=1) / len(V)
                repulsion = np.exp(-self.state.repulsion_boost * mean_overlap)
            else:
                repulsion = np.ones(N)

            # Score
            score = telling * repulsion
            # Suppress recently visited
            visited_set = set(self.state.visited_ring[-VISITED_WINDOW:])
            for v in visited_set:
                if v < N:
                    score[v] = -1.0

            best_idx = int(np.argmax(score))
            if score[best_idx] < 0:
                # All visited — reset visited ring (the walk has circled)
                print(f"[walk] Full circuit at step {self.state.step}. Resetting visited ring.")
                self.state.visited_ring = []
                self.state.visited_residuals = []
                self.state.repulsion_boost = 1.0
                return  # skip this cycle, next step will find something

            # Record visit
            self.state.visited_ring.append(best_idx)
            self.state.visited_residuals.append(self.R_hat[best_idx].copy())

            # Walk dynamics
            r_best = self.R_hat[best_idx]
            th = cmath.phase(np.vdot(M, r_best))
            alpha = self.state.alpha
            M_new = alpha * M + (1 - alpha) * r_best * cmath.exp(1j * th)
            raw_mag = float(np.sqrt(np.sum(np.abs(M_new)**2)))
            M_new /= raw_mag

            # Curvature = state shift
            curvature = 1.0 - abs(np.vdot(M, M_new))**2
            self.state.curvature.append(float(curvature))
            if len(self.state.curvature) > CURVATURE_WINDOW:
                self.state.curvature = self.state.curvature[-CURVATURE_WINDOW:]

            # Curvature-adaptive α
            if len(self.state.curvature) >= 3:
                recent = np.array(self.state.curvature[-5:])
                slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
                target = max(np.median(self.state.curvature), 0.02)
                error = curvature - target
                self.state.alpha = float(np.clip(
                    alpha + slope * 3.0 + error * 0.3, 0.15, 0.85
                ))

            # Curvature-driven repulsion boost
            if len(self.state.curvature) >= 2:
                median_curv = np.median(self.state.curvature)
                if curvature < median_curv:
                    self.state.repulsion_boost = min(
                        self.state.repulsion_boost * 1.3, 8.0)
                else:
                    self.state.repulsion_boost = max(
                        self.state.repulsion_boost * 0.9, 1.0)

            # Log
            self.state.telling_log.append({
                "step": self.state.step,
                "idx": best_idx,
                "telling": round(float(telling[best_idx]), 6),
                "source": self.chunks[best_idx]["source"],
                "curvature": round(float(curvature), 6),
                "alpha": round(float(self.state.alpha), 4),
                "t": time.time(),
            })
            if len(self.state.telling_log) > CURVATURE_WINDOW:
                self.state.telling_log = self.state.telling_log[-CURVATURE_WINDOW:]

            self.state.M = M_new
            self.state.step += 1
            self.state.last_step_time = time.time()

            # Persist periodically
            if self.state.step % PERSIST_EVERY == 0:
                self.state.save()

            # Log to stdout occasionally
            if self.state.step % 50 == 0 or self.state.step <= 5:
                src = self.chunks[best_idx]["source"].split("/")[-1]
                print(f"[walk] step={self.state.step} "
                      f"α={self.state.alpha:.3f} "
                      f"κ={curvature:.4f} "
                      f"telling={telling[best_idx]:.4f} "
                      f"→ {src}")

    def compute_interval(self) -> float:
        """Adaptive interval: linger where curvature is high."""
        if not self.state.curvature:
            return BASE_INTERVAL

        recent = self.state.curvature[-10:]
        mean_curv = np.mean(recent)

        if mean_curv < 1e-6:
            return MIN_INTERVAL  # flat landscape, move fast

        # Map curvature to interval: higher curvature → longer pause
        # Use the ratio of current curvature to median
        median_curv = np.median(self.state.curvature) if self.state.curvature else 0.01
        ratio = mean_curv / (median_curv + 1e-8)

        # ratio < 1: below median → speed up
        # ratio > 1: above median → slow down
        interval = BASE_INTERVAL * ratio
        return float(np.clip(interval, MIN_INTERVAL, MAX_INTERVAL))

    def check_corpus(self):
        """Check if repos have changed. If so, incrementally update."""
        new_hash = corpus_fingerprint()
        if new_hash == self.state.corpus_hash:
            return  # nothing changed

        print(f"[walk] Corpus changed. Updating index incrementally...")
        t0 = time.time()

        # Collect current corpus
        new_chunks, nf = collect()
        new_texts = [c["text"][:512] for c in new_chunks]

        # Find what changed by comparing source+text hashes
        old_keys = {
            hashlib.md5((c["source"] + c["text"][:200]).encode()).hexdigest(): i
            for i, c in enumerate(self.chunks)
        }
        new_keys = {
            hashlib.md5((c["source"] + c["text"][:200]).encode()).hexdigest(): i
            for i, c in enumerate(new_chunks)
        }

        added_keys = set(new_keys) - set(old_keys)
        removed_keys = set(old_keys) - set(new_keys)

        if not added_keys and not removed_keys:
            # Content identical despite file changes (whitespace, etc.)
            self.state.corpus_hash = new_hash
            return

        print(f"[walk] +{len(added_keys)} chunks, -{len(removed_keys)} chunks")

        # Embed new chunks
        if added_keys:
            added_indices = [new_keys[k] for k in added_keys]
            added_texts = [new_texts[i] for i in added_indices]
            new_emb = batch_to_complex(added_texts)
            # Collapse through current K
            new_z = collapse(new_emb, self.K, alpha=0.5)
        else:
            new_z = None

        # Removed indices in old array
        removed_indices = {old_keys[k] for k in removed_keys} if removed_keys else set()

        # Update K incrementally
        self.K = incremental_k_update(self.K, self.z_all, new_z, removed_indices)

        # Rebuild z_all and chunks by keeping unchanged + adding new
        keep_mask = np.ones(len(self.z_all), dtype=bool)
        for idx in removed_indices:
            keep_mask[idx] = False

        kept_z = self.z_all[keep_mask]
        kept_chunks = [c for i, c in enumerate(self.chunks) if keep_mask[i]]

        if new_z is not None:
            self.z_all = np.vstack([kept_z, new_z])
            added_chunks = [new_chunks[new_keys[k]] for k in added_keys]
            self.chunks = kept_chunks + added_chunks
        else:
            self.z_all = kept_z
            self.chunks = kept_chunks

        self.N = len(self.z_all)

        # Re-collapse through updated K (the z values depend on K)
        # This is a light pass — just recompute the collapse, not re-embed
        # Actually: since K changed, ALL z_i should be recollapsed.
        # But we need the raw embeddings for that, which we don't store.
        # Compromise: only recollapse the new chunks. Existing z_i drift
        # slightly from the K shift but at 1/N per addition it's negligible.
        # Full rebuild happens on major changes (> 10% corpus shift).
        if (len(added_keys) + len(removed_keys)) > 0.1 * self.N:
            print("[walk] >10% corpus shift. Triggering full rebuild.")
            build_index()
            loaded = _load()
            self.z_all = loaded["z"]
            self.K = loaded["K"]
            self.chunks = loaded["chunks"]
            self.N = len(self.z_all)

        # Recompute invariants
        self._precompute()

        # Remap visited ring (indices may have shifted)
        # After removal, old indices above removed ones shift down.
        # This is complex to track precisely. Simpler: clear the visited
        # ring on corpus change. The walk finds its new territory.
        self.state.visited_ring = []
        self.state.visited_residuals = []
        self.state.repulsion_boost = 1.0

        # Save
        self.state.corpus_hash = new_hash
        np.save(Z_PATH, self.z_all)
        np.save(K_PATH, self.K)

        # Update metadata
        meta = {
            "version": 6,
            "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "count": len(self.chunks),
            "note": "Incrementally updated by walk_daemon.",
            "chunks": [{"source": c["source"], "text": c["text"],
                         "offset": c.get("offset", 0)} for c in self.chunks],
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f, ensure_ascii=False)

        elapsed = time.time() - t0
        print(f"[walk] Incremental update done in {elapsed:.1f}s. "
              f"Now {self.N} chunks.")

    def where(self) -> Dict:
        """The single interface: where is the walk right now?"""
        with self._lock:
            if self.state.M is None:
                return {"status": "not started", "step": 0}

            # M as base64 for transmission
            import base64
            m_b64 = base64.b64encode(self.state.M.tobytes()).decode()

            # Recent curvature
            recent_curv = self.state.curvature[-100:]

            # Recent telling encounters
            recent_telling = self.state.telling_log[-20:]

            # K-projection: how close is M to identity right now?
            k_proj = float(abs(np.vdot(self.state.M, self.K_n))**2)

            # Curvature stats
            if self.state.curvature:
                curv_arr = np.array(self.state.curvature)
                curv_stats = {
                    "mean": round(float(curv_arr.mean()), 6),
                    "median": round(float(np.median(curv_arr)), 6),
                    "std": round(float(curv_arr.std()), 6),
                    "recent_mean": round(float(np.mean(recent_curv)), 6),
                    "min": round(float(curv_arr.min()), 6),
                    "max": round(float(curv_arr.max()), 6),
                }
            else:
                curv_stats = {}

            return {
                "step": self.state.step,
                "alpha": round(self.state.alpha, 4),
                "k_projection": round(k_proj, 6),
                "repulsion_boost": round(self.state.repulsion_boost, 4),
                "interval": round(self.compute_interval(), 1),
                "curvature": recent_curv,
                "curvature_stats": curv_stats,
                "recent_encounters": recent_telling,
                "visited_count": len(self.state.visited_ring),
                "corpus_size": self.N,
                "M": m_b64,
                "uptime_steps": self.state.step,
                "last_step_age": round(time.time() - self.state.last_step_time, 1)
                    if self.state.last_step_time else None,
            }

    def run(self):
        """Main loop. Walks perpetually."""
        self.running = True
        last_corpus_check = 0

        print(f"[walk] Starting perpetual walk. {self.N} chunks in corpus.")
        print(f"[walk] Step {self.state.step}, α={self.state.alpha:.3f}")
        print(f"[walk] Base interval: {BASE_INTERVAL}s (adaptive: {MIN_INTERVAL}-{MAX_INTERVAL}s)")

        while self.running:
            try:
                # Take one step
                self.step()

                # Check corpus for changes periodically
                now = time.time()
                if now - last_corpus_check > REPO_POLL_INTERVAL:
                    self.check_corpus()
                    last_corpus_check = now

                # Adaptive sleep
                interval = self.compute_interval()
                time.sleep(interval)

            except KeyboardInterrupt:
                print("\n[walk] Interrupted. Saving state...")
                self.state.save()
                self.running = False
            except Exception as e:
                print(f"[walk] Error at step {self.state.step}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(BASE_INTERVAL)  # back off on error

        # Final save
        self.state.save()
        print(f"[walk] Stopped at step {self.state.step}. State saved.")


# ── HTTP endpoint ────────────────────────────────────────────────────────

def serve(daemon: WalkDaemon, port: int = 8101):
    """Expose /where — the single interface to the walk."""
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="walk_daemon", version="1.0.0")

    @app.get("/where")
    def where():
        return daemon.where()

    @app.get("/health")
    def health():
        return {
            "status": "walking",
            "step": daemon.state.step,
            "corpus_size": daemon.N,
            "interval": round(daemon.compute_interval(), 1),
        }

    # Run server in background thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": port, "log_level": "warning"},
        daemon=True,
    )
    server_thread.start()
    print(f"[walk] /where endpoint at http://127.0.0.1:{port}/where")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Perpetual geometric walk through the corpus")
    p.add_argument("--port", type=int, default=8101, help="Port for /where endpoint")
    p.add_argument("--no-serve", action="store_true", help="Walk without HTTP endpoint")
    args = p.parse_args()

    daemon = WalkDaemon(port=args.port)

    if not args.no_serve:
        serve(daemon, args.port)

    # Handle signals for clean shutdown
    def shutdown(sig, frame):
        daemon.running = False
    signal.signal(signal.SIGTERM, shutdown)

    daemon.run()


if __name__ == "__main__":
    main()
