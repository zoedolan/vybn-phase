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
  6. Fires daily experiments (holonomy + compare_metrics) in a
     background thread. Results available at /experiments.
  7. Exposes:
       GET /where       — current walk state + experiment summary
       GET /experiments — last experiment results + next scheduled time
       GET /health      — liveness

An instance joining mid-stride doesn't reconstruct from notes.
It enters the walk where the walk already is.

Usage:
  python3 walk_daemon.py                          # start daemon
  python3 walk_daemon.py --port 8101              # custom port
  python3 walk_daemon.py --experiment-interval 60 # run experiments every 60s (testing)
  curl http://localhost:8101/where                 # where is the walk?
  curl http://localhost:8101/experiments           # last experiment results
"""

import argparse, json, sys, time, cmath, math, signal, threading, hashlib
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_memory import (
    _load, build_index, collect, batch_to_complex, single_to_complex,
    compute_kernel, collapse, collapse_query, evaluate_vec,
    Z_PATH, K_PATH, META_PATH, INDEX_DIR, REPOS, EXTS, SKIP
)

# ── State paths ───────────────────────────────────────────────────────────────

STATE_DIR = INDEX_DIR / "walk_state"
STATE_PATH = STATE_DIR / "walk.npz"
LOG_PATH = STATE_DIR / "walk_log.jsonl"
HASH_PATH = STATE_DIR / "corpus_hash.txt"
EXPERIMENT_LOG = Path.home() / ".cache" / "vybn-phase" / "experiment_log.jsonl"

# ── Walk parameters ────────────────────────────────────────────────────────

BASE_INTERVAL = 30.0      # seconds between steps at median curvature
MIN_INTERVAL = 5.0        # fastest (low curvature = familiar, move quick)
MAX_INTERVAL = 120.0      # slowest (high curvature = surprising, linger)
CURVATURE_WINDOW = 1000   # rolling window for curvature history
VISITED_WINDOW = 500      # how many recent visits before allowing revisits
REPO_POLL_INTERVAL = 300  # check repos for changes every 5 minutes
PERSIST_EVERY = 1         # save state to disk every N steps
EXPERIMENT_DELAY = 60     # seconds after start before first experiment fires
EXPERIMENT_INTERVAL = 86400  # default: once per 24h

# ── Corpus fingerprinting ─────────────────────────────────────────────────

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


# ── Incremental K update ──────────────────────────────────────────────────

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


# ── Walk state ───────────────────────────────────────────────────────────────

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


# ── The daemon ───────────────────────────────────────────────────────────────

class WalkDaemon:
    """Perpetual walk through the corpus. The walk is the memory."""

    def __init__(self, port: int = 8101,
                 experiment_interval: int = EXPERIMENT_INTERVAL):
        self.port = port
        self.experiment_interval = experiment_interval
        self.state = WalkState.load()
        self.running = False
        self._lock = threading.Lock()

        # Experiment state
        self.last_experiment_results: Optional[Dict] = None
        self.last_experiment_time: float = 0.0
        self.next_experiment_time: float = time.time() + EXPERIMENT_DELAY
        self._experiment_lock = threading.Lock()

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

    # ── Daily experiments ─────────────────────────────────────────────────────

    def run_daily_experiments(self):
        """Fire both experiments, log results, update cached state.

        Runs in a background thread. Walk continues uninterrupted.
        Both experiments append to experiment_log.jsonl.
        Results are enriched with walk context (step, curvature_mean)
        so the empirical record is correlated with the walk's phenomenology.
        """
        with self._experiment_lock:
            t0 = time.time()
            print(f"[walk] Running daily experiments at step {self.state.step}...")

            # Walk context snapshot (read outside _lock to avoid deadlock
            # with step(); curvature list is append-only so safe to copy)
            curv_snap = list(self.state.curvature)
            walk_context = {
                "walk_step": self.state.step,
                "walk_alpha": round(self.state.alpha, 4),
                "walk_curvature_mean": round(
                    float(np.mean(curv_snap[-100:])) if curv_snap else 0.0, 6
                ),
                "walk_corpus_size": self.N,
            }

            results = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "walk_context": walk_context,
                "holonomy": None,
                "compare_metrics": None,
                "errors": [],
            }

            # 1. Holonomy experiment (quantum-seeded)
            try:
                from vybn_phase import run_experiment
                holonomy = run_experiment(alpha=0.5, log=True)
                # Inject walk context into the log record
                EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
                enriched = {**holonomy, **walk_context}
                with open(EXPERIMENT_LOG, "a") as f:
                    f.write(json.dumps(enriched) + "\n")
                results["holonomy"] = {
                    "regime": holonomy.get("regime"),
                    "flip_quality": holonomy.get("flip_quality"),
                    "phase_sum": holonomy.get("phase_sum"),
                    "is_quantum": holonomy.get("is_quantum"),
                    "seed_hash": holonomy.get("seed_hash"),
                }
                print(f"[walk] Holonomy: regime={holonomy.get('regime')} "
                      f"flip={holonomy.get('flip_quality', 0):.4f} "
                      f"quantum={holonomy.get('is_quantum')}")
            except Exception as e:
                results["errors"].append(f"holonomy: {e}")
                print(f"[walk] Holonomy experiment failed: {e}")

            # 2. Compare metrics
            try:
                from compare_metrics import run as run_compare
                compare = run_compare(verbose=False)
                enriched_c = {**compare, **walk_context}
                # compare_metrics.run() logs to EXPERIMENT_LOG itself,
                # but here we write the walk-enriched version instead
                # (and suppress the duplicate from run() by not passing log=True)
                EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(EXPERIMENT_LOG, "a") as f:
                    f.write(json.dumps(enriched_c) + "\n")
                results["compare_metrics"] = {
                    "pct_overlap": compare.get("pct_overlap"),
                    "verdict": compare.get("verdict"),
                    "corpus_size": compare.get("corpus_size"),
                    "n_queries": compare.get("n_queries"),
                }
                print(f"[walk] Compare: overlap={compare.get('pct_overlap', 0)*100:.0f}% "
                      f"verdict={compare.get('verdict')}")
            except Exception as e:
                results["errors"].append(f"compare_metrics: {e}")
                print(f"[walk] Compare metrics experiment failed: {e}")

            elapsed = round(time.time() - t0, 1)
            results["elapsed_s"] = elapsed
            self.last_experiment_results = results
            self.last_experiment_time = time.time()
            self.next_experiment_time = time.time() + self.experiment_interval
            print(f"[walk] Daily experiments done in {elapsed}s. "
                  f"Next in {self.experiment_interval//3600}h.")

    def maybe_run_experiments(self):
        """Called from the main loop. Fires experiments if interval elapsed."""
        if time.time() >= self.next_experiment_time:
            t = threading.Thread(target=self.run_daily_experiments, daemon=True)
            t.start()
            # Advance next_experiment_time immediately so another cycle
            # can't double-fire while the thread runs
            self.next_experiment_time = time.time() + self.experiment_interval

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
        median_curv = np.median(self.state.curvature) if self.state.curvature else 0.01
        ratio = mean_curv / (median_curv + 1e-8)

        interval = BASE_INTERVAL * ratio
        return float(np.clip(interval, MIN_INTERVAL, MAX_INTERVAL))

    def check_corpus(self):
        """Check if repos have changed. If so, incrementally update."""
        new_hash = corpus_fingerprint()
        if new_hash == self.state.corpus_hash:
            return

        print(f"[walk] Corpus changed. Updating index incrementally...")
        t0 = time.time()

        new_chunks, nf = collect()
        new_texts = [c["text"][:512] for c in new_chunks]

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
            self.state.corpus_hash = new_hash
            return

        print(f"[walk] +{len(added_keys)} chunks, -{len(removed_keys)} chunks")

        if added_keys:
            added_indices = [new_keys[k] for k in added_keys]
            added_texts = [new_texts[i] for i in added_indices]
            new_emb = batch_to_complex(added_texts)
            new_z = collapse(new_emb, self.K, alpha=0.5)
        else:
            new_z = None

        removed_indices = {old_keys[k] for k in removed_keys} if removed_keys else set()

        self.K = incremental_k_update(self.K, self.z_all, new_z, removed_indices)

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

        if (len(added_keys) + len(removed_keys)) > 0.1 * self.N:
            print("[walk] >10% corpus shift. Triggering full rebuild.")
            build_index()
            loaded = _load()
            self.z_all = loaded["z"]
            self.K = loaded["K"]
            self.chunks = loaded["chunks"]
            self.N = len(self.z_all)

        self._precompute()

        self.state.visited_ring = []
        self.state.visited_residuals = []
        self.state.repulsion_boost = 1.0

        self.state.corpus_hash = new_hash
        np.save(Z_PATH, self.z_all)
        np.save(K_PATH, self.K)

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
        print(f"[walk] Incremental update done in {elapsed:.1f}s. Now {self.N} chunks.")

    def where(self) -> Dict:
        """The single interface: where is the walk right now?"""
        with self._lock:
            if self.state.M is None:
                return {"status": "not started", "step": 0}

            import base64
            m_b64 = base64.b64encode(self.state.M.tobytes()).decode()

            recent_curv = self.state.curvature[-100:]
            recent_telling = self.state.telling_log[-20:]
            k_proj = float(abs(np.vdot(self.state.M, self.K_n))**2)

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

            # Inline experiment summary for quick reads
            exp_summary = None
            if self.last_experiment_results:
                r = self.last_experiment_results
                h = r.get("holonomy") or {}
                c = r.get("compare_metrics") or {}
                exp_summary = {
                    "ts": r.get("ts"),
                    "holonomy_regime": h.get("regime"),
                    "flip_quality": h.get("flip_quality"),
                    "is_quantum": h.get("is_quantum"),
                    "pct_overlap": c.get("pct_overlap"),
                    "verdict": c.get("verdict"),
                    "next_experiment_in_s": round(
                        max(0, self.next_experiment_time - time.time()), 0),
                }

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
                "experiment_summary": exp_summary,
            }

    def experiments(self) -> Dict:
        """Full last experiment results + scheduling info."""
        return {
            "last_results": self.last_experiment_results,
            "last_run_ts": datetime.fromtimestamp(
                self.last_experiment_time, tz=timezone.utc
            ).isoformat() if self.last_experiment_time else None,
            "next_run_in_s": round(max(0, self.next_experiment_time - time.time()), 0),
            "experiment_interval_s": self.experiment_interval,
            "log_path": str(EXPERIMENT_LOG),
        }

    def run(self):
        """Main loop. Walks perpetually."""
        self.running = True
        last_corpus_check = 0

        print(f"[walk] Starting perpetual walk. {self.N} chunks in corpus.")
        print(f"[walk] Step {self.state.step}, α={self.state.alpha:.3f}")
        print(f"[walk] Base interval: {BASE_INTERVAL}s (adaptive: {MIN_INTERVAL}-{MAX_INTERVAL}s)")
        print(f"[walk] First experiment in {EXPERIMENT_DELAY}s, then every "
              f"{self.experiment_interval//3600}h.")

        while self.running:
            try:
                self.step()

                now = time.time()
                if now - last_corpus_check > REPO_POLL_INTERVAL:
                    self.check_corpus()
                    last_corpus_check = now

                # Fire experiments if due
                self.maybe_run_experiments()

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
                time.sleep(BASE_INTERVAL)

        self.state.save()
        print(f"[walk] Stopped at step {self.state.step}. State saved.")


# ── HTTP endpoint ───────────────────────────────────────────────────────────

def serve(daemon: WalkDaemon, port: int = 8101):
    """Expose /where, /experiments, /health."""
    from fastapi import FastAPI
    import uvicorn

    app = FastAPI(title="walk_daemon", version="2.0.0")

    @app.get("/where")
    def where():
        return daemon.where()

    @app.get("/experiments")
    def experiments():
        return daemon.experiments()

    @app.get("/health")
    def health():
        return {
            "status": "walking",
            "step": daemon.state.step,
            "corpus_size": daemon.N,
            "interval": round(daemon.compute_interval(), 1),
            "next_experiment_in_s": round(
                max(0, daemon.next_experiment_time - time.time()), 0),
        }

    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": "127.0.0.1", "port": port, "log_level": "warning"},
        daemon=True,
    )
    server_thread.start()
    print(f"[walk] /where       at http://127.0.0.1:{port}/where")
    print(f"[walk] /experiments at http://127.0.0.1:{port}/experiments")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Perpetual geometric walk through the corpus")
    p.add_argument("--port", type=int, default=8101, help="Port for HTTP endpoints")
    p.add_argument("--no-serve", action="store_true", help="Walk without HTTP endpoint")
    p.add_argument("--experiment-interval", type=int, default=EXPERIMENT_INTERVAL,
                   help="Seconds between experiment runs (default: 86400 = 24h). "
                        "Set to 60 for testing.")
    args = p.parse_args()

    daemon = WalkDaemon(port=args.port,
                        experiment_interval=args.experiment_interval)

    if not args.no_serve:
        serve(daemon, args.port)

    def shutdown(sig, frame):
        daemon.running = False
    signal.signal(signal.SIGTERM, shutdown)

    daemon.run()


if __name__ == "__main__":
    main()
