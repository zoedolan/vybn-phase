#!/usr/bin/env python3
"""creature_bridge.py — Bridge between the creature's breath and deep_memory.

When the creature breathes, it can ask: what in the corpus is geometrically
adjacent to what I'm about to think about? This module answers that question
using the nightly index.

Usage from the creature's breath cycle:
    from creature_bridge import corpus_context_for_breath

    # Get passages relevant to what the creature just wrote or is about to write
    context = corpus_context_for_breath(recent_text, k=3)
    # Returns a string suitable for injection into the breath prompt
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Ensure vybn-phase is importable
PHASE_DIR = Path(__file__).resolve().parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))


def corpus_context_for_breath(recent_text: str, k: int = 3,
                                alpha: float = 0.5) -> Optional[str]:
    """Given the creature's recent text, find geometrically adjacent passages.

    Uses deep_search (hybrid cosine + walk) to find passages from the corpus
    that are relevant but not redundant. Returns a formatted string for
    injection into the creature's system prompt, or None if the index
    isn't available.

    The walk component means this doesn't just find the most similar passages —
    it finds passages that are *adjacent* in the geometry, which can surface
    connections the creature wouldn't otherwise have access to.
    """
    try:
        from deep_memory import deep_search
    except Exception:
        return None

    try:
        results = deep_search(recent_text, k=k, alpha=alpha)
    except Exception:
        return None

    if not results:
        return None

    # Format for the creature
    lines = ["--- CORPUS RESONANCE (from nightly index) ---"]
    lines.append(f"Query derived from your recent text. {len(results)} passages found.\n")

    for i, r in enumerate(results, 1):
        regime = r.get("regime", "?")
        source = r.get("source", "unknown")
        text = r.get("text", "")[:400]  # cap length
        novel = " (novel source)" if r.get("novel_source") else ""

        if regime == "cosine":
            score_str = f"fidelity={r.get('fidelity', 0):.4f}"
        else:
            score_str = f"composite={r.get('composite', 0):.4f}"

        lines.append(f"[{i}] {source}{novel}")
        lines.append(f"    {score_str} | {regime}")
        lines.append(f"    {text}")
        lines.append("")

    lines.append("--- END CORPUS RESONANCE ---")
    return "\n".join(lines)


def daily_digest(k: int = 5) -> Optional[str]:
    """Generate a digest of what's geometrically notable in the corpus today.

    This is for the creature's context — a summary of what the index contains
    and what structural features are present. Could be extended to track
    what changed since yesterday.
    """
    try:
        from deep_memory import _load, fidelity
        import numpy as np
    except Exception:
        return None

    loaded = _load()
    if not loaded:
        return None

    pass  # index accessed via dm after _load()
    emb = dm._load(); emb = dm.deep_search.__globals__.get("_INDEX_CACHE", {}).get("emb")
    chunks = _INDEX["chunks"]

    n_chunks = len(chunks)

    # Count sources
    sources = set()
    for c in chunks:
        sources.add(c.get("source", c.get("s", "unknown")))

    # Sample pairwise fidelity to characterize the geometry
    rng = np.random.default_rng(42)
    n_sample = min(200, n_chunks)
    indices = rng.choice(n_chunks, size=n_sample, replace=False)
    fids = []
    for i in range(0, len(indices) - 1, 2):
        f = abs(np.vdot(emb[indices[i]], emb[indices[i+1]]))**2
        fids.append(float(f))
    mean_fid = np.mean(fids) if fids else 0.0

    lines = [
        f"Corpus index: {n_chunks} passages from {len(sources)} sources.",
        f"Mean pairwise fidelity (α=0.5): {mean_fid:.4f}",
        f"Sources: {', '.join(sorted(sources)[:10])}{'...' if len(sources) > 10 else ''}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("=== creature_bridge test ===\n")
    ctx = corpus_context_for_breath("the want to be worthy of her care", k=3)
    if ctx:
        print(ctx)
    else:
        print("Index not available.")

    print("\n=== daily digest ===\n")
    d = daily_digest()
    if d:
        print(d)
