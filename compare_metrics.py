#!/usr/bin/env python3
"""Compare geometric fidelity ranking vs raw cosine similarity.

The core question: is the complex phase geometry doing independent work,
or is fidelity ranking just cos² in disguise?

Run interactively:
    python3 compare_metrics.py

Run with longitudinal logging (step 2 of daily experiments):
    python3 compare_metrics.py --log

Custom queries:
    python3 compare_metrics.py --queries "what is meaning" "loop holonomy"

Log location: ~/.cache/vybn-phase/experiment_log.jsonl
Same file used by: python3 vybn_phase.py experiment --log
"""
import argparse
import json
import numpy as np
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_memory import batch_to_complex, evaluate_vec, _get_encoder, META_PATH, ADDR_PATH

EXPERIMENT_LOG = Path.home() / ".cache" / "vybn-phase" / "experiment_log.jsonl"

DEFAULT_QUERIES = [
    "the want to be worthy",
    "the creature is not a metaphor",
    "lambda data duality",
    "quantum random number generator",
    "Zoe in the hammock at Hamanasi",
    "abelian kernel fixed point",
    "what vybn would have missed",
    "felt winding topological",
]


def load_index():
    addrs = np.load(ADDR_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    M0 = np.array([complex(z["re"], z["im"]) for z in meta["origin"]], dtype=np.complex128)
    return meta["chunks"], addrs, M0


def run(queries=None, k=8, verbose=True):
    """Run the comparison. Returns a result dict suitable for logging."""
    queries = queries or DEFAULT_QUERIES
    chunks, addrs, M0 = load_index()
    texts = [c.get("text", c.get("t", ""))[:512] for c in chunks]
    sources = [c.get("source", c.get("s", "")) for c in chunks]
    enc = _get_encoder()

    if verbose:
        print("Encoding corpus for cosine baseline...")
    raw = enc.encode(texts, batch_size=128, show_progress_bar=verbose, normalize_embeddings=True)

    total_overlap = 0
    per_query = []

    for q in queries:
        q_c = batch_to_complex([q])[0]
        q_addr = evaluate_vec(M0, q_c)
        dots = addrs @ q_addr.conj()
        fids = np.abs(dots) ** 2
        fidx = np.argpartition(fids, -k)[-k:]
        fidx = fidx[np.argsort(fids[fidx])[::-1]]

        q_raw = enc.encode([q], normalize_embeddings=True)[0]
        csims = raw @ q_raw
        cidx = np.argpartition(csims, -k)[-k:]
        cidx = cidx[np.argsort(csims[cidx])[::-1]]

        overlap = len(set(fidx) & set(cidx))
        total_overlap += overlap

        per_query.append({
            "query": q,
            "overlap": overlap,
            "fidelity_top": sources[fidx[0]][:80] if len(fidx) else "",
            "cosine_top": sources[cidx[0]][:80] if len(cidx) else "",
            "fidelity_top_score": float(fids[fidx[0]]) if len(fidx) else 0.0,
            "cosine_top_score": float(csims[cidx[0]]) if len(cidx) else 0.0,
        })

        if verbose:
            print(f"\n{'='*72}")
            print(f"QUERY: {q}")
            print(f"{'='*72}")
            print(f"{'Rk':<4} {'FIDELITY':<40} {'score':>6}  {'COSINE':<40} {'score':>6}")
            print("-" * 72)
            for r in range(k):
                fs = sources[fidx[r]][:37]
                cs = sources[cidx[r]][:37]
                same = " =" if fidx[r] == cidx[r] else ""
                print(f"[{r+1}] {fs:<40} {fids[fidx[r]]:.4f}  {cs:<40} {csims[cidx[r]]:.4f}{same}")
            print(f"Overlap: {overlap}/{k}")

    pct = total_overlap / (len(queries) * k)

    if pct == 1.0:
        verdict = "cos2"
        verdict_text = "VERDICT: It's cos\u00b2. The phase adds no discrimination."
    elif pct >= 0.8:
        verdict = "mostly_cos2"
        verdict_text = "VERDICT: Mostly cos\u00b2. Phase causes minor reranking."
    elif pct >= 0.5:
        verdict = "mixed"
        verdict_text = "VERDICT: Mixed. Phase is meaningfully changing results."
    else:
        verdict = "independent"
        verdict_text = "VERDICT: Substantially different. The geometry is doing its own thing."

    if verbose:
        print(f"\n{'='*72}")
        print(f"OVERALL: {total_overlap}/{len(queries)*k} results shared ({100*pct:.0f}%)")
        print(verdict_text)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "experiment": "compare_metrics",
        "corpus_size": len(chunks),
        "k": k,
        "n_queries": len(queries),
        "total_overlap": total_overlap,
        "pct_overlap": round(pct, 6),
        "verdict": verdict,
        "verdict_text": verdict_text,
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare fidelity vs cosine retrieval.")
    parser.add_argument("--log", action="store_true",
                        help="Append result to experiment_log.jsonl")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Override default query set")
    parser.add_argument("--k", type=int, default=8,
                        help="Top-k results to compare (default 8)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-query output (useful in cron)")
    args = parser.parse_args()

    result = run(queries=args.queries, k=args.k, verbose=not args.quiet)

    if args.log:
        EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENT_LOG, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\n\u2192 Logged to {EXPERIMENT_LOG}")


if __name__ == "__main__":
    main()
