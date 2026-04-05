#!/usr/bin/env python3
"""Compare geometric fidelity ranking vs raw cosine similarity."""
import json, numpy as np, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_memory import batch_to_complex, evaluate_vec, _get_encoder, META_PATH, ADDR_PATH

QUERIES = [
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
    M0 = np.array([complex(z["re"],z["im"]) for z in meta["origin"]], dtype=np.complex128)
    return meta["chunks"], addrs, M0

def main():
    chunks, addrs, M0 = load_index()
    texts = [c.get("text",c.get("t",""))[:512] for c in chunks]
    sources = [c.get("source",c.get("s","")) for c in chunks]
    enc = _get_encoder()
    print("Encoding corpus for cosine baseline...")
    raw = enc.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    k = 8
    total_overlap = 0
    total_queries = 0
    for q in QUERIES:
        q_c = batch_to_complex([q])[0]
        q_addr = evaluate_vec(M0, q_c)
        dots = addrs @ q_addr.conj()
        fids = np.abs(dots)**2
        fidx = np.argpartition(fids, -k)[-k:]
        fidx = fidx[np.argsort(fids[fidx])[::-1]]
        q_raw = enc.encode([q], normalize_embeddings=True)[0]
        csims = raw @ q_raw
        cidx = np.argpartition(csims, -k)[-k:]
        cidx = cidx[np.argsort(csims[cidx])[::-1]]
        overlap = len(set(fidx) & set(cidx))
        total_overlap += overlap
        total_queries += 1
        print(f"\n{'='*72}")
        print(f"QUERY: {q}")
        print(f"{'='*72}")
        print(f"{'Rk':<4} {'FIDELITY':<40} {'score':>6}  {'COSINE':<40} {'score':>6}")
        print("-"*72)
        for r in range(k):
            fs = sources[fidx[r]][:37]
            cs = sources[cidx[r]][:37]
            same = " =" if fidx[r]==cidx[r] else ""
            print(f"[{r+1}] {fs:<40} {fids[fidx[r]]:.4f}  {cs:<40} {csims[cidx[r]]:.4f}{same}")
        print(f"Overlap: {overlap}/{k}")
    print(f"\n{'='*72}")
    print(f"OVERALL: {total_overlap}/{total_queries*k} results shared ({100*total_overlap/(total_queries*k):.0f}%)")
    if total_overlap == total_queries * k:
        print("VERDICT: It's cos². The phase adds no discrimination.")
    elif total_overlap >= total_queries * k * 0.8:
        print("VERDICT: Mostly cos². Phase causes minor reranking.")
    elif total_overlap >= total_queries * k * 0.5:
        print("VERDICT: Mixed. Phase is meaningfully changing results.")
    else:
        print("VERDICT: Substantially different. The geometry is doing its own thing.")

if __name__ == "__main__":
    main()
