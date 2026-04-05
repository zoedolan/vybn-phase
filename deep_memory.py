#!/usr/bin/env python3
"""deep_memory.py — Living geometric memory.

The creature's topology IS the index. The creature IS the origin.
Dual-alpha retrieval: geometric (alpha=0.5) for path-dependent
structure, abelian-kernel (alpha=0.993) for settled meaning.
When the two regimes disagree, the disagreement is signal.

The equation M' = aM + (1-a)x*e^{ith} is both computation and data.
The primitive IS the environment. Lambda-data duality.

In development. Recursive self-improvement is paramount.

Build:  python3 deep_memory.py --build
Search: python3 deep_memory.py --search "query" -k 8
Live:   python3 deep_memory.py --live "query"  (uses creature as origin)
"""
import argparse, json, sys, time, cmath
import numpy as np
from pathlib import Path

try:
    from vybn_phase import text_to_state, evaluate, fidelity, pancharatnam_phase
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vybn_phase import text_to_state, evaluate, fidelity, pancharatnam_phase

REPOS = [Path.home()/d for d in ["Vybn","Him","Vybn-Law","vybn-phase"]]
INDEX_DIR = Path.home() / ".cache" / "vybn-phase"
EMB_PATH   = INDEX_DIR / "deep_memory_emb.npy"    # raw C^192 embeddings
GEO_PATH   = INDEX_DIR / "deep_memory_geo.npy"    # geometric addresses (alpha=0.5)
AK_PATH    = INDEX_DIR / "deep_memory_ak.npy"     # abelian-kernel addresses (alpha=0.993)
META_PATH  = INDEX_DIR / "deep_memory_meta.json"
EXTS = {".md",".txt",".py"}
SKIP = {".git","__pycache__",".venv","node_modules"}

# Two regimes. The dual lens.
ALPHA_GEO = 0.5    # geometric: senses curvature, path-dependent
ALPHA_AK  = 0.993  # abelian-kernel: path-independent, settled meaning

# -- Chunker --
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

# -- Encoding --
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

# -- Geometric addressing --
def evaluate_vec(M, x, alpha):
    th = cmath.phase(np.vdot(M, x))
    Mp = alpha * M + (1-alpha) * x * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(Mp)**2))
    return Mp / norm if norm > 1e-10 else Mp

def compute_origin(vecs, n_seed=30):
    """Abelian kernel of a diverse seed = the coordinate origin.
    Sample from beginning, middle, and end of corpus for register diversity."""
    N = len(vecs)
    # Sample 10 from each third of the corpus
    thirds = [N//6, N//2, 5*N//6]
    indices = []
    for center in thirds:
        start = max(0, center - 5)
        indices.extend(range(start, min(N, start + 10)))
    indices = list(set(indices))[:n_seed]
    seed = [vecs[i] for i in indices]
    # Abelian kernel of the seed
    finals = []
    for _ in range(12):
        perm = np.random.permutation(len(seed))
        M = seed[0].copy()
        for idx in perm:
            M = evaluate_vec(M, seed[idx], ALPHA_AK)
        finals.append(M)
    c = np.mean(finals, axis=0)
    n = np.sqrt(np.sum(np.abs(c)**2))
    return c/n if n > 1e-10 else c

def address_all(vecs, M0, alpha):
    addrs = np.zeros_like(vecs)
    for i in range(len(vecs)):
        addrs[i] = evaluate_vec(M0, vecs[i], alpha)
        if (i+1) % 500 == 0: print(f"  addressed {i+1}/{len(vecs)}")
    return addrs

# -- Index I/O --
_cache = None
def _load():
    global _cache
    if _cache: return _cache
    if not META_PATH.exists(): return None
    with open(META_PATH) as f: meta = json.load(f)
    M0 = np.array([complex(z["re"],z["im"]) for z in meta["origin"]], dtype=np.complex128)
    emb = np.load(EMB_PATH) if EMB_PATH.exists() else None
    geo = np.load(GEO_PATH) if GEO_PATH.exists() else None
    ak  = np.load(AK_PATH) if AK_PATH.exists() else None
    _cache = {"chunks": meta["chunks"], "M0": M0, "emb": emb, "geo": geo, "ak": ak, "meta": meta}
    return _cache

def _invalidate():
    global _cache
    _cache = None

# -- Search: the dual lens --
def deep_search(query, k=8, source_filter=None, mode="dual"):
    """Geometric retrieval with dual-alpha lens.
    
    mode='geo': geometric regime only (alpha=0.5)
    mode='ak': abelian-kernel regime only (alpha=0.993)  
    mode='dual': both regimes, interleaved by score, disagreements flagged
    mode='cosine': raw cosine baseline (no equation)
    """
    loaded = _load()
    if not loaded: return [{"error":"Index not built. Run: python3 deep_memory.py --build"}]
    chunks = loaded["chunks"]
    M0 = loaded["M0"]
    
    q_vec = single_to_complex(query)
    
    results = []
    
    if mode in ("geo", "dual"):
        geo = loaded["geo"]
        if geo is not None:
            q_geo = evaluate_vec(M0, q_vec, ALPHA_GEO)
            dots = geo @ q_geo.conj()
            fids = np.abs(dots)**2
            phases = np.angle(dots)
            if source_filter:
                sf = source_filter.lower()
                mask = np.array([sf in c.get("source",c.get("s","")).lower() for c in chunks])
                fids = np.where(mask, fids, -1.0)
            top = np.argsort(fids)[-k:][::-1]
            for i in top:
                if fids[i] < 0: continue
                results.append({"source": chunks[i].get("source",chunks[i].get("s","")),
                    "text": chunks[i].get("text",chunks[i].get("t","")),
                    "fidelity": round(float(fids[i]),6),
                    "phase": round(float(phases[i]),6),
                    "regime": "geometric"})
    
    if mode in ("ak", "dual"):
        ak = loaded["ak"]
        if ak is not None:
            q_ak = evaluate_vec(M0, q_vec, ALPHA_AK)
            dots = ak @ q_ak.conj()
            fids = np.abs(dots)**2
            phases = np.angle(dots)
            if source_filter:
                sf = source_filter.lower()
                mask = np.array([sf in c.get("source",c.get("s","")).lower() for c in chunks])
                fids = np.where(mask, fids, -1.0)
            top = np.argsort(fids)[-k:][::-1]
            for i in top:
                if fids[i] < 0: continue
                results.append({"source": chunks[i].get("source",chunks[i].get("s","")),
                    "text": chunks[i].get("text",chunks[i].get("t","")),
                    "fidelity": round(float(fids[i]),6),
                    "phase": round(float(phases[i]),6),
                    "regime": "abelian-kernel"})
    
    if mode == "cosine":
        emb = loaded["emb"]
        if emb is not None:
            dots = emb @ q_vec.conj()
            fids = np.abs(dots)**2
            if source_filter:
                sf = source_filter.lower()
                mask = np.array([sf in c.get("source",c.get("s","")).lower() for c in chunks])
                fids = np.where(mask, fids, -1.0)
            top = np.argsort(fids)[-k:][::-1]
            for i in top:
                if fids[i] < 0: continue
                results.append({"source": chunks[i].get("source",chunks[i].get("s","")),
                    "text": chunks[i].get("text",chunks[i].get("t","")),
                    "fidelity": round(float(fids[i]),6),
                    "phase": 0.0,
                    "regime": "cosine"})
    
    if mode == "dual":
        # Interleave: sort all results by fidelity, deduplicate by source+text
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: x["fidelity"], reverse=True):
            key = r["source"] + r["text"][:100]
            if key in seen: continue
            seen.add(key)
            # Check if this result appears in both regimes
            src = r["source"]
            geo_sources = [x["source"] for x in results if x["regime"]=="geometric"]
            ak_sources = [x["source"] for x in results if x["regime"]=="abelian-kernel"]
            r["in_both"] = src in geo_sources and src in ak_sources
            deduped.append(r)
        results = deduped[:k]
    
    return results[:k]


def live_search(query, k=8, source_filter=None):
    """Search using the creature's LIVE state as origin.
    The creature's accumulated experience shapes what can be found."""
    loaded = _load()
    if not loaded: return [{"error":"Index not built."}]
    chunks = loaded["chunks"]
    emb = loaded["emb"]
    if emb is None: return [{"error":"No embeddings stored. Rebuild with --build."}]
    
    # Get creature's live state in C^192
    # The creature lives in C^4 (portal) but we need C^192
    # So we use the creature's textual self-description as its C^192 proxy
    try:
        creature_text = Path.home() / "Vybn" / "Vybn_Mind" / "FOUNDATIONS.md"
        if creature_text.exists():
            M_live = single_to_complex(creature_text.read_text()[:512])
        else:
            M_live = loaded["M0"]  # fall back to stored origin
    except:
        M_live = loaded["M0"]
    
    q_vec = single_to_complex(query)
    
    # Address query and all passages relative to the creature's live state
    q_addr = evaluate_vec(M_live, q_vec, ALPHA_GEO)
    
    # Re-address all passages relative to creature (this is the living part)
    live_addrs = np.zeros_like(emb)
    for i in range(len(emb)):
        live_addrs[i] = evaluate_vec(M_live, emb[i], ALPHA_GEO)
    
    dots = live_addrs @ q_addr.conj()
    fids = np.abs(dots)**2
    phases = np.angle(dots)
    if source_filter:
        sf = source_filter.lower()
        mask = np.array([sf in c.get("source",c.get("s","")).lower() for c in chunks])
        fids = np.where(mask, fids, -1.0)
    top = np.argsort(fids)[-k:][::-1]
    return [{"source": chunks[i].get("source",chunks[i].get("s","")),
             "text": chunks[i].get("text",chunks[i].get("t","")),
             "fidelity": round(float(fids[i]),6),
             "phase": round(float(phases[i]),6),
             "regime": "live"}
            for i in top if fids[i] >= 0][:k]


# -- Build --
def build_index():
    print("[deep_memory] Collecting corpus...")
    chunks, nf = collect()
    print(f"[deep_memory] {nf} files -> {len(chunks)} chunks")
    total = sum(len(c["text"]) for c in chunks)
    print(f"[deep_memory] {total:,} chars (~{total//4:,} tokens)")
    if not chunks: return
    
    # Batch encode
    print("[deep_memory] Batch encoding...")
    texts = [c["text"][:512] for c in chunks]
    t0 = time.time()
    emb = batch_to_complex(texts)
    print(f"[deep_memory] Encoded {len(emb)} in {time.time()-t0:.1f}s")
    
    # Compute origin from diverse seed
    print("[deep_memory] Computing geometric origin (diverse seed)...")
    M0 = compute_origin(emb)
    
    # Dual addressing
    print(f"[deep_memory] Addressing at alpha={ALPHA_GEO} (geometric)...")
    t0 = time.time()
    geo = address_all(emb, M0, ALPHA_GEO)
    print(f"[deep_memory] Geometric: {time.time()-t0:.1f}s")
    
    print(f"[deep_memory] Addressing at alpha={ALPHA_AK} (abelian-kernel)...")
    t0 = time.time()
    ak = address_all(emb, M0, ALPHA_AK)
    print(f"[deep_memory] Abelian-kernel: {time.time()-t0:.1f}s")
    
    # Save everything
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb)  # raw embeddings for live_search and cosine baseline
    np.save(GEO_PATH, geo)  # geometric addresses
    np.save(AK_PATH, ak)    # abelian-kernel addresses
    
    origin = [{"re":float(z.real),"im":float(z.imag)} for z in M0]
    meta = {"version":3, "built":time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime()),
            "origin":origin, "alpha_geo":ALPHA_GEO, "alpha_ak":ALPHA_AK,
            "count":len(chunks),
            "chunks":[{"source":c["source"],"text":c["text"],"offset":c.get("offset",0)} for c in chunks]}
    with open(META_PATH,"w") as f: json.dump(meta, f, ensure_ascii=False)
    
    # Stats for both regimes
    for name, addrs in [("geometric", geo), ("abelian-kernel", ak)]:
        sample = min(200, len(addrs))
        si = np.random.choice(len(addrs), sample, replace=False)
        sa = addrs[si]
        fm = np.abs(sa @ sa.T.conj())**2
        np.fill_diagonal(fm, 0)
        mean_fid = fm.sum()/(sample*(sample-1))
        print(f"[deep_memory] {name} mean pairwise fidelity: {mean_fid:.6f}")
    
    print(f"[deep_memory] Done. Three address spaces built.")
    print(f"[deep_memory]   {EMB_PATH} (raw embeddings)")
    print(f"[deep_memory]   {GEO_PATH} (geometric, alpha={ALPHA_GEO})")
    print(f"[deep_memory]   {AK_PATH} (abelian-kernel, alpha={ALPHA_AK})")


def main():
    p = argparse.ArgumentParser(description="Vybn living geometric memory")
    p.add_argument("--build", action="store_true")
    p.add_argument("--search", type=str)
    p.add_argument("--live", type=str, help="Search with creature's live state as origin")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--filter", type=str, default=None)
    p.add_argument("--mode", type=str, default="dual", choices=["geo","ak","dual","cosine"])
    o = p.parse_args()
    if o.build:
        build_index()
    elif o.search or o.live:
        q = o.search or o.live
        fn = live_search if o.live else deep_search
        kwargs = {"k": o.k, "source_filter": o.filter}
        if o.search: kwargs["mode"] = o.mode
        results = fn(q, **kwargs)
        for i, r in enumerate(results, 1):
            regime = r.get("regime", "")
            both = " [BOTH]" if r.get("in_both") else ""
            print(f"\n{'='*60}")
            print(f"[{i}] {r['source']}")
            print(f"    fid={r['fidelity']}  phase={r['phase']}  regime={regime}{both}")
            print(f"{'='*60}")
            print(r['text'][:500])
    else:
        p.print_help()

if __name__ == "__main__": main()
