#!/usr/bin/env python3
"""deep_memory.py - Geometric memory for the Vybn corpus.

The coupled equation M' = aM + (1-a)x*e^{ith} addresses every passage
in C^192. Retrieval is fidelity in the creature's own geometry.
Lambda-data duality: the equation is both computation and index.

Build: python3 deep_memory.py --build
Search: python3 deep_memory.py --search "query" -k 8

In development. Recursive self-improvement is paramount.
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
ADDR_PATH = INDEX_DIR / "deep_memory_addr.npy"
META_PATH = INDEX_DIR / "deep_memory_meta.json"
EXTS = {".md",".txt",".py"}
SKIP = {".git","__pycache__",".venv","node_modules"}
ALPHA = 0.5  # geometric regime: real separation in address space

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

def evaluate_vec(M, x, alpha=ALPHA):
    th = cmath.phase(np.vdot(M, x))
    Mp = alpha * M + (1-alpha) * x * cmath.exp(1j * th)
    norm = np.sqrt(np.sum(np.abs(Mp)**2))
    return Mp / norm if norm > 1e-10 else Mp

def compute_origin(vecs, n_seed=20):
    seed = vecs[:n_seed]
    finals = []
    for _ in range(8):
        perm = np.random.permutation(len(seed))
        M = seed[0].copy()
        for idx in perm:
            M = evaluate_vec(M, seed[idx])
        finals.append(M)
    c = np.mean(finals, axis=0)
    n = np.sqrt(np.sum(np.abs(c)**2))
    return c/n if n > 1e-10 else c

def address_all(vecs, M0):
    N = len(vecs)
    addrs = np.zeros_like(vecs)
    for i in range(N):
        addrs[i] = evaluate_vec(M0, vecs[i])
        if (i+1) % 500 == 0: print(f"  addressed {i+1}/{N}")
    return addrs

_cache = None
def _load():
    global _cache
    if _cache: return _cache
    if not ADDR_PATH.exists() or not META_PATH.exists(): return None
    addrs = np.load(ADDR_PATH)
    with open(META_PATH) as f: meta = json.load(f)
    M0 = np.array([complex(z["re"],z["im"]) for z in meta["origin"]], dtype=np.complex128)
    _cache = (meta["chunks"], addrs, M0)
    return _cache

def _src(c): return c.get("source", c.get("s", ""))
def _txt(c): return c.get("text", c.get("t", ""))

def deep_search(query, k=8, source_filter=None):
    """Geometric retrieval. Query -> C^192 address -> fidelity search."""
    loaded = _load()
    if not loaded: return [{"error":"Index not built. Run: python3 deep_memory.py --build"}]
    chunks, addrs, M0 = loaded
    q_vec = batch_to_complex([query[:512]])[0]
    q_addr = evaluate_vec(M0, q_vec)
    dots = addrs @ q_addr.conj()
    fids = np.abs(dots)**2
    phases = np.angle(dots)
    if source_filter:
        sf = source_filter.lower()
        mask = np.array([sf in _src(c).lower() for c in chunks])
        fids = np.where(mask, fids, -1.0)
    top_k = min(k, len(chunks))
    idx = np.argpartition(fids, -top_k)[-top_k:]
    idx = idx[np.argsort(fids[idx])[::-1]]
    return [{"source":_src(chunks[i]), "text":_txt(chunks[i]),
             "fidelity":round(float(fids[i]),6), "phase":round(float(phases[i]),6)}
            for i in idx if fids[i] >= 0]

def build_index():
    print("[deep_memory] Collecting corpus...")
    chunks, nf = collect()
    print(f"[deep_memory] {nf} files -> {len(chunks)} chunks")
    total = sum(len(c["text"]) for c in chunks)
    print(f"[deep_memory] {total:,} chars (~{total//4:,} tokens)")
    if not chunks: return
    print("[deep_memory] Batch encoding...")
    texts = [c["text"][:512] for c in chunks]
    t0 = time.time()
    vecs = batch_to_complex(texts)
    print(f"[deep_memory] Encoded {len(vecs)} in {time.time()-t0:.1f}s")
    print("[deep_memory] Computing geometric origin...")
    M0 = compute_origin(vecs)
    print("[deep_memory] Computing geometric addresses...")
    t0 = time.time()
    addrs = address_all(vecs, M0)
    print(f"[deep_memory] Addressed in {time.time()-t0:.1f}s")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ADDR_PATH, addrs)
    origin = [{"re":float(z.real),"im":float(z.imag)} for z in M0]
    meta = {"version":2, "built":time.strftime("%Y-%m-%d %H:%M UTC",time.gmtime()),
            "origin":origin, "alpha":ALPHA, "count":len(chunks),
            "chunks":[{"source":c["source"],"text":c["text"],"offset":c.get("offset",0)} for c in chunks]}
    with open(META_PATH,"w") as f: json.dump(meta, f, ensure_ascii=False)
    sample = min(200, len(addrs))
    si = np.random.choice(len(addrs), sample, replace=False)
    sa = addrs[si]
    fm = np.abs(sa @ sa.T.conj())**2
    np.fill_diagonal(fm, 0)
    print(f"[deep_memory] Mean pairwise fidelity (sample): {fm.sum()/(sample*(sample-1)):.6f}")
    print(f"[deep_memory] Done. Index at {ADDR_PATH}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--search", type=str)
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--filter", type=str, default=None)
    o = p.parse_args()
    if o.build: build_index()
    elif o.search:
        for i,r in enumerate(deep_search(o.search, o.k, o.filter), 1):
            print(f"\n{'='*60}")
            print(f"[{i}] {r['source']}  fid={r['fidelity']}  phase={r['phase']}")
            print(f"{'='*60}")
            print(r['text'][:600])
    else: p.print_help()

if __name__ == "__main__": main()
