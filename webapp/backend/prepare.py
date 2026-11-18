"""One-time preparation of the models served by the API.

Nothing here trains or changes a model. It
  1. checks the MD5 of the frozen merged model (Step 4 selection, confirmed on CONF);
  2. rebuilds the fastText-cc.bn table exactly as the thesis did (first 200k rows + every evaluation word);
  3. rebuilds the Subs2vec-w2 vocabulary and checks that the merged vocabulary is exactly the union of
     News-W2V, Informal-W2V and Subs2vec-w2 in the thesis order (proves the files belong together);
  4. writes float32 unit-row copies (fast memory-mapped search) and per-word provenance of the merged rows;
  5. fits a fixed PCA basis per model (top 20,000 words) for the 3-D view;
  6. writes data/manifest.json with checksums the server verifies at start-up.

Usage:  python prepare.py            (needs data/raw/cc.bn.300.vec.gz for the fastText table)
"""
import bz2, gzip, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

T0 = time.perf_counter()
C.DERIVED.mkdir(parents=True, exist_ok=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ words the thesis always kept (evaluation tokens)
ev = pd.read_parquet(C.EVAL_DATA / "eval_v2.parquet", columns=["tokens"])
NEEDED = {w for t in ev.tokens for w in t.split()}
for p in sorted((C.EVAL_DATA / "analogy").glob("*.txt")):
    if p.stem != "Data_Stat":
        NEEDED |= {w for l in open(p, encoding="utf-8") if len(l.split()) == 4 for w in l.split()}
NEEDED |= {w for v in json.loads((C.EVAL_DATA / "categories.json").read_text(encoding="utf-8")).values() for w in v}
log("needed words:", len(NEEDED))


def table_from_text(lines, dim, sep, keep_vectors=True):
    """Same rule as merge_src.table_from_text: keep row n if n <= 200k or the word is an evaluation word."""
    words, vecs, n = [], [], 0
    for line in lines:
        w, _, rest = line.partition(sep); n += 1
        if n <= C.TOPK_SEARCH or w in NEEDED:
            try:
                v = np.asarray(rest.split(sep)[:dim], dtype=np.float32)
                if len(v) == dim:
                    words.append(w)
                    if keep_vectors: vecs.append(v)
            except ValueError:
                pass
    return words, (np.stack(vecs) if keep_vectors else None), n


# ------------------------------------------------------------------ Subs2vec-w2 vocabulary (only for provenance + union check)
assert md5(C.SUBS_FILE) == C.SUBS_MD5, "Subs2vec-w2 file MD5 mismatch"
def subs_rows():
    with bz2.open(C.SUBS_FILE, "rt", encoding="utf-8") as f:
        next(f)
        for l in f: yield l.rstrip("\n")
subs_words, _, _ = table_from_text(subs_rows(), 300, ",", keep_vectors=False)
log("Subs2vec-w2 rows:", len(subs_words))
assert len(subs_words) == 177_376, "Subs2vec-w2 table size differs from the thesis (177,376)"

# ------------------------------------------------------------------ fastText-cc.bn table (as in the thesis)
ft_src, ft_words = C.MODELS["fasttext"]["source"], C.MODELS["fasttext"]["words"]
if not ft_src.exists():
    assert C.FT_GZ.exists(), f"download {C.FT_URL} to {C.FT_GZ} first"
    def ft_rows():
        with gzip.open(C.FT_GZ, "rt", encoding="utf-8", errors="ignore") as f:
            next(f)
            for l in f: yield l.rstrip()
    w, V, n = table_from_text(ft_rows(), 300, " ")
    nrm = np.linalg.norm(V, axis=1, keepdims=True); nrm[nrm == 0] = 1; V /= nrm
    np.save(ft_src, V.astype(np.float16)); ft_words.write_text("\n".join(w), encoding="utf-8")
    log(f"fastText table: {len(w):,} rows kept of {n:,}")
    del V

# ------------------------------------------------------------------ served copies + checks
manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
words_of = {}
for key in C.BASE_KEYS:
    spec = C.MODELS[key]
    src_md5 = md5(spec["source"])
    if key == "merged":
        assert src_md5 == C.FROZEN_MD5, "merged model is not the frozen Step 4 selection"
    words = spec["words"].read_text(encoding="utf-8").split("\n")
    U16 = np.load(spec["source"], mmap_mode="r")
    assert U16.dtype == np.float16 and U16.shape == (len(words), spec["dim"]), f"{key}: shape {U16.shape}"
    assert len(set(words)) == len(words), f"{key}: duplicate words"
    out = np.lib.format.open_memmap(C.served_path(key), mode="w+", dtype=np.float32, shape=U16.shape)
    max_dev = 0.0
    for s in range(0, len(words), 16384):
        B = np.asarray(U16[s:s + 16384], dtype=np.float32)
        n = np.linalg.norm(B, axis=1, keepdims=True)
        assert (n > 0).all(), f"{key}: zero row"
        max_dev = max(max_dev, float(np.abs(n - 1).max()))
        out[s:s + 16384] = B / n
    out.flush(); del out
    C.served_words(key).write_text("\n".join(words), encoding="utf-8")
    words_of[key] = words
    manifest["models"][key] = dict(name=spec["name"], rows=len(words), dim=spec["dim"], source=str(spec["source"].relative_to(C.REPO)),
                                   source_md5=src_md5, served_md5=md5(C.served_path(key)), max_norm_deviation_f16=max_dev)
    log(f"{key}: {len(words):,} x {spec['dim']}  source md5 {src_md5}  |norm-1| max {max_dev:.2e}")

assert manifest["models"]["fasttext"]["rows"] == 234_474, "fastText table size differs from the thesis (234,474)"
assert manifest["models"]["news"]["rows"] == 187_686 and manifest["models"]["informal"]["rows"] == 32_910

# merged vocabulary must be the union of the three sources ordered by best frequency rank (merge_src.union_words)
rank = {}
for ws in (words_of["news"], words_of["informal"], subs_words):
    for i, w in enumerate(ws):
        if i < rank.get(w, 1 << 60): rank[w] = i
assert sorted(rank, key=rank.get) == words_of["merged"], "merged vocabulary is not the union of its sources"
log("merged vocabulary == union(News-W2V, Informal-W2V, Subs2vec-w2) in thesis order: OK")

# provenance: bit 1 = News row is original, 2 = Informal original, 4 = Subs original (otherwise the block was imputed)
sets = [set(words_of["news"]), set(words_of["informal"]), set(subs_words)]
prov = np.array([sum(1 << k for k, s in enumerate(sets) if w in s) for w in words_of["merged"]], dtype=np.uint8)
np.save(C.DERIVED / "merged.provenance.npy", prov)

# block check: for words with an original News row, merged[:, :300] * sqrt(3) equals the News row (float16 rounding)
M = np.load(C.served_path("merged"), mmap_mode="r"); N = np.load(C.served_path("news"), mmap_mode="r")
nidx = {w: i for i, w in enumerate(words_of["news"])}
sample = [w for w in words_of["merged"][:50_000:97] if w in nidx]
midx = {w: i for i, w in enumerate(words_of["merged"])}
blk = np.asarray(M[[midx[w] for w in sample], :300]) * np.sqrt(3); ref = np.asarray(N[[nidx[w] for w in sample]])
cos = (blk * ref).sum(1) / np.linalg.norm(blk, axis=1) / np.linalg.norm(ref, axis=1)
assert cos.min() > 0.999, f"News block of merged rows does not match News-W2V (min cos {cos.min()})"
log(f"News block check on {len(sample)} words: min cosine {cos.min():.6f}")
manifest["checks"] = dict(union_vocab=True, news_block_min_cos=float(cos.min()), subs_rows=len(subs_words),
                          provenance_counts={"news": int((prov & 1 > 0).sum()), "informal": int((prov & 2 > 0).sum()), "subs": int((prov & 4 > 0).sum())})

# ------------------------------------------------------------------ BanglaMeta-Sim: similarity view of the merged rows
# per 300-d block: all-but-the-top (Mu & Viswanath 2018) with D directions, block re-normalised, then block weight a_k:
#   sim(w) = [ sqrt(a_k / 3) * unit(ABTT_k(block_k(w))) ]_k      (sum a_k = 3, so every row has length 1)
sim_params = {}
for k in range(3):
    bk = slice(300 * k, 300 * (k + 1))
    X = np.asarray(M[:C.SIM_FIT_ROWS, bk], np.float64)
    mu = X.mean(0); _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    for j in range(C.SIM_ABTT_D):
        if Vt[j, np.abs(Vt[j]).argmax()] < 0: Vt[j] *= -1
    sim_params[f"mu{k}"] = mu.astype(np.float32); sim_params[f"P{k}"] = Vt[:C.SIM_ABTT_D].astype(np.float32)
np.savez(C.DERIVED / "merged_sim.params.npz", **sim_params, weights=np.array(C.SIM_WEIGHTS, np.float32))
out = np.lib.format.open_memmap(C.served_path("merged_sim"), mode="w+", dtype=np.float32, shape=M.shape)
for s in range(0, M.shape[0], 16384):
    Bm = np.asarray(M[s:s + 16384], np.float32); o = np.zeros_like(Bm)
    for k in range(3):
        if C.SIM_WEIGHTS[k] == 0: continue
        bk = slice(300 * k, 300 * (k + 1))
        Y = Bm[:, bk] - sim_params[f"mu{k}"]; P = sim_params[f"P{k}"]; Y -= (Y @ P.T) @ P
        o[:, bk] = Y / np.linalg.norm(Y, axis=1, keepdims=True) * np.sqrt(C.SIM_WEIGHTS[k] / 3)
    out[s:s + 16384] = o
out.flush(); del out
C.served_words("merged_sim").write_text("\n".join(words_of["merged"]), encoding="utf-8")
manifest["models"]["merged_sim"] = dict(name=C.MODELS["merged_sim"]["name"], rows=len(words_of["merged"]), dim=900,
                                        source=manifest["models"]["merged"]["source"], source_md5=manifest["models"]["merged"]["source_md5"],
                                        served_md5=md5(C.served_path("merged_sim")), transform=dict(abtt_D=C.SIM_ABTT_D, weights=list(C.SIM_WEIGHTS),
                                        fit_rows=C.SIM_FIT_ROWS, selected_by="experiments/block_weights.py on the analogy TUNE half"))
log("merged_sim written")

# ------------------------------------------------------------------ BanglaMeta+: one model for classification and similarity
#   plus(w) = [ sqrt(eps) * mNI(w) ; sqrt(1 - eps) * sim_news(w) ; sqrt(1 - eps) * sim_subs(w) ]   (1,200-d, length 1)
#   mNI = the thesis News + Informal blocks, re-normalised (Subs2vec adds no significant classification information: P2,
#   and dropping it from the thesis part keeps every score: eval/dimension_study)
Ssim = np.load(C.served_path("merged_sim"), mmap_mode="r")
assert C.SIM_WEIGHTS[1] == 0, "the Informal block of the similarity view is expected to be all zero"
out = np.lib.format.open_memmap(C.served_path("merged_plus"), mode="w+", dtype=np.float32, shape=(M.shape[0], 1200))
a, b = np.float32(np.sqrt(C.PLUS_EPS)), np.float32(np.sqrt(1 - C.PLUS_EPS))
for s in range(0, M.shape[0], 16384):
    sm = np.asarray(Ssim[s:s + 16384])
    mni = np.asarray(M[s:s + 16384, :600]); mni = mni / np.linalg.norm(mni, axis=1, keepdims=True)
    out[s:s + 16384] = np.hstack([a * mni, b * sm[:, :300], b * sm[:, 600:900]])
out.flush(); del out
C.served_words("merged_plus").write_text("\n".join(words_of["merged"]), encoding="utf-8")
manifest["models"]["merged_plus"] = dict(name=C.MODELS["merged_plus"]["name"], rows=len(words_of["merged"]), dim=1200,
                                         source=manifest["models"]["merged"]["source"], source_md5=manifest["models"]["merged"]["source_md5"],
                                         served_md5=md5(C.served_path("merged_plus")), transform=dict(eps=C.PLUS_EPS, parts=["merged News+Informal blocks", "merged_sim News block", "merged_sim Subs block"],
                                         selected_by="experiments/combined.py on the analogy TUNE half"))
log("merged_plus written")

# ------------------------------------------------------------------ fixed PCA basis per model (top 20k words)
for key in C.ORDER:
    U = np.asarray(np.load(C.served_path(key), mmap_mode="r")[:20_000], dtype=np.float64)
    mu = U.mean(0); _, S, Vt = np.linalg.svd(U - mu, full_matrices=False)
    for j in range(3):                      # deterministic sign: largest |loading| positive
        if Vt[j, np.abs(Vt[j]).argmax()] < 0: Vt[j] *= -1
    evr = (S[:3] ** 2) / (S ** 2).sum()
    np.savez(C.DERIVED / f"{key}.pca.npz", mean=mu.astype(np.float32), comps=Vt[:3].astype(np.float32), evr=evr)
    manifest["models"][key]["pca_explained"] = [round(float(x), 4) for x in evr]

C.MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
log("wrote", C.MANIFEST)
