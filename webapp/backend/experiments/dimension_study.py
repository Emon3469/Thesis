"""Over/under-fitting and dimensionality study for BanglaMeta / BanglaMeta+.

1. Under-fitting of each 300-d source: training tokens per learned parameter, share of rare words, and analogy
   accuracy by frequency band of the answer word.
2. Over-fitting of the classifier: train-fold vs held-out-fold macro-F1 gap, and learning curves.
3. Effective dimensionality: participation ratio and dimensions needed for 90% of the variance.
4. Ten dimension variants built from the same frozen rows, all scored the same way:
   DEV and CONF macro-F1 (probe), native analogy (all, TUNE half, TEST half), similarity AUC.
Every variant row is  [ sqrt(w_p) * unit(part_p) ]_p  with sum w_p = 1, so rows have length 1 and
cos = sum_p w_p cos_p.  Parts: thesis row m (900), similarity blocks sN, sI, sS (300 each, all-but-the-top D=4).
Output: eval/dimension_study/results.json
"""
import gzip, json, math, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from app import bench
import gc


class Light:
    """One served model, loaded on demand: memory-mapped rows + its word list (no checksum, no extra indexes)."""
    def __init__(self, key):
        self.key, self.name = key, C.MODELS[key]["name"]
        self.U = np.load(C.served_path(key), mmap_mode="r")
        self.words = C.served_words(key).read_text(encoding="utf-8").split(chr(10))
        self.n = len(self.words); self.n_search = min(C.TOPK_SEARCH, self.n)
    def rows(self, ids): return np.asarray(self.U[np.asarray(ids)], np.float32)

T0 = time.perf_counter()
OUT = C.REPO / "eval" / "dimension_study"; OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(f"[{time.perf_counter() - T0:6.0f}s]", *a, flush=True)
MER = Light("merged"); N = MER.n; IDX = {w: i for i, w in enumerate(MER.words)}
PROV = np.load(C.DERIVED / "merged.provenance.npy")
Mu = MER.U                                                     # thesis rows (unit, 900)
PRM = np.load(C.DERIVED / "merged_sim.params.npz")
EPS = C.PLUS_EPS; aN, aI, aS = 1.65, 0.0, 1.35
TASKS, SEED = bench.TASKS, 42

def parts(B):
    """B: thesis rows (n, 900) -> dict of unit parts m, sN, sI, sS"""
    B = np.asarray(B, np.float32); out = {"m": B}
    for k, nm in enumerate(("sN", "sI", "sS")):
        Y = B[:, 300 * k:300 * (k + 1)] - PRM[f"mu{k}"]; P = PRM[f"P{k}"]; Y = Y - (Y @ P.T) @ P
        out[nm] = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return out

def make(weights, pca=None):
    """variant = row builder from thesis rows"""
    def rows(B):
        pt = parts(B)
        X = np.hstack([np.sqrt(w, dtype=np.float32) * pt[p] for p, w in weights.items() if w > 0])
        if pca is not None:
            mu, V = pca; Z = (X - mu) @ V; X = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return X.astype(np.float32)
    return rows

VARIANTS = {   # name: (weights, description)
    "900 thesis (BanglaMeta)":        ({"m": 1.0}, "the frozen thesis rows"),
    "600 similarity view only":       ({"sN": aN / 3, "sS": aS / 3}, "BanglaMeta-Sim without its empty block"),
    "1200 thesis + sim-News":         ({"m": EPS, "sN": 1 - EPS}, "Subs2vec similarity block dropped"),
    "1200 thesis + sim-Subs":         ({"m": EPS, "sS": 1 - EPS}, "News similarity block dropped"),
    "1200 thesis(News+Inf) + sim":    ({"mNI": EPS, "sN": (1 - EPS) * aN / 3, "sS": (1 - EPS) * aS / 3}, "Subs2vec removed from the thesis part"),
    "1500 BanglaMeta+ (current)":     ({"m": EPS, "sN": (1 - EPS) * aN / 3, "sS": (1 - EPS) * aS / 3}, "current model"),
    "1800 + sim-Informal (0.15)":     ({"m": EPS, "sN": (1 - EPS) * 1.5 / 3, "sI": (1 - EPS) * 0.15 / 3, "sS": (1 - EPS) * 1.35 / 3}, "3 x 300 similarity blocks, Informal weighted 0.15"),
    "1800 + sim equal weights":       ({"m": EPS, "sN": (1 - EPS) / 3, "sI": (1 - EPS) / 3, "sS": (1 - EPS) / 3}, "3 x 300 similarity blocks, equal weights"),
}
_orig_parts = parts
def parts(B):                                                   # add the 600-d thesis News+Informal part
    pt = _orig_parts(B); mNI = pt["m"][:, :600]; pt["mNI"] = mNI / np.linalg.norm(mNI, axis=1, keepdims=True); return pt
BUILD = {k: make(w) for k, (w, _) in VARIANTS.items()}

# PCA-compressed versions of the current 1500-d model (fitted on the 50,000 most frequent words)
cur = BUILD["1500 BanglaMeta+ (current)"]
X50 = np.vstack([cur(Mu[s:s + 5000]) for s in range(0, 30_000, 5000)])
mu50 = X50.mean(0); Xc = X50 - mu50; evals, evecs = np.linalg.eigh((Xc.T @ Xc) / (len(Xc) - 1)); del Xc
order = np.argsort(evals)[::-1]; evecs = evecs[:, order]
for k in (300, 500):
    BUILD[f"{k} PCA of BanglaMeta+"] = make(VARIANTS["1500 BanglaMeta+ (current)"][0], pca=(mu50.astype(np.float32), evecs[:, :k].astype(np.float32)))
    VARIANTS[f"{k} PCA of BanglaMeta+"] = (None, f"BanglaMeta+ compressed to {k} principal directions")
del X50
log("variants:", list(BUILD))

# ------------------------------------------------------------------ data
dev, analogy, cats = bench.load_eval("dev"); conf, _, _ = bench.load_eval("conf")
dev = dev[["task", "label", "domain", "toks"]]; conf = conf[["task", "label", "domain", "toks"]]; gc.collect()
NEED = sorted({w for d in (dev, conf) for t in d.toks for w in t if w in IDX})
needidx = {w: i for i, w in enumerate(NEED)}; need_rows = np.array([IDX[w] for w in NEED])
CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)
from scipy.sparse import csr_matrix
def incidence(df):
    r, c = [], []
    for i, toks in enumerate(df.toks):
        for w in toks:
            if w in needidx: r.append(i); c.append(needidx[w])
    A = csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(df), len(NEED)))
    n = np.asarray(A.sum(1)).ravel(); n[n == 0] = 1
    return A, n
INC = {s: incidence(df) for s, df in (("dev", dev), ("conf", conf))}
PART_DIM = {"m": 900, "sN": 300, "sI": 300, "sS": 300, "mNI": 600}
def text_means(rowfn, dim, tag):
    """mean of rowfn(word rows) over the known words of every text, both splits; cached on disk"""
    f = CACHE / f"{tag}.npz"
    if f.exists(): z = np.load(f); return {"dev": z["dev"], "conf": z["conf"]}
    out = {s: np.zeros((INC[s][0].shape[0], dim), np.float32) for s in INC}
    for a in range(0, len(NEED), 8192):
        rows = rowfn(Mu[need_rows[a:a + 8192]])
        for s in INC: out[s] += INC[s][0][:, a:a + 8192] @ rows
    for s in INC: out[s] /= INC[s][1][:, None]
    np.savez(f, **out); return out
PARTS = {p: text_means(lambda B, p=p: parts(B)[p], d, f"part_{p}") for p, d in PART_DIM.items()}
log("part sentence vectors ready")
def variant_X(name):
    w = VARIANTS[name][0]
    if w is None:                                       # PCA variant: rows are not a concatenation of parts
        k = int(name.split()[0]); return text_means(BUILD[name], k, f"pca{k}")
    return {s: np.hstack([PARTS[p][s] for p, wt in w.items() if wt > 0]) for s in ("dev", "conf")}
TASK_ROWS = {s: {t: np.flatnonzero((df.task == t).values) for t in TASKS} for s, df in (("dev", dev), ("conf", conf))}
def probe(X, y, folds, frac=1.0, rng=None):
    """out-of-fold predictions + mean train-fold F1 (for the over-fitting gap)"""
    p = np.full(len(y), -1); tr_f1 = []; alphas = []
    for tr, te in folds:
        if frac < 1: tr = np.sort(rng.choice(tr, int(len(tr) * frac), replace=False))
        sc = StandardScaler().fit(X[tr]); clf = RidgeClassifierCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr])
        p[te] = clf.predict(sc.transform(X[te])); tr_f1.append(f1_score(y[tr], clf.predict(sc.transform(X[tr])), average="macro")); alphas.append(clf.alpha_)
    return p, float(np.mean(tr_f1)), float(np.median(alphas))
FOLDS = {s: {t: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((df.task == t).sum()), df.label[df.task == t])) for t in TASKS}
         for s, df in (("dev", dev), ("conf", conf))}

def classify(XS, split, frac=1.0):
    """XS = {split: text features}; standardisation makes per-part scale irrelevant (thesis Theorem 5)"""
    df = dev if split == "dev" else conf; rng = np.random.default_rng(SEED); f1, tr, al = {}, {}, {}
    for t in TASKS:
        y = df.label.values[TASK_ROWS[split][t]].astype(int)
        p, trf, a = probe(XS[split][TASK_ROWS[split][t]], y, FOLDS[split][t], frac, rng)
        f1[t] = f1_score(y, p, average="macro"); tr[t] = trf; al[t] = a
    agg = lambda d: (d["news_topic_7"] + np.mean([d[t] for t in TASKS[1:]])) / 2
    return {"Overall": agg(f1), "D1": f1["news_topic_7"], "D2": float(np.mean([f1[t] for t in TASKS[1:]])), "train_Overall": agg(tr),
            "gap": agg(tr) - agg(f1), "alpha": al, "per_task": f1}

# analogy questions + the tune/test split used for every selection (block_weights.py)
sidx = {w: i for i, w in enumerate(MER.words[:MER.n_search])}
rngq = np.random.default_rng(42); qs, half, ansrank = [], [], []
nat = [f for f in analogy if f != "Mikolov-Filtered"]; n_q = {"tune": 0, "test": 0}
for f in nat:
    Q = analogy[f]; perm = rngq.permutation(len(Q)); tune = set(perm[: len(Q) // 2].tolist())
    n_q["tune"] += len(Q) // 2; n_q["test"] += len(Q) - len(Q) // 2
    for j, q in enumerate(Q):
        if all(w in sidx for w in q):
            qs.append([sidx[w] for w in q]); half.append("tune" if j in tune else "test")
QB = np.array(qs); HALF = np.array(half); NQ = n_q["tune"] + n_q["test"]
def analogy_ok(build):
    T = build(Mu[QB[:, 1]]) - build(Mu[QB[:, 0]]) + build(Mu[QB[:, 2]])
    bs = np.full(len(QB), -np.inf, np.float32); bi = np.zeros(len(QB), np.int64)
    for r in range(0, MER.n_search, 16384):
        e = min(r + 16384, MER.n_search); Cm = build(Mu[r:e])
        for s in range(0, len(QB), 512):
            sl = slice(s, s + 512); sim = T[sl] @ Cm.T; rr = np.arange(sim.shape[0])
            for j in (0, 1, 2):
                col = QB[sl, j]; inside = (col >= r) & (col < e); sim[rr[inside], col[inside] - r] = -np.inf
            am = sim.argmax(1); mv = sim[rr, am]; upd = mv > bs[sl]; bs[sl][upd] = mv[upd]; bi[sl][upd] = am[upd] + r
    return bi == QB[:, 3]
_vocab = {k: set(C.served_words(k).read_text(encoding="utf-8").split(chr(10))) for k in ("fasttext", "news", "informal")}
common = [(w, c) for c, ws in cats.items() for w in ws if w in IDX and all(w in v for v in _vocab.values()) and PROV[IDX[w]] & 4]
del _vocab; gc.collect()
cidx = np.array([IDX[w] for w, _ in common]); ccat = np.array([c for _, c in common]); iu = np.triu_indices(len(common), 1)
def sim_auc(build):
    U = build(Mu[cidx]); return float(roc_auc_score((ccat[iu[0]] == ccat[iu[1]]).astype(int), (U @ U.T)[iu]))

# ------------------------------------------------------------------ 4. variants
res = {"variants": {}}
for name, build in BUILD.items():
    t = time.perf_counter(); XS = variant_X(name)
    d, c = classify(XS, "dev"), classify(XS, "conf"); ok = analogy_ok(build)
    X = np.vstack([build(Mu[s:s + 5000]) for s in range(0, 20_000, 5000)]); Xc = X - X.mean(0)
    ev = np.clip(np.linalg.eigvalsh((Xc.T @ Xc) / (len(Xc) - 1))[::-1], 0, None); cs = np.cumsum(ev) / ev.sum(); del X, Xc
    res["variants"][name] = {"dim": XS["dev"].shape[1], "about": VARIANTS[name][1], "dev": d, "conf": c,
                             "analogy_all": int(ok.sum()) / NQ, "analogy_tune": int(ok[HALF == "tune"].sum()) / n_q["tune"],
                             "analogy_test": int(ok[HALF == "test"].sum()) / n_q["test"], "sim_auc": sim_auc(build),
                             "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()), "dims_90pct": int(np.searchsorted(cs, 0.9) + 1)}
    v = res["variants"][name]
    log(f"{name:<30} dim {v['dim']:>4}  DEV {d['Overall']*100:.2f} (train {d['train_Overall']*100:.2f}, gap {d['gap']*100:.2f})  CONF {c['Overall']*100:.2f}  "
        f"analogy {v['analogy_all']*100:.2f} (test half {v['analogy_test']*100:.2f})  AUC {v['sim_auc']:.4f}  PR {v['participation_ratio']:.0f}  "
        f"90% in {v['dims_90pct']}  {time.perf_counter()-t:.0f}s")
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ 2b. learning curves (DEV)
res["learning_curves"] = {}
for name in ("600 similarity view only", "900 thesis (BanglaMeta)", "1500 BanglaMeta+ (current)", "1800 + sim equal weights"):
    XS = variant_X(name)
    res["learning_curves"][name] = {str(fr): (lambda r: {"Overall": r["Overall"], "train_Overall": r["train_Overall"]})(classify(XS, "dev", fr))
                                    for fr in (0.1, 0.25, 0.5, 0.75, 1.0)}
    log("learning curve", name, {k: round(v["Overall"] * 100, 2) for k, v in res["learning_curves"][name].items()})
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

# ------------------------------------------------------------------ 1. under-fitting of each 300-d source
def corpus_counts(p):
    c = Counter()
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f: c.update(line.split())
    return c
src = {}
for nm, fn, key in (("Informal-W2V", "informal_corpus.txt.gz", "informal"), ("News-W2V", "news_corpus.txt.gz", "news")):
    cnt = corpus_counts(C.EVAL_DATA / fn); m = Light(key); V = m.n
    freqs = np.array([cnt.get(w, 0) for w in m.words]); tokens = sum(cnt.values())
    src[nm] = {"corpus_tokens": int(tokens), "vocab": V, "dims": 300, "learned_parameters": 2 * V * 300,
               "tokens_per_parameter": tokens / (2 * V * 300), "occurrences_per_word_median": float(np.median(freqs)),
               "share_words_lt_20": float((freqs < 20).mean()), "share_words_lt_50": float((freqs < 50).mean()),
               "share_tokens_by_words_lt_20": float(freqs[freqs < 20].sum() / tokens)}
    log(nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in src[nm].items()})
    del cnt, m; gc.collect()
src["Subs2vec-w2"] = {"corpus_tokens": None, "vocab": 177_376, "dims": 300, "note": "downloaded (OpenSubtitles 2018); corpus size not published in the record"}
res["sources"] = src

# analogy accuracy by frequency band of the answer word (merged rank = frequency order)
bands = [(0, 5_000), (5_000, 20_000), (20_000, 50_000), (50_000, 200_000)]
res["frequency_bands"] = {}
for key in ("news", "informal", "fasttext", "merged", "merged_sim", "merged_plus"):
    m = Light(key); sid = {w: i for i, w in enumerate(m.words[:m.n_search])}; rows = []
    for f in nat:
        for q in analogy[f]:
            rows.append((all(w in sid for w in q), IDX.get(q[3], 10 ** 9)))
    ok_full = np.zeros(len(rows), bool); can = np.array([r[0] for r in rows]); rk = np.array([r[1] for r in rows])
    allq = [q for f in nat for q in analogy[f]]
    B = np.array([[sid[w] for w in q] for q, c_ in zip(allq, can) if c_])
    T = m.rows(B[:, 1]) - m.rows(B[:, 0]) + m.rows(B[:, 2]); bs = np.full(len(B), -np.inf, np.float32); bi = np.zeros(len(B), np.int64)
    for r in range(0, m.n_search, 16384):
        e = min(r + 16384, m.n_search); Cm = np.asarray(m.U[r:e])
        for s in range(0, len(B), 512):
            sl = slice(s, s + 512); sim = T[sl] @ Cm.T; rr = np.arange(sim.shape[0])
            for j in (0, 1, 2):
                col = B[sl, j]; inside = (col >= r) & (col < e); sim[rr[inside], col[inside] - r] = -np.inf
            am = sim.argmax(1); mv = sim[rr, am]; upd = mv > bs[sl]; bs[sl][upd] = mv[upd]; bi[sl][upd] = am[upd] + r
    ok_full[can] = bi == B[:, 3]
    res["frequency_bands"][m.name] = {f"{a // 1000}k-{b // 1000}k": float(ok_full[(rk >= a) & (rk < b)].mean()) for a, b in bands}
    res["frequency_bands"][m.name]["questions"] = {f"{a // 1000}k-{b // 1000}k": int(((rk >= a) & (rk < b)).sum()) for a, b in bands}
    log("bands", m.name, res["frequency_bands"][m.name]); del m, sid, T; gc.collect()

# effective dimensionality of each 300-d source
res["source_effective_dims"] = {}
for key in ("news", "informal", "fasttext"):
    m = Light(key); X = np.asarray(m.U[:min(20_000, m.n)], np.float32); Xc = X - X.mean(0)
    ev = np.clip(np.linalg.eigvalsh((Xc.T @ Xc) / (len(Xc) - 1))[::-1], 0, None); cs = np.cumsum(ev) / ev.sum()
    res["source_effective_dims"][m.name] = {"participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()), "dims_90pct": int(np.searchsorted(cs, 0.9) + 1)}
res["notes"] = {"conf": "CONF for new variants is exploratory", "selection": "D, block weights and eps were chosen on the analogy TUNE half"}
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
log("done")
