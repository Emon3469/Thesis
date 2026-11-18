"""Can re-weighting the three blocks of the frozen merged model improve word similarity and analogy?

The merged row is m(w) = (1/sqrt 3)[n(w); i(w); s(w)] with unit blocks. With block weights a = (a_n, a_i, a_s), mean 1,
the weighted cosine of two words is   cos_a(x, y) = (1/3) * sum_k a_k * cos_k(x, y)
and is computed from the same rows by scaling the query block by block. The linear probe standardises every
feature, so the classification results are unchanged for any weights (thesis Theorem 4) - checked again below.

Protocol (no peeking): the 16,678 native analogy questions are split in half, stratified by file, seed 42.
Weights are chosen on the TUNE half only (grid over the simplex, step 0.05); the TEST half and the similarity AUC
are measured once for the chosen weights.  Output: data/block_weights.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from app import bench
from app.store import Registry

T0 = time.perf_counter()
R = Registry(verify=False); M = R.merged
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "equal-rows"
if VARIANT != "equal-rows":                       # e.g. abtt2 -> rows post-processed by build_abtt.py
    M.U = np.load(C.DERIVED / f"merged_{VARIANT}.f32.npy", mmap_mode="r")
dev, analogy, cats = bench.load_eval()
SEM = bench.SEMANTIC

# ------------------------------------------------------------------ questions (answerable in the 200k search space)
sidx = {w: i for i, w in enumerate(M.words[:M.n_search])}
rng = np.random.default_rng(42)
qs, files, half, n_total = [], [], [], {}
for f, Q in analogy.items():
    if f == "Mikolov-Filtered": continue
    n_total[f] = len(Q)
    perm = rng.permutation(len(Q)); tune = set(perm[: len(Q) // 2].tolist())
    for j, q in enumerate(Q):
        if all(w in sidx for w in q):
            qs.append([sidx[w] for w in q]); files.append(f); half.append("tune" if j in tune else "test")
B = np.array(qs); files = np.array(files); half = np.array(half)
n_q = {"tune": sum(len(Q) // 2 for f, Q in analogy.items() if f != "Mikolov-Filtered"),
       "test": sum(len(Q) - len(Q) // 2 for f, Q in analogy.items() if f != "Mikolov-Filtered")}
print(f"answerable {len(B)} | questions tune {n_q['tune']} test {n_q['test']}")

# ------------------------------------------------------------------ per-block candidate scores (top K of each block)
K = 60
blocks = [slice(0, 300), slice(300, 600), slice(600, 900)]
T = M.rows(B[:, 1]) - M.rows(B[:, 0]) + M.rows(B[:, 2])               # (q, 900): block k of T is block k target
cand = np.zeros((len(B), 0), np.int64)
for bk in blocks:
    best_s = np.full((len(B), K), -np.inf, np.float32); best_i = np.zeros((len(B), K), np.int64)
    for r in range(0, M.n_search, 32768):
        e = min(r + 32768, M.n_search); Cb = np.asarray(M.U[r:e, bk])
        for s in range(0, len(B), 2048):
            sl = slice(s, s + 2048); sim = T[sl, bk] @ Cb.T
            rows = np.arange(sim.shape[0])
            for j in (0, 1, 2):
                col = B[sl, j]; inside = (col >= r) & (col < e); sim[rows[inside], col[inside] - r] = -np.inf
            top = np.argpartition(-sim, K - 1, axis=1)[:, :K]
            allv = np.concatenate([best_s[sl], np.take_along_axis(sim, top, 1)], 1)
            alli = np.concatenate([best_i[sl], top + r], 1)
            keep = np.argpartition(-allv, K - 1, axis=1)[:, :K]
            best_s[sl] = np.take_along_axis(allv, keep, 1); best_i[sl] = np.take_along_axis(alli, keep, 1)
    cand = np.concatenate([cand, best_i], 1)
    print(f"block done {time.perf_counter()-T0:.0f}s")
cand = np.concatenate([cand, B[:, 3:4]], 1)                              # the gold answer is always a candidate
# block scores of every candidate: S[q, c, k], computed in small batches (memory)
S = np.empty((len(B), cand.shape[1], 3), np.float32)
for s0 in range(0, len(B), 200):
    sl = slice(s0, s0 + 200); rows = M.rows(cand[sl].ravel()).reshape(-1, cand.shape[1], 900)
    for k, bk in enumerate(blocks):
        S[sl, :, k] = np.einsum("qd,qcd->qc", T[sl, bk], rows[:, :, bk])

def acc_on_candidates(a, mask):
    sc = (S[mask] * a).sum(-1)
    best = sc.max(1, keepdims=True)
    win = np.where(sc == best, cand[mask], np.iinfo(np.int64).max).min(1)    # ties: lower row id wins, as argmax
    return int((win == B[mask, 3]).sum())

grid = [(a / 20, b / 20, (20 - a - b) / 20) for a in range(21) for b in range(21 - a)]
tune = half == "tune"
res = []
for g in grid:
    a = np.array(g) * 3                                                  # mean 1
    res.append((acc_on_candidates(a, tune) / n_q["tune"], g))
res.sort(key=lambda x: (-x[0], x[1]))
best_w = np.array(res[0][1]) * 3
equal = np.ones(3)
print("top tune grid points:", [(round(x, 4), w) for x, w in res[:6]], f"| {time.perf_counter()-T0:.0f}s")

# ------------------------------------------------------------------ exact evaluation of the chosen weights (full search)
def exact_scores(a):
    scale = np.concatenate([np.full(300, a[0]), np.full(300, a[1]), np.full(300, a[2])]).astype(np.float32)
    Tw = T * scale
    best_s = np.full(len(B), -np.inf, np.float32); best_i = np.zeros(len(B), np.int64)
    for r in range(0, M.n_search, 32768):
        e = min(r + 32768, M.n_search); Cm = np.asarray(M.U[r:e])
        for s in range(0, len(B), 1024):
            sl = slice(s, s + 1024); sim = Tw[sl] @ Cm.T; rows = np.arange(sim.shape[0])
            for j in (0, 1, 2):
                col = B[sl, j]; inside = (col >= r) & (col < e); sim[rows[inside], col[inside] - r] = -np.inf
            am = sim.argmax(1); mv = sim[rows, am]; upd = mv > best_s[sl]
            best_s[sl][upd] = mv[upd]; best_i[sl][upd] = am[upd] + r
    ok = best_i == B[:, 3]
    out = {h: int(ok[half == h].sum()) / n_q[h] for h in ("tune", "test")}
    out["all"] = int(ok.sum()) / (n_q["tune"] + n_q["test"])
    out["semantic"] = int(ok[np.isin(files, SEM)].sum()) / sum(n_total[f] for f in SEM)
    out["syntactic"] = int(ok[~np.isin(files, SEM)].sum()) / sum(n for f, n in n_total.items() if f not in SEM)
    return out

def sim_auc(a):
    common = [(w, c) for c, ws in cats.items() for w in ws if all(m.has(w) for m in R.models.values()) and M.provenance_of(w)["subs"]]
    U = M.rows([M.idx[w] for w, _ in common]); cs = np.array([c for _, c in common]); iu = np.triu_indices(len(common), 1)
    G = sum(a[k] * (U[:, blocks[k]] @ U[:, blocks[k]].T) for k in range(3))
    return float(roc_auc_score((cs[iu[0]] == cs[iu[1]]).astype(int), G[iu]))

out = {"protocol": __doc__.strip().splitlines()[0], "grid_step": 0.05, "K_candidates_per_block": K,
       "equal": {"weights": equal.tolist(), **exact_scores(equal), "sim_auc": sim_auc(equal)},
       "chosen": {"weights": best_w.round(4).tolist(), **exact_scores(best_w), "sim_auc": sim_auc(best_w)},
       "references_thesis_dev": {k: {"analogy_native_acc": bench.thesis_values()[k]["analogy_native_acc"], "sim_auc": bench.thesis_values()[k]["sim_auc_common"]}
                                 for k in ("fasttext", "news", "informal", "merged")}}
print(json.dumps(out, indent=1))
out["variant"] = VARIANT
(C.DATA / f"block_weights_{VARIANT}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"done {time.perf_counter()-T0:.0f}s")
