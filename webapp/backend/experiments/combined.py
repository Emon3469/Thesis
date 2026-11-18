"""One model for both jobs: c(w) = [ sqrt(eps) * m(w) ; sqrt(1 - eps) * s(w) ]   (1,800-d, unit rows)

m = frozen thesis rows (BanglaMeta), s = BanglaMeta-Sim rows. Cosine: cos_c = eps * cos_m + (1 - eps) * cos_s.
The linear probe standardises each feature, so its scores are the same for every eps > 0 (thesis Theorem 4):
they are measured once on [mean m ; mean s]. eps is chosen on the analogy TUNE half only (same split as
block_weights.py); the TEST half and the similarity AUC are reported for the chosen eps.
Output: data/combined.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from app import bench
from app.store import Registry

T0 = time.perf_counter()
R = Registry(verify=False); M, S = R.get("merged"), R.get("merged_sim")
dev, analogy, cats = bench.load_eval()
EPS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
SKIP_CLS = "--skip-classification" in sys.argv          # eps-invariant: reuse the stored values

# ------------------------------------------------------------------ classification (eps-invariant)
def probe(X, y, folds):
    return f1_score(y, bench.probe_oof(X, y, folds), average="macro")
f1 = {}
if SKIP_CLS:
    f1 = {k[3:]: v for k, v in json.loads((C.DATA / "combined.json").read_text(encoding="utf-8"))["classification_dev"].items() if k.startswith("F1_")}
for t in ([] if SKIP_CLS else bench.TASKS):
    g = dev[dev.task == t]; y = g.label.values
    Xm, _ = bench.sent_matrix(M, g.toks.tolist()); Xs, _ = bench.sent_matrix(S, g.toks.tolist())
    Xs = Xs[:, np.abs(Xs).sum(0) > 0]                          # drop the all-zero (weight 0) Informal block
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(g)), y))
    f1[t] = probe(np.hstack([Xm, Xs]), y, folds)
D1 = f1["news_topic_7"]; D2 = float(np.mean([f1[t] for t in bench.TASKS[1:]]))
cls = {"D1_F1": D1, "D2_F1": D2, "Overall_F1": (D1 + D2) / 2, **{f"F1_{t}": v for t, v in f1.items()}}
print("classification", {k: round(v, 4) for k, v in cls.items()}, f"{time.perf_counter()-T0:.0f}s")

# ------------------------------------------------------------------ analogy for every eps in one pass
sidx = {w: i for i, w in enumerate(M.words[:M.n_search])}
rng = np.random.default_rng(42)
qs, half, files, n_total = [], [], [], {}
for f, Q in analogy.items():
    if f == "Mikolov-Filtered": continue
    n_total[f] = len(Q); perm = rng.permutation(len(Q)); tune = set(perm[: len(Q) // 2].tolist())
    for j, q in enumerate(Q):
        if all(w in sidx for w in q):
            qs.append([sidx[w] for w in q]); half.append("tune" if j in tune else "test"); files.append(f)
B = np.array(qs); half = np.array(half); files = np.array(files)
n_q = {"tune": sum(n // 2 for n in n_total.values()), "test": sum(n - n // 2 for n in n_total.values())}
Tm = M.rows(B[:, 1]) - M.rows(B[:, 0]) + M.rows(B[:, 2]); Ts = S.rows(B[:, 1]) - S.rows(B[:, 0]) + S.rows(B[:, 2])
best_s = np.full((len(EPS), len(B)), -np.inf, np.float32); best_i = np.zeros((len(EPS), len(B)), np.int64)
for r in range(0, M.n_search, 32768):
    e = min(r + 32768, M.n_search); Cm = np.asarray(M.U[r:e]); Cs = np.asarray(S.U[r:e])
    for s0 in range(0, len(B), 1024):
        sl = slice(s0, s0 + 1024); sm = Tm[sl] @ Cm.T; ss = Ts[sl] @ Cs.T; rows = np.arange(sm.shape[0])
        for k, eps in enumerate(EPS):
            sim = eps * sm + (1 - eps) * ss
            for j in (0, 1, 2):
                col = B[sl, j]; inside = (col >= r) & (col < e); sim[rows[inside], col[inside] - r] = -np.inf
            am = sim.argmax(1); mv = sim[rows, am]; upd = mv > best_s[k, sl]
            best_s[k, sl][upd] = mv[upd]; best_i[k, sl][upd] = am[upd] + r
    print(f"rows {e:,} {time.perf_counter()-T0:.0f}s", flush=True)

common = [(w, c) for c, ws in cats.items() for w in ws if all(m.has(w) for m in R.models.values()) and M.provenance_of(w)["subs"]]
ids = [M.idx[w] for w, _ in common]; cs = np.array([c for _, c in common]); iu = np.triu_indices(len(common), 1)
Gm = M.rows(ids) @ M.rows(ids).T; Gs = S.rows(ids) @ S.rows(ids).T
rows_out = []
for k, eps in enumerate(EPS):
    ok = best_i[k] == B[:, 3]
    rows_out.append({"eps": eps, "tune": int(ok[half == "tune"].sum()) / n_q["tune"], "test": int(ok[half == "test"].sum()) / n_q["test"],
                     "all": int(ok.sum()) / (n_q["tune"] + n_q["test"]),
                     "semantic": int(ok[np.isin(files, bench.SEMANTIC)].sum()) / sum(n_total[f] for f in bench.SEMANTIC),
                     "sim_auc": float(roc_auc_score((cs[iu[0]] == cs[iu[1]]).astype(int), (eps * Gm + (1 - eps) * Gs)[iu]))})
chosen = max((r for r in rows_out if r["eps"] > 0), key=lambda r: (r["tune"], -r["eps"]))  # eps > 0 keeps the thesis features
out = {"protocol": __doc__.strip().splitlines()[0], "classification_dev": cls, "eps_grid": rows_out, "chosen_eps": chosen["eps"], "chosen": chosen}
for r in rows_out: print({k: round(v, 4) for k, v in r.items()})
print("chosen", chosen)
(C.DATA / "combined.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"done {time.perf_counter()-T0:.0f}s")
