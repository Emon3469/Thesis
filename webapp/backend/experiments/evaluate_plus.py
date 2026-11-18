"""Full evaluation of BanglaMeta+ against every other model, with uncertainty.

Everything is measured from the served model rows with the thesis definitions:
  * classification: linear probe (StandardScaler + RidgeClassifierCV), 5 stratified folds (seed 42), on DEV and on the held-out
    CONF half; 95% CIs from the thesis's own paired bootstrap (CONF: 10,000 resamples, seed 42, exactly as conf_src.py;
    DEV: 1,000, as merge_src.py). The script first checks that it reproduces the official CONF intervals of the thesis models.
  * word analogy: 3CosAdd over the first 200k rows (per question, so paired McNemar tests are possible)
  * word similarity: category AUC with a word-level bootstrap CI
  * coverage of DEV tokens
CONF results for BanglaMeta+ and BanglaMeta-Sim are EXPLORATORY: they were built after the thesis's one confirmatory CONF test.
Output: eval/banglameta_plus_eval/results.json, metrics_matrix.csv, predictions.npz
"""
import json, math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from app import bench
from app.store import Registry

T0 = time.perf_counter()
OUT = C.REPO / "eval" / "banglameta_plus_eval"; OUT.mkdir(parents=True, exist_ok=True)
R = Registry(verify=True)
KEYS = C.ORDER                                   # merged_plus, merged, merged_sim, fasttext, news, informal
NAMES = {k: R.get(k).name for k in KEYS}
TASKS, D2T, SEED = bench.TASKS, bench.TASKS[1:], 42
log = lambda *a: print(f"[{time.perf_counter() - T0:5.0f}s]", *a, flush=True)

def macro_f1_rows(y, p, idx, ncls):
    """Macro-F1 for each bootstrap row (identical to conf_src.macro_f1_rows)."""
    out = np.zeros(len(idx))
    for s in range(0, len(idx), 1000):
        yi, pi = y[idx[s:s + 1000]], p[idx[s:s + 1000]]; f = np.zeros((len(yi), ncls))
        for c in range(ncls):
            tp = ((yi == c) & (pi == c)).sum(1); den = (yi == c).sum(1) + (pi == c).sum(1)
            f[:, c] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0)
        out[s:s + 1000] = f.mean(1)
    return out

# ------------------------------------------------------------------ classification on DEV and CONF
res = {k: {"name": NAMES[k], "dim": R.get(k).dim, "rows": R.get(k).n} for k in KEYS}
preds, labels, boots = {}, {}, {}
for split, nb in (("dev", 1000), ("conf", 10_000)):
    df, _, _ = bench.load_eval(split)
    folds = {t: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((df.task == t).sum()), df.label[df.task == t])) for t in TASKS}
    rng = np.random.default_rng(SEED)
    BOOT = {t: rng.integers(0, (df.task == t).sum(), size=(nb, (df.task == t).sum())) for t in TASKS}
    for t in TASKS:
        labels[(split, t)] = df[df.task == t].label.values.astype(int)
    for k in KEYS:
        m = R.get(k); bf = {}
        for t in TASKS:
            g = df[df.task == t]; y = labels[(split, t)]; X, _ = bench.sent_matrix(m, g.toks.tolist())
            p = bench.probe_oof(X, y, folds[t]); ncls = int(y.max()) + 1
            preds[(k, split, t)] = p
            res[k][f"{split}_F1_{t}"] = float(f1_score(y, p, average="macro", labels=list(range(ncls)), zero_division=0))
            res[k][f"{split}_perclass_{t}"] = f1_score(y, p, average=None, labels=list(range(ncls)), zero_division=0).tolist()
            res[k][f"{split}_acc_{t}"] = float((y == p).mean())
            bf[t] = macro_f1_rows(y, p, BOOT[t], ncls)
        d1, d2 = bf["news_topic_7"], np.mean([bf[t] for t in D2T], 0); ov = (d1 + d2) / 2
        boots[(k, split)] = {"D1": d1, "D2": d2, "Overall": ov}
        r = res[k]
        r[f"{split}_D1_F1"] = r[f"{split}_F1_news_topic_7"]; r[f"{split}_D2_F1"] = float(np.mean([r[f"{split}_F1_{t}"] for t in D2T]))
        r[f"{split}_Overall_F1"] = (r[f"{split}_D1_F1"] + r[f"{split}_D2_F1"]) / 2
        for kk, b in boots[(k, split)].items():
            r[f"{split}_{kk}_lo"], r[f"{split}_{kk}_hi"] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        log(f"{split} {NAMES[k]:<15} Overall {r[f'{split}_Overall_F1']:.4f} [{r[f'{split}_Overall_lo']:.4f}, {r[f'{split}_Overall_hi']:.4f}]")

# the bootstrap must reproduce the official thesis CONF intervals (same method, same resamples)
official = pd.read_csv(C.EVAL / "conf_output" / "conf_results" / "conf_results.csv").set_index("model")
for k, nm in bench.THESIS_NAME.items():
    for kk in ("Overall_F1", "Overall_lo", "Overall_hi", "D1_F1", "D2_F1"):
        mine = res[k]["conf_" + kk]
        assert abs(mine - official.loc[nm, kk]) < 1e-9, (k, kk, mine, official.loc[nm, kk])
log("CONF point estimates and 95% intervals reproduce the official thesis table exactly")

# paired differences BanglaMeta+ minus every other model (Overall, D1, D2), DEV and CONF
deltas = []
for split, nb in (("dev", 1000), ("conf", 10_000)):
    for b in KEYS[1:]:
        for kk in ("Overall", "D1", "D2"):
            d = boots[("merged_plus", split)][kk] - boots[(b, split)][kk]
            obs = res["merged_plus"][f"{split}_{kk}_F1"] - res[b][f"{split}_{kk}_F1"]
            deltas.append(dict(split=split, vs=b, metric=kk, delta=obs, ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)),
                               p_one_sided=(1 + int((d <= 0).sum())) / (nb + 1)))
    stack = np.stack([boots[(k, split)]["Overall"] for k in KEYS]); best = stack.argmax(0)
    for i, k in enumerate(KEYS): res[k][f"{split}_P_best"] = float((best == i).mean())
log("paired bootstrap done")

# ------------------------------------------------------------------ word analogy, per question
_, analogy, cats = bench.load_eval("dev")
native_files = [f for f in analogy if f != "Mikolov-Filtered"]
def analogy_ok(m):
    sidx = {w: i for i, w in enumerate(m.words[:m.n_search])}
    qs, ans = [], []
    for f in analogy:
        for q in analogy[f]:
            ans.append(all(w in sidx for w in q)); qs.append([sidx[w] for w in q] if ans[-1] else [0, 0, 0, 0])
    Q = np.array(qs); can = np.array(ans); B = Q[can]
    T = m.rows(B[:, 1]) - m.rows(B[:, 0]) + m.rows(B[:, 2])
    best_s = np.full(len(B), -np.inf, np.float32); best_i = np.zeros(len(B), np.int64)
    for r in range(0, m.n_search, 32768):
        e = min(r + 32768, m.n_search); Cm = np.asarray(m.U[r:e])
        for s in range(0, len(B), 1024):
            sl = slice(s, s + 1024); sim = T[sl] @ Cm.T; rows = np.arange(sim.shape[0])
            for j in (0, 1, 2):
                col = B[sl, j]; inside = (col >= r) & (col < e); sim[rows[inside], col[inside] - r] = -np.inf
            am = sim.argmax(1); mv = sim[rows, am]; upd = mv > best_s[sl]
            best_s[sl][upd] = mv[upd]; best_i[sl][upd] = am[upd] + r
    ok = np.zeros(len(Q), bool); ok[can] = best_i == B[:, 3]
    return ok, can
qfile = np.array([f for f in analogy for _ in analogy[f]])
OK = {}
for k in KEYS:
    OK[k], can = analogy_ok(R.get(k))
    nat = np.isin(qfile, native_files)
    res[k]["analogy_native_acc"] = float(OK[k][nat].mean()); res[k]["analogy_answerable"] = float(can[nat].mean())
    res[k]["analogy_per_category"] = {f: float(OK[k][qfile == f].mean()) for f in analogy}
    res[k]["analogy_semantic_acc"] = float(OK[k][np.isin(qfile, bench.SEMANTIC)].mean())
    res[k]["analogy_syntactic_acc"] = float(OK[k][nat & ~np.isin(qfile, bench.SEMANTIC)].mean())
    res[k]["analogy_mikolov_acc"] = float(OK[k][qfile == "Mikolov-Filtered"].mean())
    rng = np.random.default_rng(SEED); idx = np.flatnonzero(nat)
    sample = idx[rng.integers(0, len(idx), size=(1000, len(idx)))]       # resample the native questions
    bs = OK[k][sample].mean(1)
    res[k]["analogy_lo"], res[k]["analogy_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    log(f"analogy {NAMES[k]:<15} {res[k]['analogy_native_acc']:.4f}")
assert abs(res["merged"]["analogy_native_acc"] - bench.thesis_values()["merged"]["analogy_native_acc"]) < 1e-9
mcnemar = []
nat = np.isin(qfile, native_files)
for b in KEYS[1:]:
    a_only = int((OK["merged_plus"] & ~OK[b] & nat).sum()); b_only = int((~OK["merged_plus"] & OK[b] & nat).sum())
    p = binomtest(a_only, a_only + b_only, 0.5, alternative="greater").pvalue if a_only + b_only else 1.0
    mcnemar.append(dict(vs=b, plus_only=a_only, other_only=b_only, p_one_sided=float(p)))

# ------------------------------------------------------------------ word similarity: AUC with word-level bootstrap, and score distributions
common = [(w, c) for c, ws in cats.items() for w in ws if all(R.get(k).has(w) for k in KEYS) and R.merged.provenance_of(w)["subs"]]
cs = np.array([c for _, c in common]); nW = len(common); iu = np.triu_indices(nW, 1); same = (cs[iu[0]] == cs[iu[1]])
sims_dist = {}
rng = np.random.default_rng(SEED); BW = [rng.integers(0, nW, nW) for _ in range(1000)]
for k in KEYS:
    m = R.get(k); U = m.rows([m.idx[w] for w, _ in common]); G = U @ U.T
    res[k]["sim_auc"] = float(roc_auc_score(same.astype(int), G[iu]))
    vals = []
    for bi in BW:
        Gb = G[np.ix_(bi, bi)]; cb = cs[bi]; ju = np.triu_indices(nW, 1); keep = bi[ju[0]] != bi[ju[1]]
        yb = (cb[ju[0]] == cb[ju[1]])[keep]; sb = Gb[ju][keep]
        if 0 < yb.sum() < len(yb): vals.append(roc_auc_score(yb.astype(int), sb))
    res[k]["sim_auc_lo"], res[k]["sim_auc_hi"] = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    sims_dist[k] = {"same": G[iu][same].tolist(), "diff": G[iu][~same].tolist()}
res["_common_words"] = nW

# ------------------------------------------------------------------ coverage (from the benchmark cache; vocabulary property)
bc = json.loads(C.BENCH_CACHE.read_text(encoding="utf-8"))["results"]
for k in KEYS:
    for kk in ("tok_cov_D1", "tok_cov_D2", "type_cov_D1", "type_cov_D2"):
        res[k][kk] = bc[k][kk]
    assert abs(bc[k]["Overall_F1"] - res[k]["dev_Overall_F1"]) < 1e-9, k   # same probe, same numbers as the app

# ------------------------------------------------------------------ confusion matrices of BanglaMeta+ on CONF
conf_mats = {t: confusion_matrix(labels[("conf", t)], preds[("merged_plus", "conf", t)]).tolist() for t in TASKS}

# ------------------------------------------------------------------ write
sel = {"abtt": {d: json.loads((C.DATA / f"block_weights_abtt{d}.json").read_text(encoding="utf-8")) for d in range(1, 6)},
       "eps": (lambda g: {"eps_grid": g["grid"], "chosen_eps": g["chosen"]["eps"]})(
           json.loads((C.REPO / "eval" / "dimension_study" / "eps_1200.json").read_text(encoding="utf-8")))}
out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "models": KEYS, "results": res, "deltas": deltas, "mcnemar": mcnemar,
       "confusion_conf_plus": conf_mats, "similarity_scores": sims_dist, "selection": sel,
       "class_names": {"news_topic_7": ["economy", "education", "entertainment", "international", "national", "politics", "sports"],
                       "review_sentiment_3": ["negative", "neutral", "positive"], "social_sentiment_3": ["negative", "neutral", "positive"],
                       "hate_binary_2": ["hate", "not hate"]},
       "notes": "CONF numbers for merged_plus and merged_sim are exploratory (built after the confirmatory CONF test)."}
(OUT / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
rows = []
for k in KEYS:
    r = res[k]
    rows.append({"model": NAMES[k], "dim": r["dim"], "DEV Overall F1": r["dev_Overall_F1"], "DEV D1 news F1": r["dev_D1_F1"], "DEV D2 informal F1": r["dev_D2_F1"],
                 "CONF Overall F1": r["conf_Overall_F1"], "CONF 95% CI low": r["conf_Overall_lo"], "CONF 95% CI high": r["conf_Overall_hi"],
                 "CONF D1": r["conf_D1_F1"], "CONF D2": r["conf_D2_F1"], **{f"CONF F1 {t}": r[f"conf_F1_{t}"] for t in TASKS},
                 "P(best) CONF": r["conf_P_best"], "Analogy native": r["analogy_native_acc"], "Analogy semantic": r["analogy_semantic_acc"],
                 "Analogy syntactic": r["analogy_syntactic_acc"], "Analogy Mikolov": r["analogy_mikolov_acc"], "Similarity AUC": r["sim_auc"],
                 "Token coverage D1": r["tok_cov_D1"], "Token coverage D2": r["tok_cov_D2"]})
pd.DataFrame(rows).to_csv(OUT / "metrics_matrix.csv", index=False)
np.savez_compressed(OUT / "predictions.npz", **{f"{k}|{s}|{t}": v for (k, s, t), v in preds.items()}, **{f"y|{s}|{t}": v for (s, t), v in labels.items()})
log("wrote", OUT)
