"""
PRE-EVALUATION for a Bangla two-domain meta-embedding (formal news + informal UGC/Banglish).

Pre-registered protocol (fixed before looking at confirmation data)
------------------------------------------------------------------
Data: the 2,100-text eval set. Each task is split ONCE, stratified 50/50 (seed 42):
      DEV half  -> every choice (sources, pair, fusion method, alpha) is made here
      CONF half -> touched once, for the selected configuration and its references only
Inside each half: stratified 5-fold CV. Every fusion parameter (scalers, PCA, Procrustes rotation,
autoencoder weights) and the classifier's C (inner 3-fold CV, C in 1e-4..10) are fitted on the
training folds only. Primary metric: linear-probe macro-F1. Secondary: kNN (k=10, cosine) macro-F1.
Domain score: D1 = news_topic_7 F1; D2 = mean F1 of the 3 informal tasks; Overall = (D1 + D2) / 2.

Stage 1  single-source screening (DEV)                      -> rank all sources by Overall
Stage 2  pair screening: CONC-std of every pair among the top-6 sources (DEV)
Stage 3  fusion-method comparison on the top-3 pairs + the "specialist pair"
         (best D1 source + best D2 source) (DEV)
Stage 4  selection rule: highest Overall among fused configs whose D1 and D2 are each no more than
         1.0 F1 point below the best single source on that domain (DEV)
Stage 5  confirmation (CONF): selected config vs best single source per domain and overall.
         Paired bootstrap (2,000), Holm over the 3 comparisons. Success = Overall gain with
         95% CI > 0 AND each domain's lower CI bound > -1.0 point (non-inferiority).
Controls: label-permutation (must give chance), dimension control CONC(A, Gaussian noise of B's
         size) (must not beat A), TF-IDF floor.
Power:    minimum detectable effect MDE = 2.80 x SE(Delta) (alpha .05 two-sided, power .80).
"""
from __future__ import annotations
import json, math, os, time, warnings
os.environ["JOBLIB_TEMP_FOLDER"] = str(__import__("pathlib").Path(__file__).parent / "joblib_tmp")  # C: is nearly full
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.cluster import KMeans  # noqa (kept for parity with earlier notebook)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
T0 = time.perf_counter()
SEED, NB = 42, 2000
CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
HERE = Path(__file__).parent
OUT = HERE / "preeval"; (OUT / "fig").mkdir(parents=True, exist_ok=True)
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
D2T = TASKS[1:]

ev = pd.read_parquet(HERE / "kaggle_dataset" / "eval_set.parquet")
SRC_DIRS = [HERE / "kaggle_output" / "results" / "emb", HERE / "cand_output" / "cand_results" / "emb"]
EMB = {}
for d in SRC_DIRS:
    for f in sorted(d.glob("*.npy")) if d.exists() else []:
        E = np.load(f).astype(np.float64)
        if E.shape[0] == len(ev) and np.isfinite(E).all() and np.linalg.norm(E, axis=1).min() > 0:
            EMB[f.stem] = E
        elif E.shape[0] == len(ev):                       # e.g. static model with empty texts -> keep, flag
            EMB[f.stem] = E; print(f"note: {f.stem} has {(np.linalg.norm(E, axis=1) == 0).sum()} all-zero rows")
print("sources:", {k: v.shape[1] for k, v in EMB.items()})

# ------------------------------------------------------------------ splits (fixed once)
IDX, Y, SPLIT = {}, {}, {}
for t in TASKS:
    idx = np.where(ev.task.values == t)[0]; y = ev.label.values[idx].astype(int)
    dev, conf = train_test_split(np.arange(len(idx)), test_size=0.5, stratify=y, random_state=SEED)
    IDX[t], Y[t] = idx, y
    SPLIT[t] = {"dev": np.sort(dev), "conf": np.sort(conf)}
FOLDS = {(t, h): list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(SPLIT[t][h])), Y[t][SPLIT[t][h]]))
         for t in TASKS for h in ("dev", "conf")}

# ------------------------------------------------------------------ fusion operators (fit on train fold only)
def l2(X): n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1; return X / n

class Fusion:
    def __init__(self, method, alpha=0.5, seed=SEED): self.m, self.a, self.seed = method, alpha, seed
    def fit(self, Xs):
        m = self.m
        if m in ("conc_std", "pca_conc", "conc_alpha", "conc_noise", "procrustes_avg", "aaeme", "single"):
            self.sc = [StandardScaler().fit(X) for X in Xs]
        Z = self._std(Xs) if hasattr(self, "sc") else None
        if m == "pca_conc":
            C = np.hstack(Z); self.pca = PCA(min(768, C.shape[0] - 1, C.shape[1]), random_state=self.seed).fit(C)
        if m == "procrustes_avg":
            k = min(256, min(z.shape[1] for z in Z), Z[0].shape[0] - 1)
            self.p = [PCA(k, random_state=self.seed).fit(z) for z in Z]
            A, B = (self.p[i].transform(Z[i]) for i in range(2))
            U, _, Vt = np.linalg.svd(B.T @ A); self.R = U @ Vt              # orthogonal Procrustes B -> A
        if m == "aaeme":
            self._fit_ae(Z)
        return self
    def _std(self, Xs):   # z-score per source, then 1/sqrt(d) so each source carries equal total variance
        return [s.transform(X) / math.sqrt(X.shape[1]) for s, X in zip(self.sc, Xs)]
    def transform(self, Xs):
        m = self.m
        if m == "single": return self._std(Xs)[0]
        if m == "conc_l2": return np.hstack([l2(X) for X in Xs])
        if m == "avg_l2":
            D = max(X.shape[1] for X in Xs)
            return np.mean([np.pad(l2(X), ((0, 0), (0, D - X.shape[1]))) for X in Xs], axis=0)
        Z = self._std(Xs)
        if m in ("conc_std", "conc_noise"): return np.hstack(Z)
        if m == "conc_alpha": return np.hstack([math.sqrt(self.a) * Z[0], math.sqrt(1 - self.a) * Z[1]])
        if m == "pca_conc": return self.pca.transform(np.hstack(Z))
        if m == "procrustes_avg":
            A, B = (self.p[i].transform(Z[i]) for i in range(2)); return (A + B @ self.R) / 2
        if m == "aaeme": return self._enc(Z)
        raise ValueError(m)
    # AAEME (Bollegala & Bao 2018): source-specific encoders, averaged code, reconstruct every source
    def _fit_ae(self, Z, h=512, epochs=300):
        import torch
        torch.manual_seed(self.seed); torch.set_num_threads(1)
        Zt = [torch.tensor(z, dtype=torch.float32) for z in Z]
        self.enc = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(z.shape[1], h), torch.nn.Tanh()) for z in Z])
        dec = torch.nn.ModuleList([torch.nn.Linear(h, z.shape[1]) for z in Z])
        opt = torch.optim.Adam(list(self.enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)
        for _ in range(epochs):
            code = torch.stack([e(z) for e, z in zip(self.enc, Zt)]).mean(0)
            loss = sum(((d(code) - z) ** 2).sum(1).mean() for d, z in zip(dec, Zt))
            opt.zero_grad(); loss.backward(); opt.step()
        self.ae_loss = float(loss)
    def _enc(self, Z):
        import torch
        with torch.no_grad():
            return torch.stack([e(torch.tensor(z, dtype=torch.float32)) for e, z in zip(self.enc, Z)]).mean(0).numpy()

def run_config(srcs, method, task, half, alpha=0.5, shuffle_labels=False, Xtask=None):
    """5-fold OOF predictions (probe + kNN) for one configuration on one task-half."""
    pos = SPLIT[task][half]; y = Y[task][pos].copy()
    if shuffle_labels: y = np.random.default_rng(SEED).permutation(y)
    Xs = [X[pos] for X in Xtask]          # only this task's rows are shipped to the worker
    if method == "conc_noise":
        Xs = [Xs[0], np.random.default_rng(SEED).normal(size=(len(pos), Xs[1].shape[1]))]
    probe, knn = np.full(len(y), -1), np.full(len(y), -1); Cs = []
    for tr, te in FOLDS[(task, half)]:
        if method == "ensemble":                                   # late fusion reference (not one embedding)
            P = 0
            for X in Xs:
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                           scoring="f1_macro", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
                P = P + clf.predict_proba(sc.transform(X[te]))
            probe[te] = clf.classes_[P.argmax(1)]; knn[te] = probe[te]; continue
        fu = Fusion("single" if len(srcs) == 1 and method != "conc_noise" else method, alpha).fit([X[tr] for X in Xs])
        Ftr, Fte = fu.transform([X[tr] for X in Xs]), fu.transform([X[te] for X in Xs])
        sc = StandardScaler().fit(Ftr)
        clf = LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                   scoring="f1_macro", max_iter=3000).fit(sc.transform(Ftr), y[tr])
        probe[te] = clf.predict(sc.transform(Fte)); Cs.append(float(clf.C_[0]))
        knn[te] = KNeighborsClassifier(10, metric="cosine", algorithm="brute").fit(l2(Ftr), y[tr]).predict(l2(Fte))
    assert (probe >= 0).all() and (knn >= 0).all()
    return dict(srcs="+".join(srcs), method=method, alpha=alpha if method == "conc_alpha" else np.nan, task=task, half=half,
                shuffled=shuffle_labels, y=y, probe=probe, knn=knn, C=Cs,
                probe_f1=f1_score(y, probe, average="macro"), knn_f1=f1_score(y, knn, average="macro"))

def XT(srcs, t): return [EMB[s][IDX[t]] for s in srcs]

def run_many(specs, half):
    return Parallel(n_jobs=-1)(delayed(run_config)(s, m, t, half, a, Xtask=XT(s, t)) for s, m, a in specs for t in TASKS)

def summarise(res):
    df = pd.DataFrame([{k: r[k] for k in ("srcs", "method", "alpha", "task", "probe_f1", "knn_f1")} for r in res])
    g = df.pivot_table(index=["srcs", "method", "alpha"], columns="task", values=["probe_f1", "knn_f1"], dropna=False)
    s = pd.DataFrame({"D1": g[("probe_f1", "news_topic_7")], "D2": g["probe_f1"][D2T].mean(1),
                      "kNN_D1": g[("knn_f1", "news_topic_7")], "kNN_D2": g["knn_f1"][D2T].mean(1)})
    s["Overall"] = (s.D1 + s.D2) / 2; s["kNN_Overall"] = (s.kNN_D1 + s.kNN_D2) / 2
    for t in TASKS: s[t] = g[("probe_f1", t)]
    return s.reset_index().dropna(subset=["Overall"]).sort_values("Overall", ascending=False)

# ------------------------------------------------------------------ Stage 1: single sources (DEV)
SOURCES = list(EMB)
R1 = run_many([([s], "single", .5) for s in SOURCES], "dev")
S1 = summarise(R1); S1.to_csv(OUT / "stage1_single_sources_dev.csv", index=False)
print("\nStage 1 (DEV) single sources\n", S1.round(4).to_string(index=False))
boundary = np.mean([c == CS[0] or c == CS[-1] for r in R1 for c in r["C"]])
print(f"C at grid boundary: {100*boundary:.1f}% of folds (grid {CS[0]}..{CS[-1]})")
cands = [s for s in S1.srcs if s != "TFIDF-char-SVD"]
best_D1 = S1[S1.srcs.isin(cands)].sort_values("D1", ascending=False).srcs.iat[0]
best_D2 = S1[S1.srcs.isin(cands)].sort_values("D2", ascending=False).srcs.iat[0]
best_all = S1[S1.srcs.isin(cands)].srcs.iat[0]
if best_D2 == best_D1:
    best_D2 = S1[S1.srcs.isin(cands) & (S1.srcs != best_D1)].sort_values("D2", ascending=False).srcs.iat[0]
print(f"best D1 = {best_D1} | best D2 (distinct) = {best_D2} | best overall = {best_all}")

# ------------------------------------------------------------------ complementarity (label-free CKA on all texts, DEV oracle)
def cka(X, Yv):
    X = X - X.mean(0); Yv = Yv - Yv.mean(0)
    return float(np.linalg.norm(Yv.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Yv.T @ Yv)))
CKA = {dname: pd.DataFrame([[cka(EMB[a][ev.domain.values == dname], EMB[b][ev.domain.values == dname]) for b in cands] for a in cands],
                           index=cands, columns=cands) for dname in ("D1_formal_news", "D2_informal_ugc")}
for k, v in CKA.items(): v.to_csv(OUT / f"cka_{k}.csv")
P1 = {r["srcs"] + "|" + r["task"]: r for r in R1}

# ------------------------------------------------------------------ Stage 2: pair screening (DEV)
top6 = cands[:6]
pairs = [list(p) for p in combinations(top6, 2)]
if sorted([best_D1, best_D2]) not in [sorted(p) for p in pairs]: pairs.append([best_D1, best_D2])
R2 = run_many([(p, "conc_std", .5) for p in pairs], "dev")
S2 = summarise(R2)
orc = []
for p in pairs:
    o = np.mean([((P1[p[0] + "|" + t]["probe"] == Y[t][SPLIT[t]["dev"]]) | (P1[p[1] + "|" + t]["probe"] == Y[t][SPLIT[t]["dev"]])).mean() for t in TASKS])
    orc.append(dict(srcs="+".join(p), oracle_acc=o, CKA_D1=CKA["D1_formal_news"].loc[p[0], p[1]], CKA_D2=CKA["D2_informal_ugc"].loc[p[0], p[1]]))
S2 = S2.merge(pd.DataFrame(orc), on="srcs")
for side in ("D1", "D2", "Overall"):
    S2[f"gain_{side}_vs_best_member"] = S2[side] - S2.srcs.map(lambda s: S1.set_index("srcs").loc[s.split("+"), side].max())
S2.to_csv(OUT / "stage2_pairs_dev.csv", index=False)
print("\nStage 2 (DEV) pair screening, CONC-std\n", S2.round(4).to_string(index=False))

# ------------------------------------------------------------------ Stage 3: fusion methods (DEV)
spec_pair = [best_D1, best_D2]
top_pairs = [s.split("+") for s in S2.srcs.head(3)]
if sorted(spec_pair) not in [sorted(p) for p in top_pairs]: top_pairs.append(spec_pair)
METHODS = [("conc_l2", .5), ("conc_std", .5), ("avg_l2", .5), ("pca_conc", .5), ("procrustes_avg", .5), ("aaeme", .5),
           ("conc_alpha", .25), ("conc_alpha", .75), ("ensemble", .5)]
specs = [(p, m, a) for p in top_pairs for m, a in METHODS]
R3 = run_many(specs, "dev")
S3 = summarise(R3)
S3.to_csv(OUT / "stage3_methods_dev.csv", index=False)
print("\nStage 3 (DEV) fusion methods\n", S3.round(4).to_string(index=False))

# controls on DEV
best_single = S1[S1.srcs.isin(cands)].iloc[0]
cn = [best_all, best_D2 if best_D2 != best_all else best_D1]
RC = Parallel(n_jobs=-1)(delayed(run_config)(cn, "conc_noise", t, "dev", Xtask=XT(cn, t)) for t in TASKS)
SC = summarise(RC); SC["control"] = f"CONC({best_all}, Gaussian noise)"
print("\nDimension control\n", SC.round(4).to_string(index=False))

# ------------------------------------------------------------------ Stage 4: selection rule (DEV)
bestD1v, bestD2v = S1[S1.srcs.isin(cands)].D1.max(), S1[S1.srcs.isin(cands)].D2.max()
elig = S3[(S3.method != "ensemble") & (S3.D1 >= bestD1v - 0.01) & (S3.D2 >= bestD2v - 0.01)]
pool = elig if len(elig) else S3[S3.method != "ensemble"]
SEL = pool.sort_values("Overall", ascending=False).iloc[0]
sel_srcs, sel_m, sel_a = SEL.srcs.split("+"), SEL.method, (SEL.alpha if not np.isnan(SEL.alpha) else .5)
print(f"\nSELECTED on DEV: {SEL.srcs} | {sel_m} | alpha={sel_a} | Overall={SEL.Overall:.4f} "
      f"(eligible under non-inferiority rule: {len(elig) > 0})")

# ------------------------------------------------------------------ Stage 5: confirmation (CONF, touched once)
conf_specs = {"SELECTED": (sel_srcs, sel_m, sel_a),
              f"single:{sel_srcs[0]}": ([sel_srcs[0]], "single", .5), f"single:{sel_srcs[1]}": ([sel_srcs[1]], "single", .5),
              f"single:{best_D1}": ([best_D1], "single", .5), f"single:{best_D2}": ([best_D2], "single", .5),
              f"single:{best_all}": ([best_all], "single", .5),
              "CONC-std(selected pair)": (sel_srcs, "conc_std", .5), "ENSEMBLE(selected pair)": (sel_srcs, "ensemble", .5),
              "TFIDF floor": (["TFIDF-char-SVD"], "single", .5)}
conf_specs = {k: v for k, v in conf_specs.items() if all(s in EMB for s in v[0])}
RCONF = {}
for name, (s, m, a) in conf_specs.items():
    RCONF[name] =Parallel(n_jobs=-1)(delayed(run_config)(s, m, t, "conf", a, Xtask=XT(s, t)) for t in TASKS)
perm = Parallel(n_jobs=-1)(delayed(run_config)(sel_srcs, sel_m, t, "conf", sel_a, True, Xtask=XT(sel_srcs, t)) for t in TASKS)

rng = np.random.default_rng(SEED)
BI = {t: rng.integers(0, len(SPLIT[t]["conf"]), size=(NB, len(SPLIT[t]["conf"]))) for t in TASKS}
def boot_f1(y, p, idx, C):
    out = np.empty(len(idx))
    for b, ii in enumerate(idx): out[b] = f1_score(y[ii], p[ii], average="macro", labels=list(range(C)), zero_division=0)
    return out
BF = {}
for name, res in RCONF.items():
    for r in res:
        BF[(name, r["task"])] = boot_f1(r["y"], r["probe"], BI[r["task"]], len(np.unique(r["y"])))
def dom(name, side):
    if side == "D1": return BF[(name, "news_topic_7")]
    if side == "D2": return np.mean([BF[(name, t)] for t in D2T], axis=0)
    return (dom(name, "D1") + dom(name, "D2")) / 2
def point(name, side):
    f = {r["task"]: r["probe_f1"] for r in RCONF[name]}
    d1, d2 = f["news_topic_7"], np.mean([f[t] for t in D2T])
    return {"D1": d1, "D2": d2, "Overall": (d1 + d2) / 2}[side]
TC = pd.DataFrame([dict(config=n, **{s: point(n, s) for s in ("D1", "D2", "Overall")},
                        **{f"{s}_lo": np.percentile(dom(n, s), 2.5) for s in ("D1", "D2", "Overall")},
                        **{f"{s}_hi": np.percentile(dom(n, s), 97.5) for s in ("D1", "D2", "Overall")},
                        kNN_Overall=np.mean([np.mean([r["knn_f1"] for r in RCONF[n] if r["task"] == "news_topic_7"]),
                                             np.mean([r["knn_f1"] for r in RCONF[n] if r["task"] in D2T])]))
                   for n in RCONF]).sort_values("Overall", ascending=False)
TC.to_csv(OUT / "stage5_confirmation.csv", index=False)
print("\nStage 5 (CONF) confirmation\n", TC.round(4).to_string(index=False))

comp = []
refs = {"D1": f"single:{best_D1}", "D2": f"single:{best_D2}", "Overall": f"single:{best_all}"}
for side, ref in refs.items():
    d = dom("SELECTED", side) - dom(ref, side); delta = point("SELECTED", side) - point(ref, side)
    se = d.std(ddof=1)
    comp.append(dict(domain=side, reference=ref, delta=delta, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                     p=min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))), SE=se, MDE_80pct_power=2.80 * se))
CMP = pd.DataFrame(comp); o = np.argsort(CMP.p.values); adj = np.empty(len(CMP)); run = 0
for r_, i in enumerate(o): run = max(run, (len(CMP) - r_) * CMP.p.values[i]); adj[i] = min(1, run)
CMP["p_holm"] = adj
CMP["noninferior(lo>-1pt)"] = CMP.lo > -0.01
CMP.to_csv(OUT / "stage5_selected_vs_best_single.csv", index=False)
print("\nSelected vs best single (CONF)\n", CMP.round(4).to_string(index=False))
perm_f1 = {r["task"]: r["probe_f1"] for r in perm}
print("Label-permutation control (CONF):", {k: round(v, 3) for k, v in perm_f1.items()},
      "| chance macro-F1 ≈ 1/C per task")
success = bool(CMP.set_index("domain").loc["Overall", "lo"] > 0 and CMP[CMP.domain != "Overall"]["noninferior(lo>-1pt)"].all())
print("PRE-REGISTERED SUCCESS CRITERION MET:", success)

# ------------------------------------------------------------------ save predictions + manifest
rows = []
groups = {}
for r in R1 + R2 + R3:                      # one config = 4 task results; keep all of them (keyed by config, then task)
    groups.setdefault(f"DEV:{r['srcs']}|{r['method']}|{r['alpha']}", {})[r["task"]] = r
groups.update({f"CONF:{k}": {r["task"]: r for r in v} for k, v in RCONF.items()})
for name, res in ((k, list(v.values())) for k, v in groups.items()):
    assert len(res) == 4, name
    for r in res:
        pos = SPLIT[r["task"]][r["half"]]
        rows.append(pd.DataFrame(dict(config=name, task=r["task"], uid=ev.uid.values[IDX[r["task"]][pos]], gold=r["y"],
                                      probe=r["probe"], knn=r["knn"])))
pd.concat(rows).to_parquet(OUT / "predictions.parquet", index=False)
json.dump(dict(minutes=round((time.perf_counter() - T0) / 60, 2), sources={k: v.shape[1] for k, v in EMB.items()},
               best_D1=best_D1, best_D2=best_D2, best_overall=best_all, selected=dict(srcs=sel_srcs, method=sel_m, alpha=sel_a),
               success=success, C_boundary_share=boundary, Cs=CS, seed=SEED, n_boot=NB,
               label_permutation_f1=perm_f1, dimension_control=SC.drop(columns="control").to_dict("records")),
          open(OUT / "manifest.json", "w"), indent=2, default=str)
print(f"\nDone in {(time.perf_counter()-T0)/60:.1f} min")
