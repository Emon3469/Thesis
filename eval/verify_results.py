"""Independent re-verification of the Kaggle results, run locally from the downloaded files.
Recomputes every headline number from oof_predictions.parquet + emb/*.npy and compares it with the tables."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

R = Path(__file__).parent / "kaggle_output" / "results"
SEED, N_BOOT = 42, 1000
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
ev = pd.read_parquet(Path(__file__).parent / "kaggle_dataset" / "eval_set.parquet")
oof = pd.read_parquet(R / "oof_predictions.parquet")
full = pd.read_csv(R / "table_main_full.csv")
dom = pd.read_csv(R / "table_domain.csv")
fails = []
def check(name, a, b, tol=1e-9):
    if not (abs(a - b) <= tol): fails.append(f"{name}: {a} vs {b}")

# 1. gold labels in the predictions == local eval set (same uploaded file), each uid once per model/task
g = oof.merge(ev[["uid", "label", "task"]], on="uid", suffixes=("", "_local"))
assert len(g) == len(oof) and (g["gold"] == g["label"]).all() and (g["task"] == g["task_local"]).all()
assert not oof.duplicated(["uid", "model", "task"]).any()
print(f"[1] {len(oof)} predictions match local gold labels; no duplicates")

# 2. accuracy / macro-F1 from predictions == table
for (m, t), d in oof.groupby(["model", "task"]):
    row = full[(full.model == m) & (full.task == t)].iloc[0]
    for key, col in (("probe", "pred_probe"), ("knn", "pred_knn"), ("pca", "pred_pca64")):
        check(f"{m}/{t}/{key}_acc", accuracy_score(d.gold, d[col]), row[f"{key}_acc"])
        check(f"{m}/{t}/{key}_f1", f1_score(d.gold, d[col], average="macro"), row[f"{key}_f1"])
print(f"[2] acc/F1 recomputed for {oof.groupby(['model','task']).ngroups} model-task pairs x 3 methods")

# 3. bootstrap CIs reproduced with the same seed and draw order
rng = np.random.default_rng(SEED)
y_by_task = {t: ev.loc[ev.task == t, "label"].to_numpy() for t in TASKS}
uids_by_task = {t: ev.loc[ev.task == t, "uid"].to_numpy() for t in TASKS}
BOOT = {t: rng.integers(0, len(y_by_task[t]), size=(N_BOOT, len(y_by_task[t]))) for t in TASKS}
def boot_f1(y, p, idx, C):
    out = np.empty(len(idx))
    for b, ii in enumerate(idx):
        out[b] = f1_score(y[ii], p[ii], average="macro", labels=list(range(C)), zero_division=0)
    return out
bf = {}
for (m, t), d in oof.groupby(["model", "task"]):
    d = d.set_index("uid").loc[uids_by_task[t]]
    y, p = d.gold.to_numpy(), d.pred_probe.to_numpy()
    bf[(m, t)] = boot_f1(y, p, BOOT[t], len(np.unique(y)))
    row = full[(full.model == m) & (full.task == t)].iloc[0]
    check(f"{m}/{t}/f1_lo", np.percentile(bf[(m, t)], 2.5), row.probe_f1_lo)
    check(f"{m}/{t}/f1_hi", np.percentile(bf[(m, t)], 97.5), row.probe_f1_hi)
print("[3] 95% bootstrap CIs reproduced with sklearn f1_score (independent implementation)")

# 4. domain scores = news F1 and mean of the 3 informal F1s
for _, r in dom.iterrows():
    f = {t: full[(full.model == r.model) & (full.task == t)].probe_f1.iat[0] for t in TASKS}
    check(f"{r.model}/D1", f["news_topic_7"], r.D1_news_F1)
    check(f"{r.model}/D2", np.mean([f[t] for t in TASKS[1:]]), r.D2_mean_F1)
print("[4] domain scores recomputed")

# 5. Friedman + Nemenyi from fold-level F1 (folds regenerated with the same seed)
methods = pd.read_csv(R / "table_friedman.csv").model.tolist()
F = []
for t in TASKS:
    y = y_by_task[t]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    for tr, te in folds:
        F.append([f1_score(y[te], oof[(oof.model == m) & (oof.task == t)].set_index("uid").loc[uids_by_task[t]]
                           .pred_probe.to_numpy()[te], average="macro") for m in methods])
F = np.array(F); ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, F).mean(0)
fr = stats.friedmanchisquare(*F.T); man = json.loads((R / "run_manifest.json").read_text())
check("friedman_chi2", fr.statistic, man["friedman"]["chi2"], 1e-6)
for m, r in zip(methods, ranks):
    check(f"rank/{m}", r, pd.read_csv(R / "table_friedman.csv").set_index("model").avg_rank[m], 1e-9)
CD = 2.728 * math.sqrt(5 * 6 / (6 * F.shape[0]))
check("CD", CD, man["friedman"]["CD"], 1e-3)
print(f"[5] Friedman chi2={fr.statistic:.4f} p={fr.pvalue:.4g}; CD={CD:.4f}; ranks "
      + ", ".join(f"{m}={r:.3f}" for m, r in zip(methods, ranks)))

# 6. CKA + anisotropy recomputed from the raw embeddings
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))
E = {m: np.load(R / "emb" / f"{m}.npy").astype(np.float64) for m in methods}
diag = pd.read_csv(R / "table_diagnostics.csv")
for d in ["D1_formal_news", "D2_informal_ugc"]:
    idx = np.where(ev.domain == d)[0]
    C = pd.read_csv(R / f"table_cka_{d}.csv", index_col=0)
    for a in methods:
        for b in methods:
            check(f"cka/{d}/{a}/{b}", cka(E[a][idx], E[b][idx]), C.loc[a, b], 1e-4)
        U = E[a][idx] / np.linalg.norm(E[a][idx], axis=1, keepdims=True)
        # brute force mean off-diagonal cosine (independent of the closed form used in the notebook)
        S = U @ U.T; n = len(U); aniso = (S.sum() - np.trace(S)) / (n * (n - 1))
        check(f"aniso/{d}/{a}", aniso, diag[(diag.model == a) & (diag.domain == d)].anisotropy.iat[0], 1e-5)
print("[6] CKA (50 cells) and anisotropy (brute-force pairwise) recomputed from embeddings")

print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} MISMATCHES"); [print("  ", f) for f in fails[:20]]
