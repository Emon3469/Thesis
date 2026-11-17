# %% [markdown]
# # Bangla Word2Vec merge: one-time confirmation test on the CONF half (Steps 6–7)
# **Frozen model under test**: `T: CONC` = News-W2V + Informal-W2V + Subs2vec-w2, equal-energy concatenation over the imputed
# union vocabulary (900-d), selected on the DEV half by the pre-registered rule and saved by the Step 4 kernel. It is loaded
# from the Step 4 output and checked by MD5; nothing is retrained or tuned here.
#
# **Pre-registered confirmatory family (fixed before this run)** — Overall macro-F1 (mean of D1 news and D2 informal):
# - H1: T: CONC > fastText-cc.bn (best single model on DEV, news specialist)
# - H2: T: CONC > News-W2V (its Domain-A source)
# - H3: T: CONC > Informal-W2V (its Domain-B source, informal specialist)
#
# Paired bootstrap (10,000 samples, the same resampled texts for every model), one-sided p = P(Δ ≤ 0), Holm correction at
# α = 0.05. The merged model is called better only if all three Holm-adjusted p-values are below 0.05.
#
# Reported but **not** part of the claim: per-domain differences, Subs2vec-w2, the extended model X: CONC (with fastText) and
# Q: CONC (two-model merge rebuilt from the frozen source tables; exploratory, because it was not the selected model).
#
# Evaluation is identical to DEV: mean of unit word rows per text, StandardScaler + RidgeClassifierCV, 5 stratified folds
# (seed 42) inside the CONF half, macro-F1.

# %%
import os, sys, time, json, gc, glob, math, gzip, bz2, hashlib, urllib.request, warnings
from pathlib import Path
T0 = time.perf_counter()
SMOKE = os.environ.get("EVAL_SMOKE") == "1"
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
if "ipykernel" not in sys.modules: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
show = (lambda x: display(x)) if "ipykernel" in sys.modules else print
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

SEED, ALPHA = 42, 0.05
NB = 2000 if SMOKE else 10_000
DATA = Path(sorted(glob.glob("/kaggle/input/**/eval_v2.parquet", recursive=True) or glob.glob("**/w2v_kaggle_dataset/eval_v2.parquet", recursive=True))[0]).parent
MOD = Path(sorted(glob.glob("/kaggle/input/**/selected_pure_w2v.f16.npy", recursive=True) or glob.glob("merge_output/merge_results/models/selected_pure_w2v.f16.npy"))[0]).parent
OUT = Path("/kaggle/working/conf_results" if ON_KAGGLE else ("conf_results_smoke" if SMOKE else "conf_results"))
(OUT / "fig").mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/w2v" if ON_KAGGLE else "w2v_tmp"); WORK.mkdir(exist_ok=True)
FROZEN_MD5 = {"selected_pure_w2v": "10aa206f8462c82a7b16cf60054be70f", "selected_extended": "dd6a4e253e93ae239aa8d79857643fc2",
              "News-W2V": "e43195c774d403511a67b7ce83987d7a", "Informal-W2V": "ec33a3960d435d361e5aadcdb2a08786"}
ZENODO = "https://zenodo.org/api/records/22230710/files/{f}/content"; SUBS_FILE, SUBS_MD5 = "bn_300_2_sg_wxd.csv.bz2", "7f67dc8dab206d0c1f8a83600758b374"
FT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz"
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]; D2T = TASKS[1:]
MAIN, BASELINES = "T: CONC", ["fastText-cc.bn", "News-W2V", "Informal-W2V"]
print("data:", DATA, "| frozen models:", MOD, "| SMOKE:", SMOKE)

# %%
# ---------------------------------------------------------------- CONF half (read for the first time here)
ev = pd.read_parquet(DATA / "eval_v2.parquet")
assert set(ev.split) == {"dev", "conf"}
conf = ev[ev.split == "conf"].reset_index(drop=True); n_dev = int((ev.split == "dev").sum()); del ev
conf["toks"] = conf.tokens.str.split()
if SMOKE: conf = conf.groupby("task", group_keys=False).apply(lambda g: g.sample(min(len(g), 300), random_state=SEED)).reset_index(drop=True)
NEEDED = set(w for t in conf.toks for w in t)
print(conf.groupby(["domain", "task"]).size(), f"\nCONF texts {len(conf):,} | DEV texts {n_dev:,} | distinct CONF tokens {len(NEEDED):,}")

# %%
# ---------------------------------------------------------------- tables (frozen files are MD5-checked; only rows of CONF tokens are needed)
class Table:
    def __init__(self, name, words, vecs, full_vocab_size, info):
        self.name, self.words, self.full_vocab_size, self.info = name, list(words), full_vocab_size, info
        self.U = np.asarray(vecs, dtype=np.float32); self.idx = {w: i for i, w in enumerate(self.words)}
        n = np.linalg.norm(self.U, axis=1, keepdims=True); n[n == 0] = 1; self.U /= n

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_frozen(tag, name, full=False):
    f = MOD / f"{tag}.f16.npy"; got = md5(f); assert got == FROZEN_MD5[tag], f"{tag}: md5 {got} differs from the frozen file"
    words = (MOD / f"{tag}.words.txt").read_text(encoding="utf-8").split("\n"); A = np.load(f, mmap_mode="r"); assert A.shape[0] == len(words)
    keep = np.arange(len(words)) if full else np.array([i for i, w in enumerate(words) if w in NEEDED])
    print(f"   {name}: frozen file {f.name} md5 ok, {A.shape[0]:,} x {A.shape[1]}, rows used {len(keep):,}")
    return Table(name, [words[i] for i in keep], np.asarray(A[keep], dtype=np.float32), len(words), dict(file=f.name, md5=got))

def table_from_text(name, lines_iter, dim, info, sep):
    words, vecs, n = [], [], 0
    for line in lines_iter:
        w, _, rest = line.partition(sep); n += 1
        if w in NEEDED:
            try:
                v = np.asarray(rest.split(sep)[:dim], dtype=np.float32)
                if len(v) == dim: vecs.append(v); words.append(w)
            except ValueError: pass
    return Table(name, words, np.stack(vecs), n, info)

def fetch(url, dst):
    if not dst.exists():
        t = time.perf_counter(); urllib.request.urlretrieve(url, dst); print(f"   downloaded {dst.name} in {time.perf_counter()-t:.0f}s")
    return dst

TAB = {}
TAB[MAIN] = load_frozen("selected_pure_w2v", MAIN)
TAB["X: CONC"] = load_frozen("selected_extended", "X: CONC")
NEWS_FULL, INF_FULL = load_frozen("News-W2V", "News-W2V", full=True), load_frozen("Informal-W2V", "Informal-W2V", full=True)
TAB["News-W2V"], TAB["Informal-W2V"] = NEWS_FULL, INF_FULL
local = Path("../models/B_subs2vec") / SUBS_FILE
p_subs = local if local.exists() else fetch(ZENODO.format(f=SUBS_FILE), WORK / SUBS_FILE)
assert md5(p_subs) == SUBS_MD5, "subs2vec md5 mismatch"
def subs_rows():
    with bz2.open(p_subs, "rt", encoding="utf-8") as f:
        next(f)
        for i, l in enumerate(f):
            if SMOKE and i > 30000: break
            yield l.rstrip("\n")
TAB["Subs2vec-w2"] = table_from_text("Subs2vec-w2", subs_rows(), 300, dict(file=SUBS_FILE), ",")
if not SMOKE:
    p_ft = fetch(FT_URL, WORK / "cc.bn.300.vec.gz")
    def ft_rows():
        with gzip.open(p_ft, "rt", encoding="utf-8", errors="ignore") as f:
            next(f)
            for l in f: yield l.rstrip()
    TAB["fastText-cc.bn"] = table_from_text("fastText-cc.bn", ft_rows(), 300, dict(file="cc.bn.300.vec.gz"), " ")
else:                                                            # smoke only: a stand-in so the pipeline runs end to end
    TAB["fastText-cc.bn"] = Table("fastText-cc.bn (smoke stand-in)", TAB["Subs2vec-w2"].words, TAB["Subs2vec-w2"].U[:, ::-1], 0, {})

# exploratory: Q: CONC rebuilt with the Step 4 operator (ridge-imputed union, equal energy) from the frozen source tables
def ridge_map(S, T, lam=1.0):
    sh = [w for w in S.words if w in T.idx]; X = S.U[[S.idx[w] for w in sh]].astype(np.float64); Y = T.U[[T.idx[w] for w in sh]]
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y).astype(np.float32)
def conc_impute(tabs, words):
    have = [np.fromiter((w in T.idx for w in words), bool, len(words)) for T in tabs]; out = []
    for k, T in enumerate(tabs):
        Bk = np.zeros((len(words), T.U.shape[1]), np.float32); Bk[have[k]] = T.U[[T.idx[w] for w, h in zip(words, have[k]) if h]]; todo = ~have[k]
        for j, S in enumerate(tabs):
            can = todo & have[j]
            if j == k or not can.any(): continue
            Y = S.U[[S.idx[w] for w, c in zip(words, can) if c]] @ ridge_map(S, T); n = np.linalg.norm(Y, axis=1, keepdims=True); n[n == 0] = 1
            Bk[can] = Y / n; todo &= ~can
        out.append(Bk / math.sqrt(len(tabs)))
    return np.hstack(out)
qwords = sorted(w for w in NEEDED if w in NEWS_FULL.idx or w in INF_FULL.idx)
TAB["Q: CONC (exploratory)"] = Table("Q: CONC (exploratory)", qwords, conc_impute([NEWS_FULL, INF_FULL], qwords), len(qwords), {})
print({k: len(v.words) for k, v in TAB.items()}, f"| elapsed {(time.perf_counter()-T0)/60:.1f} min")

# %%
# ---------------------------------------------------------------- evaluation (identical for every model)
FOLDS = {t: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((conf.task == t).sum()), conf.label[conf.task == t])) for t in TASKS}
rng = np.random.default_rng(SEED)
BOOT = {t: rng.integers(0, (conf.task == t).sum(), size=(NB, (conf.task == t).sum())) for t in TASKS}
def macro_f1_rows(y, p, idx, C):
    """Macro-F1 for every bootstrap row of idx (same definition as sklearn f1_score(average='macro', labels=range(C), zero_division=0))."""
    out = np.zeros(len(idx))
    for s in range(0, len(idx), 1000):
        yi, pi = y[idx[s:s + 1000]], p[idx[s:s + 1000]]; f = np.zeros((len(yi), C))
        for c in range(C):
            tp = ((yi == c) & (pi == c)).sum(1); den = (yi == c).sum(1) + (pi == c).sum(1)
            f[:, c] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0)
        out[s:s + 1000] = f.mean(1)
    return out
def sent_vecs(tab, toks_list):
    X = np.zeros((len(toks_list), tab.U.shape[1]), np.float32); empty = 0
    for i, toks in enumerate(toks_list):
        ids = [tab.idx[w] for w in toks if w in tab.idx]
        if ids: X[i] = tab.U[ids].mean(0)
        else: empty += 1
    return X, empty
def probe_oof(X, y, folds):
    p = np.full(len(y), -1)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        p[te] = RidgeClassifierCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return p

ROWS, BD1, BD2, PRED = [], {}, {}, {}
for name, tab in TAB.items():
    t0 = time.perf_counter(); r = dict(model=name, dim=tab.U.shape[1]); bf = {}
    for t in TASKS:
        g = conf[conf.task == t]; y = g.label.values.astype(int); X, empty = sent_vecs(tab, g.toks.tolist()); p = probe_oof(X, y, FOLDS[t]); C = int(y.max()) + 1
        r[f"F1_{t}"] = f1_score(y, p, average="macro", labels=list(range(C)), zero_division=0); r[f"empty_{t}"] = empty
        toks = [w for x in g.toks for w in x]; r[f"tok_cov_{t}"] = np.mean([w in tab.idx for w in toks])
        bf[t] = macro_f1_rows(y, p, BOOT[t], C); PRED[(name, t)] = p
        assert abs(macro_f1_rows(y, p, np.arange(len(y))[None], C)[0] - r[f"F1_{t}"]) < 1e-12
    r["D1_F1"] = r["F1_news_topic_7"]; r["D2_F1"] = np.mean([r[f"F1_{t}"] for t in D2T]); r["Overall_F1"] = (r["D1_F1"] + r["D2_F1"]) / 2
    BD1[name] = bf["news_topic_7"]; BD2[name] = np.mean([bf[t] for t in D2T], 0); bo = (BD1[name] + BD2[name]) / 2
    for k, b in (("D1", BD1[name]), ("D2", BD2[name]), ("Overall", bo)): r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile(b, 2.5), np.percentile(b, 97.5)
    ROWS.append(r)
    print(f"{name:<24} dim {r['dim']:>4}  D1 {r['D1_F1']:.4f}  D2 {r['D2_F1']:.4f}  Overall {r['Overall_F1']:.4f}  {time.perf_counter()-t0:.0f}s")
RES = pd.DataFrame(ROWS)

# %%
# ---------------------------------------------------------------- confirmatory tests (pre-registered family, Holm) + secondary comparisons
OV = {m: (BD1[m] + BD2[m]) / 2 for m in TAB}
pt = RES.set_index("model")
def comp(a, b, D, key, family):
    d = D[a] - D[b]; obs = pt.loc[a, key] - pt.loc[b, key]
    return dict(comparison=f"{a} vs {b}", metric=key, family=family, delta=obs, ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                p_one_sided=(1 + int((d <= 0).sum())) / (NB + 1))
TESTS = [comp(MAIN, b, OV, "Overall_F1", "confirmatory") for b in BASELINES]
ps = np.array([t["p_one_sided"] for t in TESTS]); order = np.argsort(ps); adj = np.zeros(len(ps)); run_max = 0
for rank, i in enumerate(order): run_max = max(run_max, min(1, (len(ps) - rank) * ps[i])); adj[i] = run_max
for t, a in zip(TESTS, adj): t["p_holm"] = a; t["reject_H0"] = bool(a < ALPHA)
TESTS += [comp(MAIN, "fastText-cc.bn", BD1, "D1_F1", "secondary"), comp(MAIN, "Informal-W2V", BD2, "D2_F1", "secondary"),
          comp(MAIN, "Subs2vec-w2", OV, "Overall_F1", "secondary"), comp("X: CONC", MAIN, OV, "Overall_F1", "secondary"),
          comp("X: CONC", "fastText-cc.bn", OV, "Overall_F1", "secondary"), comp(MAIN, "Q: CONC (exploratory)", OV, "Overall_F1", "exploratory"),
          comp("Q: CONC (exploratory)", "fastText-cc.bn", OV, "Overall_F1", "exploratory")]
TST = pd.DataFrame(TESTS)
CONFIRMED = bool(TST[TST.family == "confirmatory"].reject_H0.all())

# rank: share of bootstrap samples in which each model has the highest Overall macro-F1 among all models evaluated here
names = list(TAB); stack = np.stack([OV[m] for m in names]); best = stack.argmax(0)
RES["P_best_among_evaluated"] = [float((best == i).mean()) for i in range(len(names))]
RES["rank_CONF"] = RES.Overall_F1.rank(ascending=False).astype(int)
dev = pd.read_csv(MOD.parent / "merge_dev.csv").set_index("model")
dev_name = {"Q: CONC (exploratory)": "Q: CONC"}
RES["Overall_F1_DEV"] = [dev.Overall_F1.get(dev_name.get(m, m), np.nan) for m in RES.model]
RES["drop_DEV_to_CONF"] = RES.Overall_F1_DEV - RES.Overall_F1
RES.to_csv(OUT / "conf_results.csv", index=False); TST.to_csv(OUT / "conf_tests.csv", index=False)
pd.DataFrame([dict(model=m, task=t, **{"pred": p}) for (m, t), p in PRED.items()]).to_pickle(OUT / "conf_predictions.pkl")
pd.DataFrame({"task": conf.task, "label": conf.label}).to_csv(OUT / "conf_labels.csv", index=False)
np.savez_compressed(OUT / "conf_bootstrap.npz", **{f"OV|{m}": OV[m] for m in TAB}, **{f"D1|{m}": BD1[m] for m in TAB}, **{f"D2|{m}": BD2[m] for m in TAB})
show(RES.sort_values("Overall_F1", ascending=False)[["model", "dim", "D1_F1", "D2_F1", "Overall_F1", "Overall_lo", "Overall_hi", "Overall_F1_DEV",
                                                     "drop_DEV_to_CONF", "P_best_among_evaluated"]].round(4))
show(TST.round(4))
print("CONFIRMED (all three Holm-adjusted p < 0.05):", CONFIRMED)

# %%
o = RES.sort_values("Overall_F1"); fig, ax = plt.subplots(figsize=(8, 0.45 * len(o) + 1.2))
ax.errorbar(o.Overall_F1, range(len(o)), xerr=[o.Overall_F1 - o.Overall_lo, o.Overall_hi - o.Overall_F1], fmt="o",
            color="#3b6fb6", ecolor="#9db4d9", capsize=3)
ax.scatter(o.Overall_F1_DEV, range(len(o)), marker="x", color="#7f8c8d", label="DEV score")
ax.set_yticks(range(len(o))); ax.set_yticklabels(o.model, fontsize=8); ax.legend(fontsize=8)
ax.set_title("Overall macro-F1 on CONF with 95% bootstrap CI (x = DEV score)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig" / "conf_overall.png", dpi=170); plt.show(); plt.close(fig)
json.dump(dict(minutes=round((time.perf_counter() - T0) / 60, 1), smoke=SMOKE, n_boot=NB, alpha=ALPHA, main=MAIN, baselines=BASELINES,
               conf_texts=conf.groupby("task").size().to_dict(), frozen_md5=FROZEN_MD5, confirmed=CONFIRMED,
               vocab_rows_used={k: len(v.words) for k, v in TAB.items()}, numpy=np.__version__),
          open(OUT / "manifest.json", "w"), indent=2, default=str)
print(f"DONE in {(time.perf_counter()-T0)/60:.1f} min -> {OUT}")
