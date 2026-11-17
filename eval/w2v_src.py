# %% [markdown]
# # Bangla Word2Vec pre-evaluation: screening, pre-merge checks (P1–P6), 3-dataset test
# **Models**
# - **News-W2V**: Word2Vec trained here on Potrika news (eval texts removed). Replaces Model A, whose vectors are no longer downloadable.
# - **Subs2vec-w1…w6**: Model B, Bengali Word2Vec on OpenSubtitles (Zenodo 22230710), 300-d skip-gram, windows 1–6.
# - **Informal-W2V**: Word2Vec trained here on reviews + social comments (eval texts removed), same settings as News-W2V.
# - **fastText-cc.bn**: Common Crawl fastText `.vec` (word table only), reference.
#
# **Measures** (DEV half only; CONF half untouched, except the 3 small datasets drawn from it):
# word-similarity proxy (category AUC), word analogy (3CosAdd), vocabulary coverage, downstream macro-F1 on 4 tasks.
#
# **Pre-registered rules**
# - Planned pair = News-W2V + the Subs2vec window with the best Domain-2 macro-F1.
# - Specialist pair = best Domain-1 model + best Domain-2 model (if different from the planned pair, it is checked too).
# - P1 identity error < 1e-6. P2 passes if I(Y;B|A) > 0 on Domain 2 AND I(Y;A|B) > 0 on Domain 1 (95% CI lower bound > 0).
# - P3 allows SVD at k only if the risk bound < the P2 gain. P4 passes if each source holds 40–60% of top-k energy.
# - P5 passes if the median held-out imputation cosine > the 95th percentile of random-pair cosine.
# - P6 (averaging after alignment) passes if held-out Procrustes P@1 ≥ 0.50.

# %%
import os, sys, time, json, gc, glob, math, gzip, bz2, hashlib, subprocess, urllib.request, warnings
from pathlib import Path
T0 = time.perf_counter()
SMOKE = os.environ.get("EVAL_SMOKE") == "1"
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
warnings.filterwarnings("ignore")
try:
    import gensim
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gensim"]); import gensim
import numpy as np, pandas as pd
import matplotlib
if "ipykernel" not in sys.modules: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
show = (lambda x: display(x)) if "ipykernel" in sys.modules else print
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

SEED, NB, TOPK_SEARCH = 42, 1000, 200_000
W2V = dict(vector_size=300, window=4, sg=1, min_count=5, negative=10, epochs=5, seed=SEED, workers=os.cpu_count() or 4)
DATA = Path(sorted(glob.glob("/kaggle/input/**/eval_v2.parquet", recursive=True) or glob.glob("**/w2v_kaggle_dataset/eval_v2.parquet", recursive=True))[0]).parent
OUT = Path("/kaggle/working/w2v_results" if ON_KAGGLE else ("w2v_results_smoke" if SMOKE else "w2v_results")); (OUT / "fig").mkdir(parents=True, exist_ok=True)
WORK = Path("/tmp/w2v" if ON_KAGGLE else "w2v_tmp"); WORK.mkdir(exist_ok=True)
ZENODO = "https://zenodo.org/api/records/22230710/files/{f}/content"
SUBS_MD5 = {1: "6e9f97ab1293dd5e49c0c9266cf7dd6c", 2: "7f67dc8dab206d0c1f8a83600758b374", 3: "3dc9922c4fbc107af2140784f917ecec",
            4: "1dc9fb2497011265a1298261321478ce", 5: "45f95118462f408ce9d34b7fa5c271af", 6: "046c8400eddb74e42c9909ff0c350c93"}
FT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz"
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]; D2T = TASKS[1:]
exec((DATA / "tokenize.py").read_text(encoding="utf-8"))           # defines tokenize() — same function used to build the data
print("data:", DATA, "| cpus:", os.cpu_count(), "| SMOKE:", SMOKE)

# %%
# ---------------------------------------------------------------- data
ev = pd.read_parquet(DATA / "eval_v2.parquet")
ev["toks"] = ev.tokens.str.split()
dev = ev[ev.split == "dev"].reset_index(drop=True)
if SMOKE: dev = dev.groupby("task", group_keys=False).apply(lambda g: g.sample(min(len(g), 300), random_state=SEED)).reset_index(drop=True)
ANALOGY = {p.stem: [l.split() for l in open(p, encoding="utf-8") if len(l.split()) == 4] for p in sorted((DATA / "analogy").glob("*.txt")) if p.stem != "Data_Stat"}
SEMANTIC = ["Division_District", "Gender", "Number_Ordinal", "Number_Date", "Number_Female"]
CATS = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
MINI = {n: pd.read_csv(DATA / "mini" / f"{n}.csv") for n in ("domain_1", "domain_2", "merged_dataset")}
for d in MINI.values(): d["toks"] = d.tokens.astype(str).str.split()
NEEDED = set(w for t in ev.toks for w in t) | {w for q in ANALOGY.values() for l in q for w in l} | {w for v in CATS.values() for w in v}
print(ev.groupby(["task", "split"]).size().unstack(), "\nanalogy files:", len(ANALOGY), "| questions:", sum(map(len, ANALOGY.values())),
      "| needed words:", len(NEEDED))

# %%
# ---------------------------------------------------------------- embedding tables: frequency-ordered top words + every needed word
class Table:
    def __init__(self, name, words, vecs, full_vocab_size, info):
        self.name, self.words, self.full_vocab_size, self.info = name, list(words), full_vocab_size, info
        self.V = np.asarray(vecs, dtype=np.float32); self.idx = {w: i for i, w in enumerate(self.words)}
        n = np.linalg.norm(self.V, axis=1, keepdims=True); n[n == 0] = 1; self.U = self.V / n
        self.n_search = min(TOPK_SEARCH, len(self.words))          # rows are in frequency order; first n_search = analogy search space

def table_from_kv(name, kv, info):
    keep = [w for w in kv.index_to_key[:TOPK_SEARCH]] + [w for w in kv.index_to_key[TOPK_SEARCH:] if w in NEEDED]
    return Table(name, keep, np.stack([kv[w] for w in keep]), len(kv.index_to_key), info)

def table_from_text(name, lines_iter, dim, info, sep):
    """lines_iter yields raw lines; only the first field is split unless the word is kept."""
    words, vecs, n = [], [], 0
    for line in lines_iter:
        w, _, rest = line.partition(sep); n += 1
        if n <= TOPK_SEARCH or w in NEEDED:
            try:
                v = np.asarray(rest.split(sep)[:dim], dtype=np.float32)
                if len(v) == dim: vecs.append(v); words.append(w)
            except ValueError: pass
    return Table(name, words, np.stack(vecs), n, info)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def fetch(url, dst):
    if not dst.exists():
        t = time.perf_counter(); urllib.request.urlretrieve(url, dst); print(f"   downloaded {dst.name} in {time.perf_counter()-t:.0f}s")
    return dst

MODELS, INFO = {}, {}
def train_w2v(name, corpus):
    from gensim.models.word2vec import LineSentence, Word2Vec
    t = time.perf_counter(); src = DATA / corpus
    if not src.exists(): src = DATA / corpus.removesuffix(".gz")      # Kaggle decompresses .gz files on upload
    assert src.exists(), f"corpus not found: {corpus}"
    if SMOKE:
        small = WORK / f"small_{corpus}"
        opener = gzip.open if src.suffix == ".gz" else open
        with opener(src, "rt", encoding="utf-8") as f, gzip.open(small, "wt", encoding="utf-8") as g:
            for i, l in enumerate(f):
                if i >= 3000: break
                g.write(l)
        src = small
    m = Word2Vec(LineSentence(str(src)), **{**W2V, **({"epochs": 1, "min_count": 2} if SMOKE else {})})
    tab = table_from_kv(name, m.wv, dict(kind="trained word2vec", corpus=corpus, **W2V, train_seconds=round(time.perf_counter() - t),
                                         corpus_words=m.corpus_total_words))
    print(f"{name}: vocab {len(m.wv):,}, corpus {m.corpus_total_words:,} words, {time.perf_counter()-t:.0f}s"); del m; gc.collect()
    return tab

MODELS["News-W2V"] = train_w2v("News-W2V", "news_corpus.txt.gz")
MODELS["Informal-W2V"] = train_w2v("Informal-W2V", "informal_corpus.txt.gz")

for w in ([1] if SMOKE else range(1, 7)):
    fn = f"bn_300_{w}_sg_wxd.csv.bz2"; t = time.perf_counter()
    local = Path("../models/B_subs2vec") / fn
    p = local if (SMOKE and local.exists()) else fetch(ZENODO.format(f=fn), WORK / fn)
    ok = md5(p) == SUBS_MD5[w]; assert ok, f"md5 mismatch {fn}"
    def rows(p=p):
        with bz2.open(p, "rt", encoding="utf-8") as f:
            next(f)                                                   # header: word,0,1,...
            for i, l in enumerate(f):
                if SMOKE and i > 30000: break
                yield l.rstrip("\n")
    MODELS[f"Subs2vec-w{w}"] = table_from_text(f"Subs2vec-w{w}", rows(), 300, dict(kind="subs2vec (Zenodo 22230710)", file=fn, md5_ok=ok), ",")
    print(f"Subs2vec-w{w}: kept {len(MODELS[f'Subs2vec-w{w}'].words):,} of {MODELS[f'Subs2vec-w{w}'].full_vocab_size:,} words, md5 OK, {time.perf_counter()-t:.0f}s")

if not SMOKE:
    p = fetch(FT_URL, WORK / "cc.bn.300.vec.gz")
    def ft_rows():
        with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
            next(f)
            for l in f: yield l.rstrip()
    MODELS["fastText-cc.bn"] = table_from_text("fastText-cc.bn", ft_rows(), 300, dict(kind="fastText Common Crawl .vec (no subwords)"), " ")
    print(f"fastText-cc.bn: kept {len(MODELS['fastText-cc.bn'].words):,} of {MODELS['fastText-cc.bn'].full_vocab_size:,}")
print("models:", list(MODELS), f"| elapsed {(time.perf_counter()-T0)/60:.1f} min")

# %%
# ---------------------------------------------------------------- shared helpers
def sent_vecs(tab, toks_list):
    """Mean of L2-normalised word vectors of in-vocabulary tokens; zero vector if none are known."""
    X = np.zeros((len(toks_list), tab.U.shape[1]), np.float32); empty = 0
    for i, toks in enumerate(toks_list):
        ids = [tab.idx[w] for w in toks if w in tab.idx]
        if ids: X[i] = tab.U[ids].mean(0)
        else: empty += 1
    return X, empty

FOLDS = {t: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((dev.task == t).sum()), dev.label[dev.task == t]))
         for t in TASKS}
def probe_oof(X, y, folds):
    p = np.full(len(y), -1)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        p[te] = RidgeClassifierCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return p

rng = np.random.default_rng(SEED)
BOOT = {t: rng.integers(0, (dev.task == t).sum(), size=(NB, (dev.task == t).sum())) for t in TASKS}
def boot_f1(y, p, idx):
    C = int(y.max()) + 1
    return np.array([f1_score(y[i], p[i], average="macro", labels=list(range(C)), zero_division=0) for i in idx])

# %%
# ---------------------------------------------------------------- 1) independent evaluation of every model
rows, OOF, BF = [], {}, {}
for name, tab in MODELS.items():
    t0 = time.perf_counter(); r = dict(model=name, dim=tab.U.shape[1], full_vocab=tab.full_vocab_size)
    # vocabulary coverage (token and type), per domain, on DEV texts
    for dname, g in dev.groupby("domain"):
        toks = [w for t in g.toks for w in t]; types = set(toks)
        r[f"tok_cov_{dname[:2]}"] = np.mean([w in tab.idx for w in toks]); r[f"type_cov_{dname[:2]}"] = np.mean([w in tab.idx for w in types])
    # word analogy, 3CosAdd over the top-200k search space; OOV questions count as wrong
    S = tab.U[:tab.n_search]; sidx = {w: i for i, w in enumerate(tab.words[:tab.n_search])}
    res = {}
    for f, Q in ANALOGY.items():
        ok = [q for q in Q if all(w in sidx for w in q)]; correct = 0
        for s in range(0, len(ok), 512):
            B = np.array([[sidx[w] for w in q] for q in ok[s:s + 512]])
            tgt = S[B[:, 1]] - S[B[:, 0]] + S[B[:, 2]]
            sim = tgt @ S.T
            for j in (0, 1, 2): sim[np.arange(len(B)), B[:, j]] = -np.inf
            correct += int((sim.argmax(1) == B[:, 3]).sum())
        res[f] = (len(Q), len(ok), correct)
    def agg(keys):
        n = sum(res[k][0] for k in keys); a = sum(res[k][1] for k in keys); c = sum(res[k][2] for k in keys)
        return c / n, a / n, (c / a if a else np.nan)
    native = [k for k in res if k != "Mikolov-Filtered"]; syn = [k for k in native if k not in SEMANTIC]
    for lab, keys in (("analogy_native", native), ("analogy_semantic", SEMANTIC), ("analogy_syntactic", syn), ("analogy_mikolov", ["Mikolov-Filtered"])):
        r[f"{lab}_acc"], r[f"{lab}_answerable"], r[f"{lab}_acc_answerable"] = agg(keys)
    r["analogy_per_file"] = json.dumps({k: v for k, v in res.items()}, ensure_ascii=False)
    # word-similarity proxy: same-category pairs should be more similar than cross-category pairs (AUC)
    wl = [(w, c) for c, ws in CATS.items() for w in ws if w in tab.idx]
    if len({c for _, c in wl}) >= 2:
        U = tab.U[[tab.idx[w] for w, _ in wl]]; cs = np.array([c for _, c in wl]); iu = np.triu_indices(len(wl), 1)
        r["sim_auc"] = roc_auc_score((cs[iu[0]] == cs[iu[1]]).astype(int), (U @ U.T)[iu]); r["sim_words"] = len(wl)
    # downstream macro-F1 (DEV, 5-fold, same folds for every model)
    for t in TASKS:
        g = dev[dev.task == t]; X, empty = sent_vecs(tab, g.toks.tolist()); y = g.label.values
        p = probe_oof(X, y, FOLDS[t]); OOF[(name, t)] = p; BF[(name, t)] = boot_f1(y, p, BOOT[t])
        r[f"F1_{t}"] = f1_score(y, p, average="macro"); r[f"empty_{t}"] = empty
    r["D1_F1"] = r["F1_news_topic_7"]; r["D2_F1"] = np.mean([r[f"F1_{t}"] for t in D2T]); r["Overall_F1"] = (r["D1_F1"] + r["D2_F1"]) / 2
    d1b = BF[(name, "news_topic_7")]; d2b = np.mean([BF[(name, t)] for t in D2T], 0)
    r.update(D1_lo=np.percentile(d1b, 2.5), D1_hi=np.percentile(d1b, 97.5), D2_lo=np.percentile(d2b, 2.5), D2_hi=np.percentile(d2b, 97.5))
    rows.append(r); print(f"{name:<16} D1 {r['D1_F1']:.3f}  D2 {r['D2_F1']:.3f}  analogy {r['analogy_native_acc']:.3f}  "
                          f"simAUC {r.get('sim_auc', np.nan):.3f}  cov {r['tok_cov_D1']:.3f}/{r['tok_cov_D2']:.3f}  {time.perf_counter()-t0:.0f}s")
SCR = pd.DataFrame(rows); SCR.to_csv(OUT / "screening_dev.csv", index=False)
# similarity proxy on the words every model knows (fair comparison)
common = [(w, c) for c, ws in CATS.items() for w in ws if all(w in m.idx for m in MODELS.values())]
for name, tab in MODELS.items():
    U = tab.U[[tab.idx[w] for w, _ in common]]; cs = np.array([c for _, c in common]); iu = np.triu_indices(len(common), 1)
    SCR.loc[SCR.model == name, "sim_auc_common"] = roc_auc_score((cs[iu[0]] == cs[iu[1]]).astype(int), (U @ U.T)[iu])
SCR["sim_common_words"] = len(common); SCR.to_csv(OUT / "screening_dev.csv", index=False)
show(SCR.drop(columns="analogy_per_file").round(4))

# %%
# ---------------------------------------------------------------- 2) pre-registered pair selection
subs = SCR[SCR.model.str.startswith("Subs2vec")]
best_subs = subs.sort_values("D2_F1", ascending=False).model.iat[0]
best_D1 = SCR.sort_values("D1_F1", ascending=False).model.iat[0]
best_D2 = SCR[SCR.model != best_D1].sort_values("D2_F1", ascending=False).model.iat[0]
PAIRS = {"planned (News-W2V + best Subs2vec)": ("News-W2V", best_subs)}
if {best_D1, best_D2} != {"News-W2V", best_subs}: PAIRS["specialist (best D1 + best D2)"] = (best_D1, best_D2)
print("best Subs2vec window:", best_subs, "| best D1:", best_D1, "| best D2 (other):", best_D2); print("pairs to check:", PAIRS)

# %%
# ---------------------------------------------------------------- 3) pre-merge checks P1–P6 (DEV only)
def p_checks(a_name, b_name):
    A, B = MODELS[a_name], MODELS[b_name]; R = dict(pair=f"{a_name} + {b_name}", A=a_name, B=b_name)
    # P1: block-wise pooled merge == concatenation of the pooled blocks (exact identity)
    g = dev[dev.task == "news_topic_7"].head(500); Xa, _ = sent_vecs(A, g.toks.tolist()); Xb, _ = sent_vecs(B, g.toks.tolist())
    merged = np.zeros((len(g), A.U.shape[1] + B.U.shape[1]), np.float32)
    for i, toks in enumerate(g.toks):                     # build merged word rows explicitly, pool each block over its known tokens
        ra = [np.concatenate([A.U[A.idx[w]], np.zeros(B.U.shape[1], np.float32)]) for w in toks if w in A.idx]
        rb = [np.concatenate([np.zeros(A.U.shape[1], np.float32), B.U[B.idx[w]]]) for w in toks if w in B.idx]
        merged[i] = (np.mean(ra, 0) if ra else 0) + (np.mean(rb, 0) if rb else 0)
    R["P1_max_abs_error"] = float(np.abs(merged - np.hstack([Xa, Xb])).max()); R["P1_pass"] = R["P1_max_abs_error"] < 1e-6
    # P2: conditional information each model adds, via out-of-fold log-loss (nats/text), bootstrap CI
    gains, Wnorm = {}, {}
    for t in TASKS:
        g = dev[dev.task == t]; y = g.label.values; Xa, _ = sent_vecs(A, g.toks.tolist()); Xb, _ = sent_vecs(B, g.toks.tolist())
        LL = {}
        for key, X in (("A", Xa), ("B", Xb), ("AB", np.hstack([Xa, Xb]))):
            ll = np.zeros(len(y)); Wn = []
            for tr, te in FOLDS[t]:
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], cv=3, scoring="neg_log_loss", max_iter=3000, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
                P = clf.predict_proba(sc.transform(X[te])); ll[te] = -np.log(np.clip(P[np.arange(len(te)), y[te]], 1e-12, 1))
                Wn.append(np.linalg.norm(clf.coef_ / sc.scale_, 2))        # operator norm of the map from raw features to scores
            LL[key] = ll
            if key == "AB": Wnorm[t] = float(np.mean(Wn))
        gains[t] = {"B|A": LL["A"] - LL["AB"], "A|B": LL["B"] - LL["AB"]}
    def ci(vals_by_task, ts):
        pt = np.mean([vals_by_task[t].mean() for t in ts])
        bs = np.mean([vals_by_task[t][np.random.default_rng(SEED).integers(0, len(vals_by_task[t]), (2000, len(vals_by_task[t])))].mean(1) for t in ts], 0)
        return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for cond in ("B|A", "A|B"):
        for dk, ts in (("D1", ["news_topic_7"]), ("D2", D2T)):
            pt, lo, hi = ci({t: gains[t][cond] for t in ts}, ts)
            R[f"P2_I({cond})_{dk}"], R[f"P2_I({cond})_{dk}_lo"], R[f"P2_I({cond})_{dk}_hi"] = pt, lo, hi
        for t in TASKS: R[f"P2_I({cond})_{t}"] = float(gains[t][cond].mean())
    R["P2_pass"] = bool(R["P2_I(B|A)_D2_lo"] > 0 and R["P2_I(A|B)_D1_lo"] > 0)
    # P3/P4: SVD of the frequency-weighted merged word matrix over DEV vocabulary (equal energy: both blocks unit-norm rows)
    freq = pd.Series([w for t in dev.toks for w in t]).value_counts()
    vocab = [w for w in freq.index if w in A.idx or w in B.idx]
    M = np.zeros((len(vocab), A.U.shape[1] + B.U.shape[1]), np.float32)
    for i, w in enumerate(vocab):
        if w in A.idx: M[i, :A.U.shape[1]] = A.U[A.idx[w]]
        if w in B.idx: M[i, A.U.shape[1]:] = B.U[B.idx[w]]
    Mw = M * np.sqrt(freq[vocab].values.astype(np.float32))[:, None]
    _, s, Vt = np.linalg.svd(Mw, full_matrices=False); energy = s ** 2
    R["P4_raw_norm_ratio_A/B"] = float(np.linalg.norm(A.V, axis=1).mean() / np.linalg.norm(B.V, axis=1).mean())
    R["P4_energy_share_A_all"] = float((Mw[:, :A.U.shape[1]] ** 2).sum() / (Mw ** 2).sum())
    Xab = {t: np.hstack([sent_vecs(A, dev[dev.task == t].toks.tolist())[0], sent_vecs(B, dev[dev.task == t].toks.tolist())[0]]) for t in TASKS}
    gain_overall = np.mean([R["P2_I(B|A)_D1"], R["P2_I(B|A)_D2"]])                # gain of merging over A, nats/text
    R["P3_gain_reference_nats"] = gain_overall
    for k in (100, 200, 300):
        P = Vt[:k].T @ Vt[:k]
        resid = np.mean([np.linalg.norm(Xab[t] - Xab[t] @ P, axis=1).mean() * Wnorm[t] for t in TASKS])
        R[f"P3_retained_energy_k{k}"] = float(energy[:k].sum() / energy.sum())
        R[f"P3_risk_bound_k{k}"] = float(math.sqrt(2) * resid)                 # softmax CE is sqrt(2)-Lipschitz in the scores
        R[f"P3_svd_allowed_k{k}"] = bool(R[f"P3_risk_bound_k{k}"] < gain_overall)
        shareA = float((energy[:k] * (Vt[:k, :A.U.shape[1]] ** 2).sum(1)).sum() / energy[:k].sum())
        R[f"P4_share_A_top{k}"] = shareA; R[f"P4_pass_k{k}"] = bool(0.4 <= shareA <= 0.6)
    # P5: imputing a word missing from one model from its vector in the other (ridge map, held-out shared words)
    shared = [w for w in vocab if w in A.idx and w in B.idx]; prng = np.random.default_rng(SEED); prng.shuffle(shared)
    cut = int(0.8 * len(shared)); tr, te = shared[:cut], shared[cut:]
    def ridge_map(src, dst, words, lam=1.0):
        X = src.U[[src.idx[w] for w in words]]; Y = dst.U[[dst.idx[w] for w in words]]
        return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    for (src, dst, lab) in ((B, A, "B->A"), (A, B, "A->B")):
        Wm = ridge_map(src, dst, tr); Yp = src.U[[src.idx[w] for w in te]] @ Wm; Yt = dst.U[[dst.idx[w] for w in te]]
        cos = (Yp * Yt).sum(1) / (np.linalg.norm(Yp, axis=1) * np.linalg.norm(Yt, axis=1) + 1e-12)
        rp = prng.choice(len(te), (5000, 2)); rand = (Yt[rp[:, 0]] * Yt[rp[:, 1]]).sum(1)
        R[f"P5_{lab}_median_cos"], R[f"P5_{lab}_random_p95"] = float(np.median(cos)), float(np.percentile(rand, 95))
    R["P5_pass"] = bool(min(R["P5_B->A_median_cos"] - R["P5_B->A_random_p95"], R["P5_A->B_median_cos"] - R["P5_A->B_random_p95"]) > 0)
    tokA = np.mean([w in A.idx for t in dev.toks for w in t]); tokB = np.mean([w in B.idx for t in dev.toks for w in t])
    tokU = np.mean([(w in A.idx) or (w in B.idx) for t in dev.toks for w in t])
    R.update(P5_token_cov_A=tokA, P5_token_cov_B=tokB, P5_token_cov_union=tokU)
    # P6: orthogonal Procrustes B -> A on shared words (same 300-d), held-out P@1 and relative residual
    Xs, Ys = B.U[[B.idx[w] for w in tr]], A.U[[A.idx[w] for w in tr]]
    Uu, _, Vv = np.linalg.svd(Xs.T @ Ys); Rot = Uu @ Vv
    te5 = te[:5000]; Xt, Yt = B.U[[B.idx[w] for w in te5]] @ Rot, A.U[[A.idx[w] for w in te5]]
    nn = (Xt / np.linalg.norm(Xt, axis=1, keepdims=True)) @ (Yt / np.linalg.norm(Yt, axis=1, keepdims=True)).T
    R["P6_heldout_P@1"] = float((nn.argmax(1) == np.arange(len(te5))).mean())
    R["P6_relative_residual"] = float(np.linalg.norm(Xt - Yt) / np.linalg.norm(Yt)); R["P6_pass"] = R["P6_heldout_P@1"] >= 0.5
    R["shared_words_train/test"] = f"{len(tr)}/{len(te)}"
    return R

PCHK = pd.DataFrame([p_checks(a, b) for a, b in PAIRS.values()]); PCHK.insert(0, "pair_type", list(PAIRS))
PCHK.to_csv(OUT / "premerge_checks.csv", index=False); show(PCHK.T)

# %%
# ---------------------------------------------------------------- 4) the 3 small datasets (drawn from the held-out half), 3 models
best_overall_other = SCR[~SCR.model.isin(["News-W2V", best_subs])].sort_values("Overall_F1", ascending=False).model.iat[0]
MINI_MODELS = ["News-W2V", best_subs, "fastText-cc.bn" if "fastText-cc.bn" in MODELS else best_overall_other]
def retrieval(S, cls, dom):
    n = len(cls); out = []
    for i in range(n):
        o = sorted([j for j in range(n) if j != i], key=lambda j: -S[i, j]); p = next(j for j in o if cls[j] == cls[i])
        out.append(dict(rank=o.index(p) + 1, hit=int(o[0] == p), same_dom=int(dom[o[0]] == dom[i])))
    q = pd.DataFrame(out); iu = np.triu_indices(n, 1); same = cls[iu[0]] == cls[iu[1]]; w, b = S[iu][same], S[iu][~same]
    return q, dict(R1=q.hit.mean(), MRR=(1 / q["rank"]).mean(), MeanRank=q["rank"].mean(),
                   d_prime=(w.mean() - b.mean()) / math.sqrt((w.var(ddof=1) + b.var(ddof=1)) / 2), Domain1=q.same_dom.mean())
def mrr_only(S, cls):
    tot = 0.0
    for i in range(len(cls)):
        s_ = S[i].copy(); s_[i] = -np.inf; p = next(j for j in range(len(cls)) if j != i and cls[j] == cls[i]); tot += 1 / (1 + np.sum(s_ > s_[p]))
    return tot / len(cls)
mrows = []
for ds, d in MINI.items():
    cls, dom = d["class"].to_numpy(), d["domain"].to_numpy(); n = len(d)
    for m in MINI_MODELS:
        X, empty = sent_vecs(MODELS[m], d.toks.tolist()); nr = np.linalg.norm(X, axis=1, keepdims=True); nr[nr == 0] = 1; Xn = X / nr
        S = Xn @ Xn.T; q, met = retrieval(S, cls, dom)
        prng = np.random.default_rng(SEED); ge = sum(mrr_only(S, prng.permutation(cls)) >= met["MRR"] - 1e-12 for _ in range(10000))
        mrows.append(dict(dataset=ds, model=m, n=n, empty_texts=empty, **met, chance_R1=1 / (n - 1),
                          chance_MRR=sum(1 / r for r in range(1, n)) / (n - 1), p_perm=(ge + 1) / 10001))
        q.insert(0, "uid", d.uid.values); q["class"] = cls; q.to_csv(OUT / f"mini_{ds}__{m}.csv", index=False)
MR = pd.DataFrame(mrows); pv = MR.p_perm.values; o = np.argsort(pv); adj = np.empty_like(pv); run = 0
for r_, i in enumerate(o): run = max(run, (len(pv) - r_) * pv[i]); adj[i] = min(1, run)
MR["p_holm"] = adj; MR.to_csv(OUT / "mini_results.csv", index=False); show(MR.round(4))

# %%
# ---------------------------------------------------------------- 5) figures + manifest
fig, axs = plt.subplots(1, 4, figsize=(18, 4)); order = SCR.sort_values("Overall_F1").model
for ax, (c, ttl) in zip(axs, [("sim_auc_common", "Similarity proxy AUC (common words)"), ("analogy_native_acc", "Analogy accuracy (native, OOV = wrong)"),
                              ("tok_cov_D2", "Token coverage, informal DEV"), ("Overall_F1", "Downstream macro-F1 (Overall)")]):
    s = SCR.set_index("model").loc[order, c]; ax.barh(s.index, s.values, color="#3b6fb6"); ax.set_title(ttl, fontsize=9)
    for y_, v in enumerate(s.values): ax.text(v, y_, f" {v:.3f}", va="center", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig" / "screening.png", dpi=180); plt.show(); plt.close(fig)
json.dump(dict(minutes=round((time.perf_counter() - T0) / 60, 1), smoke=SMOKE, w2v_settings=W2V, topk_search=TOPK_SEARCH,
               models={k: dict(**v.info, kept=len(v.words), full_vocab=v.full_vocab_size) for k, v in MODELS.items()},
               best_subs=best_subs, best_D1=best_D1, best_D2=best_D2, pairs=PAIRS, mini_models=MINI_MODELS,
               numpy=np.__version__, gensim=gensim.__version__), open(OUT / "manifest.json", "w"), indent=2, default=str)
print(f"DONE in {(time.perf_counter()-T0)/60:.1f} min -> {OUT}")
