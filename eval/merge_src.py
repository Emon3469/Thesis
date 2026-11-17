# %% [markdown]
# # Bangla Word2Vec merging experiments (Step 4, DEV half only)
# **Sources** (all 300-d): News-W2V (Domain A, trained here), Informal-W2V (Domain B, trained here),
# Subs2vec-w2 (Domain B, conversational subtitles, Zenodo 22230710), fastText-cc.bn (reference, not Word2Vec).
#
# **Source sets**: P = News-W2V + Subs2vec-w2; Q = News-W2V + Informal-W2V; T = News-W2V + Informal-W2V + Subs2vec-w2;
# X = T + fastText (extended, not pure Word2Vec); S = fastText + Informal-W2V (extended).
#
# **Merge methods**: CONC (equal energy, union vocabulary, missing rows imputed by a ridge map), CONC-zero (missing rows = 0),
# weighted CONC, CONC + SVD (k = 100/200/300), Procrustes alignment + average, autoencoders (AAEME / CAEME style),
# joint training (one Word2Vec on both corpora) and continued training (News-W2V further trained on informal text).
#
# **Pre-registered rules (fixed before this run)**
# - Everything here uses the DEV half. The CONF half is not read.
# - Evaluation is identical for every model: same tokens, same 5 folds, same probe, same bootstrap samples.
# - Specialists = the best single model on Domain 1 and on Domain 2. A candidate is *eligible* if it is within 1 point
#   (0.01 macro-F1) of the specialist on BOTH domains.
# - Selected model = the eligible candidate with the highest Overall macro-F1 (mean of D1 and D2). Two selections are made:
#   pure Word2Vec candidates only (thesis main model) and all candidates (extended).
# - P1–P6 are computed for P, Q and (Q-merged + Subs2vec) BEFORE the merged models are evaluated.

# %%
import os, sys, time, json, gc, glob, math, gzip, bz2, hashlib, subprocess, urllib.request, warnings, shutil
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
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
show = (lambda x: display(x)) if "ipykernel" in sys.modules else print
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)

SEED, NB, TOPK_SEARCH, MARGIN, SUBS_W = 42, 1000, 200_000, 0.01, 2
W2V = dict(vector_size=300, window=4, sg=1, min_count=5, negative=10, epochs=5, seed=SEED, workers=os.cpu_count() or 4)
DATA = Path(sorted(glob.glob("/kaggle/input/**/eval_v2.parquet", recursive=True) or glob.glob("**/w2v_kaggle_dataset/eval_v2.parquet", recursive=True))[0]).parent
OUT = Path("/kaggle/working/merge_results" if ON_KAGGLE else ("merge_results_smoke" if SMOKE else "merge_results"))
(OUT / "fig").mkdir(parents=True, exist_ok=True); (OUT / "models").mkdir(exist_ok=True)
WORK = Path("/tmp/w2v" if ON_KAGGLE else "w2v_tmp"); WORK.mkdir(exist_ok=True)
ZENODO = "https://zenodo.org/api/records/22230710/files/{f}/content"
SUBS_MD5 = {2: "7f67dc8dab206d0c1f8a83600758b374"}
FT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz"
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]; D2T = TASKS[1:]
exec((DATA / "tokenize.py").read_text(encoding="utf-8"))
print("data:", DATA, "| cpus:", os.cpu_count(), "| SMOKE:", SMOKE)

# %%
# ---------------------------------------------------------------- data (DEV half only)
ev = pd.read_parquet(DATA / "eval_v2.parquet")
ev["toks"] = ev.tokens.str.split()
NEEDED = set(w for t in ev.toks for w in t)          # vocabulary to keep (tokens only; no labels are read from the CONF half)
dev = ev[ev.split == "dev"].reset_index(drop=True); del ev
if SMOKE: dev = dev.groupby("task", group_keys=False).apply(lambda g: g.sample(min(len(g), 300), random_state=SEED)).reset_index(drop=True)
ANALOGY = {p.stem: [l.split() for l in open(p, encoding="utf-8") if len(l.split()) == 4] for p in sorted((DATA / "analogy").glob("*.txt")) if p.stem != "Data_Stat"}
SEMANTIC = ["Division_District", "Gender", "Number_Ordinal", "Number_Date", "Number_Female"]
CATS = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
NEEDED |= {w for q in ANALOGY.values() for l in q for w in l} | {w for v in CATS.values() for w in v}
print(dev.groupby(["domain", "task"]).size(), "\nneeded words:", len(NEEDED))

# %%
# ---------------------------------------------------------------- embedding tables
class Table:
    """Rows are L2-normalised word vectors in frequency order; the first n_search rows are the analogy search space."""
    def __init__(self, name, words, vecs, full_vocab_size, info):
        self.name, self.words, self.full_vocab_size, self.info = name, list(words), full_vocab_size, info
        self.U = np.asarray(vecs, dtype=np.float32); self.idx = {w: i for i, w in enumerate(self.words)}
        n = np.linalg.norm(self.U, axis=1, keepdims=True); self.mean_norm = float(n.mean()); n[n == 0] = 1; self.U /= n
        self.n_search = min(TOPK_SEARCH, len(self.words))

def table_from_kv(name, kv, info):
    keep = [w for w in kv.index_to_key[:TOPK_SEARCH]] + [w for w in kv.index_to_key[TOPK_SEARCH:] if w in NEEDED]
    return Table(name, keep, np.stack([kv[w] for w in keep]), len(kv.index_to_key), info)

def table_from_text(name, lines_iter, dim, info, sep):
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

def corpus_path(corpus):
    src = DATA / corpus
    if not src.exists(): src = DATA / corpus.removesuffix(".gz")      # Kaggle decompresses .gz files on upload
    assert src.exists(), f"corpus not found: {corpus}"
    if SMOKE:
        small = WORK / f"small_{corpus}"; opener = gzip.open if src.suffix == ".gz" else open
        with opener(src, "rt", encoding="utf-8") as f, gzip.open(small, "wt", encoding="utf-8") as g:
            for i, l in enumerate(f):
                if i >= 3000: break
                g.write(l)
        src = small
    return src

from gensim.models.word2vec import LineSentence, Word2Vec
SETTINGS = {**W2V, **({"epochs": 1, "min_count": 2} if SMOKE else {})}
def train_w2v(name, src, keep_model=False):
    t = time.perf_counter(); m = Word2Vec(LineSentence(str(src)), **SETTINGS)
    tab = table_from_kv(name, m.wv, dict(kind="trained word2vec", corpus=Path(src).name, **W2V, train_seconds=round(time.perf_counter() - t),
                                         corpus_words=m.corpus_total_words))
    print(f"{name}: vocab {len(m.wv):,}, corpus {m.corpus_total_words:,} words, {time.perf_counter()-t:.0f}s")
    return (tab, m) if keep_model else tab

NEWS, INF = corpus_path("news_corpus.txt.gz"), corpus_path("informal_corpus.txt.gz")
BASE = {}
BASE["News-W2V"], m_news = train_w2v("News-W2V", NEWS, keep_model=True)
BASE["Informal-W2V"] = train_w2v("Informal-W2V", INF)
# continued training: News-W2V, vocabulary extended with informal words, then trained on the informal corpus
t = time.perf_counter(); m_news.build_vocab(LineSentence(str(INF)), update=True)
m_news.train(LineSentence(str(INF)), total_examples=m_news.corpus_count, epochs=SETTINGS["epochs"])
JOINT = {"Continued-W2V": table_from_kv("Continued-W2V", m_news.wv, dict(kind="News-W2V + continued training on informal corpus", **W2V))}
print(f"Continued-W2V: vocab {len(m_news.wv):,}, {time.perf_counter()-t:.0f}s"); del m_news; gc.collect()
# joint training: one Word2Vec on the two corpora together
both = WORK / "joint_corpus.txt"
with open(both, "w", encoding="utf-8") as g:
    for src in (NEWS, INF):
        with (gzip.open if src.suffix == ".gz" else open)(src, "rt", encoding="utf-8") as f: shutil.copyfileobj(f, g)
JOINT["Joint-W2V"] = train_w2v("Joint-W2V", both); gc.collect()

fn = f"bn_300_{SUBS_W}_sg_wxd.csv.bz2"; local = Path("../models/B_subs2vec") / fn
p_subs = local if (SMOKE and local.exists()) else fetch(ZENODO.format(f=fn), WORK / fn)
assert md5(p_subs) == SUBS_MD5[SUBS_W], "md5 mismatch"
def subs_rows():
    with bz2.open(p_subs, "rt", encoding="utf-8") as f:
        next(f)
        for i, l in enumerate(f):
            if SMOKE and i > 30000: break
            yield l.rstrip("\n")
SUBS = f"Subs2vec-w{SUBS_W}"
BASE[SUBS] = table_from_text(SUBS, subs_rows(), 300, dict(kind="subs2vec (Zenodo 22230710)", file=fn, md5_ok=True), ",")
if not SMOKE:
    p_ft = fetch(FT_URL, WORK / "cc.bn.300.vec.gz")
    def ft_rows():
        with gzip.open(p_ft, "rt", encoding="utf-8", errors="ignore") as f:
            next(f)
            for l in f: yield l.rstrip()
    BASE["fastText-cc.bn"] = table_from_text("fastText-cc.bn", ft_rows(), 300, dict(kind="fastText Common Crawl .vec (no subwords)"), " ")
FT = "fastText-cc.bn" if "fastText-cc.bn" in BASE else None
print({k: len(v.words) for k, v in {**BASE, **JOINT}.items()}, f"| elapsed {(time.perf_counter()-T0)/60:.1f} min")

# %%
# ---------------------------------------------------------------- evaluation (identical for every model)
def sent_vecs(tab, toks_list):
    X = np.zeros((len(toks_list), tab.U.shape[1]), np.float32); empty = 0
    for i, toks in enumerate(toks_list):
        ids = [tab.idx[w] for w in toks if w in tab.idx]
        if ids: X[i] = tab.U[ids].mean(0)
        else: empty += 1
    return X, empty

FOLDS = {t: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((dev.task == t).sum()), dev.label[dev.task == t])) for t in TASKS}
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
DEV_TOKS = {d: [w for t in g.toks for w in t] for d, g in dev.groupby("domain")}
COMMON = [(w, c) for c, ws in CATS.items() for w in ws if all(w in m.idx for m in {**BASE, **JOINT}.values())]
BD1, BD2 = {}, {}                                               # bootstrap distributions of D1 and D2 macro-F1 per model

def evaluate(name, tab, **meta):
    t0 = time.perf_counter(); r = dict(model=name, **meta, dim=tab.U.shape[1], vocab=len(tab.words))
    for dname, toks in DEV_TOKS.items():
        r[f"tok_cov_{dname[:2]}"] = np.mean([w in tab.idx for w in toks]); r[f"type_cov_{dname[:2]}"] = np.mean([w in tab.idx for w in set(toks)])
    S = tab.U[:tab.n_search]; sidx = {w: i for i, w in enumerate(tab.words[:tab.n_search])}; res = {}
    for f, Q in ANALOGY.items():
        ok = [q for q in Q if all(w in sidx for w in q)]; correct = 0
        for s in range(0, len(ok), 512):
            B = np.array([[sidx[w] for w in q] for q in ok[s:s + 512]])
            sim = (S[B[:, 1]] - S[B[:, 0]] + S[B[:, 2]]) @ S.T
            for j in (0, 1, 2): sim[np.arange(len(B)), B[:, j]] = -np.inf
            correct += int((sim.argmax(1) == B[:, 3]).sum())
        res[f] = (len(Q), len(ok), correct)
    def agg(keys):
        n = sum(res[k][0] for k in keys); a = sum(res[k][1] for k in keys); c = sum(res[k][2] for k in keys)
        return c / n, a / n
    native = [k for k in res if k != "Mikolov-Filtered"]; syn = [k for k in native if k not in SEMANTIC]
    for lab, keys in (("analogy_native", native), ("analogy_semantic", SEMANTIC), ("analogy_syntactic", syn), ("analogy_mikolov", ["Mikolov-Filtered"])):
        r[f"{lab}_acc"], r[f"{lab}_answerable"] = agg(keys)
    r["analogy_per_file"] = json.dumps(res, ensure_ascii=False)
    wl = [(w, c) for w, c in COMMON if w in tab.idx]
    if len({c for _, c in wl}) >= 2:
        U = tab.U[[tab.idx[w] for w, _ in wl]]; cs = np.array([c for _, c in wl]); iu = np.triu_indices(len(wl), 1)
        r["sim_auc_common"] = roc_auc_score((cs[iu[0]] == cs[iu[1]]).astype(int), (U @ U.T)[iu])
    r["sim_common_words"] = len(wl); bf = {}
    for t in TASKS:
        g = dev[dev.task == t]; X, empty = sent_vecs(tab, g.toks.tolist()); y = g.label.values
        p = probe_oof(X, y, FOLDS[t]); bf[t] = boot_f1(y, p, BOOT[t]); r[f"F1_{t}"] = f1_score(y, p, average="macro"); r[f"empty_{t}"] = empty
    r["D1_F1"] = r["F1_news_topic_7"]; r["D2_F1"] = np.mean([r[f"F1_{t}"] for t in D2T]); r["Overall_F1"] = (r["D1_F1"] + r["D2_F1"]) / 2
    BD1[name] = bf["news_topic_7"]; BD2[name] = np.mean([bf[t] for t in D2T], 0)
    r.update(D1_lo=np.percentile(BD1[name], 2.5), D1_hi=np.percentile(BD1[name], 97.5), D2_lo=np.percentile(BD2[name], 2.5), D2_hi=np.percentile(BD2[name], 97.5))
    print(f"{name:<34} dim {r['dim']:>4}  D1 {r['D1_F1']:.4f}  D2 {r['D2_F1']:.4f}  Overall {r['Overall_F1']:.4f}  analogy {r['analogy_native_acc']:.4f}  "
          f"simAUC {r.get('sim_auc_common', np.nan):.3f}  {time.perf_counter()-t0:.0f}s")
    return r

ROWS = [evaluate(n, t, family="single", method="single", pure_w2v=(n != FT)) for n, t in BASE.items()]
ROWS += [evaluate(n, t, family="joint", method=("joint training" if n == "Joint-W2V" else "continued training"), pure_w2v=True) for n, t in JOINT.items()]
SINGLE = pd.DataFrame(ROWS[:len(BASE)])
SPEC_D1, SPEC_D2 = SINGLE.sort_values("D1_F1").model.iat[-1], SINGLE.sort_values("D2_F1").model.iat[-1]
BEST_SINGLE = SINGLE.sort_values("Overall_F1").model.iat[-1]
TH1, TH2 = SINGLE.D1_F1.max() - MARGIN, SINGLE.D2_F1.max() - MARGIN
print(f"specialists: D1 = {SPEC_D1}, D2 = {SPEC_D2} | best single overall = {BEST_SINGLE} | eligibility: D1 >= {TH1:.4f} and D2 >= {TH2:.4f}")

# %%
# ---------------------------------------------------------------- pre-merge checks P1–P6 (same definitions as the pre-evaluation)
def p_checks(A, B):
    R = dict(pair=f"{A.name} + {B.name}", A=A.name, B=B.name); dA, dB = A.U.shape[1], B.U.shape[1]
    g = dev[dev.task == "news_topic_7"].head(500); Xa, _ = sent_vecs(A, g.toks.tolist()); Xb, _ = sent_vecs(B, g.toks.tolist())
    merged = np.zeros((len(g), dA + dB), np.float32)
    for i, toks in enumerate(g.toks):
        ra = [np.concatenate([A.U[A.idx[w]], np.zeros(dB, np.float32)]) for w in toks if w in A.idx]
        rb = [np.concatenate([np.zeros(dA, np.float32), B.U[B.idx[w]]]) for w in toks if w in B.idx]
        merged[i] = (np.mean(ra, 0) if ra else 0) + (np.mean(rb, 0) if rb else 0)
    R["P1_max_abs_error"] = float(np.abs(merged - np.hstack([Xa, Xb])).max()); R["P1_pass"] = R["P1_max_abs_error"] < 1e-6
    gains, Wnorm = {}, {}
    for t in TASKS:
        g = dev[dev.task == t]; y = g.label.values; Xa, _ = sent_vecs(A, g.toks.tolist()); Xb, _ = sent_vecs(B, g.toks.tolist()); LL = {}
        for key, X in (("A", Xa), ("B", Xb), ("AB", np.hstack([Xa, Xb]))):
            ll = np.zeros(len(y)); Wn = []
            for tr, te in FOLDS[t]:
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1], cv=3, scoring="neg_log_loss", max_iter=3000, n_jobs=-1).fit(sc.transform(X[tr]), y[tr])
                P = clf.predict_proba(sc.transform(X[te])); ll[te] = -np.log(np.clip(P[np.arange(len(te)), y[te]], 1e-12, 1))
                Wn.append(np.linalg.norm(clf.coef_ / sc.scale_, 2))
            LL[key] = ll
            if key == "AB": Wnorm[t] = float(np.mean(Wn))
        gains[t] = {"B|A": LL["A"] - LL["AB"], "A|B": LL["B"] - LL["AB"]}
    def ci(v, ts):
        pt = np.mean([v[t].mean() for t in ts])
        bs = np.mean([v[t][np.random.default_rng(SEED).integers(0, len(v[t]), (2000, len(v[t])))].mean(1) for t in ts], 0)
        return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for cond in ("B|A", "A|B"):
        for dk, ts in (("D1", ["news_topic_7"]), ("D2", D2T)):
            R[f"P2_I({cond})_{dk}"], R[f"P2_I({cond})_{dk}_lo"], R[f"P2_I({cond})_{dk}_hi"] = ci({t: gains[t][cond] for t in ts}, ts)
        for t in TASKS: R[f"P2_I({cond})_{t}"] = float(gains[t][cond].mean())
    R["P2_pass"] = bool(R["P2_I(B|A)_D2_lo"] > 0 and R["P2_I(A|B)_D1_lo"] > 0)
    R["P2_B_adds_any_domain"] = bool(R["P2_I(B|A)_D1_lo"] > 0 or R["P2_I(B|A)_D2_lo"] > 0)
    freq = pd.Series([w for t in dev.toks for w in t]).value_counts()
    vocab = [w for w in freq.index if w in A.idx or w in B.idx]
    M = np.zeros((len(vocab), dA + dB), np.float32)
    for i, w in enumerate(vocab):
        if w in A.idx: M[i, :dA] = A.U[A.idx[w]] * (math.sqrt(dA / 300) if dA != dB else 1)   # a merged block keeps energy proportional to its sources
        if w in B.idx: M[i, dA:] = B.U[B.idx[w]]
    Mw = M * np.sqrt(freq[vocab].values.astype(np.float32))[:, None]
    _, s, Vt = np.linalg.svd(Mw, full_matrices=False); energy = s ** 2
    R["P4_raw_norm_ratio_A/B"] = A.mean_norm / B.mean_norm; R["P4_energy_share_A_all"] = float((Mw[:, :dA] ** 2).sum() / (Mw ** 2).sum())
    R["P4_target_share_A"] = dA / (dA + dB)
    Xab = {t: np.hstack([sent_vecs(A, dev[dev.task == t].toks.tolist())[0], sent_vecs(B, dev[dev.task == t].toks.tolist())[0]]) for t in TASKS}
    gain = np.mean([R["P2_I(B|A)_D1"], R["P2_I(B|A)_D2"]]); R["P3_gain_reference_nats"] = gain
    for k in (100, 200, 300):
        Pk = Vt[:k].T @ Vt[:k]
        resid = np.mean([np.linalg.norm(Xab[t] - Xab[t] @ Pk, axis=1).mean() * Wnorm[t] for t in TASKS])
        R[f"P3_retained_energy_k{k}"] = float(energy[:k].sum() / energy.sum()); R[f"P3_risk_bound_k{k}"] = float(math.sqrt(2) * resid)
        R[f"P3_svd_allowed_k{k}"] = bool(R[f"P3_risk_bound_k{k}"] < gain)
        shareA = float((energy[:k] * (Vt[:k, :dA] ** 2).sum(1)).sum() / energy[:k].sum())
        R[f"P4_share_A_top{k}"] = shareA; R[f"P4_pass_k{k}"] = bool(abs(shareA - R["P4_target_share_A"]) <= 0.1)
    shared = [w for w in vocab if w in A.idx and w in B.idx]; prng = np.random.default_rng(SEED); prng.shuffle(shared)
    cut = int(0.8 * len(shared)); tr, te = shared[:cut], shared[cut:]
    def rmap(src, dst, words, lam=1.0):
        X = src.U[[src.idx[w] for w in words]]; Y = dst.U[[dst.idx[w] for w in words]]
        return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    for (src, dst, lab) in ((B, A, "B->A"), (A, B, "A->B")):
        Wm = rmap(src, dst, tr); Yp = src.U[[src.idx[w] for w in te]] @ Wm; Yt = dst.U[[dst.idx[w] for w in te]]
        cos = (Yp * Yt).sum(1) / (np.linalg.norm(Yp, axis=1) * np.linalg.norm(Yt, axis=1) + 1e-12)
        rp = prng.choice(len(te), (5000, 2)); rand = (Yt[rp[:, 0]] * Yt[rp[:, 1]]).sum(1)
        R[f"P5_{lab}_median_cos"], R[f"P5_{lab}_random_p95"] = float(np.median(cos)), float(np.percentile(rand, 95))
    R["P5_pass"] = bool(min(R["P5_B->A_median_cos"] - R["P5_B->A_random_p95"], R["P5_A->B_median_cos"] - R["P5_A->B_random_p95"]) > 0)
    alltok = [w for t in dev.toks for w in t]
    R.update(P5_token_cov_A=np.mean([w in A.idx for w in alltok]), P5_token_cov_B=np.mean([w in B.idx for w in alltok]),
             P5_token_cov_union=np.mean([(w in A.idx) or (w in B.idx) for w in alltok]))
    if dA == dB:
        Xs, Ys = B.U[[B.idx[w] for w in tr]], A.U[[A.idx[w] for w in tr]]; Uu, _, Vv = np.linalg.svd(Xs.T @ Ys); Rot = Uu @ Vv
        te5 = te[:5000]; Xt, Yt = B.U[[B.idx[w] for w in te5]] @ Rot, A.U[[A.idx[w] for w in te5]]
        nn = (Xt / np.linalg.norm(Xt, axis=1, keepdims=True)) @ (Yt / np.linalg.norm(Yt, axis=1, keepdims=True)).T
        R["P6_heldout_P@1"] = float((nn.argmax(1) == np.arange(len(te5))).mean()); R["P6_relative_residual"] = float(np.linalg.norm(Xt - Yt) / np.linalg.norm(Yt))
        R["P6_pass"] = bool(R["P6_heldout_P@1"] >= 0.5)
    else:
        R["P6_heldout_P@1"], R["P6_relative_residual"], R["P6_pass"] = np.nan, np.nan, False
    R["shared_words_train/test"] = f"{len(tr)}/{len(te)}"
    return R

PC = {"P": p_checks(BASE["News-W2V"], BASE[SUBS]), "Q": p_checks(BASE["News-W2V"], BASE["Informal-W2V"])}
print({k: {c: v[c] for c in ("P1_pass", "P2_pass", "P5_pass", "P6_pass")} for k, v in PC.items()}, f"| elapsed {(time.perf_counter()-T0)/60:.1f} min")

# %%
# ---------------------------------------------------------------- merge operators
MAPS = {}
def ridge_map(S, T, lam=1.0):
    """Linear map from S-space to T-space fitted on every word the two tables share."""
    if (S.name, T.name) not in MAPS:
        sh = [w for w in S.words if w in T.idx]; X = S.U[[S.idx[w] for w in sh]].astype(np.float64); Y = T.U[[T.idx[w] for w in sh]]
        MAPS[(S.name, T.name)] = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y).astype(np.float32)
    return MAPS[(S.name, T.name)]

def union_words(tabs):
    rank = {}
    for T in tabs:
        for i, w in enumerate(T.words):
            if i < rank.get(w, 1 << 60): rank[w] = i
    return sorted(rank, key=rank.get)                        # ordered by the best frequency rank in any source

def blocks(tabs, words, fill):
    """One unit-row block per source. fill='impute': a missing row is predicted from another source (ridge map) and re-normalised."""
    have = [np.fromiter((w in T.idx for w in words), bool, len(words)) for T in tabs]; out = []
    for k, T in enumerate(tabs):
        Bk = np.zeros((len(words), T.U.shape[1]), np.float32)
        Bk[have[k]] = T.U[[T.idx[w] for w, h in zip(words, have[k]) if h]]
        if fill == "impute":
            todo = ~have[k]
            for j, S in enumerate(tabs):
                can = todo & have[j]
                if j == k or not can.any(): continue
                Y = S.U[[S.idx[w] for w, c in zip(words, can) if c]] @ ridge_map(S, T)
                n = np.linalg.norm(Y, axis=1, keepdims=True); n[n == 0] = 1; Bk[can] = Y / n; todo &= ~can
        out.append(Bk)
    return out

def conc(Bs, weights=None):
    w = weights or [1 / len(Bs)] * len(Bs)
    return np.hstack([math.sqrt(wk) * b for wk, b in zip(w, Bs)])

def svd_project(M, ks):
    G = np.zeros((M.shape[1], M.shape[1]))
    for s in range(0, len(M), 50_000): c = M[s:s + 50_000].astype(np.float64); G += c.T @ c
    val, vec = np.linalg.eigh(G); vec = vec[:, ::-1]
    return {k: (M @ vec[:, :k].astype(np.float32)) for k in ks}

def procrustes_avg(A, B, words):
    sh = [w for w in B.words if w in A.idx]
    Uu, _, Vv = np.linalg.svd(B.U[[B.idx[w] for w in sh]].astype(np.float64).T @ A.U[[A.idx[w] for w in sh]]); Rot = (Uu @ Vv).astype(np.float32)
    M = np.zeros((len(words), A.U.shape[1]), np.float32); cnt = np.zeros((len(words), 1), np.float32)
    ha = np.fromiter((w in A.idx for w in words), bool, len(words)); hb = np.fromiter((w in B.idx for w in words), bool, len(words))
    M[ha] += A.U[[A.idx[w] for w, h in zip(words, ha) if h]]; M[hb] += B.U[[B.idx[w] for w, h in zip(words, hb) if h]] @ Rot
    cnt[ha] += 1; cnt[hb] += 1
    return M / cnt

def aeme(Bs, mode, epochs):
    """Autoencoded meta-embedding in the style of Bollegala & Bao (2018): one encoder per source (linear + tanh, 300-d),
    meta = concatenation (CAEME) or sum (AAEME) of the encodings, L2-normalised; linear decoders reconstruct every source."""
    import torch
    torch.manual_seed(SEED); torch.set_num_threads(os.cpu_count() or 4)
    X = [torch.from_numpy(b) for b in Bs]; h = 300
    enc = torch.nn.ModuleList([torch.nn.Linear(b.shape[1], h) for b in Bs])
    dec = torch.nn.ModuleList([torch.nn.Linear(h * len(Bs) if mode == "concat" else h, b.shape[1]) for b in Bs])
    def meta(xs):
        hs = [torch.tanh(e(x)) for e, x in zip(enc, xs)]
        return torch.nn.functional.normalize(torch.cat(hs, 1) if mode == "concat" else sum(hs), dim=1)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 1e-3); n = len(Bs[0]); g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g); tot = 0.0
        for s in range(0, n, 1024):
            i = perm[s:s + 1024]; xs = [x[i] for x in X]; m = meta(xs)
            loss = sum(((d(m) - x) ** 2).sum(1).mean() for d, x in zip(dec, xs)); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(i)
    with torch.no_grad(): out = torch.cat([meta([x[s:s + 8192] for x in X]) for s in range(0, n, 8192)]).numpy()
    return out, tot / n

# %%
# ---------------------------------------------------------------- merging experiments
BESTS = {"pure_w2v": None, "any": None}                       # running best eligible candidate per pool: (row, words, matrix)
def consider(r, tab):
    r["eligible"] = bool(r["D1_F1"] >= TH1 and r["D2_F1"] >= TH2)
    for pool in ("pure_w2v", "any"):
        if pool == "pure_w2v" and not r["pure_w2v"]: continue
        cur = BESTS[pool]
        better = cur is None or (r["eligible"], r["Overall_F1"]) > (cur[0]["eligible"], cur[0]["Overall_F1"])
        if better: BESTS[pool] = (r, tab.words, tab.U.astype(np.float16))
def run(name, words, M, **meta):
    tab = Table(name, words, M, len(words), meta); r = evaluate(name, tab, **meta); consider(r, tab); ROWS.append(r); return tab
for r_, (n_, t_) in zip(ROWS[len(BASE):], JOINT.items()): consider(r_, t_)      # joint/continued training are candidates too

SETS = {"P": ["News-W2V", SUBS], "Q": ["News-W2V", "Informal-W2V"], "T": ["News-W2V", "Informal-W2V", SUBS]}
if FT: SETS.update({"S": [FT, "Informal-W2V"], "X": [FT, "News-W2V", "Informal-W2V", SUBS]})
AE_EPOCHS = 1 if SMOKE else 10
for fam, names in SETS.items():
    tabs = [BASE[n] for n in names]; pure = FT not in names; words = union_words(tabs); tag = " + ".join(names)
    print(f"--- {fam}: {tag} | union vocabulary {len(words):,}")
    if fam == "T":                                             # certificate for adding the third source to the merged pair Q, before merging
        PC["T"] = p_checks(QTAB, BASE[SUBS]); del QTAB; gc.collect()
    Bs = blocks(tabs, words, "impute"); M = conc(Bs)
    tab = run(f"{fam}: CONC", words, M, family=fam, method="CONC (equal energy, imputed union)", pure_w2v=pure, sources=tag)
    if fam == "Q": QTAB = tab; QTAB.name = "CONC(News-W2V, Informal-W2V)"
    if fam in ("P", "Q", "T"):
        for k, Z in svd_project(M, (100, 200, 300)).items():
            run(f"{fam}: CONC+SVD{k}", words, Z, family=fam, method=f"CONC + SVD k={k}", pure_w2v=pure, sources=tag)
        del Z
    if fam in ("P", "Q"):
        for a in (0.3, 0.7):
            run(f"{fam}: wCONC a={a}", words, conc(Bs, [a, 1 - a]), family=fam, method=f"weighted CONC (weight {a} on first source)", pure_w2v=pure, sources=tag)
        for mode, lab in (("avg", "AAEME"), ("concat", "CAEME")):
            try:
                Z, loss = aeme(Bs, mode, AE_EPOCHS)
                run(f"{fam}: {lab}", words, Z, family=fam, method=f"{lab} autoencoder ({AE_EPOCHS} epochs, final loss {loss:.4f})", pure_w2v=pure, sources=tag)
            except ImportError: print("   torch not available, autoencoder skipped")
        del Bs, M; gc.collect()
        run(f"{fam}: CONC-zero", words, conc(blocks(tabs, words, "zero")), family=fam, method="CONC (missing rows = 0)", pure_w2v=pure, sources=tag)
        run(f"{fam}: Procrustes+AVG", words, procrustes_avg(tabs[0], tabs[1], words), family=fam, method="Procrustes alignment + average", pure_w2v=pure, sources=tag)
    else:
        del Bs, M
    gc.collect(); print(f"   elapsed {(time.perf_counter()-T0)/60:.1f} min")
PCHK = pd.DataFrame(list(PC.values())); PCHK.insert(0, "set", list(PC)); PCHK.to_csv(OUT / "premerge_checks.csv", index=False); show(PCHK.T)

# %%
# ---------------------------------------------------------------- selection (pre-registered rule) and paired bootstrap on DEV
RES = pd.DataFrame(ROWS); RES["eligible"] = (RES.D1_F1 >= TH1) & (RES.D2_F1 >= TH2)
def certified(r):
    c = PC.get(r.family)
    if c is None: return "n/a"
    base = c["P1_pass"] and (c["P2_pass"] if r.family != "T" else c["P2_B_adds_any_domain"])
    if r.method.startswith("CONC (equal") or r.method.startswith("weighted"): return "yes" if base and c["P5_pass"] else "no"
    if r.method.startswith("CONC (missing"): return "yes" if base else "no"
    if "SVD" in r.method: return "yes" if base and c[f"P3_svd_allowed_k{r.method.split('=')[1]}"] else "no"
    if "Procrustes" in r.method: return "yes" if base and c["P6_pass"] else "no"
    return "no certificate"
RES["certified"] = RES.apply(certified, axis=1)
def delta(a, b, D):
    d = D[a] - D[b]; return np.percentile(d, 2.5), np.percentile(d, 97.5)
for i, r in RES.iterrows():
    ov = lambda m: (BD1[m] + BD2[m]) / 2
    d = ov(r.model) - ov(BEST_SINGLE)
    RES.loc[i, "dOverall_vs_best_single"] = r.Overall_F1 - RES.set_index("model").Overall_F1[BEST_SINGLE]
    RES.loc[i, "dOverall_lo"], RES.loc[i, "dOverall_hi"] = np.percentile(d, 2.5), np.percentile(d, 97.5)
    RES.loc[i, "dD1_vs_specialist"] = r.D1_F1 - SINGLE.D1_F1.max(); RES.loc[i, "dD1_lo"], RES.loc[i, "dD1_hi"] = delta(r.model, SPEC_D1, BD1)
    RES.loc[i, "dD2_vs_specialist"] = r.D2_F1 - SINGLE.D2_F1.max(); RES.loc[i, "dD2_lo"], RES.loc[i, "dD2_hi"] = delta(r.model, SPEC_D2, BD2)
cand = RES[RES.family != "single"]
def select(pool):
    el = pool[pool.eligible]; flag = "eligible" if len(el) else "NO eligible candidate: best Overall taken"
    return (el if len(el) else pool).sort_values("Overall_F1").model.iat[-1], flag
SEL_W2V, flag_w = select(cand[cand.pure_w2v]); SEL_ANY, flag_a = select(cand)
assert BESTS["pure_w2v"][0]["model"] == SEL_W2V and BESTS["any"][0]["model"] == SEL_ANY, "running best differs from the final selection"
RES["selected"] = np.where(RES.model == SEL_W2V, "pure Word2Vec", "") + np.where(RES.model == SEL_ANY, " extended", "")
RES.to_csv(OUT / "merge_dev.csv", index=False)
cols = ["model", "dim", "D1_F1", "D2_F1", "Overall_F1", "dOverall_vs_best_single", "dOverall_lo", "dOverall_hi", "analogy_native_acc", "sim_auc_common",
        "tok_cov_D1", "tok_cov_D2", "eligible", "certified", "selected"]
show(RES.sort_values("Overall_F1", ascending=False)[cols].round(4))
print(f"SELECTED pure Word2Vec: {SEL_W2V} ({flag_w}) | SELECTED extended: {SEL_ANY} ({flag_a})")

# %%
# ---------------------------------------------------------------- save the frozen models (for the one-time CONF test) + figure + manifest
def save(tag, words, U16):
    np.save(OUT / "models" / f"{tag}.f16.npy", U16); (OUT / "models" / f"{tag}.words.txt").write_text("\n".join(words), encoding="utf-8")
save("selected_pure_w2v", BESTS["pure_w2v"][1], BESTS["pure_w2v"][2])
if SEL_ANY != SEL_W2V: save("selected_extended", BESTS["any"][1], BESTS["any"][2])
for n in ("News-W2V", "Informal-W2V"): save(n, BASE[n].words, BASE[n].U.astype(np.float16))
o = RES.sort_values("Overall_F1"); fig, ax = plt.subplots(figsize=(9, 0.28 * len(o) + 1.5))
colors = ["#c0392b" if s else ("#7f8c8d" if f == "single" else "#3b6fb6") for s, f in zip(o.selected.str.len() > 0, o.family)]
ax.barh(o.model, o.Overall_F1, color=colors); ax.set_xlim(o.Overall_F1.min() - 0.02, o.Overall_F1.max() + 0.01)
for y_, v in enumerate(o.Overall_F1): ax.text(v, y_, f" {v:.4f}", va="center", fontsize=7)
ax.set_title("Overall macro-F1 on DEV (grey = single models, red = selected)", fontsize=10); ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig" / "merge_overall.png", dpi=170); plt.show(); plt.close(fig)
json.dump(dict(minutes=round((time.perf_counter() - T0) / 60, 1), smoke=SMOKE, w2v_settings=W2V, margin=MARGIN, n_boot=NB,
               dev_texts=dev.groupby("task").size().to_dict(), specialist_D1=SPEC_D1, specialist_D2=SPEC_D2, best_single=BEST_SINGLE,
               threshold_D1=TH1, threshold_D2=TH2, selected_pure_w2v=SEL_W2V, selected_pure_w2v_flag=flag_w, selected_extended=SEL_ANY,
               selected_extended_flag=flag_a, sets=SETS, vocab={k: len(v.words) for k, v in {**BASE, **JOINT}.items()},
               full_vocab={k: v.full_vocab_size for k, v in {**BASE, **JOINT}.items()}, numpy=np.__version__, gensim=gensim.__version__),
          open(OUT / "manifest.json", "w"), indent=2, default=str)
print(f"DONE in {(time.perf_counter()-T0)/60:.1f} min -> {OUT}")
