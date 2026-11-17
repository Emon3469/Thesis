# %% [markdown]
# # Bangla 2-Domain Embedding Evaluation
# **Domain 1 — Mainstream news (formal):** Potrika, 7 topics × 150 = 1,050 texts
# **Domain 2 — Reviews & social media (informal):** BanglaBook 3×120, SentNoB 3×120, Hate Speech 2×165 = 1,050 texts
#
# | Role | Model |
# |---|---|
# | D1 candidate | `csebuetnlp/banglabert` (mean-pooled, csebuetnlp normalizer) |
# | D2 candidate | `shihab17/bangla-sentence-transformer` |
# | Comparison | `intfloat/multilingual-e5-base` (`"query: "` prefix, as its model card prescribes for classification/clustering) |
# | Comparison | fastText `cc.bn.300` (`facebook/fasttext-bn-vectors`, full subword `.bin`) |
# | Floor | TF-IDF char 2–4-grams → SVD-300 (not a candidate; a model must beat it) |
#
# **Protocol (all encoders frozen — we evaluate the embedding, not a fine-tuned model):**
# 1. Encode each text once; cache to `.npy`.
# 2. Per task, the **same** stratified 5-fold split for every model → paired comparison.
# 3. **Linear probe:** `StandardScaler → LogisticRegressionCV(C ∈ {0.01,0.1,1,10}, inner 3-fold, macro-F1)` — C is chosen *inside* each outer training fold (nested CV, no test leakage).
# 4. **kNN (k=10, cosine)** — no training; measures neighbourhood geometry.
# 5. **K-means (k = #classes)** — NMI, ARI; silhouette of gold labels (cosine).
# 6. 95% CIs: 1,000 bootstrap resamples of the out-of-fold predictions (same resample indices for all models → paired).
# 7. Significance: exact McNemar + paired bootstrap Δ macro-F1, Holm-corrected; Friedman + Nemenyi critical-difference diagram.
# 8. Limitation diagnostics from the *Limitation mitigation* notebook: unequal scales (norms, anisotropy), information loss (PCA-64 retention), vocabulary mismatch (fertility / UNK / OOV / truncation), redundancy (linear CKA, oracle accuracy).
# 9. **Audit:** every reported number is recomputed by hand from the saved predictions and asserted equal.
#
# **Kaggle settings:** Accelerator = GPU (T4 or P100), Internet = ON, add the `bangla-2domain-eval-set` dataset.

# %%
# ---------------------------------------------------------------- 1. setup
import os, sys, time, json, gc, glob, math, platform, subprocess, warnings, re, unicodedata
from pathlib import Path
T0 = time.perf_counter()
SMOKE = os.environ.get("EVAL_SMOKE") == "1"          # local pipeline test with fake encoders (no downloads)
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ   # set only inside Kaggle notebooks
ON_COLAB = "google.colab" in sys.modules
if ON_KAGGLE or ON_COLAB:
    os.environ.setdefault("HF_HOME", "/tmp/hf")      # 9 GB of weights stay out of /kaggle/working (20 GB cap)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)

def pip(*pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True, text=True)
    if r.returncode: print("pip failed:", pkgs, r.stderr[-400:])

if not SMOKE:
    try:
        import fasttext  # noqa
    except ImportError:
        pip("fasttext-wheel")
    try:
        from normalizer import normalize  # noqa  (csebuetnlp BanglaBERT normalizer)
    except ImportError:
        pip("git+https://github.com/csebuetnlp/normalizer")

import numpy as np, pandas as pd, torch, scipy
import matplotlib
if not ON_KAGGLE and not ON_COLAB and not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize as l2norm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, normalized_mutual_info_score,
                             adjusted_rand_score, silhouette_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {platform.python_version()} | torch {torch.__version__} | sklearn {sklearn.__version__} | "
      f"device {DEVICE} {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''} | SMOKE={SMOKE}")

# %%
# ---------------------------------------------------------------- 2. config
SEED, N_FOLDS, N_BOOT, KNN_K, PCA_DIM = 42, 5, 1000, 10, 64
CS = [0.01, 0.1, 1.0, 10.0]
MAX_LEN = {"D1_formal_news": 256, "D2_informal_ugc": 128}   # subword tokens
BATCH = 64
BUDGET_MIN = 30
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
TASK_DOMAIN = {"news_topic_7": "D1_formal_news", "review_sentiment_3": "D2_informal_ugc",
               "social_sentiment_3": "D2_informal_ugc", "hate_binary_2": "D2_informal_ugc"}
OUT = Path("/kaggle/working/results" if ON_KAGGLE else ("results_smoke" if SMOKE else "results"))
(OUT / "emb").mkdir(parents=True, exist_ok=True); (OUT / "fig").mkdir(exist_ok=True)

MODELS = {
    "BanglaBERT":      dict(kind="hf_mean",  repo="csebuetnlp/banglabert", role="D1 candidate", bn_norm=True),
    "BanglaSBERT":     dict(kind="st",       repo="shihab17/bangla-sentence-transformer", role="D2 candidate"),
    "mE5-base":        dict(kind="st",       repo="intfloat/multilingual-e5-base", role="comparison", prefix="query: "),
    "fastText-cc.bn":  dict(kind="fasttext", repo="facebook/fasttext-bn-vectors", file="model.bin", role="comparison"),
    "TFIDF-char-SVD":  dict(kind="tfidf",    role="floor"),
}
if SMOKE:   # same pipeline, fake encoders = seeded random projections of TF-IDF (tests code paths only)
    MODELS = {"FAKE-A": dict(kind="fake", dim=768, seed=1, role="D1 candidate"),
              "FAKE-B": dict(kind="fake", dim=768, seed=2, role="D2 candidate"),
              "FAKE-C": dict(kind="fake", dim=384, seed=3, role="comparison"),
              "FAKE-D": dict(kind="fake", dim=300, seed=4, role="comparison"),
              "TFIDF-char-SVD": dict(kind="tfidf", role="floor")}
CANDIDATES = [m for m, c in MODELS.items() if c["role"] != "floor"]
np.random.seed(SEED); torch.manual_seed(SEED)

# %%
# ---------------------------------------------------------------- 3. data
paths = sorted(glob.glob("/kaggle/input/**/eval_set.parquet", recursive=True)) or \
        sorted(glob.glob("**/eval_set.parquet", recursive=True))
assert paths, "eval_set.parquet not found — attach the Kaggle dataset"
df = pd.read_parquet(paths[0]).reset_index(drop=True)
print(paths[0], df.shape)
assert len(df) == 2100 and df["uid"].is_unique
counts = df.groupby(["domain", "source", "task", "label_name"]).size().rename("n").reset_index()
display_ = (lambda x: display(x)) if "ipykernel" in sys.modules else print
display_(counts)
TASK_IDX = {t: np.where(df["task"].values == t)[0] for t in TASKS}
Y = {t: df["label"].values[TASK_IDX[t]].astype(int) for t in TASKS}
LABELS = {t: df.iloc[TASK_IDX[t]].drop_duplicates("label").sort_values("label")["label_name"].tolist() for t in TASKS}
DOM_IDX = {d: np.where(df["domain"].values == d)[0] for d in MAX_LEN}

# %%
# ---------------------------------------------------------------- 4. encoders
_PUNCT_SPLIT = lambda t: "".join(f" {c} " if unicodedata.category(c)[0] == "P" else c for c in t)

def _domain_batches(texts):
    """Yield (domain, idx, texts, max_len) so news gets 256 tokens and UGC 128."""
    for d, idx in DOM_IDX.items():
        yield d, idx, [texts[i] for i in idx], MAX_LEN[d]

def tok_diagnostics(tokenizer, texts_dom, max_len, unk_id):
    """fertility = subword tokens / whitespace words; trunc = texts longer than max_len (incl. 2 specials)."""
    enc = tokenizer(texts_dom, add_special_tokens=False, truncation=False)["input_ids"]
    n_tok = np.array([len(e) for e in enc]); n_w = np.array([max(1, len(t.split())) for t in texts_dom])
    unk = sum(int(np.sum(np.array(e) == unk_id)) for e in enc) if unk_id is not None else 0
    return dict(fertility=float(n_tok.sum() / n_w.sum()), unk_rate=float(unk / max(1, n_tok.sum())),
                trunc_rate=float(np.mean(n_tok + 2 > max_len)))

def encode_hf_mean(cfg, texts):
    from transformers import AutoTokenizer, AutoModel
    if cfg.get("bn_norm"):
        from normalizer import normalize as bn_normalize   # BanglaBERT was pre-trained on normalized text
        texts = [bn_normalize(t) for t in texts]
    tok = AutoTokenizer.from_pretrained(cfg["repo"])
    model = AutoModel.from_pretrained(cfg["repo"]).to(DEVICE).eval()
    E = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32); diag = {}
    for d, idx, tx, L in _domain_batches(texts):
        diag[d] = tok_diagnostics(tok, tx, L, tok.unk_token_id)
        order = np.argsort([len(t) for t in tx])                       # length-sorted -> minimal padding
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            enc = tok([tx[i] for i in b], truncation=True, max_length=L, padding=True, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            E[idx[b]] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()  # masked mean pooling
    del model; gc.collect(); torch.cuda.empty_cache()
    return E, diag

def encode_st(cfg, texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["repo"], device=DEVICE)
    pre = cfg.get("prefix", "")
    E = None; diag = {}
    for d, idx, tx, L in _domain_batches(texts):
        model.max_seq_length = L
        diag[d] = tok_diagnostics(model.tokenizer, tx, L, model.tokenizer.unk_token_id)
        e = model.encode([pre + t for t in tx], batch_size=BATCH, convert_to_numpy=True,
                         normalize_embeddings=False, show_progress_bar=False)
        if E is None: E = np.zeros((len(texts), e.shape[1]), dtype=np.float32)
        E[idx] = e
    del model; gc.collect(); torch.cuda.empty_cache()
    return E, diag

def encode_fasttext(cfg, texts):
    import fasttext
    from huggingface_hub import hf_hub_download
    t = time.perf_counter()
    path = hf_hub_download(cfg["repo"], cfg["file"])
    print(f"   fastText .bin ready in {time.perf_counter()-t:.0f}s; loading ...")
    ft = fasttext.load_model(path)
    E = np.zeros((len(texts), ft.get_dimension()), dtype=np.float32); diag = {}
    for d, idx, tx, _ in _domain_batches(texts):
        toks = [_PUNCT_SPLIT(x).split() for x in tx]
        n = sum(len(tk) for tk in toks); oov = sum(ft.get_word_id(w) == -1 for tk in toks for w in tk)
        diag[d] = dict(fertility=1.0, unk_rate=float(oov / max(1, n)), trunc_rate=0.0)  # unk_rate = OOV words (served by subwords)
        for j, tk in zip(idx, toks):
            E[j] = ft.get_sentence_vector(" ".join(tk))   # official: mean of L2-normalised word vectors
    del ft; gc.collect()
    return E, diag

def encode_tfidf(cfg, texts):
    """Label-free and transductive (fit on all 2,100 texts) — stated as a floor, not a candidate."""
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit_transform(texts)
    E = TruncatedSVD(300, random_state=SEED).fit_transform(X).astype(np.float32)
    return E, {d: dict(fertility=np.nan, unk_rate=np.nan, trunc_rate=0.0) for d in DOM_IDX}

def encode_fake(cfg, texts):
    X, diag = encode_tfidf(cfg, texts)
    R = np.random.default_rng(cfg["seed"]).normal(size=(X.shape[1], cfg["dim"])).astype(np.float32)
    noise = np.random.default_rng(cfg["seed"] + 99).normal(scale=0.02 * cfg["seed"], size=(len(texts), cfg["dim"]))
    return (X @ R + noise).astype(np.float32), diag

ENCODERS = dict(hf_mean=encode_hf_mean, st=encode_st, fasttext=encode_fasttext, tfidf=encode_tfidf, fake=encode_fake)

def model_sha(repo):
    try:
        from huggingface_hub import model_info
        return model_info(repo).sha
    except Exception:
        return None

# %%
# ---------------------------------------------------------------- 5. encode all models (cached)
texts = df["text"].tolist()
EMB, DIAG, TIMING, SHA = {}, {}, {}, {}
for name, cfg in MODELS.items():
    f, fd = OUT / "emb" / f"{name}.npy", OUT / "emb" / f"{name}.diag.json"
    t = time.perf_counter()
    if f.exists() and fd.exists():
        EMB[name] = np.load(f); DIAG[name] = json.loads(fd.read_text()); TIMING[name] = 0.0
        print(f"{name:<16} cached {EMB[name].shape}"); continue
    try:
        E, diag = ENCODERS[cfg["kind"]](cfg, texts)
    except Exception as e:                             # one failing model must not kill the run
        print(f"!! {name} FAILED: {type(e).__name__}: {e}"); continue
    assert E.shape[0] == len(df) and np.isfinite(E).all(), name
    np.save(f, E); fd.write_text(json.dumps(diag))
    EMB[name], DIAG[name], TIMING[name] = E, diag, time.perf_counter() - t
    SHA[name] = model_sha(cfg["repo"]) if "repo" in cfg else None
    el = (time.perf_counter() - T0) / 60
    print(f"{name:<16} dim={E.shape[1]:<4} {TIMING[name]:6.1f}s  ({len(df)/max(TIMING[name],1e-9):7.1f} texts/s)  "
          f"elapsed {el:4.1f}/{BUDGET_MIN} min")
CANDIDATES = [m for m in CANDIDATES if m in EMB]
METHODS = [m for m in MODELS if m in EMB]
print("Encoded:", METHODS)

# %% [markdown]
# ## Evaluation
# For each (model, task) the job below returns **out-of-fold (OOF) predictions** — every text is predicted exactly once,
# by a model that never saw it. All metrics, CIs, tests and the audit are computed from these saved predictions.

# %%
# ---------------------------------------------------------------- 6. evaluation core
FOLDS = {t: list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(Y[t])), Y[t]))
         for t in TASKS}

def _probe(X, y, folds, pca_dim=None):
    oof, fold_f1, chosen_C = np.full(len(y), -1), [], []
    for tr, te in folds:
        steps = [StandardScaler()] + ([PCA(pca_dim, random_state=SEED)] if pca_dim else []) + [
            LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                 scoring="f1_macro", max_iter=3000, n_jobs=1)]
        clf = make_pipeline(*steps).fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
        fold_f1.append(f1_score(y[te], oof[te], average="macro")); chosen_C.append(float(clf[-1].C_[0]))
    return oof, fold_f1, chosen_C

def _knn(X, y, folds):
    oof = np.full(len(y), -1)
    for tr, te in folds:
        oof[te] = KNeighborsClassifier(KNN_K, metric="cosine", algorithm="brute").fit(X[tr], y[tr]).predict(X[te])
    return oof

def eval_job(name, task, X):
    y, folds = Y[task], FOLDS[task]
    Xn = l2norm(X)                                           # unit length -> scale differences removed
    p_oof, p_fold, p_C = _probe(X, y, folds)
    q_oof, q_fold, _ = _probe(X, y, folds, pca_dim=min(PCA_DIM, X.shape[1]))
    k_oof = _knn(Xn, y, folds)
    km = KMeans(len(np.unique(y)), n_init=10, random_state=SEED).fit_predict(Xn)
    return dict(model=name, task=task, probe=p_oof, probe_fold_f1=p_fold, probe_C=p_C,
                pca=q_oof, pca_fold_f1=q_fold, knn=k_oof, kmeans=km,
                nmi=normalized_mutual_info_score(y, km), ari=adjusted_rand_score(y, km),
                silhouette=silhouette_score(Xn, y, metric="cosine"))

# meta-embedding preview: CONCAT of the two domain candidates (unit-normalised first)
d1c = next((m for m in CANDIDATES if MODELS[m]["role"] == "D1 candidate"), None)
d2c = next((m for m in CANDIDATES if MODELS[m]["role"] == "D2 candidate"), None)
EXTRA = {}
if d1c and d2c:
    EXTRA[f"CONCAT({d1c}+{d2c})"] = np.hstack([l2norm(EMB[d1c]), l2norm(EMB[d2c])])
ALL_X = {**EMB, **EXTRA}

t = time.perf_counter()
jobs = [(m, tk) for m in ALL_X for tk in TASKS]
R = Parallel(n_jobs=-1, verbose=0)(delayed(eval_job)(m, tk, ALL_X[m][TASK_IDX[tk]]) for m, tk in jobs)
RES = {(r["model"], r["task"]): r for r in R}
TIMING["evaluation"] = time.perf_counter() - t
print(f"{len(jobs)} evaluation jobs in {TIMING['evaluation']:.0f}s")

# %%
# ---------------------------------------------------------------- 7. metrics + bootstrap CIs
def cm_counts(y, p, C):
    return np.bincount(y * C + p, minlength=C * C).reshape(C, C)       # rows = gold, cols = predicted

def macro_f1_from_cm(cm):
    tp = np.diag(cm).astype(float); pred = cm.sum(0); gold = cm.sum(1)
    P = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    Rc = np.divide(tp, gold, out=np.zeros_like(tp), where=gold > 0)
    F = np.divide(2 * P * Rc, P + Rc, out=np.zeros_like(tp), where=(P + Rc) > 0)
    return F.mean(), P, Rc, F

def boot_scores(y, p, C, idx):
    """Vectorised bootstrap: one confusion matrix per resample. idx shape (B, n)."""
    B = idx.shape[0]
    flat = (np.arange(B)[:, None] * C * C + y[idx] * C + p[idx]).ravel()
    cm = np.bincount(flat, minlength=B * C * C).reshape(B, C, C).astype(float)
    tp = np.einsum("bii->bi", cm); pred = cm.sum(1); gold = cm.sum(2)
    P = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    Rc = np.divide(tp, gold, out=np.zeros_like(tp), where=gold > 0)
    F = np.divide(2 * P * Rc, P + Rc, out=np.zeros_like(tp), where=(P + Rc) > 0)
    return tp.sum(1) / idx.shape[1], F.mean(1)

rng = np.random.default_rng(SEED)
BOOT_IDX = {t: rng.integers(0, len(Y[t]), size=(N_BOOT, len(Y[t]))) for t in TASKS}   # shared -> paired
BOOT = {}
rows = []
for (m, t), r in RES.items():
    y, C = Y[t], len(LABELS[t])
    row = dict(model=m, task=t, domain=TASK_DOMAIN[t], n=len(y))
    for key in ("probe", "knn", "pca"):
        acc = accuracy_score(y, r[key]); f1 = f1_score(y, r[key], average="macro")
        ba, bf = boot_scores(y, r[key], C, BOOT_IDX[t])
        BOOT[(m, t, key)] = bf
        row.update({f"{key}_acc": acc, f"{key}_acc_lo": np.percentile(ba, 2.5), f"{key}_acc_hi": np.percentile(ba, 97.5),
                    f"{key}_f1": f1, f"{key}_f1_lo": np.percentile(bf, 2.5), f"{key}_f1_hi": np.percentile(bf, 97.5)})
    row.update(probe_fold_f1_sd=float(np.std(r["probe_fold_f1"], ddof=1)), probe_C=str(r["probe_C"]),
               nmi=r["nmi"], ari=r["ari"], silhouette=r["silhouette"],
               pca_retention=row["pca_f1"] / row["probe_f1"])
    rows.append(row)
T_MAIN = pd.DataFrame(rows)
T_MAIN["model"] = pd.Categorical(T_MAIN["model"], list(ALL_X)); T_MAIN["task"] = pd.Categorical(T_MAIN["task"], TASKS)
T_MAIN = T_MAIN.sort_values(["task", "model"]).reset_index(drop=True)
T_MAIN.to_csv(OUT / "table_main_full.csv", index=False)

fmt = lambda v, lo, hi: f"{100*v:.1f} [{100*lo:.1f}, {100*hi:.1f}]"
T_SHOW = pd.DataFrame({
    "Task": T_MAIN["task"], "Model": T_MAIN["model"],
    "Probe Acc % [95% CI]": [fmt(*x) for x in T_MAIN[["probe_acc", "probe_acc_lo", "probe_acc_hi"]].values],
    "Probe Macro-F1 % [95% CI]": [fmt(*x) for x in T_MAIN[["probe_f1", "probe_f1_lo", "probe_f1_hi"]].values],
    "kNN Macro-F1 %": (100 * T_MAIN["knn_f1"]).round(1),
    "NMI": T_MAIN["nmi"].round(3), "ARI": T_MAIN["ari"].round(3), "Silhouette": T_MAIN["silhouette"].round(3)})
T_SHOW.to_csv(OUT / "table_main.csv", index=False)
display_(T_SHOW)

# %%
# ---------------------------------------------------------------- 8. domain winners
D2_TASKS = [t for t in TASKS if TASK_DOMAIN[t] == "D2_informal_ugc"]

# Plain boolean masks instead of DataFrame.query(): on Kaggle pandas routes query() through numexpr,
# which cannot compare string columns ("NumExpr 2 does not support Unicode").
def main_val(m, t, col):
    s = T_MAIN.loc[(T_MAIN["model"].astype(str) == m) & (T_MAIN["task"].astype(str) == t), col]
    assert len(s) == 1, (m, t, col); return float(s.iat[0])

def dom_val(m, col):
    s = T_DOM.loc[T_DOM["model"] == m, col]
    assert len(s) == 1, (m, col); return float(s.iat[0])

def domain_scores(m):
    f = {t: main_val(m, t, "probe_f1") for t in TASKS}
    d2b = np.mean([BOOT[(m, t, "probe")] for t in D2_TASKS], axis=0)   # independent resamples per task
    return dict(model=m, D1_news_F1=f["news_topic_7"],
                D1_lo=np.percentile(BOOT[(m, "news_topic_7", "probe")], 2.5),
                D1_hi=np.percentile(BOOT[(m, "news_topic_7", "probe")], 97.5),
                D2_mean_F1=np.mean([f[t] for t in D2_TASKS]), D2_lo=np.percentile(d2b, 2.5), D2_hi=np.percentile(d2b, 97.5),
                role=MODELS.get(m, {}).get("role", "meta-embedding preview"))
T_DOM = pd.DataFrame([domain_scores(m) for m in ALL_X])
WIN = {"D1": T_DOM[T_DOM.model.isin(CANDIDATES)].sort_values("D1_news_F1", ascending=False).model.tolist(),
       "D2": T_DOM[T_DOM.model.isin(CANDIDATES)].sort_values("D2_mean_F1", ascending=False).model.tolist()}
T_DOM.to_csv(OUT / "table_domain.csv", index=False)
display_(T_DOM.round(4))
print(f"Domain 1 (formal news) winner : {WIN['D1'][0]}   (runner-up {WIN['D1'][1]})")
print(f"Domain 2 (informal UGC) winner: {WIN['D2'][0]}   (runner-up {WIN['D2'][1]})")
floor = "TFIDF-char-SVD"
if floor in EMB:
    for dk, col in (("D1", "D1_news_F1"), ("D2", "D2_mean_F1")):
        fv = dom_val(floor, col); wv = dom_val(WIN[dk][0], col)
        print(f"  {dk}: winner beats TF-IDF floor by {100*(wv-fv):+.1f} macro-F1 points")

# %%
# ---------------------------------------------------------------- 9. significance tests
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p, dtype=float); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj

sig = []
for t in TASKS:
    order = T_MAIN[(T_MAIN.task == t) & T_MAIN.model.isin(CANDIDATES)].sort_values("probe_f1", ascending=False).model.tolist()
    a, b = order[0], order[1]
    y = Y[t]; ca = RES[(a, t)]["probe"] == y; cb = RES[(b, t)]["probe"] == y
    n01, n10 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))            # discordant pairs
    p_mc = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    d = BOOT[(a, t, "probe")] - BOOT[(b, t, "probe")]
    p_bs = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    sig.append(dict(task=t, best=a, runner_up=b, delta_F1=main_val(a, t, "probe_f1") - main_val(b, t, "probe_f1"),
                    delta_lo=np.percentile(d, 2.5), delta_hi=np.percentile(d, 97.5),
                    best_only_correct=n01, runnerup_only_correct=n10, p_mcnemar=p_mc, p_boot=p_bs))
T_SIG = pd.DataFrame(sig)
T_SIG["p_mcnemar_holm"] = holm(T_SIG.p_mcnemar); T_SIG["p_boot_holm"] = holm(T_SIG.p_boot)
T_SIG["significant@0.05"] = (T_SIG.p_mcnemar_holm < 0.05) & (T_SIG.p_boot_holm < 0.05)
T_SIG.to_csv(OUT / "table_significance.csv", index=False)
display_(T_SIG.round(4))

# Friedman + Nemenyi over blocks = (task, fold) fold-level probe macro-F1 (4 x 5 = 20 blocks)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}   # Demšar (2006), Table 5a
F = np.array([[RES[(m, t)]["probe_fold_f1"][k] for m in METHODS] for t in TASKS for k in range(N_FOLDS)])
RANKS = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, F)        # 1 = best, ties averaged
AVG_RANK = dict(zip(METHODS, RANKS.mean(0)))
fr = stats.friedmanchisquare(*F.T)
kM, NB = len(METHODS), F.shape[0]
CD = Q05[kM] * math.sqrt(kM * (kM + 1) / (6 * NB))
T_RANK = pd.DataFrame({"model": METHODS, "avg_rank": RANKS.mean(0)}).sort_values("avg_rank")
T_RANK.to_csv(OUT / "table_friedman.csv", index=False)
print(f"Friedman chi2={fr.statistic:.3f}, p={fr.pvalue:.3g}; Nemenyi CD = {Q05[kM]} * sqrt({kM}*{kM+1}/(6*{NB})) = {CD:.3f}")
print("Caveat: folds of one task share data, so blocks are not fully independent — read CD as descriptive.")
display_(T_RANK.round(3))

# %%
# ---------------------------------------------------------------- 10. limitation diagnostics
def linear_cka(X, Yv):
    X = X - X.mean(0); Yv = Yv - Yv.mean(0)
    return float(np.linalg.norm(Yv.T @ X, "fro") ** 2 / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Yv.T @ Yv, "fro")))

def anisotropy(X):
    """Exact mean cosine over all i != j pairs: (||sum u_i||^2 - n) / (n(n-1)), u_i unit vectors."""
    U = l2norm(X.astype(np.float64)); n = len(U); s = U.sum(0)
    return float((s @ s - n) / (n * (n - 1)))

diag_rows = []
for m in METHODS:
    for d, idx in DOM_IDX.items():
        nr = np.linalg.norm(EMB[m][idx], axis=1)
        dg = DIAG[m][d]
        diag_rows.append(dict(model=m, domain=d, dim=EMB[m].shape[1], norm_mean=nr.mean(), norm_cv=nr.std() / nr.mean(),
                              anisotropy=anisotropy(EMB[m][idx]), fertility=dg["fertility"],
                              unk_or_oov_rate=dg["unk_rate"], truncation_rate=dg["trunc_rate"],
                              pca64_retention=float(T_MAIN[(T_MAIN.model == m) & (T_MAIN.task.map(TASK_DOMAIN) == d)]
                                                    .pipe(lambda x: x.pca_f1.sum() / x.probe_f1.sum()))))
T_DIAG = pd.DataFrame(diag_rows)
T_DIAG.to_csv(OUT / "table_diagnostics.csv", index=False)
display_(T_DIAG.round(4))

CKA = {d: pd.DataFrame([[linear_cka(EMB[a][idx], EMB[b][idx]) for b in METHODS] for a in METHODS],
                       index=METHODS, columns=METHODS) for d, idx in DOM_IDX.items()}
for d in CKA: CKA[d].to_csv(OUT / f"table_cka_{d}.csv")
orc = []
for t in TASKS:
    for i, a in enumerate(CANDIDATES):
        for b in CANDIDATES[i + 1:]:
            ca = RES[(a, t)]["probe"] == Y[t]; cb = RES[(b, t)]["probe"] == Y[t]
            orc.append(dict(task=t, a=a, b=b, acc_a=ca.mean(), acc_b=cb.mean(), oracle_acc=(ca | cb).mean(),
                            headroom=(ca | cb).mean() - max(ca.mean(), cb.mean())))
T_ORACLE = pd.DataFrame(orc); T_ORACLE.to_csv(OUT / "table_oracle_complementarity.csv", index=False)
display_(T_ORACLE.groupby(["a", "b"])[["oracle_acc", "headroom"]].mean().sort_values("headroom", ascending=False).round(4))

# %% [markdown]
# ## Figures

# %%
# ---------------------------------------------------------------- 11. figures
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
PAL = ["#3b6fb6", "#d1495b", "#edae49", "#00798c", "#8d8d8d", "#6a4c93"]
COL = {m: PAL[i % len(PAL)] for i, m in enumerate(ALL_X)}
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / "fig" / f"{name}.png", bbox_inches="tight"); plt.show(); plt.close(fig)

# (a) main result: probe macro-F1 with 95% CI
fig, ax = plt.subplots(figsize=(10, 4)); w = 0.8 / len(ALL_X); x = np.arange(len(TASKS))
for i, m in enumerate(ALL_X):
    s = T_MAIN[T_MAIN.model == m].set_index("task").loc[TASKS]
    ax.bar(x + i * w, 100 * s.probe_f1, w, color=COL[m], label=m,
           yerr=[100 * (s.probe_f1 - s.probe_f1_lo), 100 * (s.probe_f1_hi - s.probe_f1)], capsize=2, ecolor="#333")
ax.set_xticks(x + w * (len(ALL_X) - 1) / 2, TASKS); ax.set_ylabel("Linear-probe macro-F1 (%)")
ax.set_title("Frozen-embedding linear probe, 5-fold OOF, 95% bootstrap CI"); ax.legend(ncol=3, fontsize=7, frameon=False)
save(fig, "01_probe_macro_f1")

# (b) domain scores
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
for ax, (col, lo, hi, ttl) in zip(axs, [("D1_news_F1", "D1_lo", "D1_hi", "Domain 1 — formal news"),
                                        ("D2_mean_F1", "D2_lo", "D2_hi", "Domain 2 — informal UGC (mean of 3 tasks)")]):
    s = T_DOM.sort_values(col)
    ax.barh(s.model, 100 * s[col], color=[COL[m] for m in s.model],
            xerr=[100 * (s[col] - s[lo]), 100 * (s[hi] - s[col])], capsize=3)
    for yy, (v, h) in enumerate(zip(100 * s[col], 100 * s[hi])):
        ax.text(h + 1.0, yy, f"{v:.1f}", va="center", fontsize=8)      # label right of the CI whisker
    ax.set_xlim(0, min(100, 100 * s[hi].max() + 12)); ax.set_title(ttl); ax.set_xlabel("macro-F1 (%)")
save(fig, "02_domain_scores")

# (c) confusion matrices of each task's best candidate (row-normalised, counts annotated)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, t in zip(axs, TASKS):
    best = T_MAIN[(T_MAIN.task == t) & T_MAIN.model.isin(CANDIDATES)].sort_values("probe_f1").model.iloc[-1]
    cm = cm_counts(Y[t], RES[(best, t)]["probe"], len(LABELS[t])); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]}\n{100*cmn[i,j]:.0f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(LABELS[t])), LABELS[t], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(LABELS[t])), LABELS[t], fontsize=7); ax.set_title(f"{t}\n{best}", fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
save(fig, "03_confusion_best")

# (d) critical-difference diagram
fig, ax = plt.subplots(figsize=(8, 2.2)); lo_r, hi_r = 1, kM
ax.set_xlim(lo_r - 0.3, hi_r + 0.3); ax.set_ylim(0, 1); ax.invert_xaxis(); ax.axis("off")
ax.hlines(0.8, lo_r, hi_r, color="k")
for r in range(lo_r, hi_r + 1):
    ax.vlines(r, 0.78, 0.82, color="k"); ax.text(r, 0.84, str(r), ha="center", va="bottom", fontsize=8)
srt = sorted(AVG_RANK.items(), key=lambda z: z[1])
for i, (m, r) in enumerate(srt):
    yy = 0.6 - 0.12 * i
    ax.plot([r, r], [0.8, yy], color=COL[m]); ax.text(r, yy - 0.06, f"{m} ({r:.2f})", ha="center", fontsize=7.5, color=COL[m])
ax.hlines(0.95, lo_r, lo_r + CD, color="r", lw=2); ax.text(lo_r + CD / 2, 0.98, f"CD = {CD:.2f}", ha="center", color="r", fontsize=8)
ax.set_title(f"Average rank (1 = best) over {NB} task×fold blocks — Friedman p = {fr.pvalue:.2g}", fontsize=9)
save(fig, "04_critical_difference")

# (e) redundancy: linear CKA per domain
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, (d, M) in zip(axs, CKA.items()):
    ax.imshow(M.values, cmap="viridis", vmin=0, vmax=1)
    for i in range(len(M)):
        for j in range(len(M)): ax.text(j, i, f"{M.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                                        color="white" if M.values[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(M)), M.columns, rotation=40, ha="right", fontsize=7); ax.set_yticks(range(len(M)), M.index, fontsize=7)
    ax.set_title(f"Linear CKA — {d}\n(1 = redundant, low = complementary)", fontsize=8.5)
save(fig, "05_cka_redundancy")

# (f) limitation diagnostics
fig, axs = plt.subplots(1, 4, figsize=(17, 3.4))
for ax, (col, ttl) in zip(axs, [("anisotropy", "Anisotropy (mean pairwise cosine)"), ("norm_cv", "Norm CV (scale spread)"),
                                ("pca64_retention", "Info retained at PCA-64 (F1 ratio)"), ("truncation_rate", "Truncation rate")]):
    pv = T_DIAG.pivot(index="model", columns="domain", values=col).loc[METHODS]
    pv.plot.bar(ax=ax, color=["#3b6fb6", "#d1495b"], legend=False, rot=40)
    if col == "anisotropy": fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
                                       bbox_to_anchor=(0.5, 1.08))
    ax.set_title(ttl, fontsize=8.5); ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=7)
save(fig, "06_limitation_diagnostics")

# (g) t-SNE of the two domain winners (qualitative only — distances in t-SNE are not metric)
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
for r_, m in enumerate([WIN["D1"][0], WIN["D2"][0]]):
    for c_, (d, idx) in enumerate(DOM_IDX.items()):
        Z = TSNE(2, init="pca", random_state=SEED, perplexity=30).fit_transform(l2norm(EMB[m][idx]))
        lab = df.iloc[idx]["source"] + ":" + df.iloc[idx]["label_name"]
        for k, g in enumerate(sorted(lab.unique())):
            mk = (lab == g).values
            axs[r_, c_].scatter(Z[mk, 0], Z[mk, 1], s=5, label=g, color=plt.cm.tab20(k % 20))
        axs[r_, c_].set_title(f"{m} — {d}", fontsize=9); axs[r_, c_].set_xticks([]); axs[r_, c_].set_yticks([])
        axs[r_, c_].legend(fontsize=5.5, markerscale=2, frameon=False, ncol=2)
save(fig, "07_tsne_winners")

# (h) runtime
fig, ax = plt.subplots(figsize=(7, 2.8)); tk = {k: v for k, v in TIMING.items()}
ax.barh(list(tk), list(tk.values()), color="#00798c"); ax.set_xlabel("seconds")
ax.set_title(f"Runtime — total {(time.perf_counter()-T0)/60:.1f} min (budget {BUDGET_MIN} min)")
save(fig, "08_runtime")

# %% [markdown]
# ## Audit — every reported number recomputed from the saved predictions

# %%
# ---------------------------------------------------------------- 12. audit
oof_rows = []
n_checks = 0
for (m, t), r in RES.items():
    y, C = Y[t], len(LABELS[t])
    # (1) every text predicted exactly once, by a fold that did not train on it
    assert (r["probe"] >= 0).all() and (r["knn"] >= 0).all()
    covered = np.zeros(len(y), int)
    for tr, te in FOLDS[t]:
        assert len(np.intersect1d(tr, te)) == 0; covered[te] += 1
    assert (covered == 1).all()
    row = T_MAIN[(T_MAIN.model == m) & (T_MAIN.task == t)].iloc[0]
    for key in ("probe", "knn", "pca"):
        cm = cm_counts(y, r[key], C)
        acc_hand = np.trace(cm) / cm.sum(); f1_hand = macro_f1_from_cm(cm)[0]
        # (2) hand formula == sklearn == table value
        assert abs(acc_hand - accuracy_score(y, r[key])) < 1e-12 and abs(acc_hand - row[f"{key}_acc"]) < 1e-12
        assert abs(f1_hand - f1_score(y, r[key], average="macro")) < 1e-12 and abs(f1_hand - row[f"{key}_f1"]) < 1e-12
        # (3) CI is ordered
        assert row[f"{key}_f1_lo"] <= row[f"{key}_f1_hi"]
        n_checks += 3
    # (4) bootstrap is reproducible from the seed (first task = first draw of default_rng(SEED))
    if t == TASKS[0]:
        b2 = boot_scores(y, r["probe"], C, np.random.default_rng(SEED).integers(0, len(y), size=(N_BOOT, len(y))))[1]
        assert np.array_equal(b2, BOOT[(m, t, "probe")]); n_checks += 1
    oof_rows.append(pd.DataFrame({"uid": df["uid"].values[TASK_IDX[t]], "model": m, "task": t, "gold": y,
                                  "pred_probe": r["probe"], "pred_knn": r["knn"], "pred_pca64": r["pca"],
                                  "kmeans_cluster": r["kmeans"]}))
OOF = pd.concat(oof_rows, ignore_index=True); OOF.to_parquet(OUT / "oof_predictions.parquet", index=False)
assert abs(CD - Q05[kM] * math.sqrt(kM * (kM + 1) / (6 * NB))) < 1e-12
for d in CKA: assert np.allclose(CKA[d].values, CKA[d].values.T, atol=1e-6) and np.allclose(np.diag(CKA[d].values), 1, atol=1e-6)
print(f"AUDIT PASSED — {n_checks} metric checks + fold, CKA and CD checks.")

# worked example the examiner can follow by hand
t = "hate_binary_2"; m = WIN["D2"][0]; y = Y[t]; cm = cm_counts(y, RES[(m, t)]["probe"], 2)
mf, P, Rc, Fc = macro_f1_from_cm(cm)
print(f"\nWorked example — {m} on {t} (n = {cm.sum()})")
print(pd.DataFrame(cm, index=[f"gold {l}" for l in LABELS[t]], columns=[f"pred {l}" for l in LABELS[t]]))
for i, l in enumerate(LABELS[t]):
    tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
    print(f"  {l:>9}: P = {tp}/({tp}+{fp}) = {P[i]:.4f} | R = {tp}/({tp}+{fn}) = {Rc[i]:.4f} | "
          f"F1 = 2·P·R/(P+R) = {Fc[i]:.4f}")
print(f"  macro-F1 = ({' + '.join(f'{v:.4f}' for v in Fc)}) / {len(Fc)} = {mf:.4f}")
print(f"  accuracy = {np.trace(cm)}/{cm.sum()} = {np.trace(cm)/cm.sum():.4f}")

# %%
# ---------------------------------------------------------------- 13. LaTeX + manifest
lt = T_DOM[["model", "role", "D1_news_F1", "D1_lo", "D1_hi", "D2_mean_F1", "D2_lo", "D2_hi"]].copy()
lt["Domain 1 F1 [95% CI]"] = [fmt(a, b, c) for a, b, c in lt[["D1_news_F1", "D1_lo", "D1_hi"]].values]
lt["Domain 2 F1 [95% CI]"] = [fmt(a, b, c) for a, b, c in lt[["D2_mean_F1", "D2_lo", "D2_hi"]].values]
(OUT / "table_domain.tex").write_text(lt[["model", "role", "Domain 1 F1 [95% CI]", "Domain 2 F1 [95% CI]"]]
                                      .to_latex(index=False, caption="Frozen-embedding linear-probe macro-F1 (\\%) per domain.",
                                                label="tab:domain"), encoding="utf-8")
(OUT / "table_main.tex").write_text(T_SHOW.to_latex(index=False, caption="Per-task results (5-fold OOF).",
                                                    label="tab:main"), encoding="utf-8")
TOTAL_MIN = (time.perf_counter() - T0) / 60
manifest = dict(total_minutes=round(TOTAL_MIN, 2), budget_minutes=BUDGET_MIN, device=DEVICE,
                gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, smoke=SMOKE,
                versions=dict(python=platform.python_version(), torch=torch.__version__, sklearn=sklearn.__version__,
                              numpy=np.__version__, scipy=scipy.__version__),
                seed=SEED, folds=N_FOLDS, bootstrap=N_BOOT, Cs=CS, knn_k=KNN_K, pca_dim=PCA_DIM, max_len=MAX_LEN,
                models={m: dict(**{k: v for k, v in MODELS[m].items()}, sha=SHA.get(m)) for m in METHODS},
                timing_seconds=TIMING, winners=dict(D1=WIN["D1"][0], D2=WIN["D2"][0]),
                friedman=dict(chi2=fr.statistic, p=fr.pvalue, CD=CD, blocks=NB))
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
print(f"Total runtime {TOTAL_MIN:.1f} min (budget {BUDGET_MIN}). Results in {OUT.resolve()}")
print(f"WINNERS -> Domain 1: {WIN['D1'][0]} | Domain 2: {WIN['D2'][0]}")
