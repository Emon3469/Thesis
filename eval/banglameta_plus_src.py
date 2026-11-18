# %% [markdown]
# # BanglaMeta+ (1,200-d) — building and evaluating one Bangla embedding for classification *and* word similarity
#
# This notebook rebuilds **BanglaMeta+** from the frozen thesis model, step by step, evaluates it, and exports it.
# It uses only `numpy`, `pandas`, `scikit-learn` and `gensim`, and only files in this repository.
#
# **Input.** The frozen merged model selected in Step 4 and confirmed on CONF:
# `eval/merge_output/merge_results/models/selected_pure_w2v.f16.npy` (MD5 `10aa206f8462c82a7b16cf60054be70f`),
# 273,531 words × 900 dims = [News-W2V | Informal-W2V | Subs2vec-w2], each block a unit row scaled by $1/\sqrt3$.
#
# **Recipe.**
#
# | Step | What | Formula |
# |---|---|---|
# | 1 | normalise every row | $m(w) \leftarrow m(w)/\lVert m(w)\rVert$ |
# | 2 | all-but-the-top per 300-d block $k$ | $y_k = (m_k-\mu_k) - P_k^\top P_k (m_k-\mu_k)$, $P_k$ = top $D=4$ principal directions (fitted on the 50,000 most frequent rows) |
# | 3 | block weights | $s(w) = [\sqrt{a_k/3}\; y_k/\lVert y_k\rVert]_k$, $a = (1.65,\ 0,\ 1.35)$, $\tfrac13\sum a_k = 1$ |
# | 4 | join with the thesis News+Informal blocks | $c(w) = [\sqrt\varepsilon\, \hat m_{NI}(w);\ \sqrt{1-\varepsilon}\, s_{News}(w);\ \sqrt{1-\varepsilon}\, s_{Subs}(w)]$, $\varepsilon = 0.0005$ |
#
# **Why it works.** The classification probe standardises every feature, so the tiny weight $\sqrt\varepsilon$ costs the thesis
# features nothing (thesis Theorem 5). The cosine becomes $\cos_c = \varepsilon\cos_m + (1-\varepsilon)\cos_s$, i.e. almost
# entirely the similarity-tuned view. $D$, $a$ and $\varepsilon$ were chosen on a random half of the analogy questions only
# (section 7). No training is involved and no other model's vectors are used.
#
# $\hat m_{NI}$ = the thesis News + Informal blocks re-normalised (600-d). Subs2vec stays only in the similarity part: it adds no
# significant classification information (thesis check P2), and the dimension study (`eval/dimension_study`) shows that
# dropping it from the thesis part keeps or improves every score while using 300 fewer dimensions.
#
# **Output.** `banglameta_plus_1200d.bin` (word2vec binary, 273,531 × 1,200), checked byte for byte against the released
# file `models/banglameta_plus/banglameta_plus_1200d.bin`.

# %%
import bz2, gzip, hashlib, json, math, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
T0 = time.perf_counter()

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "eval").is_dir() and (p / "models").is_dir())
SRC = REPO / "eval" / "merge_output" / "merge_results" / "models"
DATA = REPO / "eval" / "w2v_kaggle_dataset"                   # eval_v2.parquet, analogy/, categories.json
WORK = REPO / "eval" / "banglameta_plus" / "work"; WORK.mkdir(parents=True, exist_ok=True)
RELEASED = REPO / "models" / "banglameta_plus" / "banglameta_plus_1200d.bin"

FROZEN_MD5 = "10aa206f8462c82a7b16cf60054be70f"
D_ABTT, WEIGHTS, FIT_ROWS, EPS = 4, (1.65, 0.0, 1.35), 50_000, 0.0005
CHUNK, SEED, TOPK_SEARCH = 16384, 42, 200_000

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def elapsed(): return f"{time.perf_counter() - T0:.0f}s"
print("repository:", REPO)

# %% [markdown]
# ## 1. Load the frozen thesis model and verify it

# %%
assert md5(SRC / "selected_pure_w2v.f16.npy") == FROZEN_MD5, "not the frozen Step 4 model"
U16 = np.load(SRC / "selected_pure_w2v.f16.npy", mmap_mode="r")
WORDS = (SRC / "selected_pure_w2v.words.txt").read_text(encoding="utf-8").split("\n")
IDX = {w: i for i, w in enumerate(WORDS)}
N = len(WORDS)
assert U16.shape == (N, 900) and len(IDX) == N
print(f"frozen model OK: {N:,} words x {U16.shape[1]} dims, MD5 {FROZEN_MD5}")

# %% [markdown]
# ## 2. Step 1 — normalise every row (float32, written to disk to keep memory low)

# %%
M = np.lib.format.open_memmap(WORK / "merged.f32.npy", mode="w+", dtype=np.float32, shape=(N, 900))
for s in range(0, N, CHUNK):
    B = np.asarray(U16[s:s + CHUNK], dtype=np.float32)
    M[s:s + CHUNK] = B / np.linalg.norm(B, axis=1, keepdims=True)
M.flush(); M = np.load(WORK / "merged.f32.npy", mmap_mode="r")
blk = np.linalg.norm(M[:5000].reshape(5000, 3, 300), axis=2)
print(f"row length 1 ± {np.abs(np.linalg.norm(M[:5000], axis=1) - 1).max():.1e}; "
      f"each block length 1/√3 = {1 / math.sqrt(3):.4f} (mean {blk.mean():.4f})  [{elapsed()}]")

# %% [markdown]
# ## 3. Step 2 — all-but-the-top in every block
# Word vectors share a few large common directions that carry frequency, not meaning (Mu & Viswanath, 2018).
# For each 300-d block we remove the block mean and its top $D = 4$ principal directions.

# %%
PARAMS = []
for k in range(3):
    bk = slice(300 * k, 300 * (k + 1))
    X = np.asarray(M[:FIT_ROWS, bk], np.float64)
    mu = X.mean(0); _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    for j in range(D_ABTT):                                  # deterministic sign
        if Vt[j, np.abs(Vt[j]).argmax()] < 0: Vt[j] *= -1
    PARAMS.append((mu.astype(np.float32), Vt[:D_ABTT].astype(np.float32)))
    share = (S[:D_ABTT] ** 2).sum() / (S ** 2).sum()
    print(f"block {['News', 'Informal', 'Subs2vec'][k]:<9} |mean| = {np.linalg.norm(mu):.3f}   top-{D_ABTT} directions hold "
          f"{share * 100:.1f}% of the variance (removed)")

# %% [markdown]
# ## 4. Steps 3 and 4 — block weights and the BanglaMeta+ row
# Weight 0 drops the Informal block from the similarity view (it is the noisy block for word tasks), so the similarity
# part is 600-d. The thesis part keeps its News and Informal blocks (600-d, re-normalised).

# %%
a_, b_ = np.float32(np.sqrt(EPS)), np.float32(np.sqrt(1 - EPS))
C = np.lib.format.open_memmap(WORK / "banglameta_plus.f32.npy", mode="w+", dtype=np.float32, shape=(N, 1200))
for s in range(0, N, CHUNK):
    Bm = np.asarray(M[s:s + CHUNK], np.float32); sim = np.zeros_like(Bm)
    for k in range(3):
        if WEIGHTS[k] == 0: continue
        bk = slice(300 * k, 300 * (k + 1)); mu, P = PARAMS[k]
        Y = Bm[:, bk] - mu; Y -= (Y @ P.T) @ P
        sim[:, bk] = Y / np.linalg.norm(Y, axis=1, keepdims=True) * np.sqrt(WEIGHTS[k] / 3)
    mni = np.asarray(M[s:s + CHUNK, :600]); mni = mni / np.linalg.norm(mni, axis=1, keepdims=True)
    C[s:s + CHUNK] = np.hstack([a_ * mni, b_ * sim[:, :300], b_ * sim[:, 600:900]])
C.flush(); C = np.load(WORK / "banglameta_plus.f32.npy", mmap_mode="r")
lens = np.linalg.norm(C[::97], axis=1)
print(f"BanglaMeta+: {C.shape[0]:,} x {C.shape[1]}, row length 1 ± {np.abs(lens - 1).max():.1e}  [{elapsed()}]")

# %% [markdown]
# **Check the cosine identity** $\cos_c = \varepsilon\cos_{NI} + (1-\varepsilon)\cos_s$ on real word pairs ($\cos_{NI}$ = thesis News+Informal part).

# %%
def row(U, w): return np.asarray(U[IDX[w]], np.float64)
def ni(w): v = row(M, w)[:600]; return v / np.linalg.norm(v)              # thesis News+Informal part
def sim_part(w):                                          # the similarity view, rebuilt independently for one word
    m = np.asarray(M[IDX[w]], np.float32); out = []
    for k in (0, 2):
        bk = slice(300 * k, 300 * (k + 1)); mu, P = PARAMS[k]
        y = m[bk] - mu; y -= (y @ P.T) @ P; out.append(y / np.linalg.norm(y) * math.sqrt(WEIGHTS[k] / 3))
    return np.concatenate(out).astype(np.float64)
rows = []
for u, v in [("রাজা", "রানী"), ("রাজা", "পুরুষ"), ("পুরুষ", "মহিলা"), ("ঢাকা", "চট্টগ্রাম"), ("রাজা", "ভাত")]:
    cm, cs, cc = ni(u) @ ni(v), sim_part(u) @ sim_part(v), row(C, u) @ row(C, v)
    rows.append(dict(pair=f"{u} – {v}", thesis_NI_cos=cm, similarity_view_cos=cs, plus_cos=cc, formula=EPS * cm + (1 - EPS) * cs))
    assert abs(cc - (EPS * cm + (1 - EPS) * cs)) < 1e-5
pd.DataFrame(rows).round(4)

# %% [markdown]
# ## 5. Nearest neighbours (exact cosine over all 273,531 rows)

# %%
def neighbours(w, k=8):
    q = np.asarray(C[IDX[w]], np.float32); sims = np.empty(N, np.float32)
    for s in range(0, N, 32768): sims[s:s + 32768] = np.asarray(C[s:s + 32768]) @ q
    sims[IDX[w]] = -np.inf; top = np.argsort(-sims)[:k]
    return ", ".join(f"{WORDS[i]} {sims[i]:.2f}" for i in top)
for w in ["ঢাকা", "ভালো", "বৃষ্টি", "রাজা", "খেলা"]:
    print(f"{w:<8} → {neighbours(w)}")
def analogy(a, b, c, k=3):
    t = np.asarray(C[IDX[b]], np.float32) - np.asarray(C[IDX[a]], np.float32) + np.asarray(C[IDX[c]], np.float32)
    sims = np.asarray(C[:TOPK_SEARCH]) @ t
    for w in (a, b, c): sims[IDX[w]] = -np.inf
    return [WORDS[i] for i in np.argsort(-sims)[:k]]
print("রাজা : রানী :: পুরুষ : ?  →", analogy("রাজা", "রানী", "পুরুষ"))

# %% [markdown]
# ## 6. Evaluation on the DEV half (the thesis definitions)
# * **Classification:** sentence vector = mean of the word rows → `StandardScaler` → `RidgeClassifierCV`, 5 stratified folds (seed 42).
# * **Analogy:** 3CosAdd over the first 200,000 rows, inputs excluded, unanswerable questions count as wrong (16,678 native questions).
# * **Similarity AUC:** same-category word pairs ranked above different-category pairs, on the 124 category words every source table contains.

# %%
ev = pd.read_parquet(DATA / "eval_v2.parquet"); ev["toks"] = ev.tokens.str.split()
dev = ev[ev.split == "dev"].reset_index(drop=True)
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]

def sent_matrix(U, toks_list):
    X = np.zeros((len(toks_list), U.shape[1]), np.float32)
    for i, toks in enumerate(toks_list):
        ids = [IDX[w] for w in toks if w in IDX]
        if ids: X[i] = np.asarray(U[np.array(ids)]).mean(0)
    return X
def probe_oof(X, y, folds):
    p = np.full(len(y), -1)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        p[te] = RidgeClassifierCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return p
F1 = {}
for t in TASKS:
    g = dev[dev.task == t]; y = g.label.values
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(g)), y))
    F1[t] = f1_score(y, probe_oof(sent_matrix(C, g.toks.tolist()), y, folds), average="macro")
D1, D2 = F1["news_topic_7"], np.mean([F1[t] for t in TASKS[1:]])
print({t: round(v * 100, 2) for t, v in F1.items()})
print(f"DEV  D1 news {D1 * 100:.2f}   D2 informal {D2 * 100:.2f}   Overall {(D1 + D2) / 2 * 100:.2f}   [{elapsed()}]")

# %%
ANALOGY = {p.stem: [l.split() for l in open(p, encoding="utf-8") if len(l.split()) == 4]
           for p in sorted((DATA / "analogy").glob("*.txt")) if p.stem != "Data_Stat"}
NATIVE = [f for f in ANALOGY if f != "Mikolov-Filtered"]
SEMANTIC = ["Division_District", "Gender", "Number_Ordinal", "Number_Date", "Number_Female"]
sidx = {w: i for i, w in enumerate(WORDS[:TOPK_SEARCH])}
qs = [(f, [sidx[w] for w in q]) for f in NATIVE for q in ANALOGY[f] if all(w in sidx for w in q)]
Bq = np.array([q for _, q in qs]); qf = np.array([f for f, _ in qs])
T = np.asarray(C[Bq[:, 1]]) - np.asarray(C[Bq[:, 0]]) + np.asarray(C[Bq[:, 2]])
best_s = np.full(len(Bq), -np.inf, np.float32); best_i = np.zeros(len(Bq), np.int64)
for r in range(0, TOPK_SEARCH, 32768):
    e = min(r + 32768, TOPK_SEARCH); Cm = np.asarray(C[r:e])
    for s in range(0, len(Bq), 1024):
        sl = slice(s, s + 1024); sim = T[sl] @ Cm.T; rr = np.arange(sim.shape[0])
        for j in (0, 1, 2):
            col = Bq[sl, j]; inside = (col >= r) & (col < e); sim[rr[inside], col[inside] - r] = -np.inf
        am = sim.argmax(1); mv = sim[rr, am]; upd = mv > best_s[sl]
        best_s[sl][upd] = mv[upd]; best_i[sl][upd] = am[upd] + r
ok = best_i == Bq[:, 3]
n_native = sum(len(ANALOGY[f]) for f in NATIVE); n_sem = sum(len(ANALOGY[f]) for f in SEMANTIC)
print(f"analogy native {ok.sum() / n_native * 100:.2f}%  ({ok.sum():,} of {n_native:,} questions)   "
      f"semantic {ok[np.isin(qf, SEMANTIC)].sum() / n_sem * 100:.2f}%   syntactic "
      f"{ok[~np.isin(qf, SEMANTIC)].sum() / (n_native - n_sem) * 100:.2f}%   [{elapsed()}]")

# %% [markdown]
# The similarity AUC uses the thesis word set: category words present in **every** source table (News-W2V, Informal-W2V,
# Subs2vec-w2, fastText). The Subs2vec and fastText vocabularies are rebuilt with the thesis rule (first 200k rows + every evaluation word).

# %%
CATS = json.loads((DATA / "categories.json").read_text(encoding="utf-8"))
NEEDED = {w for t in ev.toks for w in t} | {w for Q in ANALOGY.values() for q in Q for w in q} | {w for v in CATS.values() for w in v}
def table_words(lines, sep):
    out = []
    for n, line in enumerate(lines, 1):
        w, _, rest = line.partition(sep)
        if n <= TOPK_SEARCH or w in NEEDED: out.append(w)
    return set(out)
with bz2.open(REPO / "models" / "B_subs2vec" / "bn_300_2_sg_wxd.csv.bz2", "rt", encoding="utf-8") as f:
    next(f); subs = table_words(f, ",")
ft_file = REPO / "webapp" / "data" / "raw" / "cc.bn.300.vec.gz"
if ft_file.exists():
    with gzip.open(ft_file, "rt", encoding="utf-8", errors="ignore") as f:
        next(f); ft = table_words(f, " ")
else:                                                   # fall back to the stored fastText table word list
    ft = set((REPO / "webapp" / "data" / "models" / "fastText-cc.bn.words.txt").read_text(encoding="utf-8").split("\n"))
news = set((SRC / "News-W2V.words.txt").read_text(encoding="utf-8").split("\n"))
inf = set((SRC / "Informal-W2V.words.txt").read_text(encoding="utf-8").split("\n"))
common = [(w, c) for c, ws in CATS.items() for w in ws if w in news and w in inf and w in subs and w in ft and w in IDX]
Uc = np.asarray(C[[IDX[w] for w, _ in common]]); cs = np.array([c for _, c in common]); iu = np.triu_indices(len(common), 1)
auc = roc_auc_score((cs[iu[0]] == cs[iu[1]]).astype(int), (Uc @ Uc.T)[iu])
toks = {d: [w for t in g.toks for w in t] for d, g in dev.groupby("domain")}
cov = {d[:2]: np.mean([w in IDX for w in t]) for d, t in toks.items()}
print(f"similarity AUC {auc:.4f} on {len(common)} category words; token coverage D1 {cov['D1'] * 100:.2f}%, D2 {cov['D2'] * 100:.2f}%  [{elapsed()}]")

# %% [markdown]
# ## 7. How $D$, the block weights and $\varepsilon$ were chosen
# All three were chosen on a random half (seed 42, stratified by file) of the native analogy questions; the other half
# and the similarity AUC were never used for the choice. The search scripts are
# `webapp/backend/experiments/block_weights.py`, `build_abtt.py` and `eps_1200.py`; their stored results are shown here.
# The epsilon grid ties on the tune half, and ties go to the smaller epsilon (the rule used from the start).

# %%
SEL = REPO / "webapp" / "data"
rows = []
for d in range(1, 6):
    j = json.loads((SEL / f"block_weights_abtt{d}.json").read_text(encoding="utf-8"))["chosen"]
    rows.append(dict(D=d, weights=j["weights"], tune_half=j["tune"], test_half=j["test"], sim_auc=j["sim_auc"]))
print("all-but-the-top D (best weights for each D, chosen on the tune half):")
display(pd.DataFrame(rows).round(4))
comb = json.loads((REPO / "eval" / "dimension_study" / "eps_1200.json").read_text(encoding="utf-8"))
print(f"epsilon grid for the 1,200-d recipe (chosen: {EPS}):")
display(pd.DataFrame(comb["grid"]).round(4))

# %% [markdown]
# ## 8. Comparison with every other model
# Measured by `webapp/backend/experiments/evaluate_plus.py` (same definitions; CONF with the thesis's 10,000-sample paired
# bootstrap, which reproduces the official thesis CONF table exactly). CONF values of the two new models are exploratory.

# %%
res = json.loads((REPO / "eval" / "banglameta_plus_eval" / "results.json").read_text(encoding="utf-8"))
R = res["results"]
tab = pd.DataFrame([{"model": R[k]["name"], "dim": R[k]["dim"], "DEV Overall F1": R[k]["dev_Overall_F1"] * 100,
                     "CONF Overall F1": R[k]["conf_Overall_F1"] * 100,
                     "CONF 95% CI": f"[{R[k]['conf_Overall_lo'] * 100:.2f}, {R[k]['conf_Overall_hi'] * 100:.2f}]",
                     "Analogy %": R[k]["analogy_native_acc"] * 100, "Similarity AUC": R[k]["sim_auc"]} for k in res["models"]])
assert abs(R["merged_plus"]["dev_Overall_F1"] - (D1 + D2) / 2) < 1e-9          # this notebook reproduces it
assert abs(R["merged_plus"]["sim_auc"] - auc) < 1e-9
tab.round(3)

# %% [markdown]
# ## 9. Export (word2vec binary) and compare with the released file

# %%
OUT = WORK / "banglameta_plus_1200d.bin"
with open(OUT, "wb") as f:
    f.write(f"{N} 1200\n".encode("utf-8"))
    for s in range(0, N, 20000):
        B = np.asarray(C[s:s + 20000], dtype="<f4")
        f.write(b"".join(f"{w} ".encode("utf-8") + r.tobytes() for w, r in zip(WORDS[s:s + 20000], B)))
mine = md5(OUT)
print(f"rebuilt: {OUT.stat().st_size:,} bytes, MD5 {mine}")
if RELEASED.exists():
    rel = md5(RELEASED)
    print(f"released: MD5 {rel} ->", "IDENTICAL, byte for byte" if rel == mine else "DIFFERENT")
from gensim.models import KeyedVectors
kv = KeyedVectors.load_word2vec_format(str(OUT), binary=True, limit=3000)
assert kv.index_to_key == WORDS[:3000] and np.array_equal(kv.vectors, np.asarray(C[:3000]))
print("gensim reads the file back exactly;", kv.most_similar("ঢাকা", topn=3))

# %% [markdown]
# ## 10. Summary

# %%
summary = pd.DataFrame([
    ("words × dims", f"{N:,} × 1,200"),
    ("DEV Overall macro-F1", f"{(D1 + D2) / 2 * 100:.2f}  (thesis model 77.12, fastText 76.20)"),
    ("CONF Overall macro-F1 (exploratory)", f"{R['merged_plus']['conf_Overall_F1'] * 100:.2f}  (thesis model 77.17: a tie)"),
    ("Word analogy, native", f"{ok.sum() / n_native * 100:.2f}%  (thesis model 6.00%, fastText 7.57%)"),
    ("Similarity AUC", f"{auc:.3f}  (thesis model 0.819, News-W2V 0.881)"),
    ("File", f"{OUT.name}, MD5 {mine}"),
    ("Run time", elapsed()),
], columns=["", "BanglaMeta+"])
summary
