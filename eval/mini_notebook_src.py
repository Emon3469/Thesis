# %% [markdown]
# # Bangla domain models on 3 small datasets — 5 experiments
# | Dataset | Texts | Models |
# |---|---|---|
# | `domain_1.csv` — formal news | 10 (5 topics × 2) | **Domain 1 model** (BanglaBERT) |
# | `domain_2.csv` — reviews, social media, Banglish | 10 (5 classes × 2) | **Domain 2 model** (BanglaSBERT) |
# | `merged_dataset.csv` — the same 20 rows | 20 (10 classes × 2) | Domain 1 model + Domain 2 model + mE5-base |
#
# **Protocol (identical in every experiment):** each class has exactly 2 texts, so every text has one correct
# *partner*. Leave-one-out retrieval by cosine similarity:
# - **R@1** = share of texts whose nearest neighbour is the partner (chance = 1/(n−1))
# - **MRR** = mean of 1/rank(partner) (chance = H(n−1)/(n−1))
# - **Mean rank** of the partner among the n−1 other texts
# - **d′** = (mean partner cosine − mean other-class cosine) / pooled SD — scale-free, comparable across models
# - **Domain@1** (merged only) = share of texts whose nearest neighbour is from the same domain
# - 95% Clopper–Pearson CI for R@1; permutation test of MRR (10,000 re-pairings), Holm-corrected over all experiments
#
# **Kaggle settings:** GPU on (CPU also works: only 20 texts), Internet ON, attach the `bangla-mini-3-datasets` dataset.

# %%
# ---------------------------------------------------------------- 1. setup
import os, sys, time, json, gc, glob, math, platform, subprocess, warnings
from pathlib import Path
T0 = time.perf_counter()
SMOKE = os.environ.get("EVAL_SMOKE") == "1"
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ   # set only inside Kaggle notebooks
if ON_KAGGLE:
    os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning)
if not SMOKE:
    try:
        from normalizer import normalize  # noqa  (csebuetnlp normalizer, required by BanglaBERT)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/csebuetnlp/normalizer"])

import numpy as np, pandas as pd, torch
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
show = (lambda x: display(x)) if "ipykernel" in sys.modules else print
print(f"python {platform.python_version()} | torch {torch.__version__} | device {DEVICE} | SMOKE={SMOKE}")

# %%
# ---------------------------------------------------------------- 2. config
SEED, N_PERM, BATCH = 42, 10_000, 32
MAX_LEN = {"D1_formal_news": 256, "D2_informal_ugc": 128}      # same as the 2,100-text evaluation
MODELS = {
    "BanglaBERT":  dict(kind="hf_mean", repo="csebuetnlp/banglabert", bn_norm=True),
    "BanglaSBERT": dict(kind="st", repo="shihab17/bangla-sentence-transformer"),
    "mE5-base":    dict(kind="st", repo="intfloat/multilingual-e5-base", prefix="query: "),
}
# Domain models as designated in the study plan. (On the 2,100-text evaluation the measured best was
# mE5-base for both domains; to test that instead set DOMAIN_MODEL = {"D1": "mE5-base", "D2": "mE5-base"}.)
DOMAIN_MODEL = {"D1": "BanglaBERT", "D2": "BanglaSBERT"}
PLAN = {"domain_1": [DOMAIN_MODEL["D1"]],
        "domain_2": [DOMAIN_MODEL["D2"]],
        "merged_dataset": list(dict.fromkeys([DOMAIN_MODEL["D1"], DOMAIN_MODEL["D2"], "BanglaBERT", "BanglaSBERT", "mE5-base"]))}
if SMOKE:
    MODELS = {m: dict(kind="fake", seed=i + 1) for i, m in enumerate(MODELS)}
OUT = Path("/kaggle/working/mini_results" if ON_KAGGLE else ("mini_results_smoke" if SMOKE else "mini_results"))
(OUT / "per_query").mkdir(parents=True, exist_ok=True); (OUT / "fig").mkdir(exist_ok=True)
for ds, ms in PLAN.items(): print(f"{ds:<15} -> {ms}")

# %%
# ---------------------------------------------------------------- 3. data
def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) or sorted(glob.glob(f"**/{name}", recursive=True))
    assert hits, f"{name} not found — attach the dataset"
    return hits[0]
DATA = {ds: pd.read_csv(find(f"{ds}.csv")) for ds in PLAN}
for ds, d in DATA.items():
    assert d["class"].value_counts().eq(2).all() and d.uid.is_unique, ds
assert set(DATA["merged_dataset"].uid) == set(DATA["domain_1"].uid) | set(DATA["domain_2"].uid)
assert len(DATA["merged_dataset"]) == 20
show(DATA["merged_dataset"][["uid", "domain", "source", "class", "script"]])

# %%
# ---------------------------------------------------------------- 4. encoders (one text -> one vector, no cross-text context)
def encode_hf_mean(cfg, texts, max_lens):
    from transformers import AutoTokenizer, AutoModel
    if cfg.get("bn_norm"):
        from normalizer import normalize as bn_normalize
        texts = [bn_normalize(t) for t in texts]
    tok = AutoTokenizer.from_pretrained(cfg["repo"]); model = AutoModel.from_pretrained(cfg["repo"]).to(DEVICE).eval()
    E = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i, (t, L) in enumerate(zip(texts, max_lens)):              # one text at a time -> no padding effects
        enc = tok(t, truncation=True, max_length=L, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            E[i] = model(**enc).last_hidden_state[0].mean(0).float().cpu().numpy()   # mean pooling
    del model; gc.collect(); torch.cuda.empty_cache()
    return E

def encode_st(cfg, texts, max_lens):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["repo"], device=DEVICE); pre = cfg.get("prefix", ""); E = None
    for i, (t, L) in enumerate(zip(texts, max_lens)):
        model.max_seq_length = L
        e = model.encode([pre + t], convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)[0]
        if E is None: E = np.zeros((len(texts), len(e)), dtype=np.float32)
        E[i] = e
    del model; gc.collect(); torch.cuda.empty_cache()
    return E

def encode_fake(cfg, texts, max_lens):      # local pipeline test only: deterministic char-hash features
    rng = np.random.default_rng(cfg["seed"]); R = rng.normal(size=(4096, 128))
    X = np.zeros((len(texts), 4096))
    for i, t in enumerate(texts):
        for k in range(len(t) - 2): X[i, hash(t[k:k + 3]) % 4096] += 1
    return (X @ R).astype(np.float32)

ENC = dict(hf_mean=encode_hf_mean, st=encode_st, fake=encode_fake)

def model_sha(repo):
    try:
        from huggingface_hub import model_info; return model_info(repo).sha
    except Exception:
        return None

# %%
# ---------------------------------------------------------------- 5. encode every needed model once on the 20 unique texts
U = DATA["merged_dataset"].reset_index(drop=True)          # 20 unique texts; the other datasets are subsets
needed = list(dict.fromkeys(m for ms in PLAN.values() for m in ms))
EMB, TIMING, SHA = {}, {}, {}
for m in needed:
    t = time.perf_counter()
    EMB[m] = ENC[MODELS[m]["kind"]](MODELS[m], U.text.tolist(), [MAX_LEN[d] for d in U.domain])
    assert np.isfinite(EMB[m]).all()
    TIMING[m] = time.perf_counter() - t; SHA[m] = model_sha(MODELS[m].get("repo", "")) if not SMOKE else None
    np.save(OUT / f"emb_{m}.npy", EMB[m])
    print(f"{m:<12} dim={EMB[m].shape[1]} {TIMING[m]:.1f}s")
ROW = {u: i for i, u in enumerate(U.uid)}

# %%
# ---------------------------------------------------------------- 6. evaluation protocol
def unit(X): return X / np.linalg.norm(X, axis=1, keepdims=True)

def evaluate(S, cls, dom):
    n = len(cls); rows = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        order = sorted(others, key=lambda j: -S[i, j])
        partner = next(j for j in others if cls[j] == cls[i]); rank = order.index(partner) + 1; nn = order[0]
        rows.append(dict(i=i, nn=nn, partner=partner, rank=rank, rr=1 / rank, hit=int(nn == partner),
                         cos_partner=S[i, partner], cos_nn=S[i, nn], same_domain_nn=int(dom[nn] == dom[i])))
    q = pd.DataFrame(rows)
    iu = np.triu_indices(n, 1); same = cls[iu[0]] == cls[iu[1]]; w, b = S[iu][same], S[iu][~same]
    return q, w, b

def metrics(q, w, b):
    pooled = math.sqrt((w.var(ddof=1) + b.var(ddof=1)) / 2)
    return dict(R1=q.hit.mean(), R1_hits=int(q.hit.sum()), MRR=q.rr.mean(), MeanRank=q["rank"].mean(),
                within_cos=w.mean(), between_cos=b.mean(), margin=w.mean() - b.mean(),
                d_prime=(w.mean() - b.mean()) / pooled, Domain1=q.same_domain_nn.mean())

def mrr_only(S, cls):
    tot = 0.0
    for i in range(len(cls)):
        s = S[i].copy(); s[i] = -np.inf; p = next(j for j in range(len(cls)) if j != i and cls[j] == cls[i])
        tot += 1 / (1 + np.sum(s > s[p]))
    return tot / len(cls)

def clopper_pearson(k, n, a=0.05):
    return (stats.beta.ppf(a / 2, k, n - k + 1) if k else 0.0, stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0)

# %%
# ---------------------------------------------------------------- 7. run the experiments
RES, QRY, SIM = [], {}, {}
for ds, models in PLAN.items():
    d = DATA[ds]; cls = d["class"].to_numpy(); dom = d["domain"].to_numpy(); n = len(d)
    for m in models:
        S = unit(EMB[m][[ROW[u] for u in d.uid]]) @ unit(EMB[m][[ROW[u] for u in d.uid]]).T
        q, w, b = evaluate(S, cls, dom); met = metrics(q, w, b)
        prng = np.random.default_rng(SEED)
        ge = sum(mrr_only(S, prng.permutation(cls)) >= met["MRR"] - 1e-12 for _ in range(N_PERM))
        lo, hi = clopper_pearson(met["R1_hits"], n)
        role = ("Domain 1 model" if m == DOMAIN_MODEL["D1"] else "") + (" / " if m == DOMAIN_MODEL["D1"] == DOMAIN_MODEL["D2"] else "") + \
               ("Domain 2 model" if m == DOMAIN_MODEL["D2"] else "")
        RES.append(dict(dataset=ds, model=m, role=role or "comparison", n=n, **met, R1_lo=lo, R1_hi=hi,
                        chance_R1=1 / (n - 1), chance_MRR=sum(1 / r for r in range(1, n)) / (n - 1),
                        p_perm_MRR=(ge + 1) / (N_PERM + 1)))
        if ds == "merged_dataset":          # break the merged result down by the query's domain
            for dname in ("D1_formal_news", "D2_informal_ugc"):
                sub = q[dom[q.i] == dname]
                RES[-1][f"MRR_{dname[:2]}_queries"] = sub.rr.mean(); RES[-1][f"R1_{dname[:2]}_queries"] = sub.hit.mean()
        q.insert(0, "query_uid", d.uid.to_numpy()[q.i]); q.insert(1, "query_class", cls[q.i])
        q["nn_uid"] = d.uid.to_numpy()[q.nn]; q["nn_class"] = cls[q.nn]; q["partner_uid"] = d.uid.to_numpy()[q.partner]
        q["query_script"] = d.script.to_numpy()[q.i]; q = q.drop(columns=["i", "nn", "partner"])
        QRY[(m, ds)] = q; SIM[(m, ds)] = S
        q.round(4).to_csv(OUT / "per_query" / f"{ds}__{m}.csv", index=False, encoding="utf-8-sig")

T = pd.DataFrame(RES)
p = T.p_perm_MRR.to_numpy(); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
for r_, i in enumerate(o):
    run = max(run, (len(p) - r_) * p[i]); adj[i] = min(1.0, run)
T["p_MRR_holm"] = adj; T["beats_chance@0.05"] = T.p_MRR_holm < 0.05
T.to_csv(OUT / "results.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for ds in PLAN:
    print(f"\n===== {ds} =====")
    cols = ["model", "role", "n", "R1_hits", "R1", "R1_lo", "R1_hi", "chance_R1", "MRR", "chance_MRR", "MeanRank",
            "within_cos", "between_cos", "d_prime", "p_perm_MRR", "p_MRR_holm"] + \
           (["Domain1", "MRR_D1_queries", "MRR_D2_queries"] if ds == "merged_dataset" else [])
    show(T[T.dataset == ds][cols].round(4))

# %%
# ---------------------------------------------------------------- 8. comparison: does each domain model keep its strength in the merged pool?
cmp_rows = []
for dk, ds in (("D1", "domain_1"), ("D2", "domain_2")):
    m = DOMAIN_MODEL[dk]; qd = QRY[(m, ds)].set_index("query_uid"); qm = QRY[(m, "merged_dataset")].set_index("query_uid")
    j = qd.join(qm, rsuffix="_merged"); assert (j.rank_merged >= j["rank"]).all()   # superset of candidates
    cmp_rows.append(dict(domain=dk, model=m, MRR_alone=j.rr.mean(), MRR_in_merged=j.rr_merged.mean(),
                         R1_alone=j.hit.mean(), R1_in_merged=j.hit_merged.mean(),
                         partners_pushed_down=int((j.rank_merged > j["rank"]).sum()),
                         cross_domain_intruders=int((j.rank_merged - j["rank"]).sum())))
CMP = pd.DataFrame(cmp_rows); CMP.to_csv(OUT / "domain_model_alone_vs_merged.csv", index=False); show(CMP.round(4))
BANG = pd.concat([q[q.query_script.str.startswith("banglish")].assign(model=m, dataset=ds)
                  [["dataset", "model", "query_uid", "query_class", "rank", "hit", "nn_class"]]
                  for (m, ds), q in QRY.items()], ignore_index=True)
BANG.to_csv(OUT / "banglish_cross_script.csv", index=False); show(BANG)

# %%
# ---------------------------------------------------------------- 9. figures
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 8})
keys = [(m, ds) for ds, ms in PLAN.items() for m in ms]
fig, axs = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 5))
for ax, (m, ds) in zip(np.atleast_1d(axs), keys):
    S = SIM[(m, ds)].copy(); np.fill_diagonal(S, np.nan); d = DATA[ds]; off = S[~np.isnan(S)]
    im = ax.imshow(S, cmap="magma", vmin=off.min(), vmax=off.max())
    labs = [f"{c} ({u[-4:]})" for c, u in zip(d["class"], d.uid)]
    ax.set_xticks(range(len(d)), labs, rotation=90, fontsize=5.5); ax.set_yticks(range(len(d)), labs, fontsize=5.5)
    for k in range(0, len(d), 2): ax.add_patch(plt.Rectangle((k - .5, k - .5), 2, 2, fill=False, ec="cyan", lw=1.2))
    r = T[(T.model == m) & (T.dataset == ds)].iloc[0]
    ax.set_title(f"{ds} | {m}\nR@1={r.R1:.2f} MRR={r.MRR:.3f} d'={r.d_prime:.2f}", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout(); fig.savefig(OUT / "fig" / "similarity_heatmaps.png", bbox_inches="tight"); plt.show(); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 3.8)); lab = [f"{ds}\n{m}" for m, ds in keys]; x = np.arange(len(keys))
COL = {"BanglaBERT": "#3b6fb6", "BanglaSBERT": "#d1495b", "mE5-base": "#edae49"}
for ax, (met, ttl) in zip(axs, [("R1", "R@1 (95% Clopper-Pearson CI)"), ("MRR", "MRR (dashed = chance)"), ("d_prime", "d'")]):
    v = [T[(T.model == m) & (T.dataset == ds)][met].iat[0] for m, ds in keys]
    err = None
    if met == "R1":
        err = [[T[(T.model == m) & (T.dataset == ds)].pipe(lambda r: (r.R1 - r.R1_lo).iat[0]) for m, ds in keys],
               [T[(T.model == m) & (T.dataset == ds)].pipe(lambda r: (r.R1_hi - r.R1).iat[0]) for m, ds in keys]]
    ax.bar(x, v, color=[COL[m] for m, _ in keys], yerr=err, capsize=3)
    for xx, vv in zip(x, v): ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=7)
    if met == "MRR":
        for xx, (m, ds) in zip(x, keys): ax.hlines(T[T.dataset == ds].chance_MRR.iat[0], xx - .4, xx + .4, ls="--", color="k")
    ax.set_xticks(x, lab, fontsize=6.5); ax.set_title(ttl); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "fig" / "metrics.png", bbox_inches="tight"); plt.show(); plt.close(fig)

# %%
# ---------------------------------------------------------------- 10. audit + worked example + manifest
checks = 0
for (m, ds), q in QRY.items():
    d = DATA[ds]; cls = d["class"].to_numpy(); S = SIM[(m, ds)]; X = EMB[m][[ROW[u] for u in d.uid]].astype(np.float64)
    S2 = np.array([[a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) for b in X] for a in X])
    assert np.allclose(S, S2, atol=1e-5); checks += 1
    r = T[(T.model == m) & (T.dataset == ds)].iloc[0]
    assert abs(r.R1 - q.hit.sum() / len(q)) < 1e-12 and abs(r.MRR - (1 / q["rank"]).mean()) < 1e-12
    assert abs(r.MRR - mrr_only(S, cls)) < 1e-9; checks += 3
    uid = d.uid.to_numpy()
    for _, row in q.iterrows():
        i = int(np.where(uid == row.query_uid)[0][0]); pp = int(np.where(uid == row.partner_uid)[0][0])
        assert row["rank"] == 1 + sum(S[i, j] > S[i, pp] for j in range(len(d)) if j not in (i, pp)); checks += 1
print(f"AUDIT PASSED — {checks} checks")
for (m, ds) in [(DOMAIN_MODEL["D1"], "domain_1"), (DOMAIN_MODEL["D2"], "domain_2")]:
    q = QRY[(m, ds)]
    print(f"\nWorked example — {m} on {ds}: partner ranks = {q['rank'].tolist()}")
    print(f"  R@1 = {q.hit.sum()}/{len(q)} = {q.hit.mean():.3f}")
    print(f"  MRR = ({' + '.join(f'{v:.4g}' for v in q.rr)}) / {len(q)} = {q.rr.mean():.4f}")
manifest = dict(minutes=round((time.perf_counter() - T0) / 60, 2), device=DEVICE, smoke=SMOKE, plan=PLAN,
                domain_model=DOMAIN_MODEL, max_len=MAX_LEN, seed=SEED, n_perm=N_PERM, timing_s=TIMING,
                model_sha=SHA, torch=torch.__version__)
(OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"\nDone in {manifest['minutes']} min -> {OUT}")
