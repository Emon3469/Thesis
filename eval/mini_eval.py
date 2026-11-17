"""
Mini evaluation: 3 models x 3 datasets = 9 experiments on 10 + 10 hand-traceable samples.

Datasets
  domain_1.csv       10 formal news texts   = 5 topics x 2           (topics drawn with seed 42)
  domain_2.csv       10 informal texts      = 5 classes x 2:
                     book:positive, book:negative, social:positive, social:negative, hate:hate
                     Banglish stratum: each book class pairs 1 romanized (Banglish) review with
                     1 Bangla-script review of the same sentiment -> tests cross-script matching.
  merged_dataset.csv the exact same 20 rows (10 classes, labels prefixed by domain)

Why this protocol: with 10 texts a trained classifier cannot be cross-validated (news would have
1-2 texts per class). Instead every class has EXACTLY 2 members, so each text has exactly one
correct "partner". Leave-one-out retrieval with cosine similarity then gives metrics that can be
checked by hand from one per-query table:
  R@1        share of texts whose nearest neighbour is its partner        (chance = 1/(n-1))
  MRR        mean of 1/rank(partner)                                     (chance = H(n-1)/(n-1))
  MeanRank   mean rank of the partner among the n-1 other texts           (best = 1)
  Margin     mean cosine(partner pairs) - mean cosine(different-class pairs)
  d'         Margin / pooled SD of the two cosine groups (scale-free; comparable across models)
  Domain@1   merged only: share of texts whose nearest neighbour is from the same domain
Statistics: Clopper-Pearson 95% CI for R@1; permutation p-value for MRR (10,000 random re-pairings).

Embeddings: the vectors produced by the Kaggle run (results/emb/*.npy, model commits in
run_manifest.json). Each text is encoded independently of every other text, so a text has the
same vector in every dataset; the dataset only decides which texts compete in retrieval.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SEED, N_PERM = 42, 10_000
HERE = Path(__file__).parent
EMB_DIR = HERE / "kaggle_output" / "results" / "emb"
OUT = HERE / "mini"; (OUT / "per_query").mkdir(parents=True, exist_ok=True); (OUT / "fig").mkdir(exist_ok=True)
MODELS = ["BanglaSBERT", "BanglaBERT", "mE5-base"]
DATASETS = ["domain_1", "domain_2", "merged_dataset"]

ev = pd.read_parquet(HERE / "kaggle_dataset" / "eval_set.parquet")
ev["row"] = np.arange(len(ev))                     # row index into the embedding matrices
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ 1. sampling
d1_pool = ev[(ev.domain == "D1_formal_news") & (ev.latin_ratio <= 0.5)]      # formal Bangla only
topics = sorted(rng.choice(sorted(d1_pool.label_name.unique()), size=5, replace=False))
d1 = pd.concat([d1_pool[d1_pool.label_name == t].sample(2, random_state=int(rng.integers(1e9))) for t in topics])
d1["class"] = d1["label_name"]

SRC = {"banglabook_sentiment": "book", "sentnob_sentiment": "social", "hate_speech": "hate"}
d2_specs = [("banglabook_sentiment", "positive"), ("banglabook_sentiment", "negative"),
            ("sentnob_sentiment", "positive"), ("sentnob_sentiment", "negative"), ("hate_speech", "hate")]
BANGLISH = {("banglabook_sentiment", "positive"): "ev01288", ("banglabook_sentiment", "negative"): "ev01710"}
parts = []
for src, lab in d2_specs:
    pool = ev[(ev.source == src) & (ev.label_name == lab)]
    if (src, lab) in BANGLISH:
        b = pool[pool.uid == BANGLISH[(src, lab)]]
        assert len(b) == 1 and b.latin_ratio.iat[0] > 0.5
        other = pool[pool.latin_ratio <= 0.05].sample(1, random_state=int(rng.integers(1e9)))
        parts.append(pd.concat([b, other]))
    else:
        parts.append(pool[pool.latin_ratio <= 0.05].sample(2, random_state=int(rng.integers(1e9))))
d2 = pd.concat(parts); d2["class"] = d2["source"].map(SRC) + ":" + d2["label_name"]

COLS = ["uid", "text", "domain", "source", "task", "label_name", "class", "latin_ratio", "row"]
d1, d2 = d1[COLS].reset_index(drop=True), d2[COLS].reset_index(drop=True)
d1["script"] = "bangla"; d2["script"] = np.where(d2.latin_ratio > 0.5, "banglish(romanized)", "bangla")
merged = pd.concat([d1.assign(**{"class": "D1:" + d1["class"]}), d2.assign(**{"class": "D2:" + d2["class"]})],
                   ignore_index=True)
DATA = {"domain_1": d1, "domain_2": d2, "merged_dataset": merged}
for name, d in DATA.items():
    assert d["class"].value_counts().eq(2).all() and d.uid.is_unique
    d.drop(columns="row").to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
assert set(merged.uid) == set(d1.uid) | set(d2.uid) and len(merged) == 20
print("Domain 1 topics:", topics)

# ------------------------------------------------------------------ 2. one evaluation function
def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def evaluate(S, cls, dom=None):
    """S: n x n cosine matrix, cls: class per text (exactly 2 per class). Returns metrics + per-query table."""
    n = len(cls); rows = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        order = sorted(others, key=lambda j: -S[i, j])                  # most similar first
        partner = next(j for j in others if cls[j] == cls[i])
        rank = order.index(partner) + 1
        nn = order[0]
        rows.append(dict(i=i, nn=nn, partner=partner, rank=rank, rr=1 / rank, hit=int(nn == partner),
                         cos_partner=S[i, partner], cos_nn=S[i, nn],
                         same_domain_nn=(int(dom[nn] == dom[i]) if dom is not None else np.nan),
                         intruders=rank - 1))
    q = pd.DataFrame(rows)
    iu = np.triu_indices(n, 1); same = cls[iu[0]] == cls[iu[1]]
    w, b = S[iu][same], S[iu][~same]
    pooled = math.sqrt((w.var(ddof=1) + b.var(ddof=1)) / 2)
    m = dict(n=n, R1=q.hit.mean(), MRR=q.rr.mean(), MeanRank=q["rank"].mean(),
             within_cos=w.mean(), between_cos=b.mean(), margin=w.mean() - b.mean(),
             d_prime=(w.mean() - b.mean()) / pooled, n_within_pairs=len(w), n_between_pairs=len(b))
    if dom is not None:
        m["Domain@1"] = q.same_domain_nn.mean()
    return m, q

def mrr_only(S, cls):
    n = len(cls); tot = 0.0
    for i in range(n):
        sims = S[i].copy(); sims[i] = -np.inf
        p = next(j for j in range(n) if j != i and cls[j] == cls[i])
        tot += 1 / (1 + np.sum(sims > sims[p]))
    return tot / n

def clopper_pearson(k, n, a=0.05):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

# ------------------------------------------------------------------ 3. the 9 experiments
EMB = {m: np.load(EMB_DIR / f"{m}.npy").astype(np.float64) for m in MODELS}
SIM, RES, QRY = {}, [], {}
for ds, d in DATA.items():
    cls = d["class"].to_numpy(); dom = d["domain"].to_numpy() if ds == "merged_dataset" else None
    n = len(d); chance_r1 = 1 / (n - 1); chance_mrr = sum(1 / r for r in range(1, n)) / (n - 1)
    for m in MODELS:
        U = unit(EMB[m][d["row"].to_numpy()]); S = U @ U.T
        met, q = evaluate(S, cls, dom)
        # permutation test: random re-pairing of texts into classes (keeps the 2-per-class structure)
        prng = np.random.default_rng(SEED); ge = 0
        for _ in range(N_PERM):
            ge += mrr_only(S, prng.permutation(cls)) >= met["MRR"] - 1e-12
        k = int(q.hit.sum()); lo, hi = clopper_pearson(k, n)
        met.update(model=m, dataset=ds, R1_hits=k, R1_lo=lo, R1_hi=hi, chance_R1=chance_r1,
                   chance_MRR=chance_mrr, p_perm_MRR=(ge + 1) / (N_PERM + 1))
        RES.append(met); SIM[(m, ds)] = S
        q.insert(0, "query_uid", d.uid.to_numpy()[q.i]); q.insert(1, "query_class", cls[q.i])
        q["nn_uid"] = d.uid.to_numpy()[q.nn]; q["nn_class"] = cls[q.nn]; q["partner_uid"] = d.uid.to_numpy()[q.partner]
        q["query_script"] = d.script.to_numpy()[q.i]
        q = q.drop(columns=["i", "nn", "partner"]); QRY[(m, ds)] = q
        q.round(4).to_csv(OUT / "per_query" / f"{ds}__{m}.csv", index=False, encoding="utf-8-sig")

T = pd.DataFrame(RES)[["dataset", "model", "n", "R1_hits", "R1", "R1_lo", "R1_hi", "chance_R1", "MRR", "chance_MRR",
                       "p_perm_MRR", "MeanRank", "within_cos", "between_cos", "margin", "d_prime", "Domain@1"]]
p = T.p_perm_MRR.to_numpy(); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
for r_, i in enumerate(o):                        # Holm step-down over the 9 experiments
    run = max(run, (len(p) - r_) * p[i]); adj[i] = min(1.0, run)
T["p_MRR_holm"] = adj; T["beats_chance@0.05"] = T.p_MRR_holm < 0.05
T.to_csv(OUT / "results_9_experiments.csv", index=False)

# merged vs single-domain: how many texts lost rank because of the other domain's texts
inter = []
for m in MODELS:
    qm = QRY[(m, "merged_dataset")].set_index("query_uid")
    for ds in ("domain_1", "domain_2"):
        qd = QRY[(m, ds)].set_index("query_uid"); j = qd.join(qm, rsuffix="_merged")
        assert (j["rank_merged"] >= j["rank"]).all()      # superset of candidates -> rank can only worsen
        inter.append(dict(model=m, domain=ds, texts=len(j), rank_worsened=int((j.rank_merged > j["rank"]).sum()),
                          hits_lost=int(((j.hit == 1) & (j.hit_merged == 0)).sum()),
                          cross_domain_intruders=int((j.rank_merged - j["rank"]).sum())))
I = pd.DataFrame(inter); I.to_csv(OUT / "merged_interference.csv", index=False)

# Banglish cross-script check
bang = []
for m in MODELS:
    for ds in ("domain_2", "merged_dataset"):
        q = QRY[(m, ds)]; b = q[q.query_script.str.startswith("banglish")]
        for _, r in b.iterrows():
            bang.append(dict(model=m, dataset=ds, banglish_uid=r.query_uid, class_=r.query_class,
                             partner_rank=r["rank"], hit=r.hit, nn_class=r.nn_class))
B = pd.DataFrame(bang); B.to_csv(OUT / "banglish_cross_script.csv", index=False)

# ------------------------------------------------------------------ 4. audit (independent re-computation)
checks = 0
for r in RES:
    m, ds = r["model"], r["dataset"]; d = DATA[ds]; cls = d["class"].to_numpy(); S = SIM[(m, ds)]
    X = EMB[m][d["row"].to_numpy()]
    S2 = np.array([[a @ b / (np.linalg.norm(a) * np.linalg.norm(b)) for b in X] for a in X])   # loop cosine
    assert np.allclose(S, S2, atol=1e-12); checks += 1
    q = QRY[(m, ds)]
    assert abs(r["R1"] - q.hit.sum() / len(q)) < 1e-12 and abs(r["MRR"] - (1 / q["rank"]).mean()) < 1e-12
    assert abs(r["MRR"] - mrr_only(S, cls)) < 1e-12; checks += 3
    # brute-force rank: count others strictly more similar than the partner
    for _, row in q.iterrows():
        i = int(np.where(d.uid.to_numpy() == row.query_uid)[0][0]); p = int(np.where(d.uid.to_numpy() == row.partner_uid)[0][0])
        assert row["rank"] == 1 + sum(S[i, j] > S[i, p] for j in range(len(d)) if j not in (i, p)); checks += 1
print(f"AUDIT PASSED: {checks} checks")

# ------------------------------------------------------------------ 5. figures
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 8})
fig, axs = plt.subplots(3, 3, figsize=(16, 15))
for r_, m in enumerate(MODELS):
    for c_, ds in enumerate(DATASETS):
        ax = axs[r_, c_]; S = SIM[(m, ds)].copy(); np.fill_diagonal(S, np.nan); d = DATA[ds]
        off = S[~np.isnan(S)]; im = ax.imshow(S, cmap="magma", vmin=off.min(), vmax=off.max())
        labs = [f"{c} ({u[-4:]})" for c, u in zip(d["class"], d.uid)]
        ax.set_xticks(range(len(d)), labs, rotation=90, fontsize=5.5); ax.set_yticks(range(len(d)), labs, fontsize=5.5)
        for k in range(0, len(d), 2):
            ax.add_patch(plt.Rectangle((k - .5, k - .5), 2, 2, fill=False, ec="cyan", lw=1.2))
        rr = T[(T.model == m) & (T.dataset == ds)].iloc[0]
        ax.set_title(f"{m} | {ds}\nR@1={rr.R1:.2f}  MRR={rr.MRR:.3f}  d'={rr.d_prime:.2f}", fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.04)
fig.suptitle("Cosine similarity (colour range per panel = that model's own min..max; cyan = partner pairs)", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig" / "similarity_3x3.png", bbox_inches="tight"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 3.8)); COL = {"BanglaSBERT": "#d1495b", "BanglaBERT": "#3b6fb6", "mE5-base": "#edae49"}
x = np.arange(3); w = 0.26
for ax, (met, ttl) in zip(axs, [("R1", "R@1 (Clopper-Pearson 95% CI)"), ("MRR", "MRR (dashed = chance)"), ("d_prime", "d' (separation, scale-free)")]):
    for i, m in enumerate(MODELS):
        s = T[T.model == m].set_index("dataset").loc[DATASETS]
        err = [s.R1 - s.R1_lo, s.R1_hi - s.R1] if met == "R1" else None
        ax.bar(x + (i - 1) * w, s[met], w, color=COL[m], label=m, yerr=err, capsize=2)
        for xx, v in zip(x + (i - 1) * w, s[met]): ax.text(xx, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    if met == "MRR":
        for xx, ds in zip(x, DATASETS): ax.hlines(T[T.dataset == ds].chance_MRR.iat[0], xx - .4, xx + .4, ls="--", color="k")
    ax.set_xticks(x, DATASETS); ax.set_title(ttl); ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig" / "metrics_9_experiments.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ 6. print
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(d1[["uid", "class"]].to_string(index=False)); print(d2[["uid", "class", "script"]].to_string(index=False))
print(T.round(4).to_string(index=False)); print(I.to_string(index=False)); print(B.to_string(index=False))
