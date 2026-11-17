"""Independent consistency checks of the Kaggle Word2Vec pre-evaluation outputs (w2v_output/w2v_results)."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).parent / "w2v_output" / "w2v_results"
S = pd.read_csv(R / "screening_dev.csv"); P = pd.read_csv(R / "premerge_checks.csv"); M = pd.read_csv(R / "mini_results.csv")
man = json.load(open(R / "manifest.json"))
fails = []
def chk(name, ok):
    if not ok: fails.append(name)

# 1. screening: domain scores are the stated averages; analogy accuracies re-derived from per-file counts
for _, r in S.iterrows():
    chk(f"{r.model} D2", abs(r.D2_F1 - np.mean([r.F1_review_sentiment_3, r.F1_social_sentiment_3, r.F1_hate_binary_2])) < 1e-12)
    chk(f"{r.model} Overall", abs(r.Overall_F1 - (r.D1_F1 + r.D2_F1) / 2) < 1e-12)
    chk(f"{r.model} D1 CI", r.D1_lo <= r.D1_F1 + 0.02 and r.D1_F1 - 0.02 <= r.D1_hi)
    pf = json.loads(r.analogy_per_file)
    native = [k for k in pf if k != "Mikolov-Filtered"]
    n = sum(pf[k][0] for k in native); c = sum(pf[k][2] for k in native); a = sum(pf[k][1] for k in native)
    chk(f"{r.model} analogy acc", abs(r.analogy_native_acc - c / n) < 1e-12)
    chk(f"{r.model} analogy answerable", abs(r.analogy_native_answerable - a / n) < 1e-12)
    chk(f"{r.model} analogy total", n == 16678)
print(f"[1] screening: {len(S)} models, domain averages and analogy accuracies re-derived from per-file counts")

# 2. pair selection re-applied
subs = S[S.model.str.startswith("Subs2vec")]
best_subs = subs.sort_values("D2_F1", ascending=False).model.iat[0]
best_D1 = S.sort_values("D1_F1", ascending=False).model.iat[0]
best_D2 = S[S.model != best_D1].sort_values("D2_F1", ascending=False).model.iat[0]
chk("best_subs", best_subs == man["best_subs"]); chk("best_D1", best_D1 == man["best_D1"]); chk("best_D2", best_D2 == man["best_D2"])
print(f"[2] selection reproduced: best Subs2vec = {best_subs}, best D1 = {best_D1}, best D2 = {best_D2}")

# 3. P-check pass flags re-derived from their pre-registered thresholds
for _, r in P.iterrows():
    chk("P1", r.P1_pass == (r.P1_max_abs_error < 1e-6))
    chk("P2", r.P2_pass == bool(r["P2_I(B|A)_D2_lo"] > 0 and r["P2_I(A|B)_D1_lo"] > 0))
    for k in (100, 200, 300):
        chk(f"P3 k{k}", r[f"P3_svd_allowed_k{k}"] == bool(r[f"P3_risk_bound_k{k}"] < r.P3_gain_reference_nats))
        chk(f"P4 k{k}", r[f"P4_pass_k{k}"] == bool(0.4 <= r[f"P4_share_A_top{k}"] <= 0.6))
        chk(f"P3 energy k{k} in [0,1]", 0 <= r[f"P3_retained_energy_k{k}"] <= 1)
    chk("P5", r.P5_pass == bool(min(r["P5_B->A_median_cos"] - r["P5_B->A_random_p95"], r["P5_A->B_median_cos"] - r["P5_A->B_random_p95"]) > 0))
    chk("P6", r.P6_pass == (r["P6_heldout_P@1"] >= 0.5))
    chk("P2 domain CI order", r["P2_I(B|A)_D2_lo"] <= r["P2_I(B|A)_D2"] <= r["P2_I(B|A)_D2_hi"])
print(f"[3] P1-P6 pass flags re-derived for {len(P)} pair(s)")

# 4. the 3 small datasets: R@1, MRR, mean rank re-derived from the saved per-text ranks; chance levels
for _, r in M.iterrows():
    q = pd.read_csv(R / f"mini_{r.dataset}__{r.model}.csv")
    chk(f"mini {r.dataset}/{r.model} R1", abs(r.R1 - q.hit.mean()) < 1e-12)
    chk(f"mini {r.dataset}/{r.model} MRR", abs(r.MRR - (1 / q["rank"]).mean()) < 1e-12)
    chk(f"mini {r.dataset}/{r.model} rank", abs(r.MeanRank - q["rank"].mean()) < 1e-12)
    chk(f"mini {r.dataset} chance", abs(r.chance_MRR - sum(1 / k for k in range(1, r.n)) / (r.n - 1)) < 1e-12)
    chk(f"mini {r.dataset} classes", q["class"].value_counts().eq(2).all())
pv = M.p_perm.values; o = np.argsort(pv); adj = np.empty_like(pv); run = 0
for i_, i in enumerate(o): run = max(run, (len(pv) - i_) * pv[i]); adj[i] = min(1, run)
chk("Holm", np.allclose(adj, M.p_holm.values))
print(f"[4] small datasets: {len(M)} experiments re-derived from per-text ranks; Holm reproduced")
print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails[:10]}")
