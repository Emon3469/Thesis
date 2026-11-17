"""Independent consistency checks of the Step 4 merging outputs (merge_output/merge_results)."""
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).parent / "merge_output" / "merge_results"
S = pd.read_csv(R / "merge_dev.csv"); P = pd.read_csv(R / "premerge_checks.csv"); man = json.load(open(R / "manifest.json"))
fails = []
def chk(name, ok):
    if not ok: fails.append(name)

# 1. scores: domain averages, analogy accuracy from per-file counts, CI sanity
for _, r in S.iterrows():
    chk(f"{r.model} D2", abs(r.D2_F1 - np.mean([r.F1_review_sentiment_3, r.F1_social_sentiment_3, r.F1_hate_binary_2])) < 1e-12)
    chk(f"{r.model} Overall", abs(r.Overall_F1 - (r.D1_F1 + r.D2_F1) / 2) < 1e-12)
    pf = json.loads(r.analogy_per_file); native = [k for k in pf if k != "Mikolov-Filtered"]
    n = sum(pf[k][0] for k in native); c = sum(pf[k][2] for k in native)
    chk(f"{r.model} analogy", abs(r.analogy_native_acc - c / n) < 1e-12 and n == 16678)
    chk(f"{r.model} CI", r.D1_lo <= r.D1_hi and r.D2_lo <= r.D2_hi and r.dOverall_lo <= r.dOverall_hi)
print(f"[1] {len(S)} models: domain averages and analogy accuracies re-derived")

# 2. specialists, thresholds, eligibility, selection (pre-registered rule re-applied)
single = S[S.family == "single"]
th1, th2 = single.D1_F1.max() - man["margin"], single.D2_F1.max() - man["margin"]
chk("thresholds", abs(th1 - man["threshold_D1"]) < 1e-12 and abs(th2 - man["threshold_D2"]) < 1e-12)
chk("specialists", single.sort_values("D1_F1").model.iat[-1] == man["specialist_D1"] and single.sort_values("D2_F1").model.iat[-1] == man["specialist_D2"])
chk("best single", single.sort_values("Overall_F1").model.iat[-1] == man["best_single"])
elig = (S.D1_F1 >= th1) & (S.D2_F1 >= th2); chk("eligibility", (elig == S.eligible).all())
cand = S[S.family != "single"]
def select(pool):
    el = pool[(pool.D1_F1 >= th1) & (pool.D2_F1 >= th2)]
    return (el if len(el) else pool).sort_values("Overall_F1").model.iat[-1]
chk("selected pure", select(cand[cand.pure_w2v]) == man["selected_pure_w2v"]); chk("selected extended", select(cand) == man["selected_extended"])
best = S.set_index("model").Overall_F1[man["best_single"]]
chk("delta overall", np.allclose(S.dOverall_vs_best_single, S.Overall_F1 - best))
chk("delta D1/D2", np.allclose(S.dD1_vs_specialist, S.D1_F1 - single.D1_F1.max()) and np.allclose(S.dD2_vs_specialist, S.D2_F1 - single.D2_F1.max()))
print(f"[2] selection reproduced: pure Word2Vec = {man['selected_pure_w2v']} | extended = {man['selected_extended']}")

# 3. Theorem 5: block weights must not change the probe (weighted CONC == CONC on every task, up to rounding flips)
for fam in ("P", "Q"):
    g = S[(S.family == fam) & (S.model.str.contains("wCONC") | (S.model == f"{fam}: CONC"))]
    for c in ("F1_news_topic_7", "F1_review_sentiment_3", "F1_social_sentiment_3", "F1_hate_binary_2"):
        chk(f"weight invariance {fam} {c}", g[c].max() - g[c].min() < 2e-3)
print("[3] weighted concatenation gives the same macro-F1 as equal-energy concatenation (Theorem 5)")

# 4. P-check flags re-derived from thresholds
for _, r in P.iterrows():
    chk("P1", r.P1_pass == (r.P1_max_abs_error < 1e-6))
    chk("P2", r.P2_pass == bool(r["P2_I(B|A)_D2_lo"] > 0 and r["P2_I(A|B)_D1_lo"] > 0))
    for k in (100, 200, 300):
        chk(f"P3 k{k}", r[f"P3_svd_allowed_k{k}"] == bool(r[f"P3_risk_bound_k{k}"] < r.P3_gain_reference_nats))
        chk(f"P4 k{k}", r[f"P4_pass_k{k}"] == bool(abs(r[f"P4_share_A_top{k}"] - r.P4_target_share_A) <= 0.1))
    chk("P5", r.P5_pass == bool(min(r["P5_B->A_median_cos"] - r["P5_B->A_random_p95"], r["P5_A->B_median_cos"] - r["P5_A->B_random_p95"]) > 0))
    if not np.isnan(r["P6_heldout_P@1"]): chk("P6", r.P6_pass == (r["P6_heldout_P@1"] >= 0.5))
print(f"[4] P1-P6 flags re-derived for {len(P)} source sets")

# 5. saved models match the table sizes
for tag, name in (("selected_pure_w2v", man["selected_pure_w2v"]), ("News-W2V", "News-W2V"), ("Informal-W2V", "Informal-W2V")):
    f = R / "models" / f"{tag}.f16.npy"
    if f.exists():
        U = np.load(f, mmap_mode="r"); words = (R / "models" / f"{tag}.words.txt").read_text(encoding="utf-8").split("\n"); row = S[S.model == name].iloc[0]
        chk(f"saved {tag}", U.shape == (row.vocab, row.dim) and len(words) == row.vocab and abs(float(np.linalg.norm(U[:1000].astype(np.float32), axis=1).mean()) - 1) < 1e-2)
print("[5] saved model files match the evaluated tables (shape, vocabulary, unit rows)")
print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails[:10]}")
