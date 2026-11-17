"""Independent checks of the CONF test outputs (conf_output/conf_results, or conf_results_smoke with argument 'smoke')."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

R = Path(__file__).parent / ("conf_results_smoke" if "smoke" in sys.argv else "conf_output/conf_results")
RES = pd.read_csv(R / "conf_results.csv"); TST = pd.read_csv(R / "conf_tests.csv"); man = json.load(open(R / "manifest.json"))
PRED = pd.read_pickle(R / "conf_predictions.pkl"); LAB = pd.read_csv(R / "conf_labels.csv"); BOOT = np.load(R / "conf_bootstrap.npz")
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
fails = []
def chk(name, ok):
    if not ok: fails.append(name)

# 1. point scores re-computed from the saved predictions and labels with sklearn
for _, r in RES.iterrows():
    for t in TASKS:
        y = LAB.label[LAB.task == t].values.astype(int); p = PRED[(PRED.model == r.model) & (PRED.task == t)].pred.iat[0]
        chk(f"{r.model} {t}", abs(f1_score(y, p, average="macro", labels=list(range(y.max() + 1)), zero_division=0) - r[f"F1_{t}"]) < 1e-12)
    chk(f"{r.model} D2", abs(r.D2_F1 - np.mean([r[f"F1_{t}"] for t in TASKS[1:]])) < 1e-12)
    chk(f"{r.model} Overall", abs(r.Overall_F1 - (r.D1_F1 + r.D2_F1) / 2) < 1e-12)
    ov = BOOT[f"OV|{r.model}"]; chk(f"{r.model} CI", abs(np.percentile(ov, 2.5) - r.Overall_lo) < 1e-12 and abs(np.percentile(ov, 97.5) - r.Overall_hi) < 1e-12)
    chk(f"{r.model} boot", np.allclose(ov, (BOOT[f"D1|{r.model}"] + BOOT[f"D2|{r.model}"]) / 2))
print(f"[1] {len(RES)} models: per-task macro-F1 recomputed from saved predictions; domain averages and CIs re-derived")

# 2. confirmatory tests: deltas, CIs, one-sided p-values and Holm adjustment re-derived from the saved bootstrap samples
conf = TST[TST.family == "confirmatory"].reset_index(drop=True); nb = man["n_boot"]; pt = RES.set_index("model").Overall_F1
for _, t in TST[TST.metric == "Overall_F1"].iterrows():
    a, b = t.comparison.split(" vs "); d = BOOT[f"OV|{a}"] - BOOT[f"OV|{b}"]
    chk(f"delta {t.comparison}", abs(t.delta - (pt[a] - pt[b])) < 1e-12)
    chk(f"ci {t.comparison}", abs(np.percentile(d, 2.5) - t.ci_lo) < 1e-12 and abs(np.percentile(d, 97.5) - t.ci_hi) < 1e-12)
    chk(f"p {t.comparison}", abs((1 + (d <= 0).sum()) / (nb + 1) - t.p_one_sided) < 1e-12)
ps = conf.p_one_sided.values; m = len(ps); order = np.argsort(ps); holm = np.zeros(m); run = 0
for k, i in enumerate(order): run = max(run, min(1, (m - k) * ps[i])); holm[i] = run
chk("holm", np.allclose(holm, conf.p_holm)); chk("reject", ((holm < man["alpha"]) == conf.reject_H0.astype(bool)).all())
chk("family", sorted(conf.comparison) == sorted(f"{man['main']} vs {b}" for b in man["baselines"]))
chk("confirmed flag", bool(conf.reject_H0.astype(bool).all()) == man["confirmed"])
print(f"[2] confirmatory family re-derived: Holm-adjusted p = {dict(zip(conf.comparison, holm.round(4)))}")

# 3. the frozen model files used are the Step 4 selections (MD5 recorded before the run)
dev_man = json.load(open(Path(__file__).parent / "merge_output/merge_results/manifest.json"))
chk("main is the DEV selection", dev_man["selected_pure_w2v"] == man["main"])
chk("md5 list", set(man["frozen_md5"]) >= {"selected_pure_w2v", "News-W2V", "Informal-W2V"})
chk("conf sizes", man["conf_texts"] == LAB.task.value_counts().to_dict())
print(f"[3] main model = DEV selection ({dev_man['selected_pure_w2v']}); CONF sizes {man['conf_texts']}")
print("\nCONFIRMED:", man["confirmed"])
print("RESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails[:10]}")
