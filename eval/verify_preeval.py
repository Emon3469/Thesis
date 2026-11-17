"""Independent checks of preeval/ outputs, recomputed from predictions.parquet."""
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

H = Path(__file__).parent; P = H / "preeval"
ev = pd.read_parquet(H / "kaggle_dataset" / "eval_set.parquet")
pr = pd.read_parquet(P / "predictions.parquet")
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
fails = []

# 1. DEV and CONF halves are disjoint, stratified, and reproducible from the seed
for t in TASKS:
    idx = np.where(ev.task.values == t)[0]; y = ev.label.values[idx]
    dev, conf = train_test_split(np.arange(len(idx)), test_size=0.5, stratify=y, random_state=42)
    du, cu = set(ev.uid.values[idx[dev]]), set(ev.uid.values[idx[conf]])
    assert not du & cu
    seen_dev = set(pr[(pr.task == t) & pr.config.str.startswith("DEV:")].uid)
    seen_conf = set(pr[(pr.task == t) & pr.config.str.startswith("CONF:")].uid)
    if seen_dev != du: fails.append(f"{t}: DEV uids differ")
    if seen_conf != cu: fails.append(f"{t}: CONF uids differ")
    if seen_dev & seen_conf: fails.append(f"{t}: DEV/CONF overlap")
print("[1] DEV/CONF disjoint and reproducible")

# 2. every text predicted exactly once per config/task; gold labels == eval set
g = pr.merge(ev[["uid", "label"]], on="uid")
assert (g.gold == g.label).all(); assert not pr.duplicated(["config", "task", "uid"]).any()
print(f"[2] {len(pr)} predictions: gold labels match, no duplicates")

# 3. reported DEV numbers == recomputed from predictions
def f1s(cfg_prefix):
    d = pr[pr.config == cfg_prefix]; return {t: f1_score(x.gold, x.probe, average="macro") for t, x in d.groupby("task")}
s3 = pd.read_csv(P / "stage3_methods_dev.csv"); s1 = pd.read_csv(P / "stage1_single_sources_dev.csv")
n = 0
for df_, kind in ((s1, "single"), (s3, None)):
    for _, r in df_.iterrows():
        a = r.alpha if not pd.isna(r.alpha) else np.nan
        key = f"DEV:{r.srcs}|{r.method}|{a if r.method == 'conc_alpha' else np.nan}"
        f = f1s(key)
        if not f: fails.append(f"missing {key}"); continue
        d1, d2 = f["news_topic_7"], np.mean([f[t] for t in TASKS[1:]])
        for name, v in (("D1", d1), ("D2", d2), ("Overall", (d1 + d2) / 2)):
            if abs(v - r[name]) > 1e-12: fails.append(f"{key} {name}: {v} vs {r[name]}")
        n += 1
print(f"[3] {n} DEV configurations recomputed")

# 4. CONF numbers recomputed
tc = pd.read_csv(P / "stage5_confirmation.csv")
for _, r in tc.iterrows():
    f = f1s(f"CONF:{r.config}"); d1, d2 = f["news_topic_7"], np.mean([f[t] for t in TASKS[1:]])
    for name, v in (("D1", d1), ("D2", d2), ("Overall", (d1 + d2) / 2)):
        if abs(v - r[name]) > 1e-12: fails.append(f"CONF {r.config} {name}")
    for s in ("D1", "D2", "Overall"):
        if not (r[f"{s}_lo"] <= r[s] + 1e-9 <= r[f"{s}_hi"] + 2e-9): fails.append(f"CONF {r.config} {s} outside CI")
print(f"[4] {len(tc)} CONF configurations recomputed; point estimates inside their CIs")

# 5. selection rule re-applied on DEV numbers gives the same selection
m = json.load(open(P / "manifest.json"))
cands = s1[s1.srcs != "TFIDF-char-SVD"]
elig = s3[(s3.method != "ensemble") & (s3.D1 >= cands.D1.max() - .01) & (s3.D2 >= cands.D2.max() - .01)]
pool = elig if len(elig) else s3[s3.method != "ensemble"]
sel = pool.sort_values("Overall", ascending=False).iloc[0]
if sel.srcs.split("+") != m["selected"]["srcs"] or sel.method != m["selected"]["method"]: fails.append("selection differs")
print(f"[5] selection rule reproduced: {sel.srcs} | {sel.method} (eligible set size {len(elig)})")

# 6. label-permutation control near chance
chance = {"news_topic_7": 1 / 7, "review_sentiment_3": 1 / 3, "social_sentiment_3": 1 / 3, "hate_binary_2": 1 / 2}
for t, v in m["label_permutation_f1"].items():
    if v > chance[t] + 0.10: fails.append(f"permutation control too high on {t}: {v}")
print("[6] permutation control:", {t: round(v, 3) for t, v in m["label_permutation_f1"].items()}, "chance:", {t: round(c, 3) for t, c in chance.items()})
print("\nRESULT:", "ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES"); [print("  ", f) for f in fails[:20]]
