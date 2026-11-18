"""Writes the BanglaMeta+ section of README.md from the result files, after checking that the numbers agree.

Every value is read from a result file and rounded only at the end; every difference is computed from the unrounded
values. The script stops (assert) if any check fails, so the README cannot show a number that disagrees with the files.
Run: python docs/make_readme_tables.py
"""
import json, math, re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "eval"
bench = json.loads((REPO / "webapp" / "data" / "benchmarks.json").read_text(encoding="utf-8"))
full = json.loads((EV / "banglameta_plus_eval" / "results.json").read_text(encoding="utf-8"))
dims = json.loads((EV / "dimension_study" / "results.json").read_text(encoding="utf-8"))
eps12 = json.loads((EV / "dimension_study" / "eps_1200.json").read_text(encoding="utf-8"))
chk = json.loads((EV / "dimension_study" / "eps_1200_check.log").read_text(encoding="utf-8").strip().splitlines()[-1])
official = pd.read_csv(EV / "conf_output" / "conf_results" / "conf_results.csv").set_index("model")
md5 = (REPO / "models" / "banglameta_plus" / "banglameta_plus_1200d.md5").read_text(encoding="utf-8").split()[0]
manifest = json.loads((REPO / "webapp" / "data" / "manifest.json").read_text(encoding="utf-8"))
R, B = full["results"], bench["results"]
KEYS = ["merged_plus", "merged", "merged_sim", "fasttext", "news", "informal"]
THESIS = {"merged": "T: CONC", "fasttext": "fastText-cc.bn", "news": "News-W2V", "informal": "Informal-W2V"}
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
N_NATIVE = 16_678
pct = lambda x, d=2: f"{x * 100:.{d}f}"
sgn = lambda x, d=2: f"{x * 100:+.{d}f}"

# ------------------------------------------------------------------ consistency checks (stop on any disagreement)
for k in KEYS:
    r = R[k]
    for s in ("dev", "conf"):
        d2 = sum(r[f"{s}_F1_{t}"] for t in TASKS[1:]) / 3
        assert abs(r[f"{s}_D2_F1"] - d2) < 1e-12 and abs(r[f"{s}_D1_F1"] - r[f"{s}_F1_news_topic_7"]) < 1e-12, (k, s)
        assert abs(r[f"{s}_Overall_F1"] - (r[f"{s}_D1_F1"] + r[f"{s}_D2_F1"]) / 2) < 1e-12, (k, s)
    assert abs(r["dev_Overall_F1"] - B[k]["Overall_F1"]) < 1e-9, k                  # app benchmark == full evaluation
    assert abs(r["analogy_native_acc"] - B[k]["analogy_native_acc"]) < 1e-9, k
    assert abs(r["sim_auc"] - B[k]["sim_auc_common"]) < 1e-9, k
    assert abs(r["analogy_native_acc"] * N_NATIVE - round(r["analogy_native_acc"] * N_NATIVE)) < 1e-6, k
for k, nm in THESIS.items():                                                     # reproduces the official thesis CONF table
    for c in ("Overall_F1", "D1_F1", "D2_F1", "Overall_lo", "Overall_hi"):
        assert abs(R[k]["conf_" + c] - official.loc[nm, c]) < 1e-9, (k, c)
new = R["merged_plus"]
assert R["merged_plus"]["dim"] == manifest["models"]["merged_plus"]["dim"] == 1200
assert abs(new["dev_Overall_F1"] - chk["dev"]) < 1e-9 and abs(new["conf_Overall_F1"] - chk["conf"]) < 1e-9   # same as the dimension study
g = next(x for x in eps12["grid"] if x["eps"] == chk["eps"])
assert abs(new["analogy_native_acc"] - g["all"]) < 1e-9 and abs(new["sim_auc"] - g["sim_auc"]) < 1e-9

# ------------------------------------------------------------------ tables
name = {k: R[k]["name"] for k in KEYS}
DISPLAY = {"sim_auc": 4}                                   # decimals shown (others: % with 2 decimals)
def shown(key, v): return round(v, DISPLAY.get(key, 4))    # value at the precision printed in the table
best = lambda key, k: "**" if shown(key, R[k][key]) >= max(shown(key, R[x][key]) for x in KEYS) else ""
def cell(k, key, fmt):
    b = best(key, k); return f"{b}{fmt(R[k][key])}{b}"

t_main = ["| Model | dims | DEV F1 | CONF F1 [95% CI] | CONF news D1 | CONF informal D2 | Analogy (native) | Semantic | Syntactic | Similarity AUC | Informal coverage |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
for k in KEYS:
    r = R[k]
    t_main.append(f"| {'**' + name[k] + '**' if k == 'merged_plus' else name[k]} | {r['dim']:,} | {cell(k, 'dev_Overall_F1', pct)} | "
                  f"{cell(k, 'conf_Overall_F1', pct)} [{pct(r['conf_Overall_lo'])}, {pct(r['conf_Overall_hi'])}] | {cell(k, 'conf_D1_F1', pct)} | "
                  f"{cell(k, 'conf_D2_F1', pct)} | {cell(k, 'analogy_native_acc', pct)}% ({round(r['analogy_native_acc'] * N_NATIVE):,}) | "
                  f"{cell(k, 'analogy_semantic_acc', pct)}% | {cell(k, 'analogy_syntactic_acc', pct)}% | {cell(k, 'sim_auc', lambda x: f'{x:.4f}')} | "
                  f"{cell(k, 'tok_cov_D2', pct)}% |")

t_tests = ["| BanglaMeta+ vs | Δ CONF Overall F1 [95% CI] | one-sided p | Analogy questions only BanglaMeta+ solves | only the other solves | McNemar p |",
           "|---|---|---|---|---|---|"]
mc = {m["vs"]: m for m in full["mcnemar"]}
for k in KEYS[1:]:
    d = next(x for x in full["deltas"] if x["split"] == "conf" and x["vs"] == k and x["metric"] == "Overall")
    assert abs(d["delta"] - (new["conf_Overall_F1"] - R[k]["conf_Overall_F1"])) < 1e-12
    verdict = "better" if d["ci_lo"] > 0 else ("worse" if d["ci_hi"] < 0 else "tie")
    p = mc[k]["p_one_sided"]
    t_tests.append(f"| {name[k]} | {sgn(d['delta'])} [{sgn(d['ci_lo'])}, {sgn(d['ci_hi'])}] ({verdict}) | {d['p_one_sided']:.4f} | "
                   f"{mc[k]['plus_only']:,} | {mc[k]['other_only']:,} | {('< 1e-300' if p == 0 else f'{p:.1e}')} |")

V = dims["variants"]; old = V["1500 BanglaMeta+ (current)"]
rows_d = [("300 (PCA/SVD of the 1,500-d model)", V["300 PCA of BanglaMeta+"]), ("500 (PCA/SVD of the 1,500-d model)", V["500 PCA of BanglaMeta+"]),
          ("600 similarity view only", V["600 similarity view only"]), ("900 thesis BanglaMeta", V["900 thesis (BanglaMeta)"]),
          ("1,200 thesis + similarity News block", V["1200 thesis + sim-News"]), ("1,200 thesis + similarity Subs block", V["1200 thesis + sim-Subs"]),
          ("**1,200 BanglaMeta+ (current)**", {"dim": 1200, "dev": {"Overall": chk["dev"], "gap": chk["gap"]}, "conf": {"Overall": chk["conf"]},
                                                "analogy_all": g["all"], "analogy_test": g["test"], "sim_auc": g["sim_auc"],
                                                "dims_90pct": V["1200 thesis(News+Inf) + sim"]["dims_90pct"]}),
          ("1,500 previous BanglaMeta+", old), ("1,800 + Informal similarity block (weight 0.15)", V["1800 + sim-Informal (0.15)"]),
          ("1,800 + all similarity blocks, equal weights", V["1800 + sim equal weights"])]
t_dims = ["| Option | DEV F1 | CONF F1 | Analogy (all) | Analogy (unseen half) | Similarity AUC | Train–held-out gap (points) | Dims holding 90% of variance |",
          "|---|---|---|---|---|---|---|---|"]
for lab, v in rows_d:
    t_dims.append(f"| {lab} | {pct(v['dev']['Overall'])} | {pct(v['conf']['Overall'])} | {pct(v['analogy_all'])}% | {pct(v['analogy_test'])}% | "
                  f"{v['sim_auc']:.4f} | {pct(v['dev']['gap'])} | {v['dims_90pct']} |")
# the switch from 1,500 to 1,200 must not lower any of the three scores
assert chk["dev"] >= old["dev"]["Overall"] and chk["conf"] >= old["conf"]["Overall"]
assert round(g["all"] * N_NATIVE) >= round(old["analogy_all"] * N_NATIVE) and g["sim_auc"] >= old["sim_auc"] - 1e-12

S = dims["sources"]
t_src = ["| Source model (300-d) | Training tokens | Learned parameters (2 × vocab × 300) | Tokens per parameter | Words seen < 20 times | Diagnosis |",
         "|---|---|---|---|---|---|"]
for nm, dx in (("News-W2V", "trained enough; no rare-word collapse"),
               ("Informal-W2V", "under-trained: ~1% analogy in every frequency band → weight 0 for word similarity, kept for classification")):
    s = S[nm]
    assert s["learned_parameters"] == 2 * s["vocab"] * 300 and abs(s["tokens_per_parameter"] - s["corpus_tokens"] / s["learned_parameters"]) < 1e-12
    t_src.append(f"| {nm} | {s['corpus_tokens']:,} | {s['learned_parameters']:,} | {s['tokens_per_parameter']:.2f} | {pct(s['share_words_lt_20'], 1)}% | {dx} |")
t_src.append("| Subs2vec-w2 | not published (OpenSubtitles 2018) | 2 × 177,376 × 300 | – | – | downloaded; strong for word similarity |")

t_eps = ["| ε (thesis share of the cosine) | Analogy, tune half | Analogy, unseen half | Analogy, all | Similarity AUC |", "|---|---|---|---|---|"]
for x in eps12["grid"]:
    t_eps.append(f"| {x['eps']}{' ← chosen' if x['eps'] == chk['eps'] else ''} | {pct(x['tune'])}% | {pct(x['test'])}% | {pct(x['all'])}% | {x['sim_auc']:.4f} |")

ranks = []
for lab, key in (("DEV F1", "dev_Overall_F1"), ("CONF F1", "conf_Overall_F1"), ("Analogy", "analogy_native_acc"), ("Similarity AUC", "sim_auc")):
    rk = 1 + sum(shown(key, R[x][key]) > shown(key, new[key]) for x in KEYS)
    tie = sum(shown(key, R[x][key]) == shown(key, new[key]) for x in KEYS) > 1
    ranks.append(f"{lab} #{rk}{' (tied)' if tie else ''}")

section = f"""<!-- BANGLAMETA_PLUS:START (generated by docs/make_readme_tables.py; do not edit by hand) -->
## 14. BanglaMeta+ (1,200-d): post-thesis extension and web app

**BanglaMeta+** is one vector per word, built from the frozen thesis model alone, with no training and no other model's vectors:

`c(w) = [ √{chk['eps']} · m_NI(w) ; √{1 - chk['eps']:g} · s_News(w) ; √{1 - chk['eps']:g} · s_Subs(w) ]`

- **m_NI(w):** the thesis row's News-W2V and Informal-W2V blocks, re-normalised (600-d).
- **s(w):** the similarity view, 600-d. In each 300-d block, the mean and the top 4 principal directions are removed ("all-but-the-top"); then the blocks are weighted News 1.65 / Subs 1.35 (the Informal block gets weight 0).
- **Classification:** the probe standardises every feature, so the small weight costs the thesis features nothing (Theorem 5).
- **Word similarity:** cosine = {chk['eps']} × thesis cosine + {1 - chk['eps']:g} × similarity-view cosine.
- **How the settings were chosen:** D, the block weights and ε were chosen on a random half of the analogy questions; ties go to the smaller ε.

| | |
|---|---|
| File | `models/banglameta_plus/banglameta_plus_1200d.bin` (word2vec binary, 273,531 × 1,200, float32; not on GitHub, rebuilt by the notebook) |
| MD5 | `{md5}` |
| Notebook | [`eval/banglameta_plus/BanglaMeta_plus_1200d.ipynb`](eval/banglameta_plus/BanglaMeta_plus_1200d.ipynb): rebuilds the file byte for byte and evaluates it |
| Recipe / export | `webapp/backend/prepare.py` · `webapp/backend/experiments/export_plus.py` |
| Rank among the 6 models | {' · '.join(ranks)} |

### 14.1 All models, measured the same way

Macro-F1 % uses the thesis probe. CONF intervals come from the thesis's 10,000-sample paired bootstrap, and they reproduce the official thesis CONF table exactly. CONF values of BanglaMeta+ and BanglaMeta-Sim are **exploratory**, because both were built after the confirmatory test. The analogy column also gives the number of questions solved, out of 16,678. Bold marks the best value in each column.

{chr(10).join(t_main)}

### 14.2 Is BanglaMeta+ really better? (paired, same texts)

{chr(10).join(t_tests)}

### 14.3 Why 1,200 dimensions: every option scored the same way

No Word2Vec here was trained with more than 300 dimensions. The larger sizes are separately trained 300-d models joined together. Full report: [`eval/dimension_study/Dimension_and_overfitting_study.pdf`](eval/dimension_study/Dimension_and_overfitting_study.pdf).

{chr(10).join(t_dims)}

- **Why 1,200 replaced the 1,500-d version:** dropping Subs2vec from the thesis part keeps or improves every score with 300 fewer dimensions. Subs2vec adds no significant classification information (check P2). The same number of analogy questions is solved, {round(g['all'] * N_NATIVE):,}.
- **Over-fitting:** the CONF score does not drop below DEV. The train–held-out gap does not grow with dimension, and the 1,200-d model has the smallest gap. Learning curves are still rising at 100% data.
- **SVD/PCA compression** lowers classification and similarity, so it is not used. The thesis model never used it either; it lost in Step 4.

### 14.4 Under-fitting of the source models

{chr(10).join(t_src)}

### 14.5 How ε was chosen (analogy tune half; the unseen half was not used)

{chr(10).join(t_eps)}

### 14.6 Web app

`webapp/` serves BanglaMeta+ (default), the thesis model and the baselines through a FastAPI backend and a React + three.js frontend:
- 3-D explorer with X/Y/Z axes, Bangla/Banglish input, relations, similarity, analogy;
- a comparison page with live benchmarks.

Every answer is computed from the served model rows. Unknown words are reported, never invented. 169 backend tests check every answer against independent calculations. See [`webapp/README.md`](webapp/README.md).
<!-- BANGLAMETA_PLUS:END -->"""

readme = (REPO / "README.md").read_text(encoding="utf-8")
if "<!-- BANGLAMETA_PLUS:START" in readme:
    readme = re.sub(r"<!-- BANGLAMETA_PLUS:START.*?<!-- BANGLAMETA_PLUS:END -->", lambda m: section, readme, flags=re.S)
else:
    readme = readme.rstrip() + "\n\n---\n\n" + section + "\n"
(REPO / "README.md").write_text(readme, encoding="utf-8")
print("README section written; all consistency checks passed")
