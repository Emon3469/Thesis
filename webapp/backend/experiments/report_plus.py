"""Figures + PDF report for the BanglaMeta+ evaluation (reads eval/banglameta_plus_eval/results.json)."""
import hashlib, json, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

EV = C.REPO / "eval" / "banglameta_plus_eval"; FIG = EV / "figures"; FIG.mkdir(exist_ok=True)
D = json.loads((EV / "results.json").read_text(encoding="utf-8"))
R, KEYS = D["results"], D["models"]
NAME = {k: R[k]["name"] for k in KEYS}
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
TASK_LABEL = {"news_topic_7": "News topic (7 classes)", "review_sentiment_3": "Review sentiment (3)", "social_sentiment_3": "Social sentiment (3)",
              "hate_binary_2": "Hate speech (2)"}

# palette (dataviz reference instance): BanglaMeta+ = slot 1 blue, thesis BanglaMeta = slot 2 orange, others neutral
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e", "#e7e6e2"
COL = {k: (BLUE if k == "merged_plus" else ORANGE if k == "merged" else GREY) for k in KEYS}
plt.rcParams.update({"font.family": "Calibri", "font.size": 10.5, "axes.edgecolor": "#bdbcb6", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight"})

def save(fig, name):
    fig.savefig(FIG / name, facecolor="white"); plt.close(fig); return name

def dotplot(ax, vals, lo=None, hi=None, title="", pct=True, fmt="{:.2f}"):
    """One row per model (fixed order), point + 95% CI; best value bold."""
    ys = np.arange(len(KEYS))[::-1]; s = 100 if pct else 1
    best = max(vals[k] for k in KEYS)
    for y, k in zip(ys, KEYS):
        v = vals[k] * s
        if lo is not None:
            ax.plot([lo[k] * s, hi[k] * s], [y, y], color=COL[k], lw=2.2, solid_capstyle="round", alpha=.55)
        ax.plot([v], [y], "o", color=COL[k], ms=8.5, mec="white", mew=1.5, zorder=3)
        ax.annotate(fmt.format(v), (v, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
                    color=INK, fontweight="bold" if vals[k] == best else "normal")
    ax.set_yticks(ys, [NAME[k] for k in KEYS]); ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="bold")
    ax.grid(axis="y", visible=False); ax.margins(x=0.12); ax.set_ylim(-0.6, len(KEYS) - 0.2)
    for lab, k in zip(ax.get_yticklabels(), KEYS):
        if k == "merged_plus": lab.set_fontweight("bold"); lab.set_color(INK)

figs = []
# ----------------------------------------------------------------------------- 1 headline scorecard
fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2))
g = lambda key: {k: R[k][key] for k in KEYS}
dotplot(axs[0, 0], g("dev_Overall_F1"), g("dev_Overall_lo"), g("dev_Overall_hi"), "Classification, DEV · macro-F1 %")
dotplot(axs[0, 1], g("conf_Overall_F1"), g("conf_Overall_lo"), g("conf_Overall_hi"), "Classification, held-out CONF · macro-F1 %")
dotplot(axs[0, 2], g("analogy_native_acc"), g("analogy_lo"), g("analogy_hi"), "Word analogy · accuracy %")
dotplot(axs[1, 0], g("sim_auc"), g("sim_auc_lo"), g("sim_auc_hi"), "Word similarity · category AUC", pct=False, fmt="{:.3f}")
dotplot(axs[1, 1], g("conf_D1_F1"), g("conf_D1_lo"), g("conf_D1_hi"), "Formal news (D1), CONF · macro-F1 %")
dotplot(axs[1, 2], g("conf_D2_F1"), g("conf_D2_lo"), g("conf_D2_hi"), "Informal text (D2), CONF · macro-F1 %")
for ax in axs[:, 1:].ravel(): ax.set_yticklabels([])
fig.suptitle("BanglaMeta+ against every model: point = measured value, line = 95% bootstrap interval", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); figs.append((save(fig, "fig01_scorecard.png"),
    "Headline metrics for all six models. Classification uses the thesis probe; CONF intervals come from the thesis's own 10,000-sample paired bootstrap "
    "and reproduce the official thesis intervals exactly for the four thesis models. CONF values of BanglaMeta+ and BanglaMeta-Sim are exploratory."))

# ----------------------------------------------------------------------------- 2 paired differences
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
others = KEYS[1:]
for ax, (split, metric, title) in zip(axs, [("dev", "Overall", "DEV · Overall F1"), ("conf", "Overall", "CONF · Overall F1"), ("conf", "D2", "CONF · informal (D2) F1")]):
    ys = np.arange(len(others))[::-1]
    for y, b in zip(ys, others):
        d = next(x for x in D["deltas"] if x["split"] == split and x["vs"] == b and x["metric"] == metric)
        c = BLUE if d["ci_lo"] > 0 else (ORANGE if d["ci_hi"] < 0 else GREY)
        ax.plot([d["ci_lo"] * 100, d["ci_hi"] * 100], [y, y], color=c, lw=2.4, solid_capstyle="round", alpha=.6)
        ax.plot([d["delta"] * 100], [y], "o", color=c, ms=8, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{d['delta'] * 100:+.2f}", (d["delta"] * 100, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(ys, [f"vs {NAME[b]}" for b in others]); ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(others) - 0.2)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK); ax.set_xlabel("BanglaMeta+ minus model (points)"); ax.margins(x=0.15)
fig.suptitle("Is BanglaMeta+ really better? Paired bootstrap 95% intervals (blue: better, grey: tie, orange: worse)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92)); figs.append((save(fig, "fig02_paired_differences.png"),
    "Differences computed on the same texts and the same resamples. An interval entirely right of 0 means BanglaMeta+ is better; an interval that "
    "contains 0 is a tie."))

# ----------------------------------------------------------------------------- 3 per task
fig, axs = plt.subplots(1, 4, figsize=(14.5, 3.8))
for ax, t in zip(axs, TASKS):
    dotplot(ax, {k: R[k][f"conf_F1_{t}"] for k in KEYS}, title=TASK_LABEL[t])
for ax in axs[1:]: ax.tick_params(labelleft=False)
fig.suptitle("Macro-F1 % per task on the held-out CONF half", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.9)); figs.append((save(fig, "fig03_per_task_conf.png"), "Per-task macro-F1 on CONF (4 tasks, 11,950 texts)."))

# ----------------------------------------------------------------------------- 4 confusion matrices
from matplotlib.colors import LinearSegmentedColormap
seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cfe0f7", "#7fb0ea", BLUE, "#174a87"])
fig, axs = plt.subplots(1, 4, figsize=(16, 4.6), gridspec_kw={"width_ratios": [2.2, 1, 1, .8]})
for ax, t in zip(axs, TASKS):
    M = np.array(D["confusion_conf_plus"][t], float); Mn = M / M.sum(1, keepdims=True); names = D["class_names"][t]
    ax.imshow(Mn, cmap=seq, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{Mn[i, j] * 100:.0f}", ha="center", va="center", fontsize=8.5, color="white" if Mn[i, j] > .55 else INK)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8.5); ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.grid(False); ax.set_title(TASK_LABEL[t], loc="left", fontsize=11, fontweight="bold", color=INK); ax.set_xlabel("predicted"); ax.set_ylabel("true")
fig.suptitle("BanglaMeta+ confusion matrices on CONF (% of each true class)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92)); figs.append((save(fig, "fig04_confusion_conf.png"),
    "Rows are true classes and sum to 100%. Most news topics are recognised well; neutral sentiment is the hardest class, as for every model."))

# ----------------------------------------------------------------------------- 5 per-class F1
fig, axs = plt.subplots(1, 4, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2.2, 1, 1, .8]})
cmp = ["merged_plus", "merged", "fasttext"]; ccol = {"merged_plus": BLUE, "merged": ORANGE, "fasttext": GREY}
for ax, t in zip(axs, TASKS):
    names = D["class_names"][t]; x = np.arange(len(names)); w = 0.26
    for j, k in enumerate(cmp):
        ax.bar(x + (j - 1) * (w + 0.02), np.array(R[k][f"conf_perclass_{t}"]) * 100, w, color=ccol[k], label=NAME[k])
    ax.set_xticks(x, names, rotation=40, ha="right", fontsize=8.5); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    ax.set_title(TASK_LABEL[t], loc="left", fontsize=11, fontweight="bold", color=INK)
axs[0].set_ylabel("F1 % (CONF)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.0))
fig.suptitle("Per-class F1 on CONF: BanglaMeta+ vs the thesis model vs fastText", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92)); figs.append((save(fig, "fig05_per_class_f1.png"), "Per-class F1 on CONF for the three strongest models."))

# ----------------------------------------------------------------------------- 6 analogy by category
LATIN = {"বিভক্তি": "bibhakti", "উপসর্গ": "upasarga", "নির্দেশক": "nirdeshak", "প্রত্যয়": "pratyay", "সাধু": "sadhu", "চলিত": "cholito"}
def latin(c):
    """chart font cannot shape Bangla: transliterate the category names"""
    import unicodedata
    c = unicodedata.normalize("NFC", c)
    for bn, la in LATIN.items(): c = c.replace(unicodedata.normalize("NFC", bn), la)
    assert not any("ঀ" <= ch <= "৿" for ch in c), c
    return c.replace("_", " ")
cats = [c for c in R["merged_plus"]["analogy_per_category"]]
cmp = ["merged_plus", "merged", "fasttext", "news"]; ccol = {"merged_plus": BLUE, "merged": ORANGE, "fasttext": GREY, "news": "#d0cfc9"}
fig, ax = plt.subplots(figsize=(13.5, 4.6)); x = np.arange(len(cats)); w = 0.2
for j, k in enumerate(cmp):
    ax.bar(x + (j - 1.5) * (w + 0.015), [R[k]["analogy_per_category"][c] * 100 for c in cats], w, color=ccol[k], label=NAME[k])
ax.set_xticks(x, [latin(c) for c in cats], rotation=25, ha="right"); ax.set_ylabel("accuracy %"); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=4, loc="upper right"); ax.set_title("Word analogy accuracy by category (3CosAdd, unanswerable = wrong)", loc="left", fontsize=13, color=INK)
figs.append((save(fig, "fig06_analogy_categories.png"), "Accuracy per analogy file. 'Mikolov-Filtered' is the translated Mikolov set (not part of the native score)."))

# ----------------------------------------------------------------------------- 7 similarity distributions
fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
for ax, k in zip(axs, ["merged", "merged_plus"]):
    sd = D["similarity_scores"][k]; bins = np.linspace(-0.3, 1, 40)
    ax.hist(sd["diff"], bins=bins, density=True, color=GREY, alpha=.75, label="different category")
    ax.hist(sd["same"], bins=bins, density=True, color=COL[k], alpha=.8, label="same category")
    ax.set_title(f"{NAME[k]} · AUC {R[k]['sim_auc']:.3f}", loc="left", fontsize=12, fontweight="bold", color=INK); ax.set_xlabel("cosine of a word pair")
    ax.legend(frameon=False); ax.grid(axis="x", visible=False)
fig.suptitle(f"Word similarity: same-category vs different-category pairs ({D['results']['_common_words']} category words)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.9)); figs.append((save(fig, "fig07_similarity_distributions.png"),
    "BanglaMeta+ pushes unrelated pairs toward 0 while related pairs stay high, so the two distributions overlap less (higher AUC)."))

# ----------------------------------------------------------------------------- 8 how the settings were chosen
fig, axs = plt.subplots(1, 3, figsize=(14, 3.7))
Ds = list(range(1, 6)); ab = D["selection"]["abtt"]
axs[0].plot(Ds, [ab[str(d)]["chosen"]["tune"] * 100 for d in Ds], "o-", color=BLUE, lw=2, label="tune half (used to choose)")
axs[0].plot(Ds, [ab[str(d)]["chosen"]["test"] * 100 for d in Ds], "o--", color=GREY, lw=2, label="test half (never used)")
axs[0].axvline(4, color=INK2, lw=1, ls=":"); axs[0].set_xlabel("directions removed per block (D)"); axs[0].set_ylabel("analogy accuracy %")
axs[0].set_title("(a) all-but-the-top: D = 4 chosen", loc="left", fontweight="bold", color=INK); axs[0].legend(frameon=False, fontsize=9)
eg = D["selection"]["eps"]["eps_grid"]; e = [r["eps"] for r in eg if r["eps"] > 0]
axs[1].semilogx(e, [r["tune"] * 100 for r in eg if r["eps"] > 0], "o-", color=BLUE, lw=2, label="tune half")
axs[1].semilogx(e, [r["test"] * 100 for r in eg if r["eps"] > 0], "o--", color=GREY, lw=2, label="test half")
axs[1].axvline(C.PLUS_EPS, color=INK2, lw=1, ls=":"); axs[1].set_xlabel("ε (thesis-row share)"); axs[1].set_ylabel("analogy accuracy %")
axs[1].set_title(f"(b) BanglaMeta+: ε = {C.PLUS_EPS} chosen (ties → smaller ε)", loc="left", fontweight="bold", color=INK); axs[1].legend(frameon=False, fontsize=9)
axs[2].semilogx(e, [r["sim_auc"] for r in eg if r["eps"] > 0], "o-", color=BLUE, lw=2)
axs[2].axvline(C.PLUS_EPS, color=INK2, lw=1, ls=":"); axs[2].set_xlabel("ε (thesis-row share)"); axs[2].set_ylabel("similarity AUC")
axs[2].set_title("(c) similarity AUC vs ε (not used to choose)", loc="left", fontweight="bold", color=INK)
fig.tight_layout(); figs.append((save(fig, "fig08_selection.png"),
    "Every setting was chosen on a random half of the analogy questions; the other half and the similarity AUC were never used for the choice."))

# ----------------------------------------------------------------------------- 9 probability of being best
fig, ax = plt.subplots(figsize=(13.5, 4.2)); ys = np.arange(len(KEYS))[::-1]
for y, k in zip(ys, KEYS):
    ax.barh(y + .18, R[k]["dev_P_best"] * 100, .34, color=COL[k], alpha=.5); ax.barh(y - .18, R[k]["conf_P_best"] * 100, .34, color=COL[k])
    ax.text(R[k]["conf_P_best"] * 100 + 1, y - .18, f"CONF {R[k]['conf_P_best'] * 100:.1f}%", va="center", fontsize=9, color=INK)
    ax.text(R[k]["dev_P_best"] * 100 + 1, y + .18, f"DEV {R[k]['dev_P_best'] * 100:.1f}%", va="center", fontsize=9, color=INK2)
ax.set_yticks(ys, [NAME[k] for k in KEYS]); ax.set_xlim(0, 110); ax.set_xlabel("% of bootstrap resamples in which the model has the highest Overall F1")
ax.grid(axis="y", visible=False); ax.set_title("How often is each model the best classifier?", loc="left", fontsize=13, color=INK)
figs.append((save(fig, "fig09_probability_best.png"), "Share of the bootstrap resamples (DEV 1,000; CONF 10,000) in which each model has the highest Overall macro-F1."))

# ----------------------------------------------------------------------------- report
metric_rows = [("DEV Overall macro-F1 %", "dev_Overall_F1", 100, 2, True), ("CONF Overall macro-F1 %", "conf_Overall_F1", 100, 2, True),
               ("CONF 95% interval", None, 100, 2, None), ("DEV news D1 %", "dev_D1_F1", 100, 2, True), ("DEV informal D2 %", "dev_D2_F1", 100, 2, True),
               ("CONF news D1 %", "conf_D1_F1", 100, 2, True), ("CONF informal D2 %", "conf_D2_F1", 100, 2, True)] + \
              [(f"CONF {TASK_LABEL[t]} %", f"conf_F1_{t}", 100, 2, True) for t in TASKS] + \
              [("Analogy, native %", "analogy_native_acc", 100, 2, True), ("Analogy, semantic %", "analogy_semantic_acc", 100, 2, True),
               ("Analogy, syntactic %", "analogy_syntactic_acc", 100, 2, True), ("Analogy, Mikolov set %", "analogy_mikolov_acc", 100, 2, True),
               ("Similarity AUC", "sim_auc", 1, 3, True), ("Token coverage, news %", "tok_cov_D1", 100, 2, True), ("Token coverage, informal %", "tok_cov_D2", 100, 2, True),
               ("P(best), CONF %", "conf_P_best", 100, 1, True), ("Dimensions", "dim", 1, 0, None)]
def cell(k, key, sc, d):
    if key is None: return f"[{R[k]['conf_Overall_lo'] * 100:.2f}, {R[k]['conf_Overall_hi'] * 100:.2f}]"
    return f"{R[k][key] * sc:,.{d}f}"
rank_summary, trs = [], []
for label, key, sc, d, rankit in metric_rows:
    vals = {k: R[k][key] for k in KEYS} if key else None
    best = max(vals.values()) if rankit else None
    tds = "".join(f"<td class='{'b' if rankit and abs(vals[k] - best) < 1e-12 else ''}{' p' if k == 'merged_plus' else ''}'>{cell(k, key, sc, d)}</td>" for k in KEYS)
    rank = (1 + sum(vals[k] > vals["merged_plus"] + 1e-12 for k in KEYS)) if rankit else None
    if rankit: rank_summary.append((label, rank))
    trs.append(f"<tr><th>{label}</th>{tds}<td class='rk'>{'#' + str(rank) if rank else ''}</td></tr>")
first = [l for l, r in rank_summary if r == 1]; notfirst = [(l, r) for l, r in rank_summary if r > 1]
mc = {x["vs"]: x for x in D["mcnemar"]}
cd = lambda vs, metric="Overall": next(x for x in D["deltas"] if x["split"] == "conf" and x["vs"] == vs and x["metric"] == metric)
binf = C.REPO / "models" / "banglameta_plus" / "banglameta_plus_1200d.bin"
md5txt = (binf.parent / "banglameta_plus_1200d.md5").read_text(encoding="utf-8").split()[0]
fig_html = "".join(f"<figure><img src='figures/{n}'><figcaption><b>Figure {i}.</b> {cap}</figcaption></figure>" for i, (n, cap) in enumerate(figs, 1))
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>BanglaMeta+ evaluation</title><style>
@page {{ size: A4 landscape; margin: 12mm; }}
body {{ font-family: Calibri, "Segoe UI", sans-serif; font-size: 10.5pt; color: #0b0b0b; margin: 0; line-height: 1.4; }}
h1 {{ font-size: 22pt; margin: 0; }} h2 {{ break-after: avoid;  font-size: 14pt; border-bottom: 2px solid {BLUE}; padding-bottom: 2px; margin: 16px 0 8px; }}
.sub {{ color: #52514e; margin: 2px 0 10px; }}
.box {{ border: 1px solid #d6d5d0; border-radius: 8px; padding: 8px 12px; background: #fafaf8; margin: 6px 0; }}
code {{ font-family: Consolas, monospace; font-size: 9.5pt; background: #f0efec; padding: 0 3px; border-radius: 3px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.6pt; }}
th, td {{ border-bottom: 1px solid #e7e6e2; padding: 3px 7px; text-align: right; }} th {{ text-align: left; font-weight: 600; }}
thead th {{ text-align: right; background: #0b0b0b; color: #fff; }} thead th:first-child {{ text-align: left; }}
td.p {{ background: #eef4fc; }} td.b {{ font-weight: 700; color: {BLUE}; }} td.p.b {{ color: {BLUE}; }} td.rk {{ color: #52514e; }}
figure {{ margin: 10px 0 14px; break-inside: avoid; }} figure img {{ width: 100%; }} figcaption {{ font-size: 9.6pt; color: #52514e; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }} ul {{ margin: 4px 0 4px 18px; padding: 0; }}
.warn {{ border-left: 4px solid {ORANGE}; padding: 4px 10px; background: #fdf3ee; }}
</style></head><body>
<h1>BanglaMeta+ · evaluation report</h1>
<div class="sub">One 1,200-dimension Bangla word embedding built from the frozen merged thesis model. Generated {D['created']} by
<code>webapp/backend/experiments/evaluate_plus.py</code> and <code>report_plus.py</code>.</div>
<div class="two"><div class="box"><b>Where the model is</b>
<ul><li>Standalone file: <code>models/banglameta_plus/banglameta_plus_1200d.bin</code> (word2vec binary, {binf.stat().st_size / 2**30:.2f} GiB, MD5 <code>{md5txt}</code>)</li>
<li>Load: <code>KeyedVectors.load_word2vec_format("banglameta_plus_1200d.bin", binary=True)</code> (gensim)</li>
<li>Served copy used by the web app: <code>webapp/data/models/merged_plus.f32.npy</code> + <code>merged_plus.words.txt</code></li>
<li>Rebuild: <code>webapp/backend/prepare.py</code> (recipe) then <code>experiments/export_plus.py</code> (export)</li></ul></div>
<div class="box"><b>What it is</b>
<ul><li>273,531 words × 1,200 dims, every row length 1</li>
<li>Row = [√0.0005 · thesis News+Informal blocks (600) ; √0.9995 · similarity-view News + Subs blocks (600)]</li><li>Subs2vec is kept only in the similarity part: it adds no significant classification information (thesis check P2), and dropping it from the thesis part keeps or improves every score (<code>eval/dimension_study</code>)</li>
<li>Similarity view = per block: mean and top 4 directions removed, weights News 1.65 / Subs 1.35 (Informal 0)</li>
<li>No training; no other model's vectors; D, weights and ε chosen on half the analogy questions</li></ul></div></div>
<h2>Metric matrix (all models measured the same way)</h2>
<table><thead><tr><th>Metric</th>{''.join(f'<th>{NAME[k]}</th>' for k in KEYS)}<th>BanglaMeta+ rank</th></tr></thead><tbody>{''.join(trs)}</tbody></table>
<p class="sub">Bold blue = best of the six. Ranks are among these six models. CONF values for BanglaMeta+ and BanglaMeta-Sim are exploratory.</p>
<h2>Verdict</h2>
<div class="two"><div class="box"><b>BanglaMeta+ is #1 of 6 on {len(first)} of {len(rank_summary)} ranked metrics:</b> {', '.join(first)}.<br>
<b>Not #1 on:</b> {'; '.join(f'{l} (#{r})' for l, r in notfirst) or 'none'}.</div>
<div class="box"><b>Significance (paired, same texts):</b>
<ul><li>CONF Overall vs fastText: {cd('fasttext')['delta'] * 100:+.2f} [{cd('fasttext')['ci_lo'] * 100:+.2f}, {cd('fasttext')['ci_hi'] * 100:+.2f}], p = {cd('fasttext')['p_one_sided']:.4f}</li>
<li>CONF Overall vs thesis BanglaMeta: {cd('merged')['delta'] * 100:+.2f} [{cd('merged')['ci_lo'] * 100:+.2f}, {cd('merged')['ci_hi'] * 100:+.2f}], a tie</li>
<li>Analogy vs fastText (McNemar): {mc['fasttext']['plus_only']} questions only BanglaMeta+ solves vs {mc['fasttext']['other_only']}, p = {mc['fasttext']['p_one_sided']:.2g}</li>
<li>Analogy vs thesis BanglaMeta: {mc['merged']['plus_only']} vs {mc['merged']['other_only']}, p = {mc['merged']['p_one_sided']:.2g}</li></ul></div></div>
<div class="warn"><b>About "top 1%".</b> These results rank BanglaMeta+ among the six models evaluated here, on the thesis's Bangla benchmarks.
They cannot rank it against every Bangla model that exists (for example transformer encoders); that would need the same benchmarks run on those models.
The honest claim is: best of the evaluated static embeddings on classification and word similarity/analogy, tied with the thesis model on held-out classification.</div>
<h2 style="break-before: page">Figures</h2>{fig_html}
</body></html>"""
(EV / "BanglaMeta_plus_evaluation.html").write_text(html, encoding="utf-8")
chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
pdf = EV / "BanglaMeta_plus_evaluation.pdf"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", (EV / "BanglaMeta_plus_evaluation.html").as_uri()],
               check=True, capture_output=True, timeout=120)
print("figures:", len(figs), "| report:", pdf, pdf.stat().st_size, "bytes")
print("rank summary:", rank_summary)
