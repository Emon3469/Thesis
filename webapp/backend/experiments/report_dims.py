"""Report: over/under-fitting and the choice of dimension (reads eval/dimension_study/*.json)."""
import json, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

EV = C.REPO / "eval" / "dimension_study"; FIG = EV / "figures"; FIG.mkdir(exist_ok=True)
S = json.loads((EV / "results.json").read_text(encoding="utf-8"))
E = json.loads((EV / "eps_1200.json").read_text(encoding="utf-8"))
chk = json.loads((EV / "eps_1200_check.log").read_text(encoding="utf-8").strip().splitlines()[-1])
V = S["variants"]
# the final 1,200-d model at the chosen eps (scores recomputed in eps_1200.py / eps_1200_check)
g = next(r for r in E["grid"] if r["eps"] == chk["eps"])
NEW = "1200 BanglaMeta+ (new)"
V[NEW] = {"dim": 1200, "about": f"thesis News+Informal + similarity view, eps = {chk['eps']}",
          "dev": {"Overall": chk["dev"], "gap": chk["gap"], "train_Overall": chk["dev"] + chk["gap"]}, "conf": {"Overall": chk["conf"]},
          "analogy_all": g["all"], "analogy_test": g["test"], "analogy_tune": g["tune"], "sim_auc": g["sim_auc"],
          "participation_ratio": V["1200 thesis(News+Inf) + sim"]["participation_ratio"], "dims_90pct": V["1200 thesis(News+Inf) + sim"]["dims_90pct"]}
V.pop("1200 thesis(News+Inf) + sim")
ORDER = ["300 PCA of BanglaMeta+", "500 PCA of BanglaMeta+", "600 similarity view only", "900 thesis (BanglaMeta)", "1200 thesis + sim-News",
         "1200 thesis + sim-Subs", NEW, "1500 BanglaMeta+ (current)", "1800 + sim-Informal (0.15)", "1800 + sim equal weights"]
LABEL = {k: k.replace(" (current)", " (old)") for k in ORDER}
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e", "#e7e6e2"
COL = {k: BLUE if k == NEW else ORANGE if k.startswith("1500") else GREY for k in ORDER}
plt.rcParams.update({"font.family": "Calibri", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": "#bdbcb6", "xtick.color": INK2, "ytick.color": INK2,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
def save(fig, n): fig.savefig(FIG / n, facecolor="white"); plt.close(fig); return n
figs = []

def dots(ax, vals, title, fmt="{:.2f}", ref=None):
    ys = np.arange(len(ORDER))[::-1]; best = max(vals.values())
    for y, k in zip(ys, ORDER):
        ax.plot([vals[k]], [y], "o", color=COL[k], ms=8.5, mec="white", mew=1.5, zorder=3)
        ax.annotate(fmt.format(vals[k]), (vals[k], y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.8,
                    fontweight="bold" if abs(vals[k] - best) < 1e-12 else "normal", color=INK)
    if ref is not None: ax.axvline(ref, color=ORANGE, lw=1, ls=":")
    ax.set_yticks(ys, [LABEL[k] for k in ORDER]); ax.grid(axis="y", visible=False); ax.margins(x=0.15); ax.set_ylim(-0.6, len(ORDER) - 0.2)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    for lab, k in zip(ax.get_yticklabels(), ORDER):
        if k == NEW: lab.set_fontweight("bold"); lab.set_color(INK)

fig, axs = plt.subplots(1, 4, figsize=(16, 5.2))
cur = V["1500 BanglaMeta+ (current)"]
dots(axs[0], {k: V[k]["dev"]["Overall"] * 100 for k in ORDER}, "Classification, DEV · F1 %", ref=cur["dev"]["Overall"] * 100)
dots(axs[1], {k: V[k]["conf"]["Overall"] * 100 for k in ORDER}, "Classification, CONF · F1 %", ref=cur["conf"]["Overall"] * 100)
dots(axs[2], {k: V[k]["analogy_all"] * 100 for k in ORDER}, "Word analogy · %", ref=cur["analogy_all"] * 100)
dots(axs[3], {k: V[k]["sim_auc"] for k in ORDER}, "Similarity AUC", fmt="{:.3f}", ref=cur["sim_auc"])
for ax in axs[1:]: ax.tick_params(labelleft=False)
fig.suptitle("Every dimension option, scored the same way (dotted line = old 1,500-d BanglaMeta+)", x=0.01, ha="left", fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
figs.append((save(fig, "fig01_dimension_options.png"), "All ten options built from the same frozen rows. The new 1,200-d model is at or above the old "
             "1,500-d model on all four scores; 1,800 loses analogy; compressing to 300/500 (SVD/PCA) loses classification and similarity."))

fig, ax = plt.subplots(figsize=(12, 4.6)); ys = np.arange(len(ORDER))[::-1]
for y, k in zip(ys, ORDER):
    tr, te = V[k]["dev"]["train_Overall"] * 100, V[k]["dev"]["Overall"] * 100
    ax.plot([te, tr], [y, y], color=COL[k], lw=2.5, alpha=.5); ax.plot([te], [y], "o", color=COL[k], ms=8, mec="white")
    ax.plot([tr], [y], "s", color=COL[k], ms=7, mec="white"); ax.annotate(f"gap {tr - te:.2f}", (tr, y), xytext=(8, -3), textcoords="offset points", fontsize=9, color=INK)
ax.set_yticks(ys, [LABEL[k] for k in ORDER]); ax.grid(axis="y", visible=False); ax.set_xlabel("macro-F1 % on DEV (circle = held-out folds, square = training folds)")
ax.set_title("Classifier over-fitting: training-fold vs held-out-fold F1", loc="left", fontsize=13, color=INK); ax.margins(x=0.12)
figs.append((save(fig, "fig02_train_vs_heldout.png"), "The gap between training-fold and held-out-fold F1 is 5–7.5 points for every size and does not grow with "
             "the number of dimensions; the new 1,200-d model has the smallest gap (least over-fitting)."))

fig, ax = plt.subplots(figsize=(10, 4.2)); fr = ["0.1", "0.25", "0.5", "0.75", "1.0"]
lc = S["learning_curves"]; lcol = {"600 similarity view only": GREY, "900 thesis (BanglaMeta)": "#6b6a65", "1500 BanglaMeta+ (current)": ORANGE, "1800 + sim equal weights": "#d0cfc9"}
for k, v in lc.items():
    ax.plot([float(f) * 100 for f in fr], [v[f]["Overall"] * 100 for f in fr], "o-", color=lcol[k], lw=2, label=LABEL[k])
ax.set_xlabel("share of the training folds used (%)"); ax.set_ylabel("held-out macro-F1 % (DEV)"); ax.legend(frameon=False)
ax.set_title("Learning curves: still rising at 100% (more data would help; no sign of over-fitting)", loc="left", fontsize=13, color=INK)
figs.append((save(fig, "fig03_learning_curves.png"), "Held-out F1 rises steadily with more training texts for every size, and the larger models are not "
             "below the smaller ones at any data size, so the extra dimensions are not over-fitted."))

fb = S["frequency_bands"]; bands = [b for b in next(iter(fb.values())) if b != "questions"]
fig, ax = plt.subplots(figsize=(11, 4)); show = ["News-W2V", "Informal-W2V", "fastText-cc.bn", "BanglaMeta", "BanglaMeta+"]
bc = {"News-W2V": "#7a7974", "Informal-W2V": "#c9c8c2", "fastText-cc.bn": GREY, "BanglaMeta": ORANGE, "BanglaMeta+": BLUE}
x = np.arange(len(bands)); w = 0.16
for j, k in enumerate(show):
    ax.bar(x + (j - 2) * (w + 0.01), [fb[k][b] * 100 for b in bands], w, color=bc[k], label=k + (" (old 1,500-d)" if k == "BanglaMeta+" else ""))
ax.set_xticks(x, [f"words ranked {b}\n({fb['News-W2V']['questions'][b]:,} questions)" for b in bands]); ax.set_ylabel("analogy accuracy %")
ax.legend(frameon=False, ncol=5, fontsize=9); ax.grid(axis="x", visible=False)
ax.set_title("Under-fitting check: analogy accuracy by how frequent the answer word is", loc="left", fontsize=13, color=INK)
figs.append((save(fig, "fig04_frequency_bands.png"), "Informal-W2V is near zero in every band: it is under-trained (too little text for 300 dimensions). "
             "The other models do not collapse on rare words."))

fig, ax = plt.subplots(figsize=(11, 4.4)); ys = np.arange(len(ORDER))[::-1]
for y, k in zip(ys, ORDER):
    ax.barh(y + .2, V[k]["dim"], .36, color="#e7e6e2"); ax.barh(y - .2, V[k]["dims_90pct"], .36, color=COL[k])
    ax.text(V[k]["dim"] + 10, y + .2, f"{V[k]['dim']:,} nominal", va="center", fontsize=8.8, color=INK2)
    ax.text(V[k]["dims_90pct"] + 10, y - .2, f"{V[k]['dims_90pct']} hold 90% · PR {V[k]['participation_ratio']:.0f}", va="center", fontsize=8.8, color=INK)
ax.set_yticks(ys, [LABEL[k] for k in ORDER]); ax.grid(axis="y", visible=False); ax.set_xlim(0, 2150); ax.set_xlabel("dimensions")
ax.set_title("Effective dimensionality: nominal size vs directions that actually carry variance", loc="left", fontsize=13, color=INK)
figs.append((save(fig, "fig05_effective_dims.png"), "Participation ratio (PR) and the number of directions holding 90% of the variance, measured on the "
             "20,000 most frequent words. The concatenated models have far fewer effective dimensions than their nominal size."))

src = S["sources"]
def row(k): v = V[k]; return (f"<tr{' class=new' if k == NEW else ''}><td>{LABEL[k]}</td><td>{v['dim']:,}</td><td>{v['dev']['Overall'] * 100:.2f}</td>"
                              f"<td>{v['conf']['Overall'] * 100:.2f}</td><td>{v['analogy_all'] * 100:.2f}</td><td>{v['analogy_test'] * 100:.2f}</td>"
                              f"<td>{v['sim_auc']:.4f}</td><td>{v['dev']['gap'] * 100:.2f}</td><td>{v['dims_90pct']}</td></tr>")
n, o = V[NEW], cur
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Dimension and over-fitting study</title><style>
@page {{ size: A4 landscape; margin: 12mm; }}
body {{ font-family: Calibri, "Segoe UI", sans-serif; font-size: 10.6pt; color: #0b0b0b; margin: 0; line-height: 1.42; }}
h1 {{ font-size: 21pt; margin: 0; }} h2 {{ font-size: 14pt; border-bottom: 2px solid {BLUE}; padding-bottom: 2px; margin: 14px 0 6px; break-after: avoid; }}
.sub {{ color: #52514e; }} table {{ border-collapse: collapse; width: 100%; font-size: 9.8pt; }}
th, td {{ border-bottom: 1px solid #e7e6e2; padding: 3px 7px; text-align: right; }} th:first-child, td:first-child {{ text-align: left; }}
thead th {{ background: #0b0b0b; color: #fff; }} tr.new td {{ background: #eef4fc; font-weight: 700; }}
.box {{ border: 1px solid #d6d5d0; border-radius: 8px; padding: 8px 12px; background: #fafaf8; margin: 6px 0; break-inside: avoid; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }} ul {{ margin: 3px 0 3px 18px; padding: 0; }}
figure {{ margin: 10px 0; break-inside: avoid; }} figure img {{ width: 100%; }} figcaption {{ font-size: 9.6pt; color: #52514e; }}
code {{ font-family: Consolas, monospace; font-size: 9.4pt; background: #f0efec; padding: 0 3px; }}
</style></head><body>
<h1>Over-fitting, under-fitting and the choice of dimension</h1>
<div class="sub">BanglaMeta / BanglaMeta+ · measured by <code>webapp/backend/experiments/dimension_study.py</code> and <code>eps_1200.py</code>.
CONF results of the new variants are exploratory (the thesis used CONF once, for its confirmatory test).</div>
<h2>Answer in one table</h2>
<table><thead><tr><th>Option</th><th>dims</th><th>DEV F1</th><th>CONF F1</th><th>Analogy all</th><th>Analogy test half</th><th>Sim AUC</th><th>train–held-out gap</th><th>dims for 90%</th></tr></thead>
<tbody>{''.join(row(k) for k in ORDER)}</tbody></table>
<div class="two">
<div class="box"><b>Decision: BanglaMeta+ is now 1,200-d.</b> Against the old 1,500-d version:
<ul><li>DEV F1 {o['dev']['Overall'] * 100:.2f} → <b>{n['dev']['Overall'] * 100:.2f}</b>; CONF F1 {o['conf']['Overall'] * 100:.2f} → <b>{n['conf']['Overall'] * 100:.2f}</b></li>
<li>Analogy {o['analogy_all'] * 100:.2f}% → <b>{n['analogy_all'] * 100:.2f}%</b> (the same number of questions solved); similarity AUC {o['sim_auc']:.4f} → <b>{n['sim_auc']:.4f}</b></li>
<li>Over-fitting gap {o['dev']['gap'] * 100:.2f} → <b>{n['dev']['gap'] * 100:.2f}</b> points; 300 fewer dimensions</li></ul>
Change: Subs2vec is removed from the thesis part and kept only in the similarity part. It adds no significant classification
information (thesis check P2: +0.003 nats, interval includes 0), so in the thesis part it was dead weight. ε = {chk['eps']} was chosen on
the analogy tune half (all values tie; ties go to the smaller ε, the rule used from the start).</div>
<div class="box"><b>Is there over-fitting?</b> Not harmful over-fitting:
<ul><li>Held-out data: DEV {n['dev']['Overall'] * 100:.2f} vs untouched CONF {n['conf']['Overall'] * 100:.2f}; an over-fitted model would drop on CONF.</li>
<li>The classifier's train/held-out gap is 5–7.5 points for every size and does not grow with dimension (Figure 2).</li>
<li>Learning curves still rise at 100% data and larger models are never below smaller ones (Figure 3).</li>
<li>Selection: the settings were chosen on half the analogy questions; on the unseen half the model still reaches {n['analogy_test'] * 100:.2f}%.</li></ul></div>
<div class="box"><b>Is there under-fitting?</b> Yes, in one source: Informal-W2V.
<ul><li>{src['Informal-W2V']['corpus_tokens']:,} tokens for {src['Informal-W2V']['learned_parameters']:,} learned parameters ({src['Informal-W2V']['tokens_per_parameter']:.2f} tokens per parameter); {src['Informal-W2V']['share_words_lt_20'] * 100:.0f}% of its words occur fewer than 20 times.</li>
<li>Its analogy accuracy is about 1% in every frequency band (Figure 4).</li>
<li>Remedy already in the model: weight 0 for word similarity; its block is kept for classification, where it adds significant information on informal text (P2: +0.068 nats).</li>
<li>News-W2V: {src['News-W2V']['corpus_tokens']:,} tokens, {src['News-W2V']['tokens_per_parameter']:.2f} tokens per parameter, no rare-word collapse.</li></ul></div>
<div class="box"><b>Why not 1,800 or SVD?</b>
<ul><li>1,800 (adding the Informal similarity block): analogy falls to {V['1800 + sim-Informal (0.15)']['analogy_all'] * 100:.2f}% / {V['1800 + sim equal weights']['analogy_all'] * 100:.2f}%: the under-trained block adds noise.</li>
<li>SVD/PCA to 300 or 500 dims: classification falls to {V['300 PCA of BanglaMeta+']['dev']['Overall'] * 100:.2f} / {V['500 PCA of BanglaMeta+']['dev']['Overall'] * 100:.2f} and similarity AUC to {V['300 PCA of BanglaMeta+']['sim_auc']:.3f}; and the 500-d version over-fits <i>more</i> (gap {V['500 PCA of BanglaMeta+']['dev']['gap'] * 100:.2f}).</li>
<li>The thesis model itself never used SVD: it was tested in Step 4 and lost (76.36 vs 76.97).</li></ul></div></div>
<h2>How to defend the size to your professor</h2>
<div class="box"><ol>
<li><b>It is not a 1,200-dimension Word2Vec.</b> No Word2Vec was trained with more than 300 dimensions. The "150–500" rule of thumb is about training one model;
every trained model here is 300-d, inside that range. BanglaMeta+ is a <i>meta-embedding</i>: four 300-d blocks derived from three separately trained 300-d models, joined without new training.</li>
<li><b>The extra dimensions carry new information, not new parameters.</b> Theorem 2: a classifier on the joined vector can always ignore a block, so joining cannot lose;
Theorem 3 + check P2 measure what each block adds. Blocks that add nothing (Subs2vec in the thesis part, Informal in the similarity part) are removed.</li>
<li><b>Effective size is small.</b> 90% of the variance of the 1,200-d model lies in {n['dims_90pct']} directions (participation ratio {n['participation_ratio']:.0f}) (Figure 5).</li>
<li><b>Dimension does not drive generalisation error here.</b> Theorem 9: the probe's error bound depends on vector length (≤ 1) and weight norm, not on dimension;
measured train/held-out gaps and learning curves confirm it, and the held-out CONF score does not drop.</li>
<li><b>Every smaller and larger option was tested</b> (600, three 1,200s, 1,500, two 1,800s, SVD to 300/500): the 1,200-d model is the smallest that keeps all three scores.</li>
<li><b>Avoid "world first".</b> Say instead: "a 1,200-d Bangla meta-embedding of separately trained 300-d Word2Vec models; best of the evaluated static embeddings
on classification, analogy and similarity." That claim is backed by these measurements; "world first" is not.</li></ol></div>
<h2 style="break-before: page">Figures</h2>
{''.join(f"<figure><img src='figures/{a}'><figcaption><b>Figure {i}.</b> {b}</figcaption></figure>" for i, (a, b) in enumerate(figs, 1))}
</body></html>"""
(EV / "Dimension_and_overfitting_study.html").write_text(html, encoding="utf-8")
pdf = EV / "Dimension_and_overfitting_study.pdf"
subprocess.run([r"C:\Program Files\Google\Chrome\Application\chrome.exe", "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", (EV / "Dimension_and_overfitting_study.html").as_uri()], check=True, capture_output=True, timeout=120)
print("report:", pdf, pdf.stat().st_size)
