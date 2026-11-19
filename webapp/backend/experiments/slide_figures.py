"""Two answer figures (held-out CONF half), read from eval/banglameta_plus_eval/results.json:
  fig10_why_it_wins.png       - news (D1) vs informal (D2) macro-F1: which models are good at both at once
  fig11_all_models_conf.png   - every model on the held-out half: Overall macro-F1 with 95% CI, DEV for comparison
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

EV = C.REPO / "eval" / "banglameta_plus_eval"; FIG = EV / "figures"
D = json.loads((EV / "results.json").read_text(encoding="utf-8")); R = D["results"]; KEYS = D["models"]
NAME = {k: R[k]["name"] + (" (thesis)" if k == "merged" else "") for k in KEYS}
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e", "#e7e6e2"
COL = {k: BLUE if k == "merged_plus" else ORANGE if k == "merged" else GREY for k in KEYS}
plt.rcParams.update({"font.family": "Calibri", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": "#bdbcb6", "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "savefig.dpi": 200, "savefig.bbox": "tight"})
p = lambda k, m: R[k][f"conf_{m}"] * 100

# ------------------------------------------------------------------ 1. why it wins
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
best_d1 = max(KEYS, key=lambda k: R[k]["conf_D1_F1"]); best_d2 = max(KEYS, key=lambda k: R[k]["conf_D2_F1"])
single = ["fasttext", "news", "informal"]
s_d1 = max(single, key=lambda k: R[k]["conf_D1_F1"]); s_d2 = max(single, key=lambda k: R[k]["conf_D2_F1"])
ax.axvline(p(s_d1, "D1_F1"), color=GREY, lw=1, ls=":"); ax.axhline(p(s_d2, "D2_F1"), color=GREY, lw=1, ls=":")
ax.text(p(s_d1, "D1_F1"), 60.6, f" best single model on news\n ({R[s_d1]['name']}, {p(s_d1, 'D1_F1'):.2f})", fontsize=9, color=INK2, va="bottom")
ax.text(82.4, p(s_d2, "D2_F1") + 0.1, f"best single model on informal ({R[s_d2]['name']}, {p(s_d2, 'D2_F1'):.2f})", fontsize=9, color=INK2, va="bottom")
for k in KEYS:
    x, y = p(k, "D1_F1"), p(k, "D2_F1")
    ax.plot([p(k, "D1_lo"), p(k, "D1_hi")], [y, y], color=COL[k], lw=2, alpha=.45)
    ax.plot([x, x], [p(k, "D2_lo"), p(k, "D2_hi")], color=COL[k], lw=2, alpha=.45)
    ax.plot([x], [y], "o", ms=11, color=COL[k], mec="white", mew=1.8, zorder=3)
    off = {"merged_plus": (30, -60), "merged": (-50, 50), "merged_sim": (14, -26), "fasttext": (-16, -30), "news": (12, 8), "informal": (14, -40)}[k]
    ax.annotate(f"{NAME[k]}\nnews {x:.2f} · informal {y:.2f}", (x, y), xytext=off, textcoords="offset points", fontsize=9.5, color=INK,
                fontweight="bold" if k in ("merged_plus", "merged") else "normal", ha="left" if off[0] > 0 else "right",
                arrowprops=dict(arrowstyle="-", color=COL[k], lw=1.3) if k in ("merged_plus", "merged") else None)
ax.set_xlabel("News (D1) macro-F1 %, CONF"); ax.set_ylabel("Informal (D2) macro-F1 %, CONF")
ax.set_xlim(80, 90.3); ax.set_ylim(60.5, 67.6)
ax.set_title("Only the merged models sit in the top-right corner", loc="left", fontsize=12.5, fontweight="bold", color=INK)

# right: each domain as a dot plot, specialist highlighted
ys = np.arange(len(KEYS))[::-1]
for j, (m, lab) in enumerate((("D1_F1", "news"), ("D2_F1", "informal"))):
    for y, k in zip(ys, KEYS):
        xv = p(k, m); yy = y + (0.17 if j == 0 else -0.17)
        lo, hi = p(k, m.replace("F1", "lo")), p(k, m.replace("F1", "hi"))
        bx.plot([lo, hi], [yy, yy], color=COL[k], lw=2, alpha=.45)
        bx.plot([xv], [yy], "o" if j == 0 else "s", ms=8, color=COL[k], mec="white", mew=1.4, zorder=3)
        bx.annotate(f"{lab} {xv:.2f}", (hi, yy), xytext=(5, -3), textcoords="offset points", fontsize=8.8, color=INK)
bx.set_yticks(ys, [NAME[k] for k in KEYS]); bx.grid(axis="y", visible=False); bx.set_xlim(60, 94)
for lab, k in zip(bx.get_yticklabels(), KEYS):
    if k in ("merged_plus", "merged"): lab.set_fontweight("bold"); lab.set_color(INK)
bx.set_xlabel("macro-F1 % on CONF (● news D1, ■ informal D2; line = 95% CI)".replace("●", "circle =").replace("■", "square ="))
bx.set_title("Per domain, with 95% intervals", loc="left", fontsize=12.5, fontweight="bold", color=INK)
fig.suptitle("Why it wins: best on news and on informal text at once (held-out CONF half, 11,950 texts)", x=0.01, ha="left", fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig10_why_it_wins.png", facecolor="white"); plt.close(fig)

# ------------------------------------------------------------------ 2. all models on the held-out half
order = sorted(KEYS, key=lambda k: -R[k]["conf_Overall_F1"])
fig, ax = plt.subplots(figsize=(12.5, 4.9)); ys = np.arange(len(order))[::-1]
for y, k in zip(ys, order):
    c, lo, hi, dv = p(k, "Overall_F1"), p(k, "Overall_lo"), p(k, "Overall_hi"), R[k]["dev_Overall_F1"] * 100
    ax.plot([lo, hi], [y, y], color=COL[k], lw=3, alpha=.45, solid_capstyle="round")
    ax.plot([c], [y], "o", ms=11, color=COL[k], mec="white", mew=1.8, zorder=3)
    ax.plot([dv], [y], "D", ms=7, mfc="white", mec=INK2, mew=1.4, zorder=4)
    ax.annotate(f"{c:.2f}  [{lo:.2f}, {hi:.2f}]   DEV {dv:.2f}   {R[k]['dim']:,}-d", (hi, y), xytext=(10, -4), textcoords="offset points",
                fontsize=9.5, color=INK, fontweight="bold" if k == "merged_plus" else "normal")
ax.set_yticks(ys, [NAME[k] for k in order]); ax.grid(axis="y", visible=False); ax.set_xlim(72, 82.2)
for lab, k in zip(ax.get_yticklabels(), order):
    if k in ("merged_plus", "merged"): lab.set_fontweight("bold"); lab.set_color(INK)
ax.set_xlabel("Overall macro-F1 %  (circle = held-out CONF with 95% bootstrap interval, open diamond = DEV)")
ax.set_title("All models on the held-out half: CONF Overall macro-F1, ranked", loc="left", fontsize=14, color=INK)
d = {x["vs"]: x for x in D["deltas"] if x["split"] == "conf" and x["metric"] == "Overall"}
note = (f"BanglaMeta+ vs fastText {d['fasttext']['delta'] * 100:+.2f} [{d['fasttext']['ci_lo'] * 100:+.2f}, {d['fasttext']['ci_hi'] * 100:+.2f}]  ·  "
        f"vs thesis BanglaMeta {d['merged']['delta'] * 100:+.2f} [{d['merged']['ci_lo'] * 100:+.2f}, {d['merged']['ci_hi'] * 100:+.2f}] (tie)  ·  "
        "CONF values of BanglaMeta+ and BanglaMeta-Sim are exploratory")
fig.text(0.01, -0.02, note, fontsize=9.5, color=INK2)
fig.savefig(FIG / "fig11_all_models_conf.png", facecolor="white"); plt.close(fig)
for k in order:
    print(f"{NAME[k]:<26} D1 {p(k,'D1_F1'):.2f}  D2 {p(k,'D2_F1'):.2f}  CONF {p(k,'Overall_F1'):.2f} [{p(k,'Overall_lo'):.2f}, {p(k,'Overall_hi'):.2f}]  DEV {R[k]['dev_Overall_F1']*100:.2f}")
