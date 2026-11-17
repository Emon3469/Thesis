"""Figures for the pre-evaluation (reads preeval/*.csv)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path(__file__).parent / "preeval"; F = P / "fig"; F.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
s1 = pd.read_csv(P / "stage1_single_sources_dev.csv"); s2 = pd.read_csv(P / "stage2_pairs_dev.csv")
s3 = pd.read_csv(P / "stage3_methods_dev.csv"); tc = pd.read_csv(P / "stage5_confirmation.csv")
cmp_ = pd.read_csv(P / "stage5_selected_vs_best_single.csv")

# 1 single-source screening: D1 vs D2 scatter
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(100 * s1.D1, 100 * s1.D2, s=40, c="#3b6fb6")
for _, r in s1.iterrows(): ax.annotate(r.srcs, (100 * r.D1, 100 * r.D2), fontsize=7, xytext=(4, 2), textcoords="offset points")
ax.set_xlabel("Domain 1 (news) probe macro-F1 %"); ax.set_ylabel("Domain 2 (informal) mean probe macro-F1 %")
ax.set_title("Stage 1 — single sources on DEV half"); fig.tight_layout(); fig.savefig(F / "s1_single_sources.png"); plt.close(fig)

# 2 pair screening: gain vs CKA
fig, ax = plt.subplots(figsize=(6.5, 4.5))
cka = (s2.CKA_D1 + s2.CKA_D2) / 2
ax.scatter(cka, 100 * s2.gain_Overall_vs_best_member, s=35, c="#d1495b")
for c, g, n in zip(cka, 100 * s2.gain_Overall_vs_best_member, s2.srcs): ax.annotate(n, (c, g), fontsize=6, xytext=(3, 2), textcoords="offset points")
ax.axhline(0, color="k", lw=.8); ax.set_xlabel("mean linear CKA of the pair (low = complementary)")
ax.set_ylabel("Overall gain of CONC-std over its better member (F1 points)")
ax.set_title("Stage 2 — is complementarity predictive of fusion gain? (DEV)")
fig.tight_layout(); fig.savefig(F / "s2_pairs_gain_vs_cka.png"); plt.close(fig)

# 3 methods: Pareto plot D1 vs D2
fig, ax = plt.subplots(figsize=(7.5, 5.5)); mk = {}
for i, (pair, g) in enumerate(s3.groupby("srcs")):
    lab = g.method + np.where(g.alpha.notna(), "(a=" + g.alpha.astype(str) + ")", "")
    ax.scatter(100 * g.D1, 100 * g.D2, s=30, label=pair)
    for x, y, l in zip(100 * g.D1, 100 * g.D2, lab): ax.annotate(l, (x, y), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
b1, b2 = s1[s1.srcs != "TFIDF-char-SVD"].D1.max(), s1[s1.srcs != "TFIDF-char-SVD"].D2.max()
ax.axvline(100 * b1, ls="--", c="gray"); ax.axhline(100 * b2, ls="--", c="gray")
ax.text(100 * b1, ax.get_ylim()[0], " best single D1", fontsize=6, color="gray"); ax.text(ax.get_xlim()[0], 100 * b2, "best single D2", fontsize=6, color="gray")
ax.set_xlabel("Domain 1 F1 %"); ax.set_ylabel("Domain 2 F1 %"); ax.legend(fontsize=6, frameon=False)
ax.set_title("Stage 3 — fusion methods (DEV); upper-right of both dashed lines = beats both specialists")
fig.tight_layout(); fig.savefig(F / "s3_methods_pareto.png"); plt.close(fig)

# 4 confirmation forest plot
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, side in zip(axs, ["D1", "D2", "Overall"]):
    t = tc.sort_values(side)
    ax.errorbar(100 * t[side], range(len(t)), xerr=[100 * (t[side] - t[f"{side}_lo"]), 100 * (t[f"{side}_hi"] - t[side])],
                fmt="o", color="#00798c", capsize=3)
    ax.set_yticks(range(len(t)), t.config, fontsize=7); ax.set_title(f"CONF half — {side} macro-F1 % (95% CI)")
    for y, v in enumerate(100 * t[side]): ax.text(v, y + 0.2, f"{v:.1f}", fontsize=6.5, ha="center")
fig.tight_layout(); fig.savefig(F / "s5_confirmation_forest.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.errorbar(100 * cmp_.delta, range(len(cmp_)), xerr=[100 * (cmp_.delta - cmp_.lo), 100 * (cmp_.hi - cmp_.delta)], fmt="s", capsize=4, color="#d1495b")
ax.axvline(0, color="k", lw=.8); ax.axvline(-1, color="gray", ls=":", lw=.8)
ax.set_yticks(range(len(cmp_)), [f"{d} vs {r}" for d, r in zip(cmp_.domain, cmp_.reference)], fontsize=7)
ax.set_xlabel("Δ macro-F1 points (selected fusion − reference), 95% bootstrap CI; dotted = −1 pt margin")
fig.tight_layout(); fig.savefig(F / "s5_delta.png"); plt.close(fig)

# 5 CKA heatmaps
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for ax, d in zip(axs, ["D1_formal_news", "D2_informal_ugc"]):
    M = pd.read_csv(P / f"cka_{d}.csv", index_col=0); ax.imshow(M.values, cmap="viridis", vmin=0, vmax=1)
    for i in range(len(M)):
        for j in range(len(M)): ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if M.values[i, j] < .6 else "k")
    ax.set_xticks(range(len(M)), M.columns, rotation=60, ha="right", fontsize=7); ax.set_yticks(range(len(M)), M.index, fontsize=7)
    ax.set_title(f"Linear CKA — {d}")
fig.tight_layout(); fig.savefig(F / "cka_all_sources.png"); plt.close(fig)
print("figures ->", F)
