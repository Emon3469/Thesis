"""epsilon for the 1,200-d variant  c(w) = [sqrt(eps) * mNI(w) ; sqrt(1-eps) * s(w)]
mNI = thesis News+Informal blocks (600, unit), s = similarity view News+Subs (600, unit, weights 1.65/1.35).
eps chosen on the analogy TUNE half (largest tune accuracy; ties -> larger eps); the TEST half, the similarity AUC and the
classification are reported for the choice. Classification is recomputed (not assumed) on DEV and CONF.
Output: eval/dimension_study/eps_1200.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
src = (Path(__file__).with_name("dimension_study.py")).read_text(encoding="utf-8")
exec(compile(src[:src.index("# ------------------------------------------------------------------ 4. variants")], "dimension_study", "exec"))

EPSS = [0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
def two(B):
    pt = parts(B); s = np.hstack([np.sqrt(aN / 3, dtype=np.float32) * pt["sN"], np.sqrt(aS / 3, dtype=np.float32) * pt["sS"]])
    return pt["mNI"], s
Am, As = two(Mu[QB[:, 0]]); Bm, Bs = two(Mu[QB[:, 1]]); Cm_, Cs = two(Mu[QB[:, 2]])
Tm, Ts = Bm - Am + Cm_, Bs - As + Cs; del Am, As, Bm, Bs, Cm_, Cs
bs = np.full((len(EPSS), len(QB)), -np.inf, np.float32); bi = np.zeros((len(EPSS), len(QB)), np.int64)
for r in range(0, MER.n_search, 16384):
    e = min(r + 16384, MER.n_search); Pm, Ps = two(Mu[r:e])
    for s0 in range(0, len(QB), 512):
        sl = slice(s0, s0 + 512); sm = Tm[sl] @ Pm.T; ss = Ts[sl] @ Ps.T; rr = np.arange(sm.shape[0])
        for k, eps in enumerate(EPSS):
            sim = eps * sm + (1 - eps) * ss
            for j in (0, 1, 2):
                col = QB[sl, j]; inside = (col >= r) & (col < e); sim[rr[inside], col[inside] - r] = -np.inf
            am = sim.argmax(1); mv = sim[rr, am]; upd = mv > bs[k, sl]; bs[k, sl][upd] = mv[upd]; bi[k, sl][upd] = am[upd] + r
gm, gs = two(Mu[cidx]); Gm, Gs = gm @ gm.T, gs @ gs.T
rows = []
for k, eps in enumerate(EPSS):
    ok = bi[k] == QB[:, 3]
    rows.append(dict(eps=eps, tune=int(ok[HALF == "tune"].sum()) / n_q["tune"], test=int(ok[HALF == "test"].sum()) / n_q["test"],
                     all=int(ok.sum()) / NQ, correct=int(ok.sum()),
                     sim_auc=float(roc_auc_score((ccat[iu[0]] == ccat[iu[1]]).astype(int), (eps * Gm + (1 - eps) * Gs)[iu]))))
    log(rows[-1])
chosen = max(rows, key=lambda r: (r["tune"], r["eps"]))
# classification for the chosen eps, recomputed from the cached part sentence vectors
w = {"mNI": chosen["eps"], "sN": (1 - chosen["eps"]) * aN / 3, "sS": (1 - chosen["eps"]) * aS / 3}
XS = {s: np.hstack([np.sqrt(w[p]) * PARTS[p][s] for p in ("mNI", "sN", "sS")]) for s in ("dev", "conf")}
d, c = classify(XS, "dev"), classify(XS, "conf")
out = {"grid": rows, "chosen": chosen, "classification": {"dev": d, "conf": c}}
(OUT / "eps_1200.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
log(f"chosen eps {chosen['eps']}: analogy {chosen['all']*100:.2f} ({chosen['correct']} correct, test half {chosen['test']*100:.2f})  AUC {chosen['sim_auc']:.4f}  "
    f"DEV {d['Overall']*100:.2f}  CONF {c['Overall']*100:.2f}  gap {d['gap']*100:.2f}")
