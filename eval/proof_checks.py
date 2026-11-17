"""Empirical checks of the conditions used in MATH_PROOF.md, for the pair A = mE5-base, B = fastText-cc.bn.
All on the 2,100-text eval set; out-of-fold (5-fold) wherever a classifier is involved."""
import os, json, math, warnings
from pathlib import Path
os.environ["JOBLIB_TEMP_FOLDER"] = str(Path(__file__).parent / "joblib_tmp")
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

H = Path(__file__).parent; SEED = 42
ev = pd.read_parquet(H / "kaggle_dataset" / "eval_set.parquet")
A = np.load(H / "kaggle_output/results/emb/mE5-base.npy").astype(np.float64)
B = np.load(H / "kaggle_output/results/emb/fastText-cc.bn.npy").astype(np.float64)
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
out = {}

# ---- Check 1 (Theorem 3, exact identity): cosine of L2-concatenation = mean of the two cosines
l2 = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
Ua, Ub = l2(A), l2(B); rng = np.random.default_rng(SEED); i, j = rng.integers(0, len(ev), (2, 20000))
cA, cB = (Ua[i] * Ua[j]).sum(1), (Ub[i] * Ub[j]).sum(1)
C = np.hstack([Ua, Ub]); cC = (C[i] * C[j]).sum(1) / (np.linalg.norm(C[i], axis=1) * np.linalg.norm(C[j], axis=1))
out["T3_max_abs_error_conc_identity"] = float(np.abs(cC - (cA + cB) / 2).max())

# ---- Check 2 (Theorem 4): size of the cross term that averaging adds on top of concatenation
Bp = np.pad(Ub, ((0, 0), (0, A.shape[1] - B.shape[1])))          # zero-pad B to 768 dims
cross = (Ua[i] * Bp[j]).sum(1) + (Bp[i] * Ua[j]).sum(1)          # <a_i,b_j> + <b_i,a_j>
signal = cA + cB
avg = (Ua + Bp) / 2; ip = (avg[i] * avg[j]).sum(1)
out["T4_identity_error"] = float(np.abs(ip - (signal + cross) / 4).max())
out["T4_cross_mean"], out["T4_cross_sd"] = float(cross.mean()), float(cross.sd if hasattr(cross, "sd") else cross.std())
out["T4_signal_sd"] = float(signal.std())
out["T4_noise_to_signal_variance"] = float(cross.var() / signal.var())
out["T4_corr_cross_signal"] = float(np.corrcoef(cross, signal)[0, 1])

# ---- Check 3 (Theorem 2): out-of-fold log-loss of the probe on A vs on [A;B]  (estimates H(Y|A) - H(Y|A,B))
def oof_logloss(X, y):
    ll = np.zeros(len(y)); pred = np.zeros(len(y), int)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1], cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                                   scoring="neg_log_loss", max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        P = clf.predict_proba(sc.transform(X[te])); ll[te] = -np.log(np.clip(P[np.arange(len(te)), y[te]], 1e-12, 1))
        pred[te] = P.argmax(1)
    return ll, pred
def std_block(X): return X       # per-fold standardisation happens inside oof_logloss
def job(t, which):
    idx = np.where(ev.task.values == t)[0]; y = ev.label.values[idx].astype(int)
    Xa, Xb = A[idx] / math.sqrt(A.shape[1]), B[idx] / math.sqrt(B.shape[1])
    X = {"A": Xa, "B": Xb, "AB": np.hstack([Xa, Xb])}[which]
    ll, pred = oof_logloss(X, y); return t, which, ll, pred, y
R = Parallel(n_jobs=6)(delayed(job)(t, w) for t in TASKS for w in ("A", "B", "AB"))
LL = {(t, w): ll for t, w, ll, _, _ in R}; PR = {(t, w): p for t, w, _, p, _ in R}; YY = {t: y for t, _, _, _, y in R}
rows = []
for t in TASKS:
    d = LL[(t, "A")] - LL[(t, "AB")]                               # per-text gain in nats; mean estimates I(Y;B|A)
    bs = np.random.default_rng(SEED).integers(0, len(d), (5000, len(d)))
    bm = d[bs].mean(1)
    ca, cb = PR[(t, "A")] == YY[t], PR[(t, "B")] == YY[t]
    rows.append(dict(task=t, n=len(d), logloss_A=LL[(t, "A")].mean(), logloss_B=LL[(t, "B")].mean(), logloss_AB=LL[(t, "AB")].mean(),
                     gain_nats=d.mean(), gain_lo=np.percentile(bm, 2.5), gain_hi=np.percentile(bm, 97.5),
                     acc_A=ca.mean(), acc_B=cb.mean(), acc_AB=(PR[(t, "AB")] == YY[t]).mean(),
                     only_A_right=int((ca & ~cb).sum()), only_B_right=int((~ca & cb).sum()), oracle=(ca | cb).mean()))
T = pd.DataFrame(rows); T.to_csv(H / "preeval" / "proof_checks_information_gain.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); print(T.round(4).to_string(index=False))

# ---- Check 4 (Theorem 5): feature radius after per-source scaling
za = StandardScaler().fit_transform(A) / math.sqrt(A.shape[1]); zb = StandardScaler().fit_transform(B) / math.sqrt(B.shape[1])
out["T5_mean_sq_norm_A"] = float((za ** 2).sum(1).mean()); out["T5_mean_sq_norm_AB"] = float((np.hstack([za, zb]) ** 2).sum(1).mean())
out["T5_radius_ratio"] = math.sqrt(out["T5_mean_sq_norm_AB"] / out["T5_mean_sq_norm_A"])
json.dump(out, open(H / "preeval" / "proof_checks.json", "w"), indent=2); print(json.dumps(out, indent=1))
