"""Quality benchmarks measured live on the served models, with the thesis's own definitions.

For every model this recomputes, from its served rows:
  * token coverage of the DEV texts (formal D1, informal D2);
  * native analogy accuracy: 3CosAdd over the first 200k rows, unanswerable questions count as wrong;
  * similarity AUC on the category word list (place / person / cardinal / ordinal / date);
  * DEV macro-F1 of the linear probe: mean of word rows -> StandardScaler -> RidgeClassifierCV, the same 5 stratified
    folds (seed 42) as merge_src.py.
The results are stored next to the values the thesis recorded (merge_dev.csv), so the page can show both and the
tests can check that this server reproduces the thesis.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import config as C
from .store import Registry

SEED = 42
TASKS = ["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"]
SEMANTIC = ["Division_District", "Gender", "Number_Ordinal", "Number_Date", "Number_Female"]
THESIS_NAME = {"merged": "T: CONC", "fasttext": "fastText-cc.bn", "news": "News-W2V", "informal": "Informal-W2V"}
# merged_sim is new (after the thesis): it has measured values only, no recorded DEV row and no CONF result


def load_eval(split: str = "dev"):
    ev = pd.read_parquet(C.EVAL_DATA / "eval_v2.parquet")
    ev["toks"] = ev.tokens.str.split()
    dev = ev[ev.split == split].reset_index(drop=True)
    analogy = {p.stem: [l.split() for l in open(p, encoding="utf-8") if len(l.split()) == 4]
               for p in sorted((C.EVAL_DATA / "analogy").glob("*.txt")) if p.stem != "Data_Stat"}
    cats = json.loads((C.EVAL_DATA / "categories.json").read_text(encoding="utf-8"))
    return dev, analogy, cats


def analogy_scores(m, analogy):
    sidx = {w: i for i, w in enumerate(m.words[:m.n_search])}
    files, qs = [], []
    for f, Q in analogy.items():
        for q in Q:
            if all(w in sidx for w in q):
                files.append(f); qs.append([sidx[w] for w in q])
    B = np.array(qs, dtype=np.int64)
    T = m.rows(B[:, 1]) - m.rows(B[:, 0]) + m.rows(B[:, 2])
    best_s = np.full(len(B), -np.inf, np.float32); best_i = np.zeros(len(B), np.int64)
    for r in range(0, m.n_search, 32768):
        e = min(r + 32768, m.n_search); Cm = np.asarray(m.U[r:e])
        for s in range(0, len(B), 1024):
            sl = slice(s, s + 1024); sim = T[sl] @ Cm.T; rows = np.arange(sim.shape[0])
            for j in (0, 1, 2):
                col = B[sl, j]; inside = (col >= r) & (col < e)
                sim[rows[inside], col[inside] - r] = -np.inf
            am = sim.argmax(1); mv = sim[rows, am]
            upd = mv > best_s[sl]                               # strict: an earlier row wins ties, as argmax does
            best_s[sl][upd] = mv[upd]; best_i[sl][upd] = am[upd] + r
    correct = best_i == B[:, 3]
    res = {}
    for f, Q in analogy.items():
        sel = np.array([x == f for x in files], bool)
        res[f] = (len(Q), int(sel.sum()), int(correct[sel].sum()))

    def agg(keys):
        n = sum(res[k][0] for k in keys); a = sum(res[k][1] for k in keys); c = sum(res[k][2] for k in keys)
        return c / n, a / n
    native = [k for k in res if k != "Mikolov-Filtered"]; syn = [k for k in native if k not in SEMANTIC]
    out = {}
    for lab, keys in (("analogy_native", native), ("analogy_semantic", SEMANTIC), ("analogy_syntactic", syn), ("analogy_mikolov", ["Mikolov-Filtered"])):
        out[f"{lab}_acc"], out[f"{lab}_answerable"] = agg(keys)
    out["analogy_questions"] = sum(res[k][0] for k in native)
    out["analogy_per_file"] = res
    return out


def sent_matrix(m, toks_list):
    X = np.zeros((len(toks_list), m.dim), np.float32); empty = 0
    for i, toks in enumerate(toks_list):
        ids = [m.idx[w] for w in toks if w in m.idx]
        if ids: X[i] = m.rows(ids).mean(0)
        else: empty += 1
    return X, empty


def probe_oof(X, y, folds):
    p = np.full(len(y), -1)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        p[te] = RidgeClassifierCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
    return p


def thesis_values():
    d = pd.read_csv(C.EVAL / "merge_output" / "merge_results" / "merge_dev.csv")
    cols = ["tok_cov_D1", "tok_cov_D2", "analogy_native_acc", "analogy_semantic_acc", "analogy_syntactic_acc", "analogy_mikolov_acc",
            "sim_auc_common", "sim_common_words", "F1_news_topic_7", "F1_review_sentiment_3", "F1_social_sentiment_3", "F1_hate_binary_2",
            "D1_F1", "D2_F1", "Overall_F1"]
    out = {}
    for key, name in THESIS_NAME.items():
        row = d[d.model == name].iloc[0]
        out[key] = {c: (float(row[c]) if pd.notna(row[c]) else None) for c in cols}
    return out


def official_conf():
    """Held-out CONF results exactly as stored by Step 5 (read, not recomputed)."""
    r = pd.read_csv(C.EVAL / "conf_output" / "conf_results" / "conf_results.csv")
    t = pd.read_csv(C.EVAL / "conf_output" / "conf_results" / "conf_tests.csv")
    keep = [c for c in r.columns if c in ("dim", "D1_F1", "D2_F1", "Overall_F1", "Overall_lo", "Overall_hi") or c.startswith("F1_")]
    models = {}
    for key, name in THESIS_NAME.items():
        row = r[r.model == name].iloc[0]
        models[key] = {c: float(row[c]) for c in keep if pd.notna(row[c])}
    tests = t[t.family == "confirmatory"][["comparison", "delta", "ci_lo", "ci_hi", "p_holm", "reject_H0"]]
    return {"models": models, "tests": json.loads(tests.to_json(orient="records")),
            "source": "eval/conf_output/conf_results/conf_results.csv, conf_tests.csv"}


def run(R: Registry, progress=lambda msg, frac: None) -> dict:
    t0 = time.perf_counter()
    dev, analogy, cats = load_eval()
    # thesis rule: the word must be in every source table, Subs2vec-w2 included (read from the merged provenance bits)
    common = [(w, c) for c, ws in cats.items() for w in ws
              if all(m.has(w) for m in R.models.values()) and R.merged.provenance_of(w)["subs"]]
    folds = {t: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((dev.task == t).sum()), dev.label[dev.task == t]))
             for t in TASKS}
    dev_toks = {d[:2]: [w for t in g.toks for w in t] for d, g in dev.groupby("domain")}
    conf, _, _ = load_eval("conf")
    conf_folds = {t: list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros((conf.task == t).sum()), conf.label[conf.task == t]))
                  for t in TASKS}
    results, steps, done = {}, len(C.ORDER) * 4, 0
    for key in C.ORDER:
        m = R.get(key); r = {}
        progress(f"{m.name}: analogy (3CosAdd over {m.n_search:,} rows)", done / steps)
        for d, toks in dev_toks.items():
            r[f"tok_cov_{d}"] = float(np.mean([m.has(w) for w in toks])); r[f"type_cov_{d}"] = float(np.mean([m.has(w) for w in set(toks)]))
        r.update(analogy_scores(m, analogy)); done += 1
        progress(f"{m.name}: similarity AUC", done / steps)
        U = m.rows([m.idx[w] for w, _ in common]); cs = np.array([c for _, c in common]); iu = np.triu_indices(len(common), 1)
        r["sim_auc_common"] = float(roc_auc_score((cs[iu[0]] == cs[iu[1]]).astype(int), (U @ U.T)[iu])); r["sim_common_words"] = len(common)
        done += 1
        progress(f"{m.name}: DEV linear probe (4 tasks x 5 folds)", done / steps)
        for t in TASKS:
            g = dev[dev.task == t]; X, empty = sent_matrix(m, g.toks.tolist()); y = g.label.values
            r[f"F1_{t}"] = float(f1_score(y, probe_oof(X, y, folds[t]), average="macro")); r[f"empty_{t}"] = empty
        r["D1_F1"] = r["F1_news_topic_7"]; r["D2_F1"] = float(np.mean([r[f"F1_{t}"] for t in TASKS[1:]])); r["Overall_F1"] = (r["D1_F1"] + r["D2_F1"]) / 2
        done += 1
        # held-out CONF half, same probe (as conf_src.py): reproduces the official CONF table for the thesis models,
        # and is EXPLORATORY for models made after the confirmatory test
        progress(f"{m.name}: CONF linear probe (4 tasks x 5 folds)", done / steps)
        for t in TASKS:
            g = conf[conf.task == t]; X, _ = sent_matrix(m, g.toks.tolist()); y = g.label.values.astype(int)
            r[f"conf_F1_{t}"] = float(f1_score(y, probe_oof(X, y, conf_folds[t]), average="macro"))
        r["conf_D1_F1"] = r["conf_F1_news_topic_7"]; r["conf_D2_F1"] = float(np.mean([r[f"conf_F1_{t}"] for t in TASKS[1:]]))
        r["conf_Overall_F1"] = (r["conf_D1_F1"] + r["conf_D2_F1"]) / 2
        done += 1
        results[key] = r
    progress("done", 1.0)
    return {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - t0, 1),
            "served_md5": {k: R.get(k).served_md5 for k in C.ORDER}, "results": results, "thesis": thesis_values()}


def load_cache(R: Registry):
    if not C.BENCH_CACHE.exists():
        return None
    b = json.loads(C.BENCH_CACHE.read_text(encoding="utf-8"))
    if b.get("served_md5") != {k: R.get(k).served_md5 for k in C.ORDER}:
        return None                                        # measured on other files: not valid for these models
    return b


def save_cache(b: dict):
    C.BENCH_CACHE.write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    import sys
    R = Registry()
    b = run(R, lambda msg, f: print(f"[{f:4.0%}] {msg}", flush=True))
    save_cache(b)
    for k in C.ORDER:
        r, t = b["results"][k], b["thesis"].get(k) or {m: float("nan") for m in b["results"][k] if isinstance(b["results"][k][m], float)}
        print(f"{k:9} analogy {r['analogy_native_acc']:.4f} (thesis {t['analogy_native_acc']:.4f})  simAUC {r['sim_auc_common']:.4f} ({t['sim_auc_common']:.4f})"
              f"  D1 {r['D1_F1']:.4f} ({t['D1_F1']:.4f})  D2 {r['D2_F1']:.4f} ({t['D2_F1']:.4f})  Overall {r['Overall_F1']:.4f} ({t['Overall_F1']:.4f})")
    sys.exit(0)
