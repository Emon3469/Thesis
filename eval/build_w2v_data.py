"""
Data for the Word2Vec pre-evaluation (run locally, low memory).

Outputs (D:/Thesis/eval/w2v_kaggle_dataset/):
  eval_v2.parquet          ~24k labelled texts, power-sized (see PRE_ANALYSIS_REPORT §6.6), with
                           split = dev/conf (stratified 50/50 per task, seed 42) and pre-tokenised text
  news_corpus.txt.gz       all Potrika articles NOT in eval_v2 (one tokenised article per line)
  informal_corpus.txt.gz   all BanglaBook/SentNoB/Hate texts NOT in eval_v2 (one tokenised text per line)
  categories.json          word-type lists for the similarity proxy, derived from the analogy data
  analogy/*.txt            the Bangla analogy set (Akter et al. 2023, MIT licence)
  tokenize.py              the tokenizer used everywhere (eval texts and corpora)
"""
import gzip, json, shutil, sys, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_eval_set import (clean, dedup_key, latin_ratio, load_banglabook, load_sentnob, load_hate,
                            NEWS_DIR, NEWS_FILES, NEWS_MIN_WORDS, NEWS_KEEP_WORDS, INFORMAL_MIN_WORDS)

SEED = 42
OUT = HERE / "w2v_kaggle_dataset"; OUT.mkdir(exist_ok=True)
PER_CLASS = {"potrika_news": 1700, "banglabook_sentiment": 1500, "sentnob_sentiment": 1000, "hate_speech": 2250}
TASK = {"potrika_news": "news_topic_7", "banglabook_sentiment": "review_sentiment_3",
        "sentnob_sentiment": "social_sentiment_3", "hate_speech": "hate_binary_2"}

TOKENIZER_SRC = '''import unicodedata
def tokenize(text):
    """NFC, drop zero-width chars, split off punctuation (Unicode P*), lowercase Latin, split on whitespace."""
    t = unicodedata.normalize("NFC", str(text))
    t = "".join(c for c in t if c not in "\\u200b\\u200c\\u200d\\ufeff")
    t = "".join(f" {c} " if unicodedata.category(c)[0] == "P" else c for c in t)
    return [w for w in t.lower().split() if not all(unicodedata.category(c)[0] == "P" for c in w)]
'''
(OUT / "tokenize.py").write_text(TOKENIZER_SRC, encoding="utf-8")
ns = {}; exec(TOKENIZER_SRC, ns); tokenize = ns["tokenize"]

# ------------------------------------------------------------- news: pass 1, reservoir pools per topic
rng = np.random.default_rng(SEED)
need = int(PER_CLASS["potrika_news"] * 1.3)
pools = []
for f in NEWS_FILES:
    pool = None
    for ch in pd.read_csv(NEWS_DIR / f, usecols=["article", "class"], chunksize=20_000):
        ch = ch.dropna(subset=["article"]).copy()
        ch["u"] = rng.random(len(ch))
        pool = ch if pool is None else pd.concat([pool, ch])
        pool = pool.nsmallest(need * 3, "u")              # keep a margin for the length filter
    pool["text"] = pool["article"].map(clean)
    pool["n_words"] = pool["text"].str.split().str.len()
    pool = pool[pool.n_words >= NEWS_MIN_WORDS].nsmallest(need, "u").copy()
    pool["text"] = pool["text"].map(lambda s: " ".join(s.split()[:NEWS_KEEP_WORDS]))
    cls = pool["class"].mode().iat[0]
    pools.append(pd.DataFrame({"text": pool.text, "label_name": str(cls).lower(), "source": "potrika_news"}))
    print(f"news pool {f}: {len(pool)}")
news = pd.concat(pools, ignore_index=True)

# ------------------------------------------------------------- informal
inf = []
for src, d in (("banglabook_sentiment", load_banglabook()), ("sentnob_sentiment", load_sentnob()), ("hate_speech", load_hate())):
    d = d.dropna(subset=["text", "label_name"]); d = d[d.text.str.split().str.len() >= INFORMAL_MIN_WORDS].copy()
    d["source"] = src; inf.append(d[["text", "label_name", "source"]])
allc = pd.concat([news] + inf, ignore_index=True)
allc["key"] = allc.text.map(dedup_key)
conf = allc.groupby("key").label_name.nunique(); allc = allc[~allc.key.isin(conf[conf > 1].index)]
cross = allc.groupby("key").source.nunique(); allc = allc[~allc.key.isin(cross[cross > 1].index)]
allc = allc.drop_duplicates("key")

picked = []
for src, k in PER_CLASS.items():
    for lab, g in allc[allc.source == src].groupby("label_name"):
        if len(g) < k: raise SystemExit(f"{src}/{lab}: only {len(g)} available, need {k}")
        picked.append(g.sample(n=k, random_state=int(rng.integers(1e9))))
ev = pd.concat(picked, ignore_index=True)
ev["task"] = ev.source.map(TASK)
ev["domain"] = np.where(ev.source == "potrika_news", "D1_formal_news", "D2_informal_ugc")
ev["label"] = -1
for t, g in ev.groupby("task"):
    m = {n: i for i, n in enumerate(sorted(g.label_name.unique()))}
    ev.loc[g.index, "label"] = g.label_name.map(m)
ev["split"] = ""
for t, g in ev.groupby("task"):
    dev, _ = train_test_split(g.index, test_size=0.5, stratify=g.label, random_state=SEED)
    ev.loc[g.index, "split"] = np.where(g.index.isin(dev), "dev", "conf")
ev["tokens"] = ev.text.map(lambda s: " ".join(tokenize(s)))
ev["latin_ratio"] = ev.text.map(latin_ratio)
ev = ev.sample(frac=1, random_state=SEED).reset_index(drop=True)
ev.insert(0, "uid", [f"w{i:05d}" for i in range(len(ev))])
EVAL_KEYS = set(ev.key)
assert ev.key.is_unique and (ev.label >= 0).all()
ev.drop(columns="key").to_parquet(OUT / "eval_v2.parquet", index=False)
print(ev.groupby(["task", "split"]).size().unstack(), "\ntotal", len(ev))

# ------------------------------------------------------------- 3 small datasets (drawn from the held-out CONF half)
mrng = np.random.default_rng(SEED + 1)
conf_ev = ev[ev.split == "conf"]
news_c = conf_ev[conf_ev.source == "potrika_news"]
topics = sorted(mrng.choice(sorted(news_c.label_name.unique()), 5, replace=False))
d1 = pd.concat([news_c[news_c.label_name == t].sample(2, random_state=int(mrng.integers(1e9))) for t in topics])
d1["class"] = d1.label_name
spec = [("sentnob_sentiment", "positive", "social:positive"), ("sentnob_sentiment", "negative", "social:negative"),
        ("sentnob_sentiment", "neutral", "social:neutral"), ("hate_speech", "hate", "comment:hate"),
        ("hate_speech", "not_hate", "comment:not_hate")]
d2 = pd.concat([conf_ev[(conf_ev.source == s) & (conf_ev.label_name == l)].sample(2, random_state=int(mrng.integers(1e9))).assign(**{"class": c})
                for s, l, c in spec])
MINI = OUT / "mini"; MINI.mkdir(exist_ok=True)
cols = ["uid", "text", "tokens", "domain", "source", "label_name", "class"]
d1, d2 = d1[cols].reset_index(drop=True), d2[cols].reset_index(drop=True)
merged = pd.concat([d1.assign(**{"class": "D1:" + d1["class"]}), d2.assign(**{"class": "D2:" + d2["class"]})], ignore_index=True)
for name, d in (("domain_1", d1), ("domain_2", d2), ("merged_dataset", merged)):
    assert d["class"].value_counts().eq(2).all()
    d.to_csv(MINI / f"{name}.csv", index=False, encoding="utf-8-sig")
assert len(merged) == 20 and set(merged.uid) == set(d1.uid) | set(d2.uid)
print("mini datasets:", len(d1), len(d2), len(merged), "| news topics:", topics)

# ------------------------------------------------------------- corpora (eval texts excluded by key)
def write_corpus(path, texts, excl_key):
    n = w = skipped = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for t in texts:
            if excl_key(t) in EVAL_KEYS: skipped += 1; continue
            toks = tokenize(t)
            if len(toks) < 3: continue
            f.write(" ".join(toks) + "\n"); n += 1; w += len(toks)
    return n, w, skipped

def news_texts():
    for f in NEWS_FILES:
        for ch in pd.read_csv(NEWS_DIR / f, usecols=["article"], chunksize=20_000):
            for a in ch.article.dropna(): yield clean(a)
stats = {}
stats["news"] = write_corpus(OUT / "news_corpus.txt.gz", news_texts(),
                             lambda t: dedup_key(" ".join(t.split()[:NEWS_KEEP_WORDS])))
inf_all = pd.concat(inf).text
stats["informal"] = write_corpus(OUT / "informal_corpus.txt.gz", inf_all, dedup_key)
for k, (n, w, s) in stats.items(): print(f"{k} corpus: {n:,} docs, {w:,} tokens, {s:,} eval texts excluded")

# ------------------------------------------------------------- analogy data + similarity categories
A = HERE.parent / "eval_resources" / "Word-Analogy-Bangla-main" / "Data"
shutil.copytree(A, OUT / "analogy", dirs_exist_ok=True)
def roles(fn):
    L = [l.split() for l in open(A / fn, encoding="utf-8") if len(l.split()) == 4]
    return {w for l in L for w in (l[0], l[2])}, {w for l in L for w in (l[1], l[3])}
div, dist = roles("Division_District.txt"); male, female = roles("Gender.txt")
card = set().union(*(roles(f)[0] for f in ("Number_Ordinal.txt", "Number_Date.txt", "Number_Female.txt")))
ordn = roles("Number_Ordinal.txt")[1]; date = roles("Number_Date.txt")[1]
cats = {"place": sorted(div | dist), "person": sorted(male | female), "cardinal": sorted(card),
        "ordinal": sorted(ordn), "date": sorted(date)}
words = [w for v in cats.values() for w in v]; assert len(words) == len(set(words)), "word in two categories"
json.dump(cats, open(OUT / "categories.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump({"per_class": PER_CLASS, "corpus_stats": stats, "n_eval": len(ev),
           "category_sizes": {k: len(v) for k, v in cats.items()}}, open(OUT / "datasheet.json", "w"), indent=1)
print({k: len(v) for k, v in cats.items()})
