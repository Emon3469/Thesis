"""
Build a small, domain-balanced Bangla evaluation set from 4 local sources.

Domain 1 (formal)   : Potrika news, 7 topics x 150                = 1,050
Domain 2 (informal) : BanglaBook 3 x 120 + SentNoB 3 x 120
                      + Bengali Hate Speech 2 x 165                = 1,050
Total                                                              = 2,100

Run:  python build_eval_set.py            (from anywhere; paths are absolute)
Out:  D:/Thesis/eval/kaggle_dataset/{eval_set.parquet, eval_set.csv, datasheet.json, datasheet.md}

Every step is deterministic (SEED = 42) so the file can be rebuilt byte-for-byte.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(r"D:/Thesis")
OUT = ROOT / "eval" / "kaggle_dataset"
OUT.mkdir(parents=True, exist_ok=True)

NEWS_DIR = ROOT / "Potrika & BARD  Soham News Datasets"
NEWS_FILES = ["Economy_40k.csv", "Education_40k.csv", "Entertainment_40k.csv",
              "International_40k.csv", "National_40k.csv", "politics_40k.csv",
              "Sports_40k.csv"]                     # *.IDAf = incomplete download, excluded
BOOK_DIR = ROOT / "Bengali Hate Speech Dataset" / "SentiNob & BanglaBook"
SENTNOB_DIR = ROOT / "Bengali Hate Speech Dataset" / "SentNoB"
HATE_FILE = ROOT / "bengali-hate-speech-dataset" / "Bengali%20hate%20speech.csv"

PER_CLASS = {"potrika_news": 150, "banglabook_sentiment": 120,
             "sentnob_sentiment": 120, "hate_speech": 165}
NEWS_MIN_WORDS, NEWS_KEEP_WORDS = 50, 400  # 400 words is > 256 subword tokens, so nothing the models see is lost
INFORMAL_MIN_WORDS = 3

# ---------------------------------------------------------------- cleaning
_ZW = re.compile(r"[\u200b\u200c\u200d\ufeff]")        # zero-width chars (ZWJ/ZWNJ break dedup)
_WS = re.compile(r"\s+")
_BN = re.compile(r"[\u0980-\u09FF]")
_LATIN = re.compile(r"[A-Za-z]")


def clean(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = _ZW.sub("", t)
    return _WS.sub(" ", t).strip()


def dedup_key(t: str) -> str:
    """Normalised key: lower-case, punctuation/symbols -> space, single spaces.
    Uses Unicode categories (P*, S*) instead of regex \\W, because Python's \\w does NOT
    match Bangla vowel signs (category Mc/Mn, e.g. 'া'), which would wrongly merge texts."""
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t.lower())
    return _WS.sub(" ", t).strip()


def latin_ratio(t: str) -> float:
    b, l = len(_BN.findall(t)), len(_LATIN.findall(t))
    return round(l / (b + l), 4) if (b + l) else 0.0


# ---------------------------------------------------------------- loaders
def load_news() -> pd.DataFrame:
    """Reservoir-style sampling without loading 1.2 GB: read in chunks, keep a random
    pool of 4x the needed rows per class (seeded), then sample from the pool."""
    rng = np.random.default_rng(SEED)
    need = PER_CLASS["potrika_news"] * 4
    frames = []
    for f in NEWS_FILES:
        pool = []
        for chunk in pd.read_csv(NEWS_DIR / f, usecols=["article", "class"], chunksize=20_000):
            chunk = chunk.dropna(subset=["article"])
            chunk["u"] = rng.random(len(chunk))       # random priority -> top-k == uniform sample
            pool.append(chunk.nsmallest(need, "u"))
            pool = [pd.concat(pool).nsmallest(need, "u")]
        p = pool[0]
        cls = p["class"].mode().iat[0]
        p = p.assign(text=p["article"].map(clean))
        p["n_words"] = p["text"].str.split().str.len()
        p = p[p["n_words"] >= NEWS_MIN_WORDS].copy()
        p["text"] = p["text"].map(lambda s: " ".join(s.split()[:NEWS_KEEP_WORDS]))
        frames.append(pd.DataFrame({"text": p["text"], "label_name": str(cls).lower(),
                                    "orig_label": cls, "orig_split": "all", "u": p["u"]}))
        print(f"  news {f:<24} pool={len(p):>4} class={cls}")
    df = pd.concat(frames, ignore_index=True)
    df["source"], df["task"], df["domain"] = "potrika_news", "news_topic_7", "D1_formal_news"
    return df


def load_banglabook() -> pd.DataFrame:
    parts = []
    for sp in ("train", "validation", "test"):
        d = pd.read_csv(BOOK_DIR / f"{sp}.csv", usecols=["Review", "sentiment"])
        d["orig_split"] = sp
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"text": d["Review"].map(clean), "label_name": d["sentiment"].str.strip(),
                         "orig_label": d["sentiment"], "orig_split": d["orig_split"],
                         "source": "banglabook_sentiment", "task": "review_sentiment_3",
                         "domain": "D2_informal_ugc"})


def load_sentnob() -> pd.DataFrame:
    names = {0: "neutral", 1: "positive", 2: "negative"}   # verified by reading samples per label
    parts = []
    for sp in ("Train", "Val", "Test"):
        d = pd.read_csv(SENTNOB_DIR / f"{sp}.csv")
        d["orig_split"] = sp.lower()
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"text": d["Data"].map(clean), "label_name": d["Label"].map(names),
                         "orig_label": d["Label"], "orig_split": d["orig_split"],
                         "source": "sentnob_sentiment", "task": "social_sentiment_3",
                         "domain": "D2_informal_ugc"})


def load_hate() -> pd.DataFrame:
    d = pd.read_csv(HATE_FILE)
    return pd.DataFrame({"text": d["sentence"].map(clean),
                         "label_name": d["hate"].map({0: "not_hate", 1: "hate"}),
                         "orig_label": d["hate"], "orig_split": "all",
                         "source": "hate_speech", "task": "hate_binary_2",
                         "domain": "D2_informal_ugc", "hate_topic": d["category"]})


# ---------------------------------------------------------------- build
def main() -> None:
    print("Loading sources ...")
    raw = {"potrika_news": load_news(), "banglabook_sentiment": load_banglabook(),
           "sentnob_sentiment": load_sentnob(), "hate_speech": load_hate()}

    stats, frames = {}, []
    for src, d in raw.items():
        n0 = len(d)
        d = d.dropna(subset=["text", "label_name"])
        d = d[d["text"].str.len() > 0].copy()
        d["n_words"] = d["text"].str.split().str.len()
        if src != "potrika_news":
            d = d[d["n_words"] >= INFORMAL_MIN_WORDS]
        d["key"] = d["text"].map(dedup_key)
        # drop keys with conflicting labels (same text, different gold label = label noise)
        conflict = d.groupby("key")["label_name"].nunique()
        d = d[~d["key"].isin(conflict[conflict > 1].index)]
        d = d.drop_duplicates("key")
        stats[src] = {"raw_rows": int(n0), "after_clean_dedup": int(len(d))}
        frames.append(d)

    allc = pd.concat(frames, ignore_index=True)
    # cross-source duplicates (e.g. the same YouTube comment in SentNoB and Hate Speech) -> drop everywhere
    cross = allc.groupby("key")["source"].nunique()
    cross_keys = set(cross[cross > 1].index)
    allc = allc[~allc["key"].isin(cross_keys)]
    print(f"Cross-source duplicate texts removed: {len(cross_keys)}")

    rng = np.random.default_rng(SEED)
    picked = []
    for src, k in PER_CLASS.items():
        d = allc[allc["source"] == src]
        for lab, g in d.groupby("label_name"):
            if len(g) < k:
                raise SystemExit(f"{src}/{lab}: only {len(g)} rows, need {k}")
            picked.append(g.sample(n=k, random_state=int(rng.integers(1e9))))
    ev = pd.concat(picked, ignore_index=True)

    # integer labels 0..C-1 per task, sorted by label name -> reproducible mapping
    ev["label"] = -1
    label_maps = {}
    for task, g in ev.groupby("task"):
        names = sorted(g["label_name"].unique())
        label_maps[task] = {n: i for i, n in enumerate(names)}
        ev.loc[g.index, "label"] = g["label_name"].map(label_maps[task])
    ev["latin_ratio"] = ev["text"].map(latin_ratio)
    ev["n_words"] = ev["text"].str.split().str.len()
    ev = ev.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle (mix sources)
    ev.insert(0, "uid", [f"ev{i:05d}" for i in range(len(ev))])
    cols = ["uid", "text", "domain", "source", "task", "label", "label_name",
            "orig_label", "orig_split", "n_words", "latin_ratio", "hate_topic"]
    ev = ev[cols]
    ev["orig_label"] = ev["orig_label"].astype(str)
    ev["hate_topic"] = ev["hate_topic"].fillna("")

    # ---------------- sanity checks (fail loudly instead of writing a bad file)
    assert ev["uid"].is_unique
    assert ev["text"].map(dedup_key).is_unique, "duplicate text survived"
    assert len(ev) == sum(k * len(label_maps[t]) for t, k in
                          zip(["news_topic_7", "review_sentiment_3", "social_sentiment_3", "hate_binary_2"],
                              PER_CLASS.values()))
    assert (ev.groupby("domain").size() == 1050).all(), ev.groupby("domain").size()

    ev.to_parquet(OUT / "eval_set.parquet", index=False)
    ev.to_csv(OUT / "eval_set.csv", index=False, encoding="utf-8-sig")   # BOM -> Excel shows Bangla

    # ---------------- datasheet
    counts = ev.groupby(["domain", "source", "label_name"]).size().rename("n").reset_index()
    length = ev.groupby("source")["n_words"].describe()[["mean", "50%", "min", "max"]].round(1)
    sheet = {
        "seed": SEED, "n_total": int(len(ev)),
        "per_domain": ev.groupby("domain").size().to_dict(),
        "per_source": ev.groupby("source").size().to_dict(),
        "label_maps": label_maps, "cleaning_stats": stats,
        "cross_source_duplicates_removed": len(cross_keys),
        "share_romanized_texts(latin_ratio>0.5)": ev.groupby("source")["latin_ratio"]
            .apply(lambda s: round(float((s > 0.5).mean()), 4)).to_dict(),
        "rules": {"unicode": "NFC + zero-width removed + whitespace collapsed",
                  "dedup": "lower-case, punctuation-stripped exact key; within and across sources; "
                           "conflicting-label duplicates dropped",
                  "news": f">= {NEWS_MIN_WORDS} words, first {NEWS_KEEP_WORDS} words kept",
                  "informal": f">= {INFORMAL_MIN_WORDS} words",
                  "sampling": "uniform random, stratified by label, equal per class"},
    }
    (OUT / "datasheet.json").write_text(json.dumps(sheet, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Bangla 2-domain evaluation set — datasheet", "",
          f"Total texts: **{len(ev)}** (seed {SEED})", "",
          "## Counts", "", counts.to_markdown(index=False), "",
          "## Length (words)", "", length.to_markdown(), "",
          "## Cleaning", "", pd.DataFrame(stats).T.to_markdown(), "",
          f"Cross-source duplicates removed: {len(cross_keys)}", "",
          "## Label maps", "", "```json", json.dumps(label_maps, ensure_ascii=False, indent=2), "```"]
    (OUT / "datasheet.md").write_text("\n".join(md), encoding="utf-8")
    print(counts.to_string(index=False))
    print(length)
    print(f"\nWrote {len(ev)} rows -> {OUT}")


if __name__ == "__main__":
    main()
