"""Paths and constants shared by the preparation script, the API server and the tests."""
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
WEBAPP = BACKEND.parent
REPO = WEBAPP.parent
EVAL = REPO / "eval"
SRC_MODELS = EVAL / "merge_output" / "merge_results" / "models"   # frozen Step 4 outputs (float16, unit rows)
EVAL_DATA = EVAL / "w2v_kaggle_dataset"                            # eval_v2.parquet, analogy/, categories.json, tokenize.py
DATA = WEBAPP / "data"
DERIVED = DATA / "models"                                          # float32 unit-row copies served by the API
RAW = DATA / "raw"
MANIFEST = DATA / "manifest.json"
BENCH_CACHE = DATA / "benchmarks.json"

FROZEN_MD5 = "10aa206f8462c82a7b16cf60054be70f"   # selected_pure_w2v.f16.npy, the model chosen in Step 4 and confirmed on CONF
TOPK_SEARCH = 200_000                             # analogy search space used in the thesis (first 200k rows)
SUBS_FILE = REPO / "models" / "B_subs2vec" / "bn_300_2_sg_wxd.csv.bz2"
SUBS_MD5 = "7f67dc8dab206d0c1f8a83600758b374"
FT_GZ = RAW / "cc.bn.300.vec.gz"
FT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz"

# key -> how the model is described and where its frozen source lives
MODELS = {
    "merged_plus": dict(
        name="BanglaMeta+", label="BanglaMeta+ (one model: thesis rows + similarity view)", dim=1200, color="#ff3d6e",
        source=SRC_MODELS / "selected_pure_w2v.f16.npy", words=SRC_MODELS / "selected_pure_w2v.words.txt",
        about="One vector per word: [sqrt(0.0005) x the thesis News+Informal blocks (600) ; sqrt(0.9995) x the BanglaMeta-Sim "
              "News+Subs blocks (600)], 1,200-d. Classification keeps the thesis features that carry information; cosine is 0.0005 thesis + 0.9995 "
              "similarity view. eps chosen on the analogy TUNE half (experiments/combined.py)."),
    "merged": dict(
        name="BanglaMeta", label="BanglaMeta (merged Word2Vec)", dim=900, color="#d7263d",
        source=SRC_MODELS / "selected_pure_w2v.f16.npy", words=SRC_MODELS / "selected_pure_w2v.words.txt",
        about="Concatenation of News-W2V, Informal-W2V and Subs2vec-w2 (1/sqrt(3) per block); the model selected on DEV and confirmed on CONF."),
    "merged_sim": dict(
        name="BanglaMeta-Sim", label="BanglaMeta-Sim (similarity view of the merged model)", dim=900, color="#ff8fa3",
        source=SRC_MODELS / "selected_pure_w2v.f16.npy", words=SRC_MODELS / "selected_pure_w2v.words.txt",
        about="The same frozen merged rows, post-processed for word similarity: per block all-but-the-top (mean and top 4 "
              "directions removed), then block weights News 1.65 / Informal 0 / Subs 1.35. Chosen on half of the analogy "
              "questions (experiments/block_weights.py); no training, no other model's vectors."),
    "fasttext": dict(
        name="fastText-cc.bn", label="fastText cc.bn (baseline)", dim=300, color="#5b6b73",
        source=DERIVED / "fastText-cc.bn.f16.npy", words=DERIVED / "fastText-cc.bn.words.txt",
        about="Facebook fastText Common Crawl Bangla vectors (.vec, no subwords); first 200k rows plus every evaluation word, as in the thesis."),
    "news": dict(
        name="News-W2V", label="News-W2V (formal)", dim=300, color="#1b7f5b",
        source=SRC_MODELS / "News-W2V.f16.npy", words=SRC_MODELS / "News-W2V.words.txt",
        about="Word2Vec skip-gram trained on the Potrika news corpus (65.3 M tokens)."),
    "informal": dict(
        name="Informal-W2V", label="Informal-W2V (informal)", dim=300, color="#e0a526",
        source=SRC_MODELS / "Informal-W2V.f16.npy", words=SRC_MODELS / "Informal-W2V.words.txt",
        about="Word2Vec skip-gram trained on book reviews, social comments and hate-speech text (3.19 M tokens)."),
}
ORDER = ["merged_plus", "merged", "merged_sim", "fasttext", "news", "informal"]
BASE_KEYS = ["merged", "fasttext", "news", "informal"]       # models stored as their own files
SIM_ABTT_D = 4                                               # chosen on the analogy TUNE half: D = 1..5 -> 4 best
SIM_WEIGHTS = (1.65, 0.0, 1.35)                              # chosen on the same TUNE half (grid step 0.05, mean 1)
SIM_FIT_ROWS = 50_000                                        # ABTT mean / directions fitted on the most frequent rows
PLUS_EPS = 0.0005                                            # thesis-row share of the BanglaMeta+ cosine (TUNE half; experiments/eps_1200.py)
DERIVED_KEYS = ("merged", "merged_sim", "merged_plus")     # all rows come from the frozen merged model


def served_path(key: str) -> Path:
    return DERIVED / f"{key}.f32.npy"


def served_words(key: str) -> Path:
    return DERIVED / f"{key}.words.txt"
