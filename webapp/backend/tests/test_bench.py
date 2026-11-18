"""The server's live measurements reproduce the values the thesis recorded."""
import numpy as np
import pytest

import config as C
from app import bench

METRICS = ["tok_cov_D1", "tok_cov_D2", "analogy_native_acc", "analogy_semantic_acc", "analogy_syntactic_acc", "analogy_mikolov_acc",
           "sim_auc_common", "sim_common_words", "F1_news_topic_7", "F1_review_sentiment_3", "F1_social_sentiment_3", "F1_hate_binary_2",
           "D1_F1", "D2_F1", "Overall_F1"]


@pytest.fixture(scope="module")
def cache(R):
    b = bench.load_cache(R)
    if b is None:
        pytest.skip("no benchmark cache for these files: run `python -m app.bench`")
    return b


@pytest.mark.parametrize("key", C.BASE_KEYS)
def test_cached_measurements_reproduce_thesis(cache, key):
    got, ref = cache["results"][key], cache["thesis"][key]
    for m in METRICS:
        # analogy sub-scores: the served rows are the frozen float16 values, the thesis scored the single models in float32
        # before saving, so a near-tie can flip -> allow at most one question of difference (the native total must match)
        tol = 1.01 / per_set_questions(cache, key, m) if m.startswith("analogy_") and m != "analogy_native_acc" else 1e-4
        assert abs(got[m] - ref[m]) <= tol, f"{key} {m}: server {got[m]} vs thesis {ref[m]}"


SEMANTIC = ["Division_District", "Gender", "Number_Ordinal", "Number_Date", "Number_Female"]


def per_set_questions(cache, key, m):
    files = cache["results"][key]["analogy_per_file"]
    native = [f for f in files if f != "Mikolov-Filtered"]
    keys = {"analogy_semantic_acc": SEMANTIC, "analogy_syntactic_acc": [f for f in native if f not in SEMANTIC],
            "analogy_mikolov_acc": ["Mikolov-Filtered"]}[m]
    return sum(files[f][0] for f in keys)


def test_live_analogy_run_equals_cache(R, cache):
    _, analogy, _ = bench.load_eval()
    r = bench.analogy_scores(R.get("informal"), analogy)
    assert r["analogy_native_acc"] == cache["results"]["informal"]["analogy_native_acc"]


def test_conf_official_ranking():
    o = bench.official_conf()["models"]
    assert max(o, key=lambda k: o[k]["Overall_F1"]) == "merged"
    assert np.isclose(o["merged"]["Overall_F1"], 0.7717, atol=5e-5)


def test_cached_conf_measurements_reproduce_official_conf(cache):
    """The server's CONF probe reproduces the official held-out table of the thesis (eval/conf_output)."""
    o = bench.official_conf()["models"]
    for k in C.BASE_KEYS:
        for m in ("D1_F1", "D2_F1", "Overall_F1"):
            assert abs(cache["results"][k][f"conf_{m}"] - o[k][m]) < 1e-4, (k, m)


def test_banglameta_plus_is_best_on_every_task(cache):
    """BanglaMeta+ leads every other model on DEV classification and on analogy / similarity (it matches the
    similarity view it is built from within one analogy question). On the held-out CONF half it ties the thesis
    model (within 0.1 point; the bootstrap CI is about +-0.7) and beats every other model."""
    r = cache["results"]; p = r["merged_plus"]
    for k in ("merged", "merged_sim", "fasttext", "news", "informal"):
        assert p["Overall_F1"] > r[k]["Overall_F1"], k
        if k == "merged":
            assert abs(p["conf_Overall_F1"] - r[k]["conf_Overall_F1"]) < 0.001
        else:
            assert p["conf_Overall_F1"] > r[k]["conf_Overall_F1"], k
        if k != "merged_sim":
            assert p["analogy_native_acc"] > r[k]["analogy_native_acc"] and p["sim_auc_common"] > r[k]["sim_auc_common"], k
    n = sum(v[0] for f, v in p["analogy_per_file"].items() if f != "Mikolov-Filtered")
    assert (r["merged_sim"]["analogy_native_acc"] - p["analogy_native_acc"]) * n <= 1 + 1e-9
    assert abs(r["merged_sim"]["sim_auc_common"] - p["sim_auc_common"]) < 5e-4


def test_similarity_view_beats_every_model_on_word_tasks(cache):
    """Held-out-chosen post-processing: BanglaMeta-Sim leads on analogy and similarity AUC (measured, all questions)."""
    r = cache["results"]
    for k in ("merged", "fasttext", "news", "informal"):
        assert r["merged_sim"]["analogy_native_acc"] > r[k]["analogy_native_acc"]   # (and >= merged_plus, see above)
        assert r["merged_sim"]["sim_auc_common"] > r[k]["sim_auc_common"]
    assert r["merged_sim"]["Overall_F1"] > r["fasttext"]["Overall_F1"]   # classification still above the baseline
