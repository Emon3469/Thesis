"""Every answer equals an independent computation on the frozen model rows."""
import numpy as np
import pytest

import config as C
from app import service as S
from app import text as T

WORDS = ["বাংলাদেশ", "ভালো", "ঢাকা", "খেলা"]


def reference_rows(key, B):
    """Rows as the thesis files define them; for merged_sim the documented transform is applied here, independently."""
    B = np.asarray(B, np.float32); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    if key == "merged_plus":
        S = reference_rows("merged_sim", B)
        NI = B[:, :600] / np.linalg.norm(B[:, :600], axis=1, keepdims=True)
        return np.hstack([np.sqrt(C.PLUS_EPS) * NI, np.sqrt(1 - C.PLUS_EPS) * S[:, :300], np.sqrt(1 - C.PLUS_EPS) * S[:, 600:]])
    if key != "merged_sim":
        return B
    p = np.load(C.DERIVED / "merged_sim.params.npz"); out = np.zeros_like(B)
    for k in range(3):
        if C.SIM_WEIGHTS[k] == 0:
            continue
        Y = B[:, 300 * k:300 * (k + 1)] - p[f"mu{k}"]; Y -= (Y @ p[f"P{k}"].T) @ p[f"P{k}"]
        out[:, 300 * k:300 * (k + 1)] = Y / np.linalg.norm(Y, axis=1, keepdims=True) * np.sqrt(C.SIM_WEIGHTS[k] / 3)
    return out


def brute_neighbors(key, word, k):
    """Reference: rows read from the frozen float16 file, normalised (and for merged_sim transformed) here; full scan."""
    words = C.MODELS[key]["words"].read_text(encoding="utf-8").split("\n")
    U = np.load(C.MODELS[key]["source"], mmap_mode="r"); n = len(words)
    i = words.index(word)
    q = reference_rows(key, U[i:i + 1])[0]; q /= np.linalg.norm(q)
    sims = np.empty(n, np.float32)
    for s in range(0, n, 20000):
        B = reference_rows(key, U[s:s + 20000]); sims[s:s + len(B)] = B @ q
    sims[i] = -np.inf
    order = np.lexsort((np.arange(n), -sims))[:k]
    return [words[j] for j in order], sims[order]


@pytest.mark.parametrize("key", C.ORDER)
@pytest.mark.parametrize("word", WORDS)
def test_neighbors_match_brute_force(R, key, word):
    if not R.get(key).has(word):
        pytest.skip("not in this model")
    got = S.neighbors(R, key, [word], 15)[0]
    ref_w, ref_s = brute_neighbors(key, word, 15)
    assert [n["word"] for n in got["neighbors"]] == ref_w
    assert np.allclose([n["cos"] for n in got["neighbors"]], ref_s, atol=2e-5)
    assert word not in [n["word"] for n in got["neighbors"]]
    cs = [n["cos"] for n in got["neighbors"]]
    assert cs == sorted(cs, reverse=True) and all(-1 <= c <= 1.00001 for c in cs)


def test_neighbors_cosines_recomputed(R):
    m = R.merged
    r = S.neighbors(R, "merged", ["ঢাকা"], 10)[0]
    for n in r["neighbors"]:
        assert abs(float(m.vec("ঢাকা") @ m.vec(n["word"])) - n["cos"]) < 1e-5
        assert m.idx[n["word"]] == n["rank"]


def test_oov_is_reported_not_invented(R):
    r = S.neighbors(R, "informal", ["সংবিধানপ্রণেতা"], 5)[0]
    assert r["in_model"] is False and r["neighbors"] == [] and r["row"] is None


def test_word_missing_in_one_model_only(R):
    w = next(w for w in R.merged.words[:5000] if not R.get("informal").has(w) and all("ঀ" <= c <= "৿" for c in w))
    c = S.compare_word(R, w, 5, C.ORDER)
    assert c["models"]["merged"]["in_model"] and not c["models"]["informal"]["in_model"]
    assert c["models"]["informal"]["neighbors"] == []


def test_banglish_and_bangla_give_identical_answers(R):
    a = S.neighbors(R, "merged", ["bangladesh"], 10)[0]
    b = S.neighbors(R, "merged", ["বাংলাদেশ"], 10)[0]
    assert a["word"] == b["word"] == "বাংলাদেশ" and a["neighbors"] == b["neighbors"]


def test_analogy_matches_reference(R):
    m = R.merged
    res = S.analogy(R, "merged", "রাজা", "রানী", "পুরুষ", 5)
    assert res["ok"]
    t = m.vec("রানী") - m.vec("রাজা") + m.vec("পুরুষ")
    sims = np.empty(m.n_search, np.float32)
    for s in range(0, m.n_search, 32768):
        sims[s:s + 32768] = np.asarray(m.U[s:min(s + 32768, m.n_search)]) @ (t / np.linalg.norm(t))
    for w in ("রাজা", "রানী", "পুরুষ"):
        sims[m.idx[w]] = -np.inf
    ref = [m.words[j] for j in np.lexsort((np.arange(len(sims)), -sims))[:5]]
    assert [x["word"] for x in res["results"]] == ref


def test_analogy_missing_word(R):
    res = S.analogy(R, "informal", "রাজা", "রানী", "সংবিধানপ্রণেতা", 5)
    assert res["ok"] is False and res["results"] == [] and res["missing"]


def test_sentence_vector_is_mean_of_rows(R):
    m = R.merged
    out = S.sentences(R, "merged", ["আমি তোমাকে খুব ভালোবাসি", "দেশের অর্থনীতি দ্রুত বাড়ছে"], 5)
    for s, v in zip(out["sentences"], out["_vectors"]):
        words = [t["word"] for t in s["tokens"] if t["in_model"]]
        assert np.allclose(v, m.rows([m.idx[w] for w in words]).mean(0), atol=1e-7)
        assert s["used"] == len(words)
    a, b = out["_vectors"]
    assert abs(out["similarity"][0][1] - float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))) < 1e-5
    assert out["similarity"][0][1] == out["similarity"][1][0] and abs(out["similarity"][0][0] - 1) < 1e-5


def test_sentence_with_unknown_words_only(R):
    out = S.sentences(R, "informal", ["সংবিধানপ্রণেতা", "আমি ভালো আছি"], 5)
    assert out["_vectors"][0] is None and out["sentences"][0]["coverage"] == 0 and out["sentences"][0]["nearest"] == []
    assert out["similarity"][0][1] is None


def test_banglish_sentence_uses_resolved_words(R):
    out = S.sentences(R, "merged", ["ami bhalo achi"], 5)
    assert [t["word"] for t in out["sentences"][0]["tokens"]] == ["আমি", "ভাল", "আছি"]


def test_similarity_symmetric(R):
    r = S.similarity(R, [("ঢাকা", "চট্টগ্রাম"), ("চট্টগ্রাম", "ঢাকা"), ("বই", "বই")], C.ORDER)
    for k in C.ORDER:
        assert r[0]["cos"][k] == r[1]["cos"][k]
        assert abs(r[2]["cos"][k] - 1) < 1e-5


@pytest.mark.parametrize("method", ["pca", "pca-local", "tsne"])
def test_explore_shapes_and_determinism(R, method):
    args = (R, "merged", ["ঢাকা", "khela"], ["ami bhalo achi", "খেলাটা দারুণ ছিল"], 8, method, 3)
    a = S.explore(*args); b = S.explore(*args)
    assert a == b
    ids = {p["id"] for p in a["points"]}
    assert len(ids) == len(a["points"])
    assert all(e["source"] in ids and e["target"] in ids for e in a["edges"])
    xyz = np.array([p["xyz"] for p in a["points"]])
    assert np.isfinite(xyz).all() and xyz.shape[1] == 3
    if method != "pca":
        assert np.abs(xyz).max() <= 1.0 + 1e-6
    assert {p["label"] for p in a["points"] if p["kind"] == "query"} == {"ঢাকা", "খেলা"}


def test_global_pca_matches_cloud(R):
    c = S.cloud(R, "merged", 500)
    e = S.explore(R, "merged", [c["words"][10]], [], 3, "pca", 3)
    p = next(p for p in e["points"] if p["kind"] == "query")
    a, b = np.array(p["xyz"]), np.array(c["points"][10])           # same fixed axes, only the zoom differs
    assert abs(a @ b / np.linalg.norm(a) / np.linalg.norm(b) - 1) < 1e-4


def test_two_d_projection(R):
    e = S.explore(R, "merged", ["ঢাকা"], [], 5, "pca", 2)
    assert all(p["xyz"][2] == 0 for p in e["points"])


def test_input_errors(R):
    for bad in ["two words", "привет", ""]:
        with pytest.raises(T.InputError):
            S.resolve_word(R, bad)
    with pytest.raises(T.InputError):
        S.explore(R, "merged", ["ঢাকা"], [], 1, "tsne", 3)      # 2 points: too few for t-SNE


def test_pair_similarity_word_word_equals_dot(R):
    m = R.merged
    r = S.pair_similarity(R, "merged", {"kind": "word", "text": "ঢাকা"}, {"kind": "word", "text": "chottogram"})
    assert r["b"]["word"] == "চট্টগ্রাম"
    assert abs(r["cos"] - float(m.vec("ঢাকা") @ m.vec("চট্টগ্রাম"))) < 1e-5
    assert abs(np.cos(np.radians(r["angle_deg"])) - r["cos"]) < 1e-4


def test_pair_similarity_sentence_matches_sentences_endpoint(R):
    a, b = "ami tomake khub bhalobashi", "খেলাটা আজ দারুণ জমেছিল"
    r = S.pair_similarity(R, "merged", {"kind": "sentence", "text": a}, {"kind": "sentence", "text": b})
    ref = S.sentences(R, "merged", [a, b], 3)["similarity"][0][1]
    assert abs(r["cos"] - ref) < 1e-6


def test_pair_similarity_word_sentence(R):
    m = R.merged
    r = S.pair_similarity(R, "merged", {"kind": "word", "text": "ভালো"}, {"kind": "sentence", "text": "আমি ভালো আছি"})
    v = m.rows([m.idx[w] for w in ["আমি", "ভালো", "আছি"]]).mean(0)
    assert abs(r["cos"] - float(m.vec("ভালো") @ v / np.linalg.norm(v))) < 1e-5


def test_pair_similarity_missing(R):
    r = S.pair_similarity(R, "informal", {"kind": "word", "text": "সংবিধানপ্রণেতা"}, {"kind": "word", "text": "ভালো"})
    assert r["cos"] is None and r["a"]["in_model"] is False


def test_explore_only_user_input(R):
    words, sents = ["ঢাকা", "khela"], ["ami bhalo achi", "খেলাটা দারুণ ছিল"]
    only = S.explore(R, "merged", words, sents, 8, "pca", 3, similar=False)
    allowed = {"ঢাকা", "খেলা"} | {t["word"] for s in only["sentences"] for t in s["tokens"] if t["in_model"]}
    labels = {p["label"] for p in only["points"] if p["kind"] != "sentence"}
    assert labels == allowed and not any(p["kind"] == "neighbor" for p in only["points"])
    assert sum(p["kind"] == "sentence" for p in only["points"]) == 2
    full = S.explore(R, "merged", words, sents, 8, "pca", 3, similar=True)
    near = {n["word"] for s in full["sentences"] for n in s["nearest"]} | {n["word"] for w in full["words"] for n in w["neighbors"]}
    assert near <= {p["label"] for p in full["points"]}
    # the same word gets the same position in both modes (fixed global PCA basis)
    # same fixed axes in both modes: positions differ only by one zoom factor
    pos = {p["label"]: np.array(p["xyz"]) for p in full["points"]}
    pairs = [(pos[p["label"]], np.array(p["xyz"])) for p in only["points"] if p["kind"] != "sentence"]
    ratios = [np.linalg.norm(o) / np.linalg.norm(f) for f, o in pairs if np.linalg.norm(f) > 1e-3]
    assert np.allclose(ratios, ratios[0], rtol=1e-3)
    assert all(np.allclose(f / np.linalg.norm(f), o / np.linalg.norm(o), atol=1e-3) for f, o in pairs if np.linalg.norm(f) > 1e-3)
    assert max(np.abs(np.array([p["xyz"] for p in only["points"]])).max(), 0) == pytest.approx(1.0, abs=1e-4)


def test_relations_between_user_words(R):
    m = R.merged
    e = S.explore(R, "merged", ["রাজা", "পুরুষ", "রানী"], ["ami bhat khai"], 5, "pca", 3, similar=False)
    rel = {(r["source"], r["target"]): r for r in e["relations"]}
    assert len(rel) == 6                                   # 3 words + 1 sentence -> 6 pairs
    r = rel[("w:রাজা", "w:পুরুষ")]
    assert abs(r["cos"] - float(m.vec("রাজা") @ m.vec("পুরুষ"))) < 1e-5
    full = np.empty(m.n, np.float32)
    for s0 in range(0, m.n, 32768):
        full[s0:s0 + 32768] = np.asarray(m.U[s0:s0 + 32768]) @ m.vec("রাজা")
    full[m.idx["রাজা"]] = -np.inf
    assert r["rank_ab"] == int((full > full[m.idx["পুরুষ"]]).sum()) + 1
    assert rel[("w:রাজা", "w:রানী")]["rank_ab"] < 10 and rel[("w:রাজা", "w:রানী")]["strength"] == "very strong"
    for b in r["bridges"]:
        assert b["word"] not in ("রাজা", "পুরুষ", "রানী")
        assert abs(b["cos_a"] - float(m.vec("রাজা") @ m.vec(b["word"]))) < 1e-4
    mins = [min(b["cos_a"], b["cos_b"]) for b in r["bridges"]]
    assert mins == sorted(mins, reverse=True)
    sent = rel[("w:রাজা", "s:0")]
    assert sent["rank_ab"] is None and sent["rank_ba"] is not None
