"""The served model files are exactly the frozen thesis models."""
import hashlib
import struct

import numpy as np
import pytest

import config as C


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def test_frozen_merged_checksum():
    assert md5(C.MODELS["merged"]["source"]) == C.FROZEN_MD5


@pytest.mark.parametrize("key", C.BASE_KEYS)
def test_served_rows_equal_frozen_source(R, key):
    m = R.get(key)
    src = np.load(C.MODELS[key]["source"], mmap_mode="r")
    words = C.MODELS[key]["words"].read_text(encoding="utf-8").split("\n")
    assert words == m.words
    ids = np.unique(np.r_[0, 1, 2, np.arange(0, m.n, 997), m.n - 1])
    X = np.asarray(src[ids], np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert np.array_equal(X, m.rows(ids)), "served row differs from frozen source / norm"


@pytest.mark.parametrize("key", C.ORDER)
def test_rows_are_unit_length(R, key):
    m = R.get(key)
    n = np.linalg.norm(m.rows(np.arange(0, m.n, 101)), axis=1)
    assert np.allclose(n, 1, atol=1e-5)


def test_published_word2vec_bin_matches_served_merged(R):
    """models/merged_w2v_T_conc/bangla_merged_w2v_900d.bin (the released file) has the same rows."""
    m = R.merged
    with open(C.REPO / "models" / "merged_w2v_T_conc" / "bangla_merged_w2v_900d.bin", "rb") as f:
        n, d = map(int, f.readline().split())
        assert (n, d) == (m.n, m.dim)
        for i in range(3000):
            w = bytearray()
            while (ch := f.read(1)) != b" ": w += ch
            v = np.frombuffer(f.read(4 * d), dtype="<f4")
            word = w.decode("utf-8").lstrip("\n")
            assert word == m.words[i]
            assert np.allclose(v / np.linalg.norm(v), m.rows([i])[0], atol=1e-6)


def test_merged_vocab_is_union_and_news_block_is_original(R):
    m, news = R.merged, R.get("news")
    sample = [w for w in m.words[:200_000:503] if m.provenance_of(w)["news"]]
    assert sample
    for w in sample:
        assert news.has(w)
        blk = m.vec(w)[:300]
        cos = blk @ news.vec(w) / np.linalg.norm(blk)
        assert cos > 0.9999
    # provenance says "not in News" -> News-W2V really lacks it
    missing = [w for w in m.words[:50_000] if not m.provenance_of(w)["news"]][:200]
    assert missing and not any(news.has(w) for w in missing)
    inf = R.get("informal")
    assert all(inf.has(w) == m.provenance_of(w)["informal"] for w in m.words[:20_000])


def test_manifest_counts_match_thesis(R):
    rows = {k: R.get(k).n for k in C.ORDER}
    assert rows == {"merged_plus": 273_531, "merged": 273_531, "merged_sim": 273_531, "fasttext": 234_474, "news": 187_686, "informal": 32_910}
    assert R.manifest["checks"]["union_vocab"] is True


def test_similarity_view_is_the_documented_transform_of_the_frozen_rows(R):
    """merged_sim rows = per-block ABTT (D=4) + weights, recomputed here from the frozen float16 file."""
    src = np.load(C.MODELS["merged"]["source"], mmap_mode="r")
    p = np.load(C.DERIVED / "merged_sim.params.npz")
    assert tuple(p["weights"]) == pytest.approx(C.SIM_WEIGHTS) and p["P0"].shape == (C.SIM_ABTT_D, 300)
    # the ABTT directions are the top principal directions of the first 50k frozen rows of each block
    X = np.asarray(src[:C.SIM_FIT_ROWS], np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    for k in range(3):
        Xb = X[:, 300 * k:300 * (k + 1)].astype(np.float64)
        assert np.allclose(Xb.mean(0), p[f"mu{k}"], atol=1e-5)
        _, _, Vt = np.linalg.svd(Xb - Xb.mean(0), full_matrices=False)
        assert np.allclose(np.abs(Vt[:C.SIM_ABTT_D] @ p[f"P{k}"].T), np.eye(C.SIM_ABTT_D), atol=1e-3)
    ids = np.arange(0, R.merged.n, 1013)
    Bm = np.asarray(src[ids], np.float32); Bm /= np.linalg.norm(Bm, axis=1, keepdims=True)
    exp = np.zeros_like(Bm)
    for k in range(3):
        if C.SIM_WEIGHTS[k] == 0: continue
        Y = Bm[:, 300 * k:300 * (k + 1)] - p[f"mu{k}"]; Y -= (Y @ p[f"P{k}"].T) @ p[f"P{k}"]
        exp[:, 300 * k:300 * (k + 1)] = Y / np.linalg.norm(Y, axis=1, keepdims=True) * np.sqrt(C.SIM_WEIGHTS[k] / 3)
    assert np.allclose(exp, R.get("merged_sim").rows(ids), atol=1e-6)
    assert np.allclose(np.linalg.norm(exp, axis=1), 1, atol=1e-5)


def test_banglameta_plus_is_the_documented_combination(R):
    m, s, p = R.get("merged"), R.get("merged_sim"), R.get("merged_plus")
    ids = np.arange(0, m.n, 997)
    mni = m.rows(ids)[:, :600]; mni = mni / np.linalg.norm(mni, axis=1, keepdims=True)
    exp = np.hstack([np.sqrt(C.PLUS_EPS) * mni, np.sqrt(1 - C.PLUS_EPS) * s.rows(ids)[:, :300], np.sqrt(1 - C.PLUS_EPS) * s.rows(ids)[:, 600:]])
    assert np.allclose(exp, p.rows(ids), atol=1e-7)
    assert np.allclose(np.linalg.norm(p.rows(ids), axis=1), 1, atol=1e-5)
    assert np.allclose(s.rows(ids)[:, 300:600], 0)                 # the dropped block really is empty
    # cosine of the combination = eps * thesis cosine + (1 - eps) * similarity-view cosine
    ni = lambda w: m.vec(w)[:600] / np.linalg.norm(m.vec(w)[:600])
    a, b = p.vec("রাজা") @ p.vec("রানী"), C.PLUS_EPS * (ni("রাজা") @ ni("রানী")) + (1 - C.PLUS_EPS) * (s.vec("রাজা") @ s.vec("রানী"))
    assert abs(a - b) < 1e-5
