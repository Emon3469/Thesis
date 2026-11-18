"""HTTP layer: valid requests succeed, invalid ones get a clear 422, concurrent use is consistent."""
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_health_and_models(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["frozen_md5_verified"]
    m = client.get("/api/models").json()
    assert m["primary"] == "merged_plus" and m["models"]["merged_plus"]["dim"] == 1200
    assert m["models"]["merged"]["rows"] == 273_531 and m["models"]["merged"]["dim"] == 900


@pytest.mark.parametrize("path,body", [
    ("/api/neighbors", {"words": ["ভালো", "kemon"], "k": 5}),
    ("/api/explore", {"words": ["ঢাকা"], "sentences": ["ami bhalo achi", "খেলা শেষ"], "k": 5, "projection": "pca", "dims": 3}),
    ("/api/explore", {"words": [], "sentences": ["ami bhalo achi", "খেলা শেষ", "দাম বাড়ছে"], "k": 5, "projection": "pca-local", "dims": 2}),
    ("/api/sentences", {"sentences": ["আমি ভালো আছি"], "k": 5}),
    ("/api/analogy", {"a": "রাজা", "b": "রানী", "c": "purush"}),
    ("/api/similarity", {"pairs": [["ঢাকা", "chittagong"]]}),
    ("/api/compare/word", {"word": "valo", "k": 5}),
    ("/api/compare/sentences", {"sentences": ["ami bhalo achi", "আমি ভালো আছি"]}),
    ("/api/pair", {"a": {"kind": "word", "text": "ভালো"}, "b": {"kind": "sentence", "text": "ami bhalo achi"}}),
    ("/api/pair", {"a": {"kind": "word", "text": "raja"}, "b": {"kind": "word", "text": "rani"}, "model": "merged_sim"}),
    ("/api/explore", {"words": ["রাজা", "rani"], "sentences": [], "k": 5, "model": "merged_sim"}),
    ("/api/analogy", {"a": "রাজা", "b": "রানী", "c": "পুরুষ", "model": "merged_sim"}),
])
def test_valid_requests(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("path,body", [
    ("/api/neighbors", {"words": ["привет"]}),
    ("/api/neighbors", {"words": ["two words"]}),
    ("/api/neighbors", {"words": []}),
    ("/api/neighbors", {"words": ["ভালো"] * 13}),
    ("/api/neighbors", {"words": ["ভালো"], "k": 0}),
    ("/api/neighbors", {"words": ["ভালো"], "k": 51}),
    ("/api/neighbors", {"words": ["ভালো"], "model": "glove"}),
    ("/api/explore", {"words": [], "sentences": []}),
    ("/api/explore", {"words": ["ভালো"], "projection": "umap"}),
    ("/api/sentences", {"sentences": ["hello 😀"]}),
    ("/api/sentences", {"sentences": ["ক" * 601]}),
    ("/api/analogy", {"a": "", "b": "রানী", "c": "পুরুষ"}),
    ("/api/compare/sentences", {"sentences": ["একটি"]}),
    ("/api/pair", {"a": {"kind": "word", "text": "привет"}, "b": {"kind": "word", "text": "ভালো"}}),
    ("/api/pair", {"a": {"kind": "image", "text": "ভালো"}, "b": {"kind": "word", "text": "ভালো"}}),
])
def test_invalid_requests_get_422(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]


def test_resolve_endpoint(client):
    r = client.get("/api/resolve", params={"text": "ami valo achi"}).json()
    assert [t["word"] for t in r["tokens"]] == ["আমি", "ভাল", "আছি"]
    assert client.get("/api/resolve", params={"text": "नमस्ते"}).status_code == 422


def test_cloud(client):
    c = client.get("/api/cloud", params={"model": "merged", "n": 1000}).json()
    assert len(c["words"]) == len(c["points"]) == 1000


def test_concurrent_requests_are_consistent(client):
    words = ["ভালো", "ঢাকা", "বই", "খেলা", "টাকা", "মা", "সরকার", "দেশ"]
    ref = {w: client.post("/api/neighbors", json={"words": [w], "k": 10}).json() for w in words}
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda w: (w, client.post("/api/neighbors", json={"words": [w], "k": 10}).json()), words * 4))
    assert all(r == ref[w] for w, r in got)


def test_benchmarks_endpoint(client):
    b = client.get("/api/benchmarks").json()
    assert b["official_conf"]["models"]["merged"]["Overall_F1"] > b["official_conf"]["models"]["fasttext"]["Overall_F1"]
    assert all(t["reject_H0"] for t in b["official_conf"]["tests"])
