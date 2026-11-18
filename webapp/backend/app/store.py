"""Model store. Every number the API returns is computed here from the served model rows.

Each model is a float32 matrix of unit-length rows, memory-mapped from data/models/<key>.f32.npy. The copies are
made by prepare.py from the frozen float16 files; their checksums are verified when the server starts, and the
tests check them row by row against the frozen sources.
"""
import hashlib
import json
import threading

import numpy as np

import config as C
from .translit import Transliterator

QUERY_BLOCK = 64                 # queries per matrix product
SEARCH_SLOTS = threading.BoundedSemaphore(2)   # at most two full-vocabulary scans at once (memory)


def md5(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Embedding:
    def __init__(self, key: str, entry: dict, verify: bool = True):
        spec = C.MODELS[key]
        self.key, self.name, self.label, self.color, self.about = key, spec["name"], spec["label"], spec["color"], spec["about"]
        path = C.served_path(key)
        if verify:
            got = md5(path)
            if got != entry["served_md5"]:
                raise RuntimeError(f"{key}: served file checksum {got} != manifest {entry['served_md5']}; run prepare.py again")
            if key in C.DERIVED_KEYS and entry["source_md5"] != C.FROZEN_MD5:
                raise RuntimeError("merged: manifest source is not the frozen Step 4 model")
        self.U = np.load(path, mmap_mode="r")
        self.words = C.served_words(key).read_text(encoding="utf-8").split("\n")
        if self.U.shape != (len(self.words), spec["dim"]) or len(self.words) != entry["rows"]:
            raise RuntimeError(f"{key}: shape {self.U.shape} does not match {len(self.words)} words / manifest")
        self.idx = {w: i for i, w in enumerate(self.words)}
        self.n, self.dim = len(self.words), spec["dim"]
        self.n_search = min(C.TOPK_SEARCH, self.n)
        self.source_md5, self.served_md5 = entry["source_md5"], entry["served_md5"]
        p = np.load(C.DERIVED / f"{key}.pca.npz")
        self.pca_mean, self.pca_comps, self.pca_evr = p["mean"], p["comps"], p["evr"]
        self.provenance = np.load(C.DERIVED / "merged.provenance.npy") if key in C.DERIVED_KEYS else None
        cloud = self.pca(self.rows(np.arange(min(5000, self.n))))
        self.pca_scale = float(np.abs(cloud).max())

    # ---------------------------------------------------------------- rows
    def has(self, w: str) -> bool:
        return w in self.idx

    def rows(self, ids) -> np.ndarray:
        return np.asarray(self.U[np.asarray(ids, dtype=np.int64)], dtype=np.float32)

    def vec(self, w: str) -> np.ndarray:
        return np.asarray(self.U[self.idx[w]], dtype=np.float32)

    def pca(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, np.float32) - self.pca_mean) @ self.pca_comps.T

    def provenance_of(self, w: str):
        if self.provenance is None or w not in self.idx:
            return None
        b = int(self.provenance[self.idx[w]])
        return {"news": bool(b & 1), "informal": bool(b & 2), "subs": bool(b & 4)}

    # ---------------------------------------------------------------- exact cosine search
    def search(self, Q, k: int, exclude=None, limit: int | None = None):
        """Exact top-k cosine search over the first `limit` rows (all rows by default).

        Q: (m, dim) query vectors (any length; normalised here). exclude: per query, a set of row ids to skip.
        Returns per query a list of (row id, cosine), cosine descending, ties broken by row id (frequency order).
        """
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float32))
        norms = np.linalg.norm(Q, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("zero query vector")
        Q = Q / norms
        limit = self.n if limit is None else min(limit, self.n)
        exclude = exclude or [set()] * len(Q)
        out = []
        with SEARCH_SLOTS:
            for s in range(0, len(Q), QUERY_BLOCK):
                Qb = Q[s:s + QUERY_BLOCK]
                S = np.empty((len(Qb), limit), dtype=np.float32)
                for r in range(0, limit, 32768):
                    e = min(r + 32768, limit)
                    S[:, r:e] = (self.U[r:e] @ Qb.T).T
                for j in range(len(Qb)):
                    ex = [i for i in exclude[s + j] if i < limit]
                    if ex:
                        S[j, ex] = -np.inf
                    kk = min(k, limit - len(ex))
                    top = np.argpartition(-S[j], kk - 1)[:kk] if kk < limit else np.arange(limit)
                    order = np.lexsort((top, -S[j, top]))
                    out.append([(int(top[o]), float(S[j, top[o]])) for o in order])
        return out


    def all_sims(self, Q) -> np.ndarray:
        """Cosine of each query vector with every row: (m, n) float32. Used for ranks and bridge words."""
        Q = np.atleast_2d(np.asarray(Q, dtype=np.float32))
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        out = np.empty((len(Q), self.n), dtype=np.float32)
        with SEARCH_SLOTS:
            for r in range(0, self.n, 32768):
                e = min(r + 32768, self.n)
                out[:, r:e] = (self.U[r:e] @ Q.T).T
        return out


class Registry:
    def __init__(self, verify: bool = True):
        if not C.MANIFEST.exists():
            raise RuntimeError("data/manifest.json not found: run `python prepare.py` first")
        self.manifest = json.loads(C.MANIFEST.read_text(encoding="utf-8"))
        self.models = {k: Embedding(k, self.manifest["models"][k], verify) for k in C.ORDER}
        self.merged = self.models["merged"]          # resolver vocabulary (identical for every merged-derived model)
        self.translit = Transliterator(self.merged.words)

    def get(self, key: str) -> Embedding:
        if key not in self.models:
            raise KeyError(key)
        return self.models[key]
