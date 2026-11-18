"""Query logic behind the API. Pure functions of (Registry, input) so the tests can call them directly."""
import numpy as np

from . import text as T
from .store import Embedding, Registry

ANALOGY_NOTE = "3CosAdd (b - a + c) over the first 200,000 rows, inputs excluded: the rule used for the thesis analogy scores."


# ---------------------------------------------------------------------------------------------- input resolution
def resolve_token(R: Registry, tok: str) -> dict:
    """Map one token to a row of the merged model. Bangla is used as typed; Banglish goes through the transliterator,
    whose candidates are all merged-model words. Nothing is invented: unknown words stay unknown."""
    kind = T.token_kind(tok)
    options: list[str] = []
    if kind == "banglish":
        options = [w for w, _, _ in R.translit.candidates(tok)]
        if R.merged.has(tok) and tok not in options:
            options.append(tok)                       # a Latin word the model itself contains
    elif R.merged.has(tok):
        options = [tok]
    word = options[0] if options else None
    return {"input": tok, "kind": kind, "word": word, "options": options, "found": word is not None}


def resolve_word(R: Registry, raw: str) -> dict:
    t = T.check_text(raw, T.MAX_WORD_CHARS, "word")
    toks = T.tokenize(t)
    if len(toks) != 1:
        raise T.InputError(f"Enter one word at a time (got {len(toks)} tokens from '{t}').")
    return resolve_token(R, toks[0])


def resolve_sentence(R: Registry, raw: str) -> dict:
    t = T.check_text(raw, T.MAX_SENTENCE_CHARS, "sentence")
    toks = [resolve_token(R, w) for w in T.tokenize(t)]
    if not toks:
        raise T.InputError(f"The sentence '{t}' has no words.")
    return {"text": t, "tokens": toks}


# ---------------------------------------------------------------------------------------------- word queries
def neighbor_list(m: Embedding, hits) -> list[dict]:
    return [{"word": m.words[i], "cos": round(c, 6), "rank": i} for i, c in hits]


def neighbors(R: Registry, model: str, words: list[str], k: int) -> list[dict]:
    m = R.get(model)
    res = [resolve_word(R, w) for w in words]
    ok = [r for r in res if r["found"] and m.has(r["word"])]
    hits = m.search(np.stack([m.vec(r["word"]) for r in ok]), k, [{m.idx[r["word"]]} for r in ok]) if ok else []
    it = iter(hits)
    out = []
    for r in res:
        inm = r["found"] and m.has(r["word"])
        out.append({**r, "in_model": inm, "row": m.idx[r["word"]] if inm else None,
                    "provenance": m.provenance_of(r["word"]) if inm else None,
                    "neighbors": neighbor_list(m, next(it)) if inm else []})
    return out


def cosine(m: Embedding, a: str, b: str):
    if not (m.has(a) and m.has(b)):
        return None
    return round(float(m.vec(a) @ m.vec(b)), 6)


def similarity(R: Registry, pairs: list[tuple[str, str]], models: list[str]) -> list[dict]:
    out = []
    for a, b in pairs:
        ra, rb = resolve_word(R, a), resolve_word(R, b)
        row = {"a": ra, "b": rb, "cos": {}}
        for k in models:
            m = R.get(k)
            row["cos"][k] = cosine(m, ra["word"], rb["word"]) if ra["found"] and rb["found"] else None
        out.append(row)
    return out


def analogy(R: Registry, model: str, a: str, b: str, c: str, k: int) -> dict:
    """a : b :: c : ?   ->  nearest rows to (b - a + c)."""
    m = R.get(model)
    rs = [resolve_word(R, x) for x in (a, b, c)]
    missing = [r["input"] for r in rs if not (r["found"] and m.has(r["word"]))]
    base = {"model": model, "a": rs[0], "b": rs[1], "c": rs[2], "rule": ANALOGY_NOTE}
    if missing:
        return {**base, "ok": False, "missing": missing, "results": []}
    va, vb, vc = (m.vec(r["word"]) for r in rs)
    target = vb - va + vc
    ids = {m.idx[r["word"]] for r in rs}
    hits = m.search(target[None], k, [ids], limit=m.n_search)[0]
    pts = project_vectors(m, np.stack([va, vb, vc, target / np.linalg.norm(target)] + [m.vec(m.words[i]) for i, _ in hits[:1]]), "pca", 3)
    return {**base, "ok": True, "missing": [], "results": neighbor_list(m, hits), "points": pts.tolist()}


# ---------------------------------------------------------------------------------------------- sentences
def sentence_vector(m: Embedding, words: list[str]):
    """Mean of the unit rows of the words the model has (the thesis sentence vector). None if no word is known."""
    ids = [m.idx[w] for w in words if w is not None and w in m.idx]
    if not ids:
        return None, 0
    return m.rows(ids).mean(0), len(ids)


def sentences(R: Registry, model: str, texts: list[str], k: int) -> dict:
    m = R.get(model)
    sents = [resolve_sentence(R, t) for t in texts]
    vecs, out = [], []
    for s in sents:
        words = [t["word"] for t in s["tokens"]]
        v, used = sentence_vector(m, words)
        for t in s["tokens"]:
            t["in_model"] = t["found"] and m.has(t["word"])
        vecs.append(v)
        out.append({**s, "used": used, "coverage": round(used / len(words), 4)})
    ok = [i for i, v in enumerate(vecs) if v is not None]
    if ok:
        hits = m.search(np.stack([vecs[i] for i in ok]), k)
        for i, h in zip(ok, hits):
            out[i]["nearest"] = neighbor_list(m, h)
    for s in out:
        s.setdefault("nearest", [])
    n = len(vecs)
    sim = [[None] * n for _ in range(n)]
    for i in ok:
        for j in ok:
            sim[i][j] = round(float(vecs[i] @ vecs[j] / np.linalg.norm(vecs[i]) / np.linalg.norm(vecs[j])), 6)
    return {"model": model, "sentences": out, "similarity": sim, "_vectors": vecs}


# ---------------------------------------------------------------------------------------------- projection
def project_vectors(m: Embedding, X: np.ndarray, method: str, dims: int, fit: bool = True) -> np.ndarray:
    """Coordinates for the 3-D view. 'pca' = the model's fixed PCA basis (same space as the background cloud),
    'pca-local' = PCA of these points only, 'tsne' = t-SNE of these points (seed 42)."""
    X = np.asarray(X, np.float32)
    if method == "pca":
        # fixed global axes (the model's own PCA basis); zoom = fit the points being shown
        P = m.pca(X)[:, :dims]
        scale = float(np.abs(P).max()) if fit else m.pca_scale
        return np.round(P / (scale if scale > 1e-6 else m.pca_scale), 5)
    if method == "pca-local":
        if len(X) < 3:
            raise T.InputError("Local PCA needs at least 3 points.")
        Y = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Y, full_matrices=False)
        for j in range(min(dims, len(Vt))):
            if Vt[j, np.abs(Vt[j]).argmax()] < 0: Vt[j] *= -1
        P = Y @ Vt[:dims].T
    elif method == "tsne":
        if len(X) < 5:
            raise T.InputError("t-SNE needs at least 5 points.")
        from sklearn.manifold import TSNE
        P = TSNE(n_components=dims, perplexity=float(max(2, min(30, (len(X) - 1) / 3))), init="pca", metric="cosine",
                 random_state=42, max_iter=1000).fit_transform(X)
    else:
        raise T.InputError(f"Unknown projection '{method}'.")
    if P.shape[1] < dims:
        P = np.hstack([P, np.zeros((len(P), dims - P.shape[1]), np.float32)])
    scale = float(np.abs(P).max()) or 1.0
    return np.round(P / scale, 5)


def cloud(R: Registry, model: str, n: int) -> dict:
    m = R.get(model)
    n = int(min(max(n, 100), 5000, m.n))
    P = project_vectors(m, m.rows(np.arange(n)), "pca", 3, fit=False)
    return {"model": model, "words": m.words[:n], "points": P.tolist(), "explained": [float(x) for x in m.pca_evr]}


def explore(R: Registry, model: str, words: list[str], texts: list[str], k: int, method: str, dims: int, similar: bool = True) -> dict:
    """Everything the explorer page draws, with 3-D positions.

    similar=False: only the user's input (query words, sentences and the words inside the sentences).
    similar=True : also the nearest words of every query word and of every sentence vector."""
    m = R.get(model)
    W = neighbors(R, model, words, k) if words else []
    S = sentences(R, model, texts, k) if texts else {"sentences": [], "similarity": [], "_vectors": []}
    points, vecs, edges, seen = [], [], [], {}

    def add(word, kind, group):
        if word in seen:
            p = points[seen[word]]
            if kind == "query": p["kind"] = "query"
            p["groups"] = sorted(set(p["groups"]) | {group})
            return seen[word]
        seen[word] = len(points)
        points.append({"id": f"w:{word}", "label": word, "kind": kind, "groups": [group], "row": m.idx[word]})
        vecs.append(m.vec(word))
        return seen[word]

    for gi, w in enumerate(W):
        if not w["in_model"]:
            continue
        q = add(w["word"], "query", f"w{gi}")
        for nb in (w["neighbors"] if similar else []):
            j = add(nb["word"], "neighbor", f"w{gi}")
            edges.append({"source": points[q]["id"], "target": points[j]["id"], "cos": nb["cos"]})
    for si, (s, v) in enumerate(zip(S["sentences"], S["_vectors"])):
        if v is None:
            continue
        pid = f"s:{si}"
        points.append({"id": pid, "label": s["text"][:40], "kind": "sentence", "groups": [f"s{si}"], "sentence": si})
        vecs.append(v)
        for t in s["tokens"]:
            if t["in_model"]:
                j = add(t["word"], "token", f"s{si}")
                edges.append({"source": pid, "target": points[j]["id"], "cos": round(float(m.vec(t["word"]) @ v / np.linalg.norm(v)), 6)})
        for nb in (s["nearest"] if similar else []):
            j = add(nb["word"], "neighbor", f"s{si}")
            edges.append({"source": pid, "target": points[j]["id"], "cos": nb["cos"]})
    rel_items = [(points[seen[w["word"]]]["id"], m.vec(w["word"]), {m.idx[w["word"]]}) for w in W if w["in_model"]]
    rel_items += [(f"s:{si}", v, set()) for si, v in enumerate(S["_vectors"]) if v is not None]
    relations = relations_between(m, rel_items[:MAX_REL_ITEMS])
    coords = project_vectors(m, np.stack(vecs), method, dims).tolist() if vecs else []
    for p, c in zip(points, coords):
        p["xyz"] = c + [0.0] * (3 - len(c))
    S.pop("_vectors", None)
    return {"model": model, "similar": similar, "projection": {"method": method, "dims": dims,
                                           "explained": [float(x) for x in m.pca_evr] if method == "pca" else None},
            "words": W, "sentences": S["sentences"], "sentence_similarity": S["similarity"], "points": points, "edges": edges, "relations": relations}


# ---------------------------------------------------------------------------------------------- comparison
def compare_word(R: Registry, word: str, k: int, models: list[str]) -> dict:
    r = resolve_word(R, word)
    cols = {}
    for key in models:
        m = R.get(key)
        inm = r["found"] and m.has(r["word"])
        hits = m.search(m.vec(r["word"])[None], k, [{m.idx[r["word"]]}])[0] if inm else []
        cols[key] = {"in_model": inm, "row": m.idx[r["word"]] if inm else None, "neighbors": neighbor_list(m, hits),
                     "provenance": m.provenance_of(r["word"]) if inm else None}
    overlap = {}
    for a in models:
        for b in models:
            A = {x["word"] for x in cols[a]["neighbors"]}; B = {x["word"] for x in cols[b]["neighbors"]}
            overlap[f"{a}|{b}"] = round(len(A & B) / len(A | B), 4) if A and B else None
    return {"query": r, "k": k, "models": cols, "overlap": overlap}


def compare_sentences(R: Registry, texts: list[str], models: list[str]) -> dict:
    out = {}
    for key in models:
        s = sentences(R, key, texts, 5)
        out[key] = {"similarity": s["similarity"], "coverage": [x["coverage"] for x in s["sentences"]],
                    "nearest": [x["nearest"][:5] for x in s["sentences"]]}
    tokens = [resolve_sentence(R, t) for t in texts]
    return {"sentences": tokens, "models": out}


# ---------------------------------------------------------------------------------------------- similarity of any two items
def item_vector(R: Registry, m: Embedding, item: dict):
    """item = {"kind": "word" | "sentence", "text": ...}. Returns (description, vector or None)."""
    if item.get("kind") == "word":
        r = resolve_word(R, item.get("text", ""))
        inm = r["found"] and m.has(r["word"])
        return {**r, "kind_of_item": "word", "in_model": inm}, (m.vec(r["word"]) if inm else None)
    if item.get("kind") == "sentence":
        s = resolve_sentence(R, item.get("text", ""))
        for t in s["tokens"]:
            t["in_model"] = t["found"] and m.has(t["word"])
        v, used = sentence_vector(m, [t["word"] for t in s["tokens"]])
        return {**s, "kind_of_item": "sentence", "used": used, "in_model": v is not None}, v
    raise T.InputError("Each item must be a word or a sentence.")


def pair_similarity(R: Registry, model: str, a: dict, b: dict) -> dict:
    """Cosine similarity of two words, two sentences, or a word and a sentence, in the full model space."""
    m = R.get(model)
    da, va = item_vector(R, m, a)
    db, vb = item_vector(R, m, b)
    if va is None or vb is None:
        return {"model": model, "a": da, "b": db, "cos": None, "angle_deg": None}
    cos = float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb))
    cos = max(-1.0, min(1.0, cos))
    return {"model": model, "a": da, "b": db, "cos": round(cos, 6), "angle_deg": round(float(np.degrees(np.arccos(cos))), 3)}



# ---------------------------------------------------------------------------------------------- relationships between the user's items
MAX_REL_ITEMS = 12


def strength(rank: int) -> str:
    """Plain-language reading of a neighbour rank (rank 1 = the single nearest row)."""
    return "very strong" if rank <= 10 else "strong" if rank <= 100 else "moderate" if rank <= 1000 else "weak" if rank <= 10000 else "not close"


def relations_between(m: Embedding, items: list) -> list[dict]:
    """For every pair of the user's items (id, vector, own row ids): cosine in the full space, how high each one ranks
    among the other's neighbours, and bridge words (rows near both: highest min(cos to A, cos to B))."""
    if len(items) < 2:
        return []
    sims = m.all_sims(np.stack([v for _, v, _ in items]))
    own = set().union(*[o for _, _, o in items])
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ia, va, oa), (ib, vb, ob) = items[i], items[j]
            cos = float(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb))
            # rank of B among A's neighbours (A's own row not counted); only defined when B is a word row
            # rank of B among A's neighbours = rows (other than A itself) scoring strictly above B's own row, + 1;
            # only defined when B is a word row
            def rank(row_sims, own_rows, other_rows):
                thr = row_sims[next(iter(other_rows))]
                return int((row_sims > thr).sum()) - sum(1 for o in own_rows if row_sims[o] > thr) + 1
            rank_ab = rank(sims[i], oa, ob) if ob else None
            rank_ba = rank(sims[j], ob, oa) if oa else None
            both = np.minimum(sims[i], sims[j])
            both[list(own)] = -np.inf
            top = np.argpartition(-both, 5)[:5]
            top = top[np.lexsort((top, -both[top]))]
            best = min(r for r in (rank_ab, rank_ba) if r is not None) if (rank_ab or rank_ba) else None
            out.append({"source": ia, "target": ib, "cos": round(cos, 6), "angle_deg": round(float(np.degrees(np.arccos(max(-1, min(1, cos))))), 2),
                        "rank_ab": rank_ab, "rank_ba": rank_ba, "strength": strength(best) if best else None,
                        "bridges": [{"word": m.words[t], "cos_a": round(float(sims[i, t]), 4), "cos_b": round(float(sims[j, t]), 4)} for t in top]})
    return out
