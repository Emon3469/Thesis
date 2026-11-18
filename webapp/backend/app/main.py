"""BanglaMeta Explorer API. Run:  uvicorn app.main:app --port 8000   (from webapp/backend)"""
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config as C
from . import bench, service as S
from . import text as T
from .store import Registry

ModelKey = Literal["merged_plus", "merged", "merged_sim", "fasttext", "news", "informal"]
Projection = Literal["pca", "pca-local", "tsne"]
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    R = Registry(verify=os.environ.get("BM_SKIP_VERIFY") != "1")
    STATE.update(R=R, started=time.strftime("%Y-%m-%d %H:%M:%S"), load_seconds=round(time.perf_counter() - t, 1),
                 bench=bench.load_cache(R), job={"running": False, "message": "", "progress": 0.0, "error": None},
                 lock=threading.Lock(), clouds={})
    yield
    STATE.clear()


app = FastAPI(title="BanglaMeta Explorer API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(T.InputError)
async def input_error(_: Request, e: T.InputError):
    return JSONResponse(status_code=422, content={"detail": str(e)})


def R() -> Registry:
    return STATE["R"]


# ------------------------------------------------------------------------------------------------ request bodies
class WordsReq(BaseModel):
    words: list[str] = Field(min_length=1, max_length=T.MAX_WORDS)
    model: ModelKey = "merged"
    k: int = Field(10, ge=1, le=50)


class ExploreReq(BaseModel):
    words: list[str] = Field(default_factory=list, max_length=T.MAX_WORDS)
    sentences: list[str] = Field(default_factory=list, max_length=T.MAX_SENTENCES)
    similar: bool = True          # False = map only the user's own words and sentences
    model: ModelKey = "merged"
    k: int = Field(10, ge=1, le=30)
    projection: Projection = "pca"
    dims: Literal[2, 3] = 3


class SentencesReq(BaseModel):
    sentences: list[str] = Field(min_length=1, max_length=T.MAX_SENTENCES)
    model: ModelKey = "merged"
    k: int = Field(10, ge=1, le=50)


class AnalogyReq(BaseModel):
    a: str; b: str; c: str
    model: ModelKey = "merged"
    k: int = Field(10, ge=1, le=50)


class PairsReq(BaseModel):
    pairs: list[tuple[str, str]] = Field(min_length=1, max_length=20)
    models: list[ModelKey] = Field(default_factory=lambda: list(C.ORDER))


class Item(BaseModel):
    kind: Literal["word", "sentence"]
    text: str = Field(max_length=T.MAX_SENTENCE_CHARS)


class PairReq(BaseModel):
    a: Item
    b: Item
    model: ModelKey = "merged"


class CompareWordReq(BaseModel):
    word: str
    k: int = Field(10, ge=1, le=50)
    models: list[ModelKey] = Field(default_factory=lambda: list(C.ORDER))


class CompareSentReq(BaseModel):
    sentences: list[str] = Field(min_length=2, max_length=10)
    models: list[ModelKey] = Field(default_factory=lambda: list(C.ORDER))


# ------------------------------------------------------------------------------------------------ routes
@app.get("/api/health")
def health():
    r = R()
    return {"ok": True, "started": STATE["started"], "load_seconds": STATE["load_seconds"],
            "models": {k: {"rows": m.n, "dim": m.dim, "served_md5": m.served_md5} for k, m in r.models.items()},
            "frozen_md5_verified": r.merged.source_md5 == C.FROZEN_MD5}


@app.get("/api/models")
def models():
    r = R()
    return {"order": C.ORDER, "primary": "merged_plus", "frozen_md5": C.FROZEN_MD5, "checks": r.manifest.get("checks"),
            "models": {k: {"key": k, "name": m.name, "label": m.label, "color": m.color, "about": m.about, "rows": m.n, "dim": m.dim,
                           "source_md5": m.source_md5, "served_md5": m.served_md5, "analogy_search_rows": m.n_search,
                           "pca_explained": [float(x) for x in m.pca_evr]} for k, m in r.models.items()}}


@app.get("/api/resolve")
def resolve(text: str):
    t = T.check_text(text, T.MAX_SENTENCE_CHARS, "text")
    return {"text": t, "tokens": [S.resolve_token(R(), w) for w in T.tokenize(t)]}


@app.post("/api/neighbors")
def neighbors(q: WordsReq):
    return {"model": q.model, "k": q.k, "results": S.neighbors(R(), q.model, q.words, q.k)}


@app.post("/api/explore")
def explore(q: ExploreReq):
    if not q.words and not q.sentences:
        raise T.InputError("Add at least one word or one sentence.")
    return S.explore(R(), q.model, q.words, q.sentences, q.k, q.projection, q.dims, q.similar)


@app.post("/api/sentences")
def sentences(q: SentencesReq):
    out = S.sentences(R(), q.model, q.sentences, q.k)
    out.pop("_vectors", None)
    return out


@app.post("/api/analogy")
def analogy(q: AnalogyReq):
    return S.analogy(R(), q.model, q.a, q.b, q.c, q.k)


@app.post("/api/similarity")
def similarity(q: PairsReq):
    return {"results": S.similarity(R(), q.pairs, q.models)}


@app.post("/api/pair")
def pair(q: PairReq):
    return S.pair_similarity(R(), q.model, q.a.model_dump(), q.b.model_dump())


@app.get("/api/cloud")
def cloud(model: ModelKey = "merged", n: int = 2500):
    key = (model, n)
    if key not in STATE["clouds"]:
        STATE["clouds"][key] = S.cloud(R(), model, n)
    return STATE["clouds"][key]


@app.post("/api/compare/word")
def compare_word(q: CompareWordReq):
    return S.compare_word(R(), q.word, q.k, q.models)


@app.post("/api/compare/sentences")
def compare_sentences(q: CompareSentReq):
    return S.compare_sentences(R(), q.sentences, q.models)


@app.get("/api/benchmarks")
def benchmarks():
    return {"measured": STATE["bench"], "job": STATE["job"], "official_conf": bench.official_conf(), "thesis_dev": bench.thesis_values()}


@app.post("/api/benchmarks/run")
def run_benchmarks():
    with STATE["lock"]:
        if STATE["job"]["running"]:
            return {"started": False, "job": STATE["job"]}
        STATE["job"] = {"running": True, "message": "starting", "progress": 0.0, "error": None}

    def work():
        try:
            def prog(msg, f): STATE["job"].update(message=msg, progress=round(f, 3))
            b = bench.run(R(), prog)
            bench.save_cache(b)
            STATE["bench"] = b
            STATE["job"].update(running=False, message="done", progress=1.0)
        except Exception as e:                                   # reported to the page, never swallowed
            STATE["job"].update(running=False, error=f"{type(e).__name__}: {e}")
    threading.Thread(target=work, daemon=True).start()
    return {"started": True, "job": STATE["job"]}


# ------------------------------------------------------------------------------------------------ built frontend (production)
DIST = C.WEBAPP / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
