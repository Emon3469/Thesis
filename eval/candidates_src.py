# %% [markdown]
# # Candidate screening — encode the 2,100-text eval set with 6 more Bangla encoders
# Same protocol as `bangla-2domain-embedding-eval` (news 256 tokens, informal 128, frozen encoders,
# E5-family `"query: "` prefix, BanglaBERT normalizer). Output: `cand_results/emb/<model>.npy` + diagnostics.
# The fusion pre-evaluation is run offline from these files.

# %%
import os, sys, time, json, gc, glob, subprocess, unicodedata, zipfile, warnings
from pathlib import Path
T0 = time.perf_counter()
ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
SMOKE = os.environ.get("EVAL_SMOKE") == "1"
if ON_KAGGLE: os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false"); warnings.filterwarnings("ignore")
def pip(*p):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *p], capture_output=True, text=True)
    if r.returncode: print("pip failed", p, r.stderr[-300:])
if not SMOKE:
    pip("git+https://github.com/csebuetnlp/normalizer")
    pip("-U", "sentence-transformers>=5.1")         # kazalbrur models use the new ST module layout
    try:
        import gensim  # noqa
    except ImportError:
        pip("gensim")
import numpy as np, pandas as pd, torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import sentence_transformers
print("device", DEVICE, "| torch", torch.__version__, "| sentence-transformers", sentence_transformers.__version__)

# %%
MAX_LEN = {"D1_formal_news": 256, "D2_informal_ugc": 128}; BATCH = 64
CANDS = {
    "BanglaBERT-large":     dict(kind="hf_mean", repo="csebuetnlp/banglabert_large", bn_norm=True, role="D1 contextual (formal)"),
    "mE5-large":            dict(kind="st", repo="intfloat/multilingual-e5-large", prefix="query: ", role="general multilingual"),
    "BnEmbed-e5s-banglish": dict(kind="st", repo="kazalbrur/bangla-embed-e5-small-banglish", prefix="query: ", role="D2 / Banglish"),
    "BnEmbed-e5s":          dict(kind="st", repo="kazalbrur/bangla-embed-e5-small", prefix="query: ", role="Bangla native"),
    "L3Cube-BnSBERT":       dict(kind="st", repo="l3cube-pune/bengali-sentence-similarity-sbert", role="Bangla STS"),
    "bnlp-word2vec":        dict(kind="w2v", repo="sagorsarker/bangla_word2vec", file="bangla_word2vec_gen4.zip", role="static (wiki)"),
}
OUT = Path("/kaggle/working/cand_results" if ON_KAGGLE else "cand_results_smoke"); (OUT / "emb").mkdir(parents=True, exist_ok=True)
paths = sorted(glob.glob("/kaggle/input/**/eval_set.parquet", recursive=True)) or sorted(glob.glob("**/eval_set.parquet", recursive=True))
df = pd.read_parquet(paths[0]).reset_index(drop=True); assert len(df) == 2100
texts = df.text.tolist(); DOM_IDX = {d: np.where(df.domain.values == d)[0] for d in MAX_LEN}
if SMOKE:
    CANDS = {"FAKE": dict(kind="fake", role="smoke")}

# %%
_PSPLIT = lambda t: "".join(f" {c} " if unicodedata.category(c)[0] == "P" else c for c in t)
def tok_diag(tok, tx, L):
    enc = tok(tx, add_special_tokens=False, truncation=False)["input_ids"]
    n = np.array([len(e) for e in enc]); w = np.array([max(1, len(t.split())) for t in tx])
    unk = sum(int(np.sum(np.array(e) == tok.unk_token_id)) for e in enc) if tok.unk_token_id is not None else 0
    return dict(fertility=float(n.sum() / w.sum()), unk_rate=float(unk / max(1, n.sum())), trunc_rate=float(np.mean(n + 2 > L)))

def enc_hf_mean(cfg):
    from transformers import AutoTokenizer, AutoModel
    tx_all = texts
    if cfg.get("bn_norm"):
        from normalizer import normalize; tx_all = [normalize(t) for t in texts]
    tok = AutoTokenizer.from_pretrained(cfg["repo"]); m = AutoModel.from_pretrained(cfg["repo"]).to(DEVICE).eval()
    E = np.zeros((len(texts), m.config.hidden_size), np.float32); diag = {}
    for d, idx in DOM_IDX.items():
        tx = [tx_all[i] for i in idx]; L = MAX_LEN[d]; diag[d] = tok_diag(tok, tx, L); order = np.argsort([len(t) for t in tx])
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            e = tok([tx[i] for i in b], truncation=True, max_length=L, padding=True, return_tensors="pt").to(DEVICE)
            with torch.inference_mode(): h = m(**e).last_hidden_state
            mk = e["attention_mask"].unsqueeze(-1).float(); E[idx[b]] = ((h * mk).sum(1) / mk.sum(1)).cpu().numpy()
    del m; gc.collect(); torch.cuda.empty_cache(); return E, diag

def enc_st(cfg):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg["repo"], device=DEVICE); pre = cfg.get("prefix", ""); E = None; diag = {}
    for d, idx in DOM_IDX.items():
        tx = [texts[i] for i in idx]; m.max_seq_length = MAX_LEN[d]; diag[d] = tok_diag(m.tokenizer, tx, MAX_LEN[d])
        e = m.encode([pre + t for t in tx], batch_size=BATCH, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        if E is None: E = np.zeros((len(texts), e.shape[1]), np.float32)
        E[idx] = e
    del m; gc.collect(); torch.cuda.empty_cache(); return E, diag

def enc_w2v(cfg):
    from huggingface_hub import hf_hub_download
    from gensim.models import Word2Vec, KeyedVectors
    z = hf_hub_download(cfg["repo"], cfg["file"]); dst = Path("/tmp/w2v" if ON_KAGGLE else "w2v_tmp"); zipfile.ZipFile(z).extractall(dst)
    mf = [p for p in dst.rglob("*") if p.suffix in (".model", ".bin", ".kv")]; print("   w2v files:", [p.name for p in mf])
    f = next(p for p in mf if p.suffix == ".model")
    try: kv = Word2Vec.load(str(f)).wv
    except Exception: kv = KeyedVectors.load(str(f))
    E = np.zeros((len(texts), kv.vector_size), np.float32); diag = {}
    for d, idx in DOM_IDX.items():
        n = oov = 0
        for i in idx:
            toks = _PSPLIT(texts[i]).split(); vs = []
            for w in toks:
                n += 1
                if w in kv.key_to_index:
                    v = kv[w]; nv = np.linalg.norm(v)
                    if nv > 0: vs.append(v / nv)
                else: oov += 1
            if vs: E[i] = np.mean(vs, 0)          # same rule as fastText get_sentence_vector
        diag[d] = dict(fertility=1.0, unk_rate=oov / max(1, n), trunc_rate=0.0, empty_texts=int(sum((E[idx] == 0).all(1))))
    return E, diag

def enc_fake(cfg):
    return np.random.default_rng(0).normal(size=(len(texts), 32)).astype(np.float32), {d: {} for d in DOM_IDX}

ENC = dict(hf_mean=enc_hf_mean, st=enc_st, w2v=enc_w2v, fake=enc_fake)

# %%
TIM, SHA, FAILED = {}, {}, {}
for name, cfg in CANDS.items():
    t = time.perf_counter()
    try:
        E, diag = ENC[cfg["kind"]](cfg)
        assert E.shape[0] == len(df) and np.isfinite(E).all()
    except Exception as e:
        FAILED[name] = f"{type(e).__name__}: {e}"; print(f"!! {name} FAILED: {FAILED[name][:300]}"); continue
    np.save(OUT / "emb" / f"{name}.npy", E); (OUT / "emb" / f"{name}.diag.json").write_text(json.dumps(diag))
    TIM[name] = time.perf_counter() - t
    try:
        from huggingface_hub import model_info; SHA[name] = model_info(cfg["repo"]).sha
    except Exception: SHA[name] = None
    print(f"{name:<22} dim={E.shape[1]:<5} {TIM[name]:6.1f}s  elapsed {(time.perf_counter()-T0)/60:.1f} min  diag={diag}")
(OUT / "manifest.json").write_text(json.dumps(dict(device=DEVICE, minutes=round((time.perf_counter() - T0) / 60, 2),
    models={k: dict(**v, sha=SHA.get(k)) for k, v in CANDS.items()}, timing_s=TIM, failed=FAILED,
    st_version=sentence_transformers.__version__, torch=torch.__version__), indent=2, default=str))
print("done", round((time.perf_counter() - T0) / 60, 1), "min; failed:", list(FAILED))
