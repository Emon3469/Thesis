# BanglaMeta Explorer

An interactive web app for the merged Bangla Word2Vec model (BanglaMeta, 273,531 words × 900 dimensions). It combines
a 3-D embedding projector with a word2vec playground, and adds a page that compares BanglaMeta with fastText-cc.bn,
News-W2V and Informal-W2V.

Every answer comes from the served model rows. Nothing is generated, guessed or filled in. A word the model does not
contain is reported as missing.

## Pages

| Page | What it does |
|---|---|
| **Explorer** (`/`) | **Word field:** add Bangla or Banglish words to see each word and its nearest neighbours in 3-D. **Sentence field:** add one sentence per line to get sentence vectors, a sentence-to-sentence cosine heatmap and the words nearest each sentence. **Projections:** global PCA, local PCA, or t-SNE, with X/Y/Z axes, in 3-D or 2-D. **Analogy tool:** a : b :: c : ?. |
| **Compare** (`/compare`) | Nearest neighbours from all five models side by side, with overlap. Word-pair cosines. Sentence similarity per model. Quality measured live on this server next to the values the thesis recorded. The official held-out (CONF) results and tests. |

## Models built from the merged model

All three are built from the frozen merged rows alone: no training, and no other model's vectors.

| Model | What it is | Classification (DEV / CONF) | Analogy | Similarity AUC |
|---|---|---|---|---|
| **BanglaMeta+** (default) | one 1,200-d vector per word: [√0.0005 · thesis News+Informal blocks ; √0.9995 · BanglaMeta-Sim News+Subs blocks] | **77.34** / **77.22** | **8.98%** | **0.891** |
| BanglaMeta (thesis) | the frozen thesis rows, unchanged | 77.12 / **77.17** | 6.00% | 0.819 |
| BanglaMeta-Sim | thesis rows, per block: all-but-the-top (mean and top 4 directions removed), then weights News 1.65 / Informal 0 / Subs 1.35 | 76.25 / 76.28 | 8.99% | 0.891 |
| fastText-cc.bn | baseline | 76.20 / 76.44 | 7.57% | 0.867 |

How BanglaMeta+ works:
- **Classification:** the probe standardises every feature, so the tiny thesis-row share costs nothing. The probe still sees all the thesis features.
- **Word tasks:** cosine is 0.0005 × thesis cosine + 0.9995 × similarity-view cosine, which keeps the similarity view's strength.
- **Why 1,200:** see `eval/dimension_study/Dimension_and_overfitting_study.pdf`. Dropping Subs2vec from the thesis part keeps or improves every score compared with the earlier 1,500-d version.

How the settings were chosen:
- **D, the block weights and ε** were chosen on a random half of the analogy questions (`backend/experiments/`: `block_weights.py`, `build_abtt.py`, `eps_1200.py`; results in `data/block_weights_*.json` and `eval/dimension_study/eps_1200.json`). The similarity AUC was never used for the choice.
- **CONF scores** of the new models are exploratory, because the thesis used the CONF half once for its confirmatory test. On CONF, BanglaMeta+ scores 77.22 against 77.17 for the thesis model, a tie within the confidence interval.
- **Coverage:** the vocabulary is the merged one, so informal-token coverage (94.90%) stays below fastText's (96.08%).

Note: cosine measures how often two words share contexts, so "রাজা" and "পুরুষ" are not close in any of these models
(fastText 0.18, News-W2V 0.20). Their relation is the male/female direction, which the Analogy tool finds
(রাজা : রানী :: পুরুষ : মহিলা).

## Input: Bangla or Banglish only

- **Accepted:** Bangla script, Latin letters (Banglish), digits, punctuation and spaces. Anything else (other scripts, emoji, accented letters) is rejected with a message, both in the browser and on the server.
- **Bangla words:** looked up exactly as typed, after the thesis tokenizer. The server runs `eval/w2v_kaggle_dataset/tokenize.py` itself.
- **Banglish words:** converted by a rule-based phonetic table plus a beam search that keeps only spellings that are words of the merged model. Candidates are ranked by frequency in the model. The page shows what each Banglish word became (`bhalo → ভাল`) and lets you pick another candidate.

## Integrity

`prepare.py` checks the files and writes `data/manifest.json`. It confirms:
- the merged model is the frozen Step 4 file (MD5 `10aa206f8462c82a7b16cf60054be70f`);
- the merged vocabulary is exactly the union of News-W2V, Informal-W2V and Subs2vec-w2, in the thesis order;
- the News block of each merged row equals the News-W2V row;
- the fastText table is rebuilt with the thesis rule (first 200k rows plus every evaluation word = 234,474 rows).

The server re-checks the file checksums every time it starts.

The tests (`backend/tests`, 169 tests) check that:
- served rows equal the frozen sources, and the released `.bin` file;
- every neighbour list and analogy result equals an independent brute-force computation;
- sentence vectors are the mean of their word rows, as in the thesis;
- the tokenizer matches the thesis tokens on 3,000 evaluation texts;
- Banglish candidates are always model words;
- invalid input gets a clear 422 error;
- concurrent requests return identical answers;
- the live benchmarks reproduce every DEV score the thesis recorded.

On the DEV half, the server reproduces the thesis scores to 4 decimals. The one exception: one analogy sub-score for News-W2V differs by a single question, because the served rows are the frozen float16 values.

## Run

The first time only. This needs `data/raw/cc.bn.300.vec.gz` from
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz (about 880 MB).

```bash
cd webapp/backend
python prepare.py
python -m app.bench
```

`python prepare.py` checks the files and writes the fast-search copies (about 2 minutes). `python -m app.bench` measures quality on the DEV half and stores the results (about 2 minutes).

Production uses one server on http://127.0.0.1:8000:

```bash
cd webapp/frontend && npm install && npm run build
```

```bash
cd webapp/backend && python -m uvicorn app.main:app --port 8000
```

For development, run the API on port 8000 and the frontend with hot reload on port 5173:

```bash
cd webapp/frontend && npm run dev
```

Tests:

```bash
cd webapp/backend && python -m pytest -q tests
```

## Stack

- **Backend:** FastAPI and NumPy. Models are memory-mapped float32 unit rows. Search is exact cosine over the full vocabulary (about 0.15 s on this PC). Analogies use the thesis rule: 3CosAdd over the first 200k rows.
- **Frontend:** React, TypeScript and Vite. The 3-D view uses Three.js through react-three-fiber and drei. UI animation uses Framer Motion. Fonts are Anek Bangla, Fraunces and IBM Plex Mono, bundled locally.
