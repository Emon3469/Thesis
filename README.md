# Merging Domain-Specific Bangla Word2Vec Models

**A news Word2Vec and a conversational Word2Vec, merged into one 900-dimensional Bangla embedding that is confirmed on held-out data to beat each source model and the fastText Common Crawl baseline.**

This repository contains the full, reproducible pipeline of an MSc thesis on *meta-embedding* for Bangla:

- building the evaluation data;
- training and screening the source models;
- checking mathematically that a merge can help, before merging;
- running 28 merge experiments;
- selecting a model by a rule fixed in advance;
- a single confirmation test on data never used for any choice.

**Post-thesis extension (§14):** **BanglaMeta+**, a 1,200-d embedding built from the frozen merged model. It keeps the merged model's classification accuracy and adds the best word-analogy and word-similarity scores of the evaluated models. An interactive web app (`webapp/`) serves it.

Every number in this README comes from a CSV or JSON in this repository. Each stage has an independent verification script (`eval/verify_*.py`), and all of them pass.

---

## Contents

1. [Main result](#1-main-result)
2. [Repository layout](#2-repository-layout)
3. [Data](#3-data)
4. [Source models](#4-source-models)
5. [How the merges are built](#5-how-the-merges-are-built)
6. [Why concatenation: the mathematics](#6-why-concatenation-the-mathematics)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [Pipeline, step by step](#8-pipeline-step-by-step)
9. [Results](#9-results)
10. [Using the merged model](#10-using-the-merged-model)
11. [Reproducing the experiments](#11-reproducing-the-experiments)
12. [Limitations](#12-limitations)
13. [Sources and references](#13-sources-and-references)
14. [BanglaMeta+ (1,200-d): post-thesis extension and web app](#14-banglameta-1200-d-post-thesis-extension-and-web-app)

---

## 1. Main result

**Final model: `T: CONC`**
- News-W2V + Informal-W2V + Subs2vec-w2, merged by equal-energy concatenation over a ridge-imputed union vocabulary.
- 900 dimensions, 273,531 words.

**Confirmation test** (CONF half: 11,950 texts that played no part in any choice; macro-F1; 10,000 paired bootstrap samples; Holm correction):

| Pre-registered hypothesis | Merged | Baseline | Δ Overall [95% CI] | Holm-adjusted p | Result |
|---|---|---|---|---|---|
| H1: merged > fastText-cc.bn | 0.7717 | 0.7644 | +0.0073 [+0.0017, +0.0128] | 0.0044 | confirmed |
| H2: merged > News-W2V | 0.7717 | 0.7477 | +0.0241 [+0.0186, +0.0295] | 0.0003 | confirmed |
| H3: merged > Informal-W2V | 0.7717 | 0.7326 | +0.0391 [+0.0335, +0.0448] | 0.0003 | confirmed |

**Why it wins.** Each source is strong on one domain only. The merge matches the best model on news (fastText) and the best model on informal text (Informal-W2V) at the same time, and no single model does both.

| CONF macro-F1 | News (D1) | Informal (D2) | Overall |
|---|---|---|---|
| **Merged T: CONC** | **0.8840** | **0.6595** | **0.7717** |
| fastText-cc.bn | 0.8815 | 0.6474 | 0.7644 |
| News-W2V | 0.8791 | 0.6162 | 0.7477 |
| Subs2vec-w2 | 0.8656 | 0.6206 | 0.7431 |
| Informal-W2V | 0.8073 | 0.6579 | 0.7326 |

**The DEV score held up.** The model scored 0.7712 on DEV and 0.7717 on CONF, so selecting among 28 candidates did not inflate the result.

---

## 2. Repository layout

```
README.md                      this file
EXECUTION_PLAN.md              the pre-registered research plan
RESEARCH_DIRECTION.md          scope and positioning of the thesis
models/merged_w2v_T_conc/      model card of the final merged model (weights not stored on GitHub, see §10)
models/banglameta_plus/        model card + MD5 of BanglaMeta+ 1,200-d (weights rebuilt by the notebook, see §14)
webapp/                        FastAPI backend + React/three.js frontend serving the models (tests in webapp/backend/tests)
docs/                          mathematics exercise set with worked solutions, README table generator
eval/
  build_eval_set.py            loaders and cleaning for every source dataset
  build_w2v_data.py            builds the evaluation set (DEV/CONF) and the two training corpora
  w2v_src.py                   Steps 1-3: train/download source models, screening, pre-merge checks P1-P6
  merge_src.py                 Step 4: all merge experiments + pre-registered selection   <- the merge code
  conf_src.py                  Steps 6-7: one-time confirmation test on the CONF half
  export_w2v.py                exports the frozen merged model to word2vec .bin format
  make_notebook.py             turns a *_src.py file into a Kaggle notebook
  verify_w2v.py                independent check of Steps 1-3 outputs
  verify_merge.py              independent check of Step 4 outputs
  verify_conf.py               independent check of Steps 6-7 outputs
  proof_checks.py              numeric checks of the theorems
  w2v_kaggle_kernel/           notebook + metadata that ran Steps 1-3 on Kaggle
  merge_kaggle_kernel/         notebook + metadata that ran Step 4 on Kaggle
  conf_kaggle_kernel/          notebook + metadata that ran Steps 6-7 on Kaggle
  w2v_output/                  Steps 1-3 results (screening_dev.csv, premerge_checks.csv, figures, log)
  merge_output/                Step 4 results (merge_dev.csv, premerge_checks.csv, manifest.json, figure, log)
  conf_output/                 Steps 6-7 results (conf_results.csv, conf_tests.csv, manifest.json, figure, log)
  W2V_PREEVAL_REPORT.md        report of Steps 1-3
  MERGE_PROOF.md               the merge formula and its 10 theorems
  MERGE_RESULTS.md             report of Step 4
  CONF_RESULTS.md              report of Steps 6-7
  banglameta_plus/             BanglaMeta_plus_1200d.ipynb: builds, evaluates and exports BanglaMeta+ (source: banglameta_plus_src.py)
  banglameta_plus_eval/        full evaluation of BanglaMeta+ (report PDF, figures, metrics_matrix.csv, results.json)
  dimension_study/             over/under-fitting and dimension study (report PDF, figures, results.json)
  preeval/                     earlier exploratory stage with sentence encoders (BanglaBERT, mE5, ...)
  notebook_src.py, candidates_src.py, mini_*.py, preeval*.py   code of that earlier stage
```

### Where each model is built

All in [`eval/merge_src.py`](eval/merge_src.py):

| Model | Code |
|---|---|
| News-W2V, Informal-W2V (trained) | `train_w2v`, lines 121-131 |
| Continued-W2V (News-W2V trained further on informal text) | lines 133-136 |
| Joint-W2V (one Word2Vec on both corpora) | lines 138-142 |
| Subs2vec-w2, fastText-cc.bn (downloaded, MD5-checked) | lines 144-161 |
| Merge operators: `ridge_map`, `union_words`, `blocks`, `conc`, `svd_project`, `procrustes_avg`, `aeme` | lines 318-385 |
| Source sets P, Q, T, S, X | line 401 |
| Every merge (CONC, SVD, weighted CONC, AAEME/CAEME, CONC-zero, Procrustes+AVG) | lines 404-429 |
| Selection rule | lines 434-465 |

---

## 3. Data

### 3.1 Domains and source datasets

| Domain | Dataset | Content | Task | Classes |
|---|---|---|---|---|
| D1: formal news | Potrika | Bangla news articles | topic | 7 (economy, education, entertainment, international, national, politics, sports) |
| D2: informal text | BanglaBook | book reviews | sentiment | 3 |
| D2: informal text | SentNoB | social-media comments | sentiment | 3 |
| D2: informal text | Bengali hate speech | social-media comments | hate / not hate | 2 |

The raw datasets are **not** redistributed here; see §13 for where to get them.

### 3.2 Evaluation set (`build_w2v_data.py`)

- **Size:** 23,900 labelled texts, balanced per class, sampled with seed 42.
- **Composition:**

  | Task | Texts per class | Total | Mean tokens |
  |---|---|---|---|
  | News topic (7 classes) | 1,700 | 11,900 (49.8%) | 216.7 (truncated) |
  | Review sentiment (3) | 1,500 | 4,500 | 23.1 |
  | Social sentiment (3) | 1,000 | 3,000 | 13.6 |
  | Hate speech (2) | 2,250 | 4,500 | 15.4 |

  News makes up 11,900 texts (49.8%); the three informal tasks make up 12,000 texts (50.2%).
- **Cleaning:** duplicate texts and texts with conflicting labels are removed.
- **Split:** every task is split once, stratified 50/50 (seed 42), into **DEV** (11,950 texts: all experiments and choices) and **CONF** (11,950 texts: used once, for the final test).
- **Why this size:** a power analysis set it at about 12k texts per half, enough to detect a 1-2 point difference in macro-F1.
- **Tokenizer** (`tokenize.py`, used everywhere): NFC normalisation, zero-width characters removed, punctuation split off, Latin lowercased, split on whitespace.

### 3.3 Training corpora

**Every evaluation text is removed from the training corpora**, so no model has seen a test text.

| Corpus | Source | Documents | Tokens | Eval texts removed |
|---|---|---|---|---|
| `news_corpus` | Potrika | 271,910 | 65,324,912 | 11,984 |
| `informal_corpus` | BanglaBook + SentNoB + hate speech | 149,423 | 3,192,884 | 13,616 |

### 3.4 Intrinsic test data

- **Word analogy:** the Bangla analogy set of Akter et al. (2023), 16,678 native questions, scored by 3CosAdd over the 200k most frequent words. A question with an unknown word counts as wrong.
- **Word similarity proxy:** AUC of cosine similarity for "same category vs different category" word pairs, built from the analogy word lists. It uses only the 124 words present in every model.

---

## 4. Source models

| Model | Type | Training data | Vocabulary | Role |
|---|---|---|---|---|
| **News-W2V** | Word2Vec, trained here | news corpus | 187,686 | Domain A (news) |
| **Informal-W2V** | Word2Vec, trained here | informal corpus | 32,910 | Domain B (informal) |
| **Subs2vec-w2** | Word2Vec, downloaded | OpenSubtitles (Zenodo record 22230710) | 177,376 | Domain B (conversational) |
| fastText-cc.bn | fastText `.vec`, downloaded | Common Crawl + Wikipedia | 1,468,578 | baseline (not Word2Vec) |
| Joint-W2V | Word2Vec, trained here | news + informal corpus together | 199,071 | merge alternative |
| Continued-W2V | News-W2V, trained further | + informal corpus | 194,383 | merge alternative |

- **Word2Vec settings** (gensim 4.4.0, identical for all trained models): skip-gram, 300 dimensions, window 4, min_count 5, negative sampling 10, 5 epochs, seed 42.
- **Which Subs2vec model:** Subs2vec has six context windows (w1-w6). w2 was chosen by the pre-registered rule "best informal-domain F1 on DEV" (Step 2).
- **Why News-W2V is trained here:** the originally proposed news model (alamkhaan) can no longer be downloaded, so a Word2Vec with the same recipe was trained on Potrika.

---

## 5. How the merges are built

### 5.1 The main formula (`CONC`)

For every word `w` in the union of the source vocabularies, with K sources:

```
m(w) = (1/√K) · [ ã₁(w) ; ã₂(w) ; … ; ã_K(w) ]
```

1. **Normalise:** every source row is scaled to length 1 (`Table`, `merge_src.py`).
2. **Union vocabulary:** every word known to at least one source (`union_words`).
3. **Complete missing rows by ridge imputation:** if a source *k* lacks the word, its block is predicted from a source that has it. The prediction uses a linear map fitted on all shared words, and the result is re-normalised:
   ```
   M_{S→T} = argmin_M Σ_{w ∈ S∩T} ‖s(w)·M − t(w)‖² + λ‖M‖²  =  (SᵀS + λI)⁻¹ SᵀT,   λ = 1
   ```
   (`ridge_map`, `blocks(..., fill="impute")`)
4. **Concatenate with equal energy:** each block is scaled by `1/√K`. Every merged row then has length 1, and each source contributes exactly 1/K of it (`conc`).

For the final model T (K = 3): columns 1-300 come from News-W2V, 301-600 from Informal-W2V and 601-900 from Subs2vec-w2.

### 5.2 Source sets

| Set | Sources | Pure Word2Vec |
|---|---|---|
| **P** | News-W2V + Subs2vec-w2 (the professor's news + conversational pair) | yes |
| **Q** | News-W2V + Informal-W2V (the two domain specialists among Word2Vec models) | yes |
| **T** | News-W2V + Informal-W2V + Subs2vec-w2 | yes |
| S | fastText + Informal-W2V | no (extended) |
| X | fastText + News-W2V + Informal-W2V + Subs2vec-w2 | no (extended) |

### 5.3 All merge methods tested (28 candidates)

| Method | What it does | Sets |
|---|---|---|
| **CONC** | equal-energy concatenation, ridge-imputed union (§5.1) | P, Q, T, S, X |
| CONC-zero | same, but missing rows are left as zeros | P, Q |
| weighted CONC | blocks scaled by √α and √(1−α), α = 0.3 or 0.7 | P, Q |
| CONC + SVD | CONC projected onto its top-k right singular vectors, k = 100, 200, 300 (`svd_project`) | P, Q, T |
| Procrustes + AVG | the second source is rotated onto the first by orthogonal Procrustes on shared words, then the two are averaged (`procrustes_avg`) | P, Q |
| AAEME | autoencoder: one linear + tanh encoder per source (300-d), encodings summed, L2-normalised, linear decoders reconstruct each source; 10 epochs, Adam (`aeme`, Bollegala & Bao 2018) | P, Q |
| CAEME | the same, but encodings concatenated (600-d) | P, Q |
| Joint-W2V | one Word2Vec trained on the news + informal corpora together | — |
| Continued-W2V | News-W2V, vocabulary extended, trained further on the informal corpus | — |

---

## 6. Why concatenation: the mathematics

The full statements and proofs are in [`eval/MERGE_PROOF.md`](eval/MERGE_PROOF.md). In short:

| # | Result | Meaning |
|---|---|---|
| 1 | The merged sentence vector equals the concatenation of the source sentence vectors | merging words = merging sentence features (checked: max error 0.0) |
| 2 | R(H_AB) ≤ min(R(H_A), R(H_B)) | with enough data, a linear probe on the merge is never worse than on either source |
| 3 | Gain = I(Y; X_B \| X_A) ≥ 0 | the gain equals the label information the second model adds; measured, not assumed |
| 4 | Every linear merge (averaging, alignment + averaging, SVD, weighting, linear autoencoder) is L·m(w) | none of them can reach a lower loss than concatenation |
| 5 | Block weights cancel under standardisation | equal weights need no tuning (confirmed: weighted CONC gives identical F1) |
| 6 | cos(m(u), m(v)) = average of source cosines | merged similarity is less noisy if source errors are not perfectly correlated |
| 7 | SVD risk bound (Eckart-Young) | how much compression can lose |
| 8 | Imputation error bound | missing-word completion adds only bounded error |
| 9 | Dimension-free generalisation bound | 600-900 dimensions are safe with ~6,000 texts per domain |
| 10 | Hoeffding + union bound | differences among 28 DEV candidates below ~2.4 points are not guaranteed, hence the CONF test |

### Pre-merge checks (P1-P6)

These are computed before any merged model is evaluated (`p_checks` in `merge_src.py`).

| Check | Question | P | Q | T |
|---|---|---|---|---|
| P1 | Pooling identity holds? | ✓ | ✓ | ✓ |
| P2 | Does each source add label information (CI > 0)? | ✓ | ✓ (informal: 0.068 nats; news: 0.198 nats) | ✗ (Subs2vec adds 0.003 [−0.001, 0.006] nats) |
| P3 | Is SVD guaranteed safe? | ✗ | ✗ | ✗ |
| P4 | Is energy balanced between sources? | ✓ | ✓ | ✓ |
| P5 | Is imputation better than chance? | ✓ | ✓ | ✓ |
| P6 | Is alignment good enough for averaging (P@1 ≥ 0.5)? | ✗ (0.44) | ✗ (0.08) | n/a |

The checks predicted the experimental outcome. Concatenation is certified, while SVD and averaging are not, and in the experiments they scored lower. The third source in T was not certified, and in the experiments it added a non-significant +0.26 points.

---

## 7. Evaluation protocol

Every model, single or merged, is evaluated the same way:

- **Sentence vector:** the mean of the unit-length rows of the text's tokens that are in the vocabulary.
- **Probe:** `StandardScaler` + `RidgeClassifierCV` (α from 10⁻² to 10⁴), 5 stratified folds (seed 42), out-of-fold predictions.
- **Metric:** macro-F1.
  - D1 = news topic.
  - D2 = mean of review sentiment, social sentiment and hate speech.
  - **Overall = (D1 + D2) / 2.**
- **Uncertainty:** paired bootstrap over texts. All models use the same resampled texts (1,000 samples on DEV, 10,000 on CONF).
- **Selection rule (pre-registered):**
  1. Specialists = the best single model on D1 and on D2.
  2. A candidate is *eligible* if it is within 0.01 of the specialist on **both** domains.
  3. Selected = the eligible candidate with the highest Overall F1. This is done twice: among pure Word2Vec candidates (main model) and among all candidates (extended model).
- **Confirmation (pre-registered):** the selected model is frozen (MD5-checked) and tested once on CONF against fastText, News-W2V and Informal-W2V. It counts as better only if all three Holm-adjusted one-sided p-values are below 0.05.

---

## 8. Pipeline, step by step

| Step | What | Code | Output | Report |
|---|---|---|---|---|
| 0 | Build evaluation set + corpora | `build_w2v_data.py` | `w2v_kaggle_dataset/` (local, not on GitHub) | — |
| 1 | Download / train and verify source models | `w2v_src.py` | `w2v_output/` | `W2V_PREEVAL_REPORT.md` |
| 2 | Evaluate each model alone (DEV) | `w2v_src.py` | `screening_dev.csv` | same |
| 3 | Prove a merge can add information (P1-P6) | `w2v_src.py`, `proof_checks.py` | `premerge_checks.csv` | same + `MERGE_PROOF.md` |
| 4 | 28 merge experiments (DEV) | `merge_src.py` | `merge_output/` | `MERGE_RESULTS.md` |
| 5 | Select by the pre-registered rule | `merge_src.py` | `manifest.json` | same |
| 6-7 | One-time test on CONF | `conf_src.py` | `conf_output/` | `CONF_RESULTS.md` |
| — | Export the final model | `export_w2v.py` | `bangla_merged_w2v_900d.bin` | `models/merged_w2v_T_conc/README.md` |

All heavy steps ran on Kaggle CPU notebooks (Step 4: 130 min; CONF test: 5.5 min). Each `*_src.py` also runs locally, and `EVAL_SMOKE=1` gives a fast end-to-end smoke test.

---

## 9. Results

### 9.1 Single models on DEV (Step 2)

| Model | Similarity AUC | Analogy | Token coverage news / informal | D1 | D2 | Overall |
|---|---|---|---|---|---|---|
| fastText-cc.bn | 0.867 | 0.076 | 0.986 / 0.961 | 0.879 | 0.645 | 0.762 |
| News-W2V | 0.881 | 0.063 | 0.985 / 0.912 | 0.874 | 0.623 | 0.749 |
| Subs2vec-w2 | 0.858 | 0.059 | 0.960 / 0.915 | 0.859 | 0.626 | 0.742 |
| Informal-W2V | 0.692 | 0.008 | 0.839 / 0.914 | 0.815 | 0.645 | 0.730 |

No single Word2Vec model is best on both domains. That is the case for merging.

### 9.2 Merge experiments on DEV (Step 4, top of 33 rows; full table in `eval/MERGE_RESULTS.md`)

| Candidate | Dim | D1 | D2 | Overall | Δ vs fastText [95% CI] |
|---|---|---|---|---|---|
| X: CONC (extended) | 1200 | 0.8885 | 0.6582 | 0.7733 | +0.0113 [+0.0065, +0.0163] |
| **T: CONC (selected)** | 900 | 0.8823 | 0.6601 | 0.7712 | +0.0092 [+0.0037, +0.0151] |
| Q: CAEME | 600 | 0.8817 | 0.6599 | 0.7708 | +0.0088 [+0.0028, +0.0145] |
| Q: CONC | 600 | 0.8799 | 0.6594 | 0.7697 | +0.0077 [+0.0018, +0.0136] |
| Joint-W2V | 300 | 0.8752 | 0.6570 | 0.7661 | +0.0041 [−0.0015, +0.0098] |
| Continued-W2V | 300 | 0.8734 | 0.6567 | 0.7651 | +0.0031 [−0.0027, +0.0088] |
| Q: CONC+SVD300 | 300 | 0.8787 | 0.6485 | 0.7636 | +0.0016 [−0.0046, +0.0076] |
| Q: Procrustes+AVG | 300 | 0.8717 | 0.6538 | 0.7628 | +0.0008 [−0.0051, +0.0067] |
| P: CONC | 600 | 0.8793 | 0.6384 | 0.7589 | −0.0031 [−0.0088, +0.0026] |

What the experiments showed:
- **Concatenation beat every compression.** SVD, alignment + averaging and the 300-d autoencoder all scored below plain concatenation (Theorem 4).
- **Weights made no difference.** Weighted CONC gave identical F1 to equal weights (Theorem 5).
- **The informal gain comes from Informal-W2V.** News + Subs2vec (P) beats its own sources but not fastText.

### 9.3 Confirmation on CONF (Steps 6-7)

See §1. All three pre-registered hypotheses were confirmed. Secondary results:
- **Adding Subs2vec as a third model:** T vs Q = +0.0026 [−0.0010, +0.0062], not significant.
- **Adding fastText:** X vs T = +0.0066 [+0.0030, +0.0102], significant. X is the best model measured, but it is not pure Word2Vec.

---

## 10. Using the merged model

The weights (≈ 1 GB) are too large for GitHub. They are produced by:

```bash
python eval/export_w2v.py
```

This writes `models/merged_w2v_T_conc/bangla_merged_w2v_900d.bin` from the frozen Step 4 output (MD5-checked).

```python
from gensim.models import KeyedVectors
import numpy as np

kv = KeyedVectors.load_word2vec_format("models/merged_w2v_T_conc/bangla_merged_w2v_900d.bin", binary=True)
print(len(kv), kv.vector_size)            # 273531 900

def sentence_vector(tokens):              # the representation used in all experiments
    rows = [kv[w] for w in tokens if w in kv]
    return np.mean(rows, axis=0) if rows else np.zeros(kv.vector_size, dtype=np.float32)
```

Use the same tokenizer as the experiments (`tokenize.py`, shown in `eval/build_w2v_data.py`).

**Good for:** features for Bangla text classification across news and informal text.

**Not the best choice for:** word similarity and analogy, where News-W2V and fastText score higher (§12). For those, use **BanglaMeta+** (§14), which keeps the classification accuracy and leads on analogy and similarity:

```python
kv = KeyedVectors.load_word2vec_format("models/banglameta_plus/banglameta_plus_1200d.bin", binary=True)   # 273531 x 1200
```

---

## 11. Reproducing the experiments

**Requirements:** Python 3.10+, `numpy`, `pandas`, `scikit-learn`, `gensim 4.4`, `torch` (autoencoders only), `matplotlib`, `pyarrow`, and the `kaggle` CLI for the cloud runs.

```bash
# 0. build data (needs the raw datasets of §13 in the folders named in eval/build_eval_set.py)
python eval/build_w2v_data.py

# smoke test any stage locally (minutes)
EVAL_SMOKE=1 python eval/merge_src.py

# full runs on Kaggle: build the notebook, then push it
python eval/make_notebook.py merge_src.py merge_kaggle_kernel/bangla-w2v-merge.ipynb
kaggle kernels push -p eval/merge_kaggle_kernel
kaggle kernels output <user>/bangla-w2v-merge -p eval/merge_output

# independent verification of every reported number
python eval/verify_w2v.py
python eval/verify_merge.py
python eval/verify_conf.py
```

**Reproducibility note.** Word2Vec training with several worker threads is not bit-for-bit reproducible, so re-training gives slightly different source models (e.g. News-W2V D1: 0.876 in Step 1, 0.874 in Step 4). For that reason the confirmation test loads the **frozen** files saved by Step 4 and checks them by MD5; it does not retrain.

---

## 12. Limitations

- **The thesis model's gain is for classification only.** Its similarity AUC is 0.819 (News-W2V: 0.881) and its analogy accuracy is 0.060 (fastText: 0.076). The noisier Informal-W2V block lowers the intrinsic scores, which is the limit case of Theorem 6. BanglaMeta+ (§14) removes this limitation: analogy 0.0898, similarity AUC 0.8914. Its CONF result is exploratory, because it was built after the confirmatory test.
- **The gain over fastText is small** (+0.7 points Overall) though significant. It comes from informal text; on news the merged model and fastText are equal.
- **The probe is linear.** It runs on averaged static vectors, so it measures embedding quality, not the best accuracy reachable with fine-tuned models.
- **Not compared with transformers on this benchmark.** BanglaBERT, multilingual E5 and others were explored only in an earlier stage on a different evaluation set (`eval/preeval/`). No claim of being the best Bangla model overall is made.
- **Three source models.** Adding Subs2vec as a third Word2Vec gave no significant improvement over the two-model merge (News + Informal).

---

## 13. Sources and references

**Data and models**
- Potrika: Bangla news dataset (topic-labelled news articles).
- BanglaBook: Bangla book-review sentiment dataset.
- SentNoB: Bangla social-media sentiment dataset.
- Bengali Hate Speech dataset.
- Subs2vec Bangla Word2Vec (OpenSubtitles), Zenodo record 22230710.
- fastText Common Crawl vectors `cc.bn.300.vec` (Grave et al., 2018).
- Bangla word-analogy dataset: Akter et al., 2023 (MIT licence).

**Meta-embedding literature** (found through SciSpace)
- Coates & Bollegala (2018), *Frustratingly Easy Meta-Embedding*: https://scispace.com/papers/frustratingly-easy-meta-embedding-computing-meta-embeddings-1m3jmrdh39
- Bollegala & Bao (2018), *Learning Word Meta-Embeddings by Autoencoding*: https://scispace.com/papers/learning-word-meta-embeddings-by-autoencoding-zj3zydu0iu
- Bollegala (2022), *Learning Meta Word Embeddings by Unsupervised Weighted Concatenation of Source Embeddings*: https://scispace.com/papers/learning-meta-word-embeddings-by-unsupervised-weighted-x4edqcl9
- Jawanpuria et al. (2020), *Learning Geometric Word Meta-Embeddings*: https://scispace.com/papers/learning-geometric-word-meta-embeddings-1z33jexmcs
- Raunak et al. (2019), *Effective Dimensionality Reduction for Word Embeddings*: https://scispace.com/papers/effective-dimensionality-reduction-for-word-embeddings-1ht9cklr8o
- Mu & Viswanath (2018), *All-but-the-Top: Simple and Effective Postprocessing for Word Representations* (used by BanglaMeta+): https://scispace.com/papers/all-but-the-top-simple-and-effective-postprocessing-for-word-5e3j95bkmx

---

<!-- BANGLAMETA_PLUS:START (generated by docs/make_readme_tables.py; do not edit by hand) -->
## 14. BanglaMeta+ (1,200-d): post-thesis extension and web app

**BanglaMeta+** is one vector per word, built from the frozen thesis model alone, with no training and no other model's vectors:

`c(w) = [ √0.0005 · m_NI(w) ; √0.9995 · s_News(w) ; √0.9995 · s_Subs(w) ]`

- **m_NI(w):** the thesis row's News-W2V and Informal-W2V blocks, re-normalised (600-d).
- **s(w):** the similarity view, 600-d. In each 300-d block, the mean and the top 4 principal directions are removed ("all-but-the-top"); then the blocks are weighted News 1.65 / Subs 1.35 (the Informal block gets weight 0).
- **Classification:** the probe standardises every feature, so the small weight costs the thesis features nothing (Theorem 5).
- **Word similarity:** cosine = 0.0005 × thesis cosine + 0.9995 × similarity-view cosine.
- **How the settings were chosen:** D, the block weights and ε were chosen on a random half of the analogy questions; ties go to the smaller ε.

| | |
|---|---|
| File | `models/banglameta_plus/banglameta_plus_1200d.bin` (word2vec binary, 273,531 × 1,200, float32; not on GitHub, rebuilt by the notebook) |
| MD5 | `a30a9c38b5a7964fa5ab2d4b44f2c701` |
| Notebook | [`eval/banglameta_plus/BanglaMeta_plus_1200d.ipynb`](eval/banglameta_plus/BanglaMeta_plus_1200d.ipynb): rebuilds the file byte for byte and evaluates it |
| Recipe / export | `webapp/backend/prepare.py` · `webapp/backend/experiments/export_plus.py` |
| Rank among the 6 models | DEV F1 #1 · CONF F1 #1 · Analogy #2 · Similarity AUC #1 (tied) |

### 14.1 All models, measured the same way

Macro-F1 % uses the thesis probe. CONF intervals come from the thesis's 10,000-sample paired bootstrap, and they reproduce the official thesis CONF table exactly. CONF values of BanglaMeta+ and BanglaMeta-Sim are **exploratory**, because both were built after the confirmatory test. The analogy column also gives the number of questions solved, out of 16,678. Bold marks the best value in each column.

| Model | dims | DEV F1 | CONF F1 [95% CI] | CONF news D1 | CONF informal D2 | Analogy (native) | Semantic | Syntactic | Similarity AUC | Informal coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| **BanglaMeta+** | 1,200 | **77.34** | **77.22** [76.49, 77.92] | **88.61** | 65.83 | 8.98% (1,498) | **7.80**% | 9.49% | **0.8914** | 94.90% |
| BanglaMeta | 900 | 77.12 | 77.17 [76.45, 77.88] | 88.40 | **65.95** | 6.00% (1,001) | 6.53% | 5.77% | 0.8191 | 94.90% |
| BanglaMeta-Sim | 900 | 76.25 | 76.28 [75.55, 76.97] | 88.39 | 64.18 | **8.99**% (1,499) | **7.80**% | **9.50**% | **0.8914** | 94.90% |
| fastText-cc.bn | 300 | 76.20 | 76.44 [75.71, 77.14] | 88.15 | 64.74 | 7.57% (1,262) | 4.09% | 9.07% | 0.8669 | **96.08**% |
| News-W2V | 300 | 74.85 | 74.77 [74.03, 75.49] | 87.91 | 61.62 | 6.27% (1,045) | 6.18% | 6.30% | 0.8809 | 91.24% |
| Informal-W2V | 300 | 73.04 | 73.26 [72.47, 74.02] | 80.73 | 65.79 | 0.81% (135) | 0.42% | 0.98% | 0.6921 | 91.38% |

### 14.2 Is BanglaMeta+ really better? (paired, same texts)

| BanglaMeta+ vs | Δ CONF Overall F1 [95% CI] | one-sided p | Analogy questions only BanglaMeta+ solves | only the other solves | McNemar p |
|---|---|---|---|---|---|
| BanglaMeta | +0.05 [-0.26, +0.35] (tie) | 0.3826 | 682 | 185 | 7.2e-68 |
| BanglaMeta-Sim | +0.94 [+0.50, +1.37] (better) | 0.0001 | 0 | 1 | 1.0e+00 |
| fastText-cc.bn | +0.78 [+0.25, +1.31] (better) | 0.0027 | 997 | 761 | 9.9e-09 |
| News-W2V | +2.45 [+1.95, +2.96] (better) | 0.0001 | 682 | 229 | 3.2e-53 |
| Informal-W2V | +3.96 [+3.37, +4.58] (better) | 0.0001 | 1,389 | 26 | < 1e-300 |

### 14.3 Why 1,200 dimensions: every option scored the same way

No Word2Vec here was trained with more than 300 dimensions. The larger sizes are separately trained 300-d models joined together. Full report: [`eval/dimension_study/Dimension_and_overfitting_study.pdf`](eval/dimension_study/Dimension_and_overfitting_study.pdf).

| Option | DEV F1 | CONF F1 | Analogy (all) | Analogy (unseen half) | Similarity AUC | Train–held-out gap (points) | Dims holding 90% of variance |
|---|---|---|---|---|---|---|---|
| 300 (PCA/SVD of the 1,500-d model) | 75.40 | 75.44 | 9.10% | 8.87% | 0.8878 | 6.33 | 236 |
| 500 (PCA/SVD of the 1,500-d model) | 75.70 | 75.77 | 9.05% | 8.81% | 0.8868 | 7.52 | 389 |
| 600 similarity view only | 76.25 | 76.28 | 8.99% | 8.66% | 0.8914 | 6.39 | 449 |
| 900 thesis BanglaMeta | 77.12 | 77.17 | 6.00% | 5.62% | 0.8191 | 6.44 | 532 |
| 1,200 thesis + similarity News block | 77.23 | 77.14 | 6.87% | 6.56% | 0.8817 | 5.30 | 230 |
| 1,200 thesis + similarity Subs block | 76.95 | 77.23 | 6.40% | 6.36% | 0.8701 | 5.69 | 247 |
| **1,200 BanglaMeta+ (current)** | 77.34 | 77.22 | 8.98% | 8.66% | 0.8914 | 5.22 | 450 |
| 1,500 previous BanglaMeta+ | 77.16 | 77.15 | 8.98% | 8.65% | 0.8913 | 6.06 | 450 |
| 1,800 + Informal similarity block (weight 0.15) | 77.16 | 77.29 | 8.86% | 8.45% | 0.8928 | 6.62 | 479 |
| 1,800 + all similarity blocks, equal weights | 77.16 | 77.29 | 6.62% | 6.30% | 0.8818 | 6.62 | 540 |

- **Why 1,200 replaced the 1,500-d version:** dropping Subs2vec from the thesis part keeps or improves every score with 300 fewer dimensions. Subs2vec adds no significant classification information (check P2). The same number of analogy questions is solved, 1,498.
- **Over-fitting:** the CONF score does not drop below DEV. The train–held-out gap does not grow with dimension, and the 1,200-d model has the smallest gap. Learning curves are still rising at 100% data.
- **SVD/PCA compression** lowers classification and similarity, so it is not used. The thesis model never used it either; it lost in Step 4.

### 14.4 Under-fitting of the source models

| Source model (300-d) | Training tokens | Learned parameters (2 × vocab × 300) | Tokens per parameter | Words seen < 20 times | Diagnosis |
|---|---|---|---|---|---|
| News-W2V | 65,324,912 | 112,611,600 | 0.58 | 56.6% | trained enough; no rare-word collapse |
| Informal-W2V | 3,192,884 | 19,746,000 | 0.16 | 63.9% | under-trained: ~1% analogy in every frequency band → weight 0 for word similarity, kept for classification |
| Subs2vec-w2 | not published (OpenSubtitles 2018) | 2 × 177,376 × 300 | – | – | downloaded; strong for word similarity |

### 14.5 How ε was chosen (analogy tune half; the unseen half was not used)

| ε (thesis share of the cosine) | Analogy, tune half | Analogy, unseen half | Analogy, all | Similarity AUC |
|---|---|---|---|---|
| 0.0005 ← chosen | 9.31% | 8.66% | 8.98% | 0.8914 |
| 0.001 | 9.31% | 8.66% | 8.98% | 0.8914 |
| 0.002 | 9.31% | 8.63% | 8.97% | 0.8914 |
| 0.003 | 9.31% | 8.63% | 8.97% | 0.8913 |
| 0.005 | 9.31% | 8.62% | 8.96% | 0.8913 |
| 0.01 | 9.31% | 8.61% | 8.96% | 0.8911 |

### 14.6 Web app

`webapp/` serves BanglaMeta+ (default), the thesis model and the baselines through a FastAPI backend and a React + three.js frontend:
- 3-D explorer with X/Y/Z axes, Bangla/Banglish input, relations, similarity, analogy;
- a comparison page with live benchmarks.

Every answer is computed from the served model rows. Unknown words are reported, never invented. 169 backend tests check every answer against independent calculations. See [`webapp/README.md`](webapp/README.md).
<!-- BANGLAMETA_PLUS:END -->
