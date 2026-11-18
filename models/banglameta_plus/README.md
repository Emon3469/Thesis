# BanglaMeta+: one Bangla word embedding for classification and word similarity (1,200-d)

- **File:** `banglameta_plus_1200d.bin`, in the standard word2vec binary format: 273,531 words × 1,200 dimensions, float32, 1.23 GiB.
  - MD5: `a30a9c38b5a7964fa5ab2d4b44f2c701` (also in `banglameta_plus_1200d.md5`).
  - gensim reads it back exactly.
- **Load:**
  ```python
  from gensim.models import KeyedVectors
  kv = KeyedVectors.load_word2vec_format("banglameta_plus_1200d.bin", binary=True)
  ```
- **Same model in the web app:** `webapp/data/models/merged_plus.f32.npy` + `merged_plus.words.txt`.
- **Earlier 1,500-d version:** archived in `archive_1500d/`, superseded (see below).

## How it is built

The model is built from the frozen thesis model alone: no training, and no other model's vectors. The thesis model is `models/merged_w2v_T_conc`, frozen file MD5 `10aa206f8462c82a7b16cf60054be70f`.

```
c(w) = [ √0.0005 · m_NI(w) ; √0.9995 · s_News(w) ; √0.9995 · s_Subs(w) ]      every row has length 1
```

- **m_NI(w):** the thesis row's News-W2V and Informal-W2V blocks, re-normalised (600-d).
- **s(w):** the similarity view. In each 300-d block, the block mean and its top 4 principal directions are removed; then the blocks are re-weighted News 1.65 / Informal 0 / Subs 1.35. Only the News and Subs blocks are kept (600-d).
- **Classification:** it keeps the thesis features that carry information. The probe standardises every feature, so the small weight costs nothing.
- **Word similarity:** cosine = 0.0005 · thesis cosine + 0.9995 · similarity-view cosine.
- **No Word2Vec was trained with more than 300 dimensions.** Every trained source model is 300-d; the 1,200 dimensions are four 300-d blocks derived from three separately trained models.

How the settings were chosen:
- **D = 4, the block weights and ε = 0.0005** were chosen on a random half of the 16,678 native analogy questions only. The ε values all tied on that half, and ties go to the smaller ε, the rule used from the start.

Code: recipe in `webapp/backend/prepare.py`; export in `webapp/backend/experiments/export_plus.py`; notebook in `eval/banglameta_plus/BanglaMeta_plus_1200d.ipynb` (rebuilds this file byte for byte).

## Why 1,200 and not 1,500, 1,800, 900 or an SVD-compressed size

`eval/dimension_study/Dimension_and_overfitting_study.pdf` scores ten options built from the same rows, all measured the same way:

| Option | DEV F1 | CONF F1 | Analogy | Sim AUC | Train–held-out gap |
|---|---|---|---|---|---|
| **1,200 BanglaMeta+ (this file)** | **77.34** | **77.22** | **8.98%** | **0.8914** | **5.22** |
| 1,500 (previous version) | 77.16 | 77.15 | 8.98% | 0.8913 | 6.06 |
| 1,800 (+ Informal similarity block) | 77.16 | 77.29 | 8.86% | 0.8928 | 6.62 |
| 900 thesis BanglaMeta | 77.12 | 77.17 | 6.00% | 0.819 | 6.44 |
| 500 (PCA/SVD of 1,500) | 75.70 | 75.77 | 9.05% | 0.887 | 7.52 |

- **1,500 → 1,200:** this drops Subs2vec from the thesis part. Subs2vec adds no significant classification information (thesis check P2), and dropping it keeps or improves every score while reducing over-fitting.
- **Over-fitting:**
  - Held-out CONF does not drop (77.34 on DEV, 77.22 on CONF).
  - The train–held-out gap does not grow with dimension.
  - Learning curves still rise at 100% data.
- **Under-fitting:** only Informal-W2V is under-trained: 3.19 M tokens for 19.7 M parameters, with about 1% analogy accuracy in every frequency band. It therefore has weight 0 for word similarity; it is kept for classification, where it adds significant information.

## Evaluation (full report: `eval/banglameta_plus_eval/BanglaMeta_plus_evaluation.pdf`)

| Metric | BanglaMeta+ | Thesis BanglaMeta | fastText-cc.bn | News-W2V |
|---|---|---|---|---|
| DEV Overall macro-F1 | **77.34** | 77.12 | 76.20 | 74.85 |
| CONF Overall macro-F1 (exploratory) | **77.22** [76.49, 77.92] | 77.17 | 76.44 | 74.77 |
| Word analogy, native | **8.98%** | 6.00% | 7.57% | 6.27% |
| Similarity AUC | **0.891** | 0.819 | 0.867 | 0.881 |

- **Rank:** #1 of the six evaluated models on 11 of 18 metrics.
- **CONF classification:**
  - vs fastText: +0.78 [+0.25, +1.31], p = 0.003.
  - vs the thesis model: +0.05 [−0.26, +0.35], a tie.
- **Analogy vs fastText:** 997 vs 761 questions solved by only one of the two, McNemar p = 1e-8.
- **CONF numbers are exploratory:** this model was built after the thesis's one confirmatory CONF test.
