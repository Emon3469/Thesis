# Pre-Analysis: Merging Domain-Specific Bangla Embedding Models into a Meta-Embedding

**Scope.** Before building the full meta-embedding (formal + informal anchors, then healthcare, legal-Khatian and dialect
niches), test on existing data whether merging current Bangla encoders gives a real, measurable gain, and which design
to carry forward. Everything below comes from runs in `D:\Thesis\eval` and was re-checked by `verify_preeval.py`.

---

## 0. Decision summary

| Question | Answer from the evidence |
|---|---|
| Does a 2-model merge beat the best single model? | **Not shown.** The pre-registered criterion (Overall gain with 95% CI > 0, no domain worse by > 1 pt) was **not met** on the held-out half. |
| Is there a merge worth carrying forward? | **Yes: CONC-std of `mE5-base` + `fastText-cc.bn`.** It was the best pair on DEV (+1.5 pts over its better member) and held up on CONF (74.6 vs 74.2 for mE5-base alone), but within noise. |
| Which single encoders are the strongest anchors? | `mE5-base`, `mE5-large` (formal news), `L3Cube-BnSBERT` (informal). **Not** BanglaBERT/BanglaBERT-large as frozen embeddings. |
| Why no confirmed gain? | The study is **underpowered**. The minimum detectable effect is 2.3 Overall F1 points (6.7 on Domain 2). Plausible fusion gains are 1–2 pts. |
| What must change before the main study? | (1) Enlarge the evaluation set ~5–11× (power analysis in §6.6). (2) Use repeated splits. (3) Add retrieval/STS tasks. (4) Add a real Banglish test set. |
| **Go / No-go** | **Go, with conditions.** Carry CONC-std (mE5-base + fastText) as the baseline merge. Do not claim superiority until the powered study confirms it. |

---

## 1. Research questions and pre-registered hypotheses

| ID | Hypothesis | Test | Outcome |
|---|---|---|---|
| H1 | Existing Bangla encoders differ in quality by domain (a formal and an informal specialist exist) | Stage 1, DEV | **Supported.** Best D1 = mE5-large (86.8), best D2 = L3Cube-BnSBERT (63.6) |
| H2 | Merging two complementary encoders beats the best single encoder on both domains | Stages 3–5, CONF | **Not supported** at this sample size (Δ Overall −0.7 pts, 95% CI [−2.2, +0.9]) |
| H3 | Fusion gain grows as the pair's similarity (CKA) falls | Stage 2, 15 pairs | **Direction as expected, not significant** (Spearman ρ = −0.34, p = 0.22) |
| H4 | Gains are information, not just extra dimensions | Dimension control | **Supported.** CONC(mE5-base, Gaussian noise) = 72.7 < mE5-base alone 74.4 |

---

## 2. Claim audit of the starting brief (`eval.txt`)

| Claim | Status | Evidence |
|---|---|---|
| `Kowsher/BnVec`, `Kowsher/Bangla-Fasttext`, `sagorbrur/bnlp` model IDs | **Do not exist** on Hugging Face | Hub lookups returned not-found |
| bnlp Word2Vec is "300d, news + Wikipedia" | **Incorrect** | `sagorsarker/bangla_word2vec` model card: 100 d, Wikipedia only |
| BanglaEmbed-MSE / -MNR (arXiv:2411.15270) | **Not released** under these names | Hub search finds `kazalbrur/bangla-embed-e5-small(-banglish)` and `Srimon12/bangla-embed-e5-small-mnrl` |
| `kazalbrur/bangla-embed-e5-small-banglish` acc@1 = 0.96 on Banglish | **Self-reported** on the model card; not independently tested here | Our eval set is < 1.5% romanized, so it cannot test this claim |
| BanglaBERT is "undisputed SOTA" | **True for fine-tuning, false as a frozen embedding** | Frozen mean-pooled BanglaBERT 66.0, BanglaBERT-large 59.7 Overall, both below mE5 (74.4) |
| "Near-ceiling 92–97% accuracy" on these domains | **Not reproduced with frozen embeddings** | Best frozen news F1 ≈ 87–89%. Informal 3-class sentiment ≈ 55–66% |
| "> 85% of Bangla pretraining corpora come from these two domains" | **Unsourced** | No citation given; do not use in the thesis without one |

**Implication:** cite only the verified models and numbers. The brief's accuracy figures are mostly fine-tuned-classifier
results and are not comparable to frozen-embedding quality.

---

## 3. Literature positioning (38 papers in the SciSpace export)

| Line of work | Representative papers (export) | Level | Sources fused | Domain-aware | Bangla |
|---|---|---|---|---|---|
| Unsupervised meta-embedding | Learning Word Meta-Embeddings (2016); Frustratingly Easy Meta-Embedding (2018); AEME by Autoencoding (2018); Locally Linear Meta-embedding (2018); Geometric Meta-Embeddings (2020); Unsupervised Weighted Concatenation (2022) | word | static | no | no |
| Task/supervised meta-embedding | Dynamic Meta-Embeddings (2018); Word Prisms (2020); Adversarial Feature-based Meta-Embeddings (2020) | word → sentence model | static | partly | no |
| Domain-specific fusion | Fusing Vector Space Models for Domain-Specific Applications (2019); Task-oriented Domain-specific Meta-Embedding (2020) | word | static | **yes** | no |
| Survey | A Survey on Word Meta-Embedding Learning (2022) | — | — | — | — |
| Bangla static embeddings | Towards Bengali Word Embedding (2020); Word2Vec tuning for Bangla (2020/22); Enhanced Neural Word Embedding / BanglaFastText (2022); Domain-related Corpus (2023); Comprehensive Study of Bengali Embedding Models (2024) | word | single model | partly | **yes** |
| Bangla sentence embedding | Bangla SBERT via multilingual KD (2024) | sentence | single model | no | **yes** |
| Niche-domain Bangla benchmarks | BanglaMedNER (2023); Vashantor (2025); ANCHOLIK-NER (2025); KhatianDoc (2026); MTS-Dialog-Bangla (2026) | task | — | **yes** | **yes** |

**Gap, as the export supports it:**
- Meta-embedding work is almost entirely **word-level and static**. The 2022 weighted-concatenation paper and the 2022 survey list contextualised/sentence-level meta-embeddings as out of scope or open.
- No paper in the set fuses **contextual and static encoders at sentence level for Bangla**, or evaluates a merge **across formal vs informal Bangla**, or carries it to **niche domains**.
- That combination is the defensible novelty. The 2019 domain-fusion paper also reports that individual domain-specific embeddings underperform general ones while their combination helps. This matches our Stage 1 result (general mE5 > Bangla-specific single models).

---

## 4. Data audit

| Item | Finding | Risk / action |
|---|---|---|
| Size | 2,100 texts: 1,050 news (7 topics × 150), 1,050 informal (BanglaBook 360, SentNoB 360, Hate 330) | Too small for 1–2 pt effects (§6.6) |
| Cleaning | NFC, zero-width removal, near-duplicate removal within and across sources (47 cross-source duplicates), conflicting-label duplicates dropped | OK |
| Banglish | < 1.5% romanized (5 texts) | Domain 2's "Banglish" claim is **untested**. Add a human-typed Banglish set |
| Truncation | 50–78% of news articles exceed 256 tokens; models see only the opening | Report it; consider chunk-and-pool in the main study |
| Label ambiguity | Neutral class is the main confusion (44% of SentNoB neutral predicted negative) | Report per-class; consider a binary-polarity variant |
| Possible pretraining exposure | BanglaBERT's Bangla2B+ and mE5's web data may include the same news sites | Cannot be ruled out. Note as a threat; prefer post-2023 or held-out sources for the niche study |
| OOV (static) | bnlp-word2vec misses 10.1% of informal words vs 3.9% of news words; fastText 3.4% vs 1.2% | Confirms the informal-text vocabulary gap that motivates subword/contextual fusion |

---

## 5. Protocol (fixed before the confirmation half was used)

1. **Two halves.** Each task is split once, stratified 50/50 (seed 42), into **DEV** (all choices) and **CONF** (touched once).
2. **Frozen encoders, linear probe.** Standardise, then `LogisticRegressionCV` with C ∈ {1e-4 … 10}, chosen by an inner 3-fold CV inside each outer training fold. Outer: stratified 5-fold.
   - The C grid was widened after an earlier run hit its lower edge in 68% of folds. Now 21.8% of folds hit an edge.
3. **All fusion parameters are fitted on training folds only** (scalers, PCA, Procrustes rotation, autoencoder).
4. **Metrics.**
   - D1 = news macro-F1; D2 = mean macro-F1 of the 3 informal tasks; Overall = (D1 + D2)/2.
   - Secondary metric: kNN (k = 10, cosine).
5. **Stages.**
   - (1) Rank 11 sources.
   - (2) CONC-std for all 15 pairs among the top 6.
   - (3) 9 fusion methods on the top 3 pairs plus the specialist pair.
   - (4) Selection rule: best Overall with D1 and D2 each ≥ best single − 1 pt.
   - (5) Confirmation with paired bootstrap (2,000) and Holm correction.
6. **Controls.**
   - Label permutation, which must give chance. Result: 0.142 / 0.389 / 0.277 / 0.473 vs chance 0.143 / 0.333 / 0.333 / 0.5.
   - The dimension control.
   - A TF-IDF floor.

Fusion methods tested:
- CONC-L2, CONC-std (z-score + 1/√d per source), CONC-α (0.25 / 0.75)
- AVG-L2 (zero-pad + mean, as in Frustratingly Easy Meta-Embedding)
- PCA-CONC (as in the 2019 domain-fusion paper)
- Procrustes-AVG (as in Geometric Meta-Embeddings)
- AAEME (as in AEME by Autoencoding)
- Late-fusion ENSEMBLE, as a reference that does not produce one embedding

---

## 6. Results

### 6.1 Stage 1 — single sources (DEV, probe macro-F1 %)

| Source | D1 news | D2 informal | Overall | kNN Overall |
|---|---|---|---|---|
| **mE5-base** | 85.6 | 63.3 | **74.4** | 72.0 |
| mE5-large | **86.8** | 60.8 | 73.8 | 71.2 |
| fastText-cc.bn | 85.9 | 60.1 | 73.0 | 69.9 |
| L3Cube-BnSBERT | 80.7 | **63.6** | 72.1 | 66.5 |
| BnEmbed-e5s-banglish | 81.9 | 61.3 | 71.6 | 70.1 |
| BnEmbed-e5s | 82.4 | 60.5 | 71.5 | 69.8 |
| BanglaSBERT | 79.3 | 56.3 | 67.8 | 61.3 |
| BanglaBERT | 77.1 | 54.9 | 66.0 | 56.1 |
| TF-IDF floor | 78.3 | 53.2 | 65.7 | 65.4 |
| bnlp-word2vec | 80.8 | 50.4 | 65.6 | 60.6 |
| BanglaBERT-large | 68.0 | 51.4 | 59.7 | 49.5 |

Three sources do not beat the TF-IDF floor: BanglaBERT, bnlp-word2vec and BanglaBERT-large. Mean-pooled ELECTRA
discriminators are poor frozen sentence embeddings.

### 6.2 Stage 2 — pairs (DEV, CONC-std; top 5 of 15)

| Pair | Overall | Gain over better member | mean CKA |
|---|---|---|---|
| mE5-base + fastText-cc.bn | **75.9** | +1.5 | 0.46 |
| mE5-base + L3Cube-BnSBERT | 75.7 | +1.3 | 0.59 |
| mE5-large + L3Cube-BnSBERT | 75.4 | +1.6 | 0.61 |
| L3Cube-BnSBERT + BnEmbed-e5s-banglish | 75.1 | +3.0 | 0.68 |
| mE5-large + fastText-cc.bn | 75.0 | +1.2 | 0.47 |

Redundant pairs gain nothing:
- mE5-base + mE5-large: CKA 0.87, gain −0.2
- BnEmbed-e5s + its Banglish variant: CKA 0.94, gain +0.3

### 6.3 Stage 3 — fusion methods (DEV, best per method on the leading pair)

| Method (mE5-base + fastText) | D1 | D2 | Overall |
|---|---|---|---|
| AVG-L2 | **88.7** | 64.8 | **76.7** (selected) |
| CONC-std / CONC-α | 87.2 | 64.7 | 75.9 |
| CONC-L2 | 87.2 | 64.6 | 75.9 |
| AAEME | 88.0 | 63.4 | 75.7 |
| ENSEMBLE (late fusion) | 85.9 | 60.9 | 73.4 |
| Procrustes-AVG | 83.5 | 60.0 | 71.8 |
| PCA-CONC | 80.8 | 57.0 | 68.9 |

**Notes:**
- CONC-α equals CONC-std for the probe. The probe re-standardises features, so source weights only matter for similarity use (kNN/retrieval).
- PCA and Procrustes reductions hurt. This matches the survey's finding that SVD-style reduction loses accuracy.

### 6.4 Stage 5 — confirmation (CONF half, macro-F1 %, 95% bootstrap CI)

| Configuration | D1 | D2 | Overall |
|---|---|---|---|
| single: mE5-large | 85.8 [82.7, 88.6] | **64.6** [60.4, 68.3] | **75.2** [72.6, 77.5] |
| CONC-std (mE5-base + fastText) | 86.9 [83.9, 89.5] | 62.3 [57.9, 66.2] | 74.6 [71.9, 76.9] |
| single: mE5-base | **87.7** [84.8, 90.3] | 60.8 [56.6, 64.5] | 74.2 [71.7, 76.4] |
| **SELECTED: AVG-L2 (mE5-base + fastText)** | 85.6 [82.6, 88.5] | 61.5 [57.1, 65.4] | 73.5 [70.8, 76.0] |
| ENSEMBLE (same pair) | 86.3 | 60.5 | 73.4 |
| single: L3Cube-BnSBERT | 81.7 | 62.3 | 72.0 |
| single: fastText-cc.bn | 84.3 | 58.3 | 71.3 |
| TF-IDF floor | 79.1 | 52.5 | 65.8 |

| Selected vs best single | Δ (pts) | 95% CI | p (Holm) | MDE (80% power) |
|---|---|---|---|---|
| D1 vs mE5-large | −0.2 | [−2.4, +2.1] | 1.00 | 3.2 |
| D2 vs L3Cube-BnSBERT | −0.8 | [−5.7, +3.8] | 1.00 | 6.7 |
| Overall vs mE5-base | −0.7 | [−2.2, +0.9] | 1.00 | 2.3 |

### 6.5 What the confirmation tells us

- **The DEV winner fell back:** AVG-L2 dropped from 76.7 (DEV) to 73.5 (CONF). It was picked from 36 configurations, and that search inflated its DEV score (the winner's curse). This is why the held-out half exists.
- **The simpler CONC-std is more stable** (75.9 → 74.6). It is the safer default to carry forward.
- **Single-model rankings also move between halves** by up to ±1.7 pts: mE5-large +1.4, fastText −1.7. Half-to-half noise is as large as the fusion gains being sought.

### 6.6 Power analysis — how big must the main study be?

This is a **pilot-based** power analysis: the 2,100-text study estimates the variance, which then sets the size of
the main study. This is the procedure recommended for NLP by Card et al. (2020), *With Little Power Comes Great
Responsibility* (https://scispace.com/papers/with-little-power-comes-great-responsibility-41dj40g7qr), and for
probing/embedding experiments by Zhu et al. (2022), *On the data requirements of probing*
(https://scispace.com/papers/on-the-data-requirements-of-probing-2q5qnbtj). The significance-test choice follows
Dror et al. (2018) (https://scispace.com/papers/the-hitchhiker-s-guide-to-testing-statistical-significance-5azh1weopx).

MDE = 2.80 × SE(Δ) at α = .05 (two-sided) and power .80. SE shrinks with √n, so detecting a smaller effect needs
(MDE / target)² times more texts in the confirmation half.

| Domain | Current MDE | Needed for a 2-pt effect | Needed for a 1-pt effect |
|---|---|---|---|
| Overall | 2.3 pts | 1.3× (≈ 1,400 CONF texts) | 5.3× (≈ 5,600) |
| D1 news | 3.2 pts | 2.6× (≈ 1,400 news) | 10.2× (≈ 5,400 news) |
| D2 informal | 6.7 pts | 11.2× (≈ 5,900 informal) | 44.9× (≈ 23,600 informal) |

**Recommendation:** target **≈ 12,000 texts per half** (24,000 total; ≈ 6,000 news + 6,000 informal in each). That
detects a 2-pt effect on each domain and a 1-pt effect Overall.
- The sources have enough: after cleaning, BanglaBook 90,096, SentNoB 13,004 and Hate Speech 27,999. Potrika has 7 topic files of up to 40k articles each (by file name; count after cleaning still to be measured).
- Measured encoding cost on a Kaggle T4 was 31–88 s per model for 2,100 texts. At 24k texts that is about 6–17 min per model.

---

## 7. Threats to validity

1. **Low power** (§6.6): the main reason H2 is undecided rather than rejected.
2. **Selection bias**: mitigated by the DEV/CONF split; confirmed in practice (AVG-L2's fall).
3. **Single split**: one 50/50 split. Use repeated splits (e.g. 5 × 2 CV) in the main study.
4. **Frozen-probe only**: no retrieval/STS/clustering-as-primary tasks yet. A merge can help retrieval even when a re-standardising probe shows no gain.
5. **Truncation** of news at 256 tokens (50–78% of articles).
6. **Banglish untested** (< 1.5% of texts).
7. **Possible pretraining exposure** of news sources.
8. **Label ambiguity** in the neutral class.

---

## 8. Plan for the main study

1. **Scale the evaluation set** to ≈ 24k texts with the same builder (`build_eval_set.py`, larger `PER_CLASS`). Add:
   - a human-typed Banglish set (romanized ↔ native pairs)
   - a Bangla STS/retrieval set (e.g. the MTEB Bengali tasks cited on the BanglaEmbed model card)
2. **Repeated splits** (5 × 2 CV), keeping the pre-registered selection rule and success criterion.
3. **Candidate fusions:**
   - CONC-std as the baseline merge
   - AAEME and a learned attention fusion (the DME family) as trainable merges
   - sentence-level unsupervised attention meta-embedding (the 2022 sentence-level line) as the closest prior work
4. **Anchors:**
   - formal: mE5-base / mE5-large
   - informal: L3Cube-BnSBERT, and BnEmbed-e5s-banglish if the Banglish set shows its advantage
   - subword: fastText
   - Drop frozen BanglaBERT variants as embedding sources.
5. **Niche extension:** use the merged anchor to align healthcare (BanglaMedNER, MTS-Dialog-Bangla), legal (KhatianDoc text fields) and dialect (Vashantor, ANCHOLIK-NER) embeddings. Report the same D1/D2 non-inferiority checks so the niche gain does not cost the general domains.

**Thesis framing:** the evidence supports "domain-bridging meta-embedding" and a measured, complementarity-informed choice of
sources. It does **not yet** support "complementarity predicts gain" (H3: ρ = −0.34, p = 0.22) or "beats the best single
model" (H2). Keep those as hypotheses for the powered study.

---

## 9. Reproducibility

| Item | Location |
|---|---|
| Eval set + datasheet | `../kaggle_dataset/` (seed 42) |
| Embeddings, 5 sources | `../kaggle_output/results/emb/` (Kaggle `notebook85fb1b3325` v4) |
| Embeddings, 6 candidates | `../cand_output/cand_results/emb/` (Kaggle `bangla-candidate-encoders` v1, T4, 5.8 min) |
| Pre-evaluation | `../preeval.py` → `preeval/*.csv`, `predictions.parquet`, `manifest.json` (≈ 22 min on 12 CPU threads) |
| Independent checks | `../verify_preeval.py`: **all passed**. 60,900 saved predictions; 38 DEV and 8 CONF configurations recomputed; DEV/CONF disjoint; selection rule reproduced; a second full run gave identical numbers |
| Figures | `preeval/fig/` |
| Model commits | recorded in each run's manifest |
