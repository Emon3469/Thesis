# Word2Vec Pre-Evaluation Report (before merging)

Run: Kaggle kernel `emon3469/bangla-w2v-preeval` v2 (private), 76.3 min, 4 CPUs.
Raw outputs: `w2v_output/w2v_results/`. Every number below is copied from those CSVs.
`verify_w2v.py` independently re-derives them: **ALL CHECKS PASSED**. It recomputes domain averages, analogy accuracies from per-file counts, the pair selection, every P1–P6 pass flag, the mini R@1/MRR/rank values from per-text ranks, and the Holm correction.

All screening and P-check numbers use the **DEV half only** (≈12k texts). The CONF half is untouched apart from the 20 mini texts, which were drawn from it and used only for the illustrative 3-dataset test.

---

## Step 1: Download and verify the models

| Model | What it is | Vocabulary | Verified |
|---|---|---|---|
| News-W2V | Our skip-gram Word2Vec (300-d, window 4, min_count 5, neg 10, 5 epochs, seed 42), trained on Potrika news with all evaluation texts removed | 187,686 (65.3M training tokens) | Trained in the run |
| Informal-W2V | Same settings, trained on BanglaBook, SentNoB and hate-speech texts not in the evaluation set | 32,910 (3.2M training tokens) | Trained in the run |
| Subs2vec-w1…w6 | Model B: Zenodo 22230710, OpenSubtitles skip-gram 300-d, windows 1–6 | 177,376 | MD5 OK for all 6 files |
| fastText-cc.bn | Common Crawl + Wikipedia `.vec` reference (whole words only, no subwords) | 1,468,578 | Loaded |

**Model A (alamkhaan) could not be used.** Its Google Drive vector files return 404, so the model does not load. News-W2V replaces it: same domain, a larger corpus (65.3M vs 42.5M words), known settings, and no leakage from the evaluation set.

---

## Step 2: Each model evaluated on its own (DEV half)

### 2a. Word similarity (proxy)
There is no public Bangla word-similarity dataset with human scores. As a proxy, we measure the AUC of same-category vs cross-category cosine over 5 word categories taken from the analogy data: place, person, cardinal, ordinal, date. The fair comparison is on the 124 words that every model covers.

| Model | AUC (124 common words) | AUC (own coverage) | words |
|---|---|---|---|
| **News-W2V** | **0.880** | 0.865 | 178 |
| Subs2vec-w5 | 0.871 | 0.854 | 179 |
| fastText-cc.bn | 0.867 | 0.836 | 211 |
| Subs2vec-w4 | 0.867 | 0.855 | 179 |
| Subs2vec-w3 | 0.863 | 0.849 | 179 |
| Subs2vec-w6 | 0.860 | 0.848 | 179 |
| Subs2vec-w1 | 0.858 | 0.840 | 179 |
| Subs2vec-w2 | 0.858 | 0.848 | 179 |
| Informal-W2V | 0.694 | 0.697 | 125 |

### 2b. Word analogy (Akter et al. 2023)
- Data: the native set of 16,678 questions, scored with 3CosAdd over a top-200k search space.
- Unanswerable (OOV) questions count as wrong.
- "Ans." is the share of questions the model can answer.

| Model | Accuracy | Ans. | Acc. if answerable | Semantic | Syntactic | Mikolov-filtered set |
|---|---|---|---|---|---|---|
| **fastText-cc.bn** | **0.0757** | 0.780 | 0.0971 | 0.0409 | **0.0908** | 0.0990 |
| Subs2vec-w4 | 0.0675 | 0.723 | 0.0933 | **0.0691** | 0.0667 | 0.1096 |
| Subs2vec-w3 | 0.0655 | 0.723 | 0.0906 | 0.0649 | 0.0658 | 0.1077 |
| Subs2vec-w5 | 0.0654 | 0.723 | 0.0904 | 0.0647 | 0.0656 | 0.1233 |
| Subs2vec-w6 | 0.0629 | 0.723 | 0.0870 | 0.0618 | 0.0634 | **0.1321** |
| News-W2V | 0.0600 | 0.686 | 0.0875 | 0.0608 | 0.0597 | 0.0691 |
| Subs2vec-w2 | 0.0585 | 0.723 | 0.0809 | 0.0594 | 0.0582 | 0.0744 |
| Subs2vec-w1 | 0.0456 | 0.723 | 0.0631 | 0.0407 | 0.0478 | 0.0358 |
| Informal-W2V | 0.0077 | 0.336 | 0.0228 | 0.0048 | 0.0089 | 0.0037 |

Absolute accuracies are low for every model under this strict protocol. The comparison between models is what matters.

### 2c. Vocabulary coverage (DEV texts)

| Model | Token cov. D1 news | Type cov. D1 | Token cov. D2 informal | Type cov. D2 |
|---|---|---|---|---|
| **fastText-cc.bn** | **0.986** | **0.837** | **0.961** | **0.826** |
| News-W2V | 0.985 | 0.800 | 0.912 | 0.675 |
| Subs2vec (all windows) | 0.960 | 0.616 | 0.915 | 0.668 |
| Informal-W2V | 0.839 | 0.239 | 0.914 | 0.609 |

### 2d. Downstream macro-F1 (5-fold probe on DEV, 95% bootstrap CI)

| Model | News topic (D1) | Review sent. | Social sent. | Hate | **D2 mean** | **Overall** |
|---|---|---|---|---|---|---|
| **fastText-cc.bn** | **0.879** [0.871, 0.887] | **0.567** | 0.571 | 0.796 | 0.645 [0.633, 0.657] | **0.762** |
| News-W2V | 0.876 [0.867, 0.884] | 0.564 | 0.555 | 0.741 | 0.620 [0.607, 0.633] | 0.748 |
| Subs2vec-w3 | 0.867 | 0.554 | 0.556 | 0.765 | 0.625 | 0.746 |
| Subs2vec-w6 | 0.869 | 0.550 | 0.552 | 0.758 | 0.620 | 0.745 |
| Subs2vec-w5 | 0.866 | 0.544 | 0.552 | 0.762 | 0.619 | 0.743 |
| Subs2vec-w4 | 0.864 | 0.547 | 0.549 | 0.767 | 0.621 | 0.743 |
| Subs2vec-w2 | 0.859 [0.850, 0.867] | 0.562 | 0.549 | 0.766 | 0.625 [0.613, 0.637] | 0.742 |
| Subs2vec-w1 | 0.853 | 0.555 | 0.540 | 0.765 | 0.620 | 0.736 |
| Informal-W2V | 0.814 [0.803, 0.824] | 0.565 | **0.572** | **0.804** | **0.647** [0.635, 0.659] | 0.731 |

**Takeaways from Step 2**
- **Formal specialist:**
  - News-W2V has the best word-similarity proxy, and its news F1 (0.876) is statistically tied with fastText (0.879; the CIs overlap).
  - Analogy accuracy is 0.060, and coverage is the second best.
- **Informal specialist:**
  - Informal-W2V has the best D2 F1 (0.647), but it is tied with fastText (0.645).
  - Its intrinsic quality is poor (analogy 0.008, similarity 0.694), because its training corpus is only 3.2M tokens.
  - Subs2vec (Model B) is a better *general* model than Informal-W2V, but on informal tasks it is no better than News-W2V (D2 0.625 vs 0.620).
- **Subs2vec window:**
  - By the pre-registered rule (best D2), the selected window is **w2** (0.6255).
  - w3 is essentially equal (0.6248). w4–w6 are better on analogy.
- **fastText-cc.bn** is the strongest single model overall. It must be one of the baselines the merged model has to beat.

---

## Step 3: Can merging add information? Pre-merge certificate P1–P6

Two pairs were checked:
- **Planned pair:** News-W2V + Subs2vec-w2.
- **Specialist pair:** chosen by the rule "best D1 + best D2", which gives fastText-cc.bn + Informal-W2V.

| Check | What it tests | Planned: News-W2V + Subs2vec-w2 | Specialist: fastText + Informal-W2V |
|---|---|---|---|
| **P1** concatenation identity | pool(concat) = concat(pool) | max error 0.0 → **PASS** | 0.0 → **PASS** |
| **P2** conditional information I(Y;B\|A), in nats (out-of-fold log-loss drop, 95% CI) | Does B add label information beyond A (and A beyond B)? | B\|A on D2: 0.026 [0.019, 0.033]; A\|B on D1: 0.045 [0.039, 0.051] → **PASS** | B\|A on D2: 0.021 [0.014, 0.027]; A\|B on D1: 0.210 [0.197, 0.223] → **PASS** |
| **P3** SVD risk bound vs P2 gain | Can truncation to k dims provably keep the gain? | Bound 16.7 / 12.9 / 10.0 nats (k = 100/200/300) vs gain 0.021 → **not certified** | 15.9 / 12.2 / 9.2 vs 0.014 → **not certified** |
| **P4** scale balance (A's energy share in the top-k directions) | No single source dominates | 0.515 / 0.507 / 0.504 → **PASS** | 0.501 / 0.523 / 0.542 → **PASS** |
| **P5** imputation of missing words (median held-out cosine vs 95th percentile of random pairs) | Can a union vocabulary be filled reliably? | B→A 0.630 vs 0.392; A→B 0.587 vs 0.322 → **PASS** | B→A 0.407 vs 0.204; A→B 0.802 vs 0.761 → **PASS** (narrow margin) |
| **P6** orthogonal Procrustes alignment (held-out P@1 ≥ 0.5) | Are the two spaces rotations of each other? | 0.441 → **FAIL** | 0.117 → **FAIL** |

Union token coverage:
- Planned pair: 0.982 (A alone 0.979, B alone 0.957).
- Specialist pair: 0.985 (fastText alone 0.984, Informal-W2V alone 0.844).

**What the certificate allows (pre-registered decision rule)**
- **Merging is justified for both pairs (P1 + P2 pass).** In each domain, the second model adds measurable label information that the first lacks, and the CIs exclude 0.
  - One exception within the planned pair: on the hate task, News-W2V adds nothing beyond Subs2vec (I(A|B) = −0.0013).
- **Concatenation with equal-energy scaling** is the certified method. P4 shows that scaling gives each source about half of the variance.
- **A union vocabulary with imputation of missing rows** is allowed (P5).
- **SVD is not certified at any k.**
  - The P3 bound is a worst-case bound. It is about 500 times larger than the gain, so it cannot guarantee that SVD keeps the gain.
  - This does *not* show that SVD hurts. Under the pre-registered rule, CONC+SVD can only be reported as an exploratory ablation, not as the main method.
  - At k = 300, SVD keeps 86–89% of the energy.
- **Averaging after alignment is not justified (P6 fails).**
  - The spaces are not rotations of each other: relative residual about 1.0.
  - So averaging methods (Procrustes/CCA + average) are expected to lose information. They should be tested only as comparison baselines.

---

## Additional test: 3 small datasets × 3 models

**Datasets** (`w2v_kaggle_dataset/mini/`), drawn once from the CONF half with seed 43:
- `domain_1.csv`: 10 news texts, 2 from each of 5 topics (economy, education, entertainment, national, sports).
- `domain_2.csv`: 10 informal texts, 2 from each class: social positive/negative/neutral (SentNoB) and comment hate/not-hate.
- `merged_dataset.csv`: the same 20 texts. Each class is prefixed with its domain.

**Models:** News-W2V (formal specialist), Subs2vec-w2 (Model B, conversational), fastText-cc.bn (reference).

**Method:** identical for every model and dataset.
- Each text in turn is a query. All other texts are ranked by cosine similarity, and the query's same-class partner should be ranked first.
- R@1: share of queries whose partner is ranked first.
- MRR: mean of 1/rank of the partner.
- d′: separation of same-class vs different-class cosines.
- Domain@1: share of queries whose nearest neighbour comes from the same domain.
- p: label-permutation test (10,000 permutations), Holm-corrected over the 9 experiments.

| Dataset | Model | R@1 | MRR | Mean rank | d′ | Domain@1 | p (Holm) |
|---|---|---|---|---|---|---|---|
| Domain 1 (news) | News-W2V | 0.60 | 0.678 | 3.00 | 0.67 | — | 0.020 |
| | **Subs2vec-w2** | **0.70** | **0.758** | **2.60** | **0.83** | — | **0.006** |
| | fastText-cc.bn | 0.60 | 0.699 | 2.90 | 0.41 | — | 0.023 |
| | *chance* | *0.111* | *0.314* | *5.0* | *0* | | |
| Domain 2 (conversational) | News-W2V | 0.10 | 0.335 | 4.80 | −0.06 | — | 1.000 |
| | Subs2vec-w2 | 0.00 | 0.237 | 5.30 | −0.09 | — | 1.000 |
| | **fastText-cc.bn** | **0.20** | **0.341** | 5.40 | −0.10 | — | 1.000 |
| | *chance* | *0.111* | *0.314* | *5.0* | *0* | | |
| Merged (20) | News-W2V | 0.35 | 0.456 | **6.80** | **0.50** | 0.70 | 0.001 |
| | Subs2vec-w2 | 0.35 | 0.442 | 7.10 | 0.35 | 0.70 | 0.001 |
| | **fastText-cc.bn** | **0.40** | **0.486** | 7.05 | 0.42 | **0.80** | 0.001 |
| | *chance* | *0.053* | *0.187* | *10.0* | *0* | *≈0.47* | |

**Comparative analysis**
- **Domain 1 (news):**
  - All three models retrieve the same-topic partner far above chance (R@1 0.6–0.7 vs 0.11), and all are significant after Holm.
  - Subs2vec-w2 is numerically best here, but the difference is 1 text out of 10.
  - Topic is carried by content words, which every model covers well.
- **Domain 2 (conversational):**
  - No model beats chance; all d′ ≤ 0.
  - Sentiment and hate are expressed by a few polarity words inside short noisy texts. An unweighted mean of word vectors is dominated by topic words, so same-*class* pairs are not closer than other pairs.
  - This matches the 12k-text DEV screening, where D2 F1 (0.62–0.65) is far below D1 (0.86–0.88). Informal text is the domain where a single model is weakest and where merging has room to help. P2 confirms this: the D2 gain is significant for both pairs.
- **Merged dataset:**
  - All models are significant. Much of this comes from separating the domains: 70–80% of nearest neighbours fall in the same domain, and D1 pairs are easy.
  - fastText is numerically best (R@1 0.40, Domain@1 0.80). It is the only model with broad coverage of both registers.
  - The specialists do well on one side only. News-W2V has the best mean rank and d′ (driven by the news half).
- **Caution:** with 10–20 texts, one text changes R@1 by 0.05–0.10. These mini results **illustrate** the pattern but cannot rank models.
  - The ranking evidence is the DEV screening above (≈12k texts with CIs).
  - It will be confirmed on the CONF half.

---

## Conclusions of the pre-evaluation (steps 1–3 of 7)

1. **No single Word2Vec model is best on both domains.**
   - The news specialist (News-W2V) and the informal specialist (Informal-W2V) each lead or tie on their own domain and lose clearly on the other: D2 0.620 vs 0.647, and D1 0.814 vs 0.876.
2. **Combining them can add useful information. This is proved before merging (P1, P2)** for both candidate pairs, with significant conditional information in both domains.
3. **The certified merge is equal-energy concatenation over a union vocabulary with imputation** (P1, P2, P4, P5).
   - SVD reduction and averaging after alignment are not certified (P3, P6). They will be run only as ablations or baselines.
4. **The bar to beat is fastText-cc.bn** (Overall 0.762), the strongest single model, together with each specialist on its own domain.

## Honest limitations
- Model A (alamkhaan) could not be loaded (Drive files deleted). It was replaced by News-W2V, a model trained on a larger, cleaned news corpus.
- Word similarity is a **category proxy**, not human-rated similarity (no Bangla human-rated set exists). It is labelled as a proxy throughout.
- Informal-W2V is trained on only 3.2M tokens. More informal text (BD-SHS, BanglaLM) would likely improve it.
- The P3 bound is worst-case and loose. Failing it means "not guaranteed", not "harmful".
- "Top 1%" cannot be proved by a pre-evaluation. It is earned only by beating all baselines on the untouched CONF half (Step 7).

## Next (steps 4–7)
- **Step 4 (DEV half):** build and score on DEV:
  - CONC (equal energy) for both pairs, plus the 3-model concatenation fastText + News-W2V + Informal-W2V;
  - ablations: CONC+SVD (k = 100/200/300), weighted CONC, Procrustes/CCA + average, AEME autoencoder, and continued training of News-W2V on informal text.
- **Step 5:** select one model with the pre-registered rule (best Overall, each domain within 1 point of its specialist).
- **Steps 6–7:** test once on the CONF half against fastText, News-W2V and Informal-W2V, using paired bootstrap with Holm correction.
