# Execution Plan: Merging Two Domain-Specific Bangla Word2Vec Models

Builds on `RESEARCH_DIRECTION.md`. Changes requested now:
- evaluate several merging methods and pick the best;
- prove mathematically, **before merging**, that the chosen merge should work;
- list what is needed and what the current folder already provides.

---

## 1. The two models you proposed: what they actually are

| | Model A: news (formal) | Model B: conversational (informal proxy) |
|---|---|---|
| Source | `alamkhaan/Bengali-Word2Vec-on-Bangla-News-Dataset` (GitHub) | Zenodo record 22230710, "Bengali word2vec embeddings trained on OpenSubtitles" (Grim & Buchanan, 2026) |
| Corpus | Prothom Alo, Jugantor, Bangladesh Pratidin, Daily Star: 3,248,295 sentences, 42,466,428 words | OpenSubtitles 2018 (film and TV subtitles); size not stated |
| Format | gensim `word2vec2.model` **plus two extra files on Google Drive** (needed to load) | 60 CSV files (`bn_{dim}_{window}_{cbow|sg}_wxd.csv.bz2`): dims 50–500, windows 1–6, CBOW and skip-gram; 10 GB total |
| Size | 250-d, 461,498 words | 300-d files ≈ 218 MB each (compressed); vocabulary size not stated |
| Training settings | Not stated (skip-gram/CBOW, window, epochs unknown) | Per file; other settings not stated |
| Reported accuracy | **None** (only example neighbours) | **None** on the record |
| License | MIT | CC BY 4.0 |

**Assessment.**
- Both are usable, but **neither has a reported accuracy**, so neither can be called "best" yet. They must be screened.
- Model B is a **proxy**. Subtitles are scripted speech, not social media: little slang spelling, no Banglish, possibly translated dialogue. It may still help Domain 2, but it must be tested against a model trained on real informal text.
- Subtitle embeddings in general perform comparably to Wikipedia ones on standard benchmarks (van Paridon & Thompson 2021, the original subs2vec paper; see §8). That supports trying Model B, not choosing it blindly.

---

## 2. What the current folder already gives us

| Resource | Size (measured) | Enough for |
|---|---|---|
| Potrika news, 7 files | **285,715 articles, 67.4M words** | Training our own news Word2Vec, on a corpus **larger** than Model A's 42.5M words |
| BanglaBook reviews | 158,065 rows, 2.77M words | Evaluation; small for training |
| SentNoB social comments | 15,728 rows, 0.24M words | Evaluation |
| Hate speech comments | 30,000 rows, 0.44M words | Evaluation |
| **Informal total** | **≈204k texts, 3.45M words** | Enough to **evaluate** and to scale the eval set to ~24k texts (power analysis). **Too small to train a strong informal Word2Vec alone**: published Bangla Word2Vec corpora are 20M–240M words |
| Eval set builder, pre-registered protocol, verification scripts | — | Reused unchanged |
| Proof base (`eval/preeval/MATH_PROOF.md`) | — | Extended to word level (§4) |
| fastText cc.bn | sentence vectors only | Reference; word-level use needs the model file |

**Machine limits:** C: has **481 MB free**, so every model goes on D: (80 GB free). RAM is 8 GB, so word matrices will be restricted to the vocabulary we actually need. Heavy steps can run on Kaggle (≈29 GB RAM).

---

## 3. What I need from you

| # | Item | Why | Where to put it |
|---|---|---|---|
| 1 | **Model A**: `word2vec2.model` from the repo **and both Google Drive files** linked in its README ("First File", "Second File") | The model will not load without them. Google Drive downloads need a browser | `D:\Thesis\models\A_news_w2v\` |
| 2 | **Model B**: the six skip-gram 300-d files `bn_300_1_sg … bn_300_6_sg` (≈1.3 GB) | Screening across window sizes. I can download these from Zenodo myself if you approve | `D:\Thesis\models\B_subs2vec\` |
| 3 | **Bangla analogy set** (Akter et al. 2023, 16,678 analogies) and, if available, the Bülbül et al. 2024 question sets | Intrinsic evaluation the professor asked for | `D:\Thesis\eval_resources\` |
| 4 | **Optional: more informal text**: e.g. the BD-SHS social-media data (its authors trained an embedding on 1.47M comments) or the BanglaLM corpus (social media + blogs) | Lets us train a real informal Word2Vec to test against Model B | `D:\Thesis\corpora\informal\` |
| 5 | **Professor's agreement** on: Case A; subtitles as an informal proxy; contextual models (mE5, BanglaBERT) as baselines only | Avoids redoing work later | — |
| 6 | **Free space on C:** (≈481 MB left) | Python and some tools still write there | — |

If items 3 or 4 are not publicly downloadable, I can search for them and draft a request email to the authors.

---

## 4. Mathematical proof **before** merging: the pre-merge certificate

Proofs that hold for any pair are not enough; we need checks that a **specific** pair passes before we build the merged model.
Each check below is computed from the two source matrices and the DEV half only.

| # | Statement (word level) | What is proved | Check computed before merging | Pass condition |
|---|---|---|---|---|
| P1 | Concatenation is lossless | Mean-pooling is linear, so pool(concat) = concat(pool); Theorem 1 of `MATH_PROOF.md` then applies: the merged model cannot lose in the population | Exact identity check on the eval set | Error < 1e-12 |
| P2 | Gain = conditional information | Merged gain = I(Y; B \| A) ≥ 0, strictly > 0 iff B adds label information | Out-of-fold log-loss of A vs [A;B] on DEV, with bootstrap CI | CI excludes 0 on each domain's tasks |
| P3 | **SVD loss bound** | For truncation to k dims, ‖x − P_k x‖² summed over words = Σ_{i>k} σ_i² (Eckart–Young). For an L-Lipschitz loss and probe weights ‖W‖ ≤ Λ, the risk increase is at most L·Λ·E‖x − P_k x‖, and averaging over a sentence cannot increase this residual (triangle inequality) | Singular values of the concatenated matrix → bound for k = 100, 200, 300 | Bound below the P2 gain, otherwise SVD may cancel the gain |
| P4 | Scale balance | With per-source equal-energy scaling, each source contributes exactly half of the total variance; without it the larger-norm source dominates the leading SVD directions | Norm ratio and variance share of each source in the top-k SVD directions | Each source's share within 40–60% after scaling |
| P5 | Vocabulary imputation error | For union vocabularies, a linear map fitted on shared words imputes missing rows; its error on held-out shared words bounds the error added for missing words | Held-out imputation error and OOV rate per domain | Imputation error well below the typical distance between unrelated words |
| P6 | Alignment validity | Orthogonal Procrustes has a closed-form optimum; its residual measures how far the two spaces are from a rotation of each other | Procrustes residual on shared words | Low residual → averaging after alignment is justified; high → keep concatenation |

**Pre-registered decision rule:** a pair is merged only if P1–P2 pass. SVD is used only at a k where P3 passes. Averaging methods are used only if P6 passes.

**Honest limit:** these prove *when the merge must help* and *how much SVD can lose*. They cannot prove "top 1%". That is decided by the selection rule and the held-out confirmation (Stages 4–5).

---

## 5. Model choice: my recommendation

| Domain | Candidates to screen | Why |
|---|---|---|
| **Formal (news)** | **A**: alamkhaan news Word2Vec (250-d); **A′**: our own Word2Vec on Potrika (67.4M words, 300-d skip-gram, window 4, the best Bangla setting reported by Rahman 2020); bnlp Wikipedia Word2Vec (100-d) as a weaker reference | A′ is larger, documented, and lets us remove eval texts from training. A has unknown settings |
| **Informal** | **B**: subs2vec OpenSubtitles (6 skip-gram 300-d windows); **B′**: our Word2Vec on informal text (3.45M words; larger if item 4 arrives), also as continued training from B; BD-SHS embedding if public | Tests whether a subtitle proxy beats real informal text |
| **Reference** | fastText cc.bn (subword) | Handles unknown words; the strongest static source in the pre-evaluation |

The best formal and best informal source are chosen by the pre-registered screening (Stage 1), not in advance.

---

## 6. Realigned stages

| Stage | What happens | Data | Output |
|---|---|---|---|
| 0. Acquire and verify | Load each model, check checksums, dimensions and vocabulary size | Items 1–2 | Model inventory |
| 1. Source screening | Each candidate alone: analogy accuracy, word similarity, OOV rate per domain, probe F1 on the 4 tasks | DEV half only | Best formal + best informal source |
| 2. **Pre-merge certificate** | Checks P1–P6 for the top pair(s) | DEV half | Go / no-go per pair and per method |
| 3. Merge methods | CONC (equal energy); CONC + SVD at k = 100/200/300; weighted CONC; Procrustes or CCA + average; autoencoder (AEME); continued training | DEV half | Score table |
| 4. Selection | Pre-registered rule: best two-domain average with each domain within 1 point of its specialist | DEV half | One merged model |
| 5. Confirmation | Selected model vs both specialists and baselines; paired bootstrap, Holm; power-sized (~12k texts per half, built from the local data) | CONF half, used once | Main result |
| 6. Niche extension | Medical and dialect first (most public data) | Niche corpora | Thesis chapter / paper section |

**Leakage rule:** texts in the evaluation set are removed from any corpus we train on. Model A's corpus overlaps the same newspapers as Potrika and cannot be cleaned, so this is reported as a limitation.

---

## 7. Answers in one line each

- **Is the current data enough?** For evaluation, yes. For training a strong informal Word2Vec, no: 3.45M words is small.
- **Are the two proposed models the best?** Unknown: neither reports any accuracy. Both enter screening alongside our own controlled models.
- **Can the merge be proved before merging?** Its *conditions* can be (P1–P6). "Top 1%" cannot be proved; it is earned in the confirmation stage.

---

## 8. Sources (SciSpace)

- van Paridon & Thompson 2021, subs2vec: Word embeddings from subtitles in 55 languages — https://scispace.com/papers/subs2vec-word-embeddings-from-subtitles-in-55-languages-s0kvcl8vxo
- Romim et al. 2022, BD-SHS (embedding trained on 1.47M social-media comments) — https://scispace.com/papers/bd-shs-a-benchmark-dataset-for-learning-to-detect-online-3bikwps9
- Kowsher et al. 2021, BanglaLM corpus — https://scispace.com/papers/banglalm-data-mining-based-bangla-corpus-for-language-model-4w36c9hkc5
- Rahman 2020, Robust and Consistent Estimation of Word Embedding for Bangla — https://scispace.com/papers/robust-and-consistent-estimation-of-word-embedding-for-59iulpo7yp
- Akter et al. 2023, On Evaluation of Bangla Word Analogies — https://scispace.com/papers/on-evaluation-of-bangla-word-analogies-3aotqjyj
- Raunak et al. 2019, Effective Dimensionality Reduction for Word Embeddings — https://scispace.com/papers/effective-dimensionality-reduction-for-word-embeddings-1ht9cklr8o
- Gohourou & Kuwabara 2023, Performance Impact of Dimensionality Reduction on Word Vectors — https://scispace.com/papers/investigating-the-performance-impact-of-dimensionality-16og95h996
- Seyler & Zhai 2020, Methods for Domain-Aware Word Embeddings — https://scispace.com/papers/a-study-of-methods-for-the-generation-of-domain-aware-word-1k8dc2ect0
- Sarma et al. 2018, Domain Adapted Word Embeddings — https://scispace.com/papers/domain-adapted-word-embeddings-for-improved-sentiment-56fsw2ruzt
