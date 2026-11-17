# Research Direction Check and Realigned Plan

Sources: `professor_instruction.txt`, our pre-evaluation (`eval/preeval/`), and SciSpace searches (links in §8).

---

## 1. Verdict: are we on the right track?

**Partly. The framing is right, but the level of the models is wrong for your professor's instruction.**

| Aspect | Professor's instruction | What we did | Verdict |
|---|---|---|---|
| Unit being merged | Two **Word2Vec word-embedding matrices** (V × d₁ and V × d₂) | A **sentence encoder** (mE5-base) with fastText sentence vectors | **Off-track.** mE5 is a transformer sentence model, not Word2Vec |
| Merge method | Concatenation, then SVD to k dims; alignment + combination; joint/continued training | 9 sentence-level fusions (CONC, AVG, PCA, Procrustes, AAEME…) | Methods transfer; must be redone on word matrices |
| Vocabulary alignment | Required (intersection or union + imputation) | Not needed for sentence vectors, so not done | **Missing** |
| Evaluation | Word similarity, analogy, **OOV coverage**, plus downstream tasks, same splits for all | Downstream classification only | **Intrinsic tests missing** |
| Baselines | Word2Vec vs fastText vs a contextual model (BanglaBERT) | Many contextual + fastText + bnlp Word2Vec | Good; keep contextual models as **baselines**, not as sources |
| Two domains (news vs social media) | Yes: "Case A, independently trained" | Yes: formal news vs informal UGC | **On-track** |
| Rigour (splits, CIs, power) | "Same train/test splits and metrics" | Pre-registered DEV/CONF, bootstrap, Holm, power analysis | **On-track, and stronger than asked** |

**Answer to the professor's closing question:** our two models are **Case A**: trained on **different corpora** (news vs social media/reviews). His Case A guidance is that concatenation + SVD is valid once vocabularies are aligned, and that we must show the merge improves downstream results.

**What carries over unchanged:**
- the evaluation set builder and the 4 local datasets;
- the pre-registered protocol (DEV/CONF split, controls, paired bootstrap, Holm);
- the power analysis (~12,000 texts per half);
- the proof (`eval/preeval/MATH_PROOF.md`). Theorems 1–5 hold for word matrices too, because averaging word vectors over a sentence is linear: averaging concatenated word vectors gives exactly the concatenation of the averaged vectors.

**One new caution from the proof:** Theorem 1 ("merging never loses") does **not** cover the SVD step. SVD keeps the directions of largest variance, not the most useful ones. Our sentence-level PCA-CONC lost 7 F1 points (75.9 → 68.9). The word-level SVD must be tested at several k, with retained variance reported. This is the professor's "information loss" row.

---

## 2. Which Bangla domains already have Word2Vec / fastText models

Found through SciSpace. Repositories in the professor's file were checked and all exist.

| Domain | Existing word-embedding resources | Evidence |
|---|---|---|
| **News** (formal) | Word2Vec on 5 years of daily newspapers (Ahmad & Amin 2016); Word2Vec/GloVe/fastText on 180M words (Hossain & Hoque 2020); Word2Vec/fastText/GloVe on 240M words from ~900k articles (Bülbül et al. 2024); `alamkhaan/Bengali-Word2Vec-on-Bangla-News-Dataset` (250-d, 461,498 words, per the professor's file); `soham96/Bangla2Vec` (news + Wikipedia) | Rich. **Best-covered domain** |
| **Wikipedia** (encyclopedic) | bnlp Word2Vec (`sagorsarker/bangla_word2vec`, 100-d, Wikipedia only) | Model card checked |
| **Web / Common Crawl** | fastText `cc.bn.300` (`facebook/fasttext-bn-vectors`) | Used in our pre-evaluation |
| **Large mixed corpora** | BanglaFastText skip-gram/CBOW (Kowsher et al. 2022); BengFastText from 250M articles (Karim et al. 2020) | Toolkit papers |
| **Social media / sentiment** (informal) | Word2Vec for Bangla sentiment, 83.79% accuracy (Sumit et al. 2018); Word2Vec/doc2vec on 10,500 Facebook posts, 77.34% with BiLSTM (Hoque et al. 2026) | **Thin.** No large public social-media Word2Vec found |

**Implication:** Domain 1 has ready-made Word2Vec models. Domain 2 does not have a good public one, so the plan below trains one on our informal corpora (the professor's "train your own Word2Vec on a domain-specific corpus" option).

---

## 3. Niche domains: weak performance and little or no word-embedding support

| Niche domain | Best reported results found | Word-embedding status | Why it suits the thesis |
|---|---|---|---|
| **Medical / healthcare** | Bangla medical NER: best fine-tuned F1 0.596 (Abdullah & Maruf 2026); BanglaMedNER BiLSTM-CRF macro-F1 75% (export); multi-task triage F1 0.631 (FastMedBangla 2026) | One unsupervised "medical nearest-word embedding" paper (Mandal et al. 2024); no standard medical Word2Vec | Clear accuracy gap; specialised vocabulary means high OOV |
| **Legal** (incl. Khatian land records) | Legal document classification: 67.96% accuracy, F1 0.65 (Dhrubo et al. 2026); KhatianDoc: six multimodal LLMs scored zero on 39.3% of one task (export) | No Bangla legal Word2Vec found | Lowest scores found; formal register close to Domain 1 |
| **Regional dialects** (Chittagong, Sylhet, Noakhali, Barishal…) | Dialect classification 84.8–94.1% (several 2025–26 papers); regional NER F1 ~82% (ANCHOLIK-NER, export) | Work is transformer-based; **no dialect Word2Vec found** | Dialect vocabulary is OOV for news-trained models |
| **Banglish / code-mixed** | Code-mixed sentiment via data augmentation with word embeddings (IEEE Access 2023); Banglish dialect detection 81.6% (BRRD-Net 2026) | Word-embedding work exists but small | Romanized spelling breaks Bangla-script vocabularies |

**Honest caveat:** for legal and dialect text, "has Word2Vec but low accuracy" is **not** what SciSpace shows. Those domains have weak results and **no** domain Word2Vec. That is a stronger gap: you would build the first one.

---

## 4. The realigned research questions

- **RQ1.** Does merging two independently trained, domain-specific Bangla Word2Vec models (news + informal) give one embedding that is not worse than each specialist in its own domain, and better across both?
- **RQ2.** Which merge works best: concatenation, concatenation + SVD (at which k), weighted concatenation, alignment (Procrustes / CCA) + average, autoencoder, or joint/continued training? How much does SVD lose?
- **RQ3.** How should vocabulary mismatch be handled: intersection, or union with imputation (zero, linear mapping from the other space, fastText subwords)? What does each cost in OOV coverage?
- **RQ4.** Does the merged anchor help niche domains (medical, legal, dialect, Banglish) when merged with a small niche Word2Vec?

---

## 5. The plan

### Phase 0 — align with the professor (week 1)
- Show him §1 and answer his question: **Case A**.
- Agree that contextual models (mE5, BanglaBERT) stay as **baselines**, and the merged model is word-level.

### Phase 1 — source models (weeks 2–3)
Two kinds of pair, so a reviewer cannot say the gain comes from different training settings:

| Pair | Model 1 (formal) | Model 2 (informal) | Purpose |
|---|---|---|---|
| **Controlled** | Word2Vec skip-gram trained by us on news (Potrika) | Same algorithm and settings, trained by us on reviews + social media (BanglaBook, SentNoB, Hate train portions) | Isolates the **domain** effect |
| **Off-the-shelf** | `alamkhaan` news Word2Vec (250-d) or Bangla2Vec | the informal model above | Shows the method works with public models |

- Settings for the controlled pair: one fixed configuration for both, e.g. 300-d skip-gram, window 4 (the best Bangla setting reported by Rahman 2020).
- **Leakage rule:** texts in the evaluation set are removed from the Word2Vec training corpora.
- Reference baselines: bnlp Word2Vec (Wikipedia), fastText `cc.bn.300`, and BanglaBERT / mE5 as contextual baselines.

### Phase 2 — vocabulary alignment (week 4)
- Intersection; union + zero fill; union + linear-map imputation (learn W on shared words, predict missing rows); union + fastText-subword imputation.
- Report vocabulary overlap and OOV rate per domain for each.

### Phase 3 — merge methods (weeks 5–6)
- CONC (each source L2-normalised and scaled to equal energy, fixing the "unequal scales" limitation)
- **CONC + SVD** at k ∈ {100, 200, 300}, with retained variance (the professor's formulation)
- Weighted CONC (Bollegala 2022)
- Alignment + average: Procrustes; CCA / domain-adapted embeddings (Sarma et al. 2018)
- Autoencoded meta-embedding (AEME)
- Continued training of the news model on informal text (professor's option 3)
- Seyler & Zhai (2020) compared concatenation, data fusion and aligned interpolation for general + domain embeddings. Use it as the closest prior design.

### Phase 4 — evaluation (weeks 7–9)
| Type | Test | Resource |
|---|---|---|
| Intrinsic | Word analogy | Bangla analogy set, 16,678 items (Akter et al. 2023); semantic/syntactic question sets (Bülbül et al. 2024) |
| Intrinsic | Word similarity / relatedness | Intrinsic Bangla evaluation tasks (Sadman et al. 2019). If too small, build a 300–500 pair human-rated set with 3 raters and report agreement |
| Intrinsic | **OOV coverage** per domain | Our eval set + niche corpora |
| Extrinsic | News topic, review sentiment, social sentiment, hate speech | Our eval set, scaled to ~24k texts (power analysis) |

- Same splits for every model (as the professor asks). Keep the pre-registered DEV/CONF rule, paired bootstrap, Holm, label-permutation and dimension controls.
- Success criterion, fixed in advance: the merged model is non-inferior (within 1 point) to each specialist in its own domain and better on the two-domain average.

### Phase 5 — niche domains (weeks 10–13)
- Pick 2 of the 4 niche domains (medical and dialect have the most public data: BanglaMedNER, Bangla MedER, Vashantor, ANCHOLIK-NER, BIDWESH).
- Train a small niche Word2Vec; merge it with the unified anchor; measure OOV reduction and downstream F1.

### Phase 6 — writing (weeks 14–16)
- Thesis chapters follow RQ1–RQ4.
- Q1 paper: the controlled-pair result plus the SVD information-loss analysis plus one niche domain. Candidate venues to check: ACM TALLIP, Natural Language Engineering, Language Resources and Evaluation, Expert Systems with Applications. **Verify each venue's current SJR quartile before submitting.**

---

## 6. Professor's limitation table → the experiment that answers each

| Limitation (professor) | Experiment |
|---|---|
| Different embedding spaces | Procrustes / CCA alignment vs plain concatenation (Phase 3) |
| Redundancy | CKA between the two matrices + conditional-information test from `MATH_PROOF.md` T2 |
| Information loss (SVD) | SVD at k = 100/200/300 vs no reduction; retained variance vs task F1 |
| Unequal scales | Per-source L2 normalisation and equal-energy scaling vs raw |
| Vocabulary mismatch | The 4 alignment strategies in Phase 2 |
| Model identity | State clearly: the output is a combined embedding matrix, not a retrained Word2Vec; compare with continued training |

---

## 7. Questions to ask in SciSpace

1. "Which methods combine a general-domain and a domain-specific word embedding (concatenation, SVD, CCA, Procrustes), and which works best for classification in low-resource languages?"
2. "How much task-relevant information does truncated SVD or PCA lose when reducing concatenated word embeddings, and how should k be chosen?"
3. "How should missing words be imputed when merging two word embeddings with different vocabularies?"
4. "What Bangla word similarity, relatedness, and analogy benchmarks exist, and how large are they?"
5. "Are there Word2Vec or fastText models trained on Bangla social media, reviews, or Banglish text, and are they public?"
6. "What word embeddings exist for Bangla medical, legal, or dialect text, and what accuracies do they report?"
7. "Does continued training of a pretrained Word2Vec model on a new domain corpus outperform merging two separate domain models?"
8. "What are the theoretical guarantees and failure cases of meta-embeddings built by concatenation or averaging?"
9. "Which Q1 journals publish low-resource word-embedding and meta-embedding research, and what evaluation standards do they expect?"

---

## 8. Sources (SciSpace)

- Bülbül et al. 2024, A Comprehensive Study of Bengali Embedding Models — https://scispace.com/papers/a-comprehensive-study-of-bengali-embedding-models-insights-3y7nfshqebdj
- Hossain & Hoque 2020, Towards Bengali Word Embedding — https://scispace.com/papers/towards-bengali-word-embedding-corpus-creation-intrinsic-and-2bo042panu
- Ahmad & Amin 2016, Bengali word embeddings for document classification — https://scispace.com/papers/bengali-word-embeddings-and-it-s-application-in-solving-3clp064v6p
- Kowsher et al. 2022, An Enhanced Neural Word Embedding Model (BanglaFastText) — https://scispace.com/papers/an-enhanced-neural-word-embedding-model-for-transfer-iiv04kmg
- Karim et al. 2020, Classification Benchmarks for Under-resourced Bengali (BengFastText) — https://scispace.com/papers/classification-benchmarks-for-under-resourced-bengali-10qt6y4f9q
- Rahman 2020, Robust and Consistent Estimation of Word Embedding for Bangla — https://scispace.com/papers/robust-and-consistent-estimation-of-word-embedding-for-59iulpo7yp
- Sumit et al. 2018, Exploring Word Embedding for Bangla Sentiment Analysis — https://scispace.com/papers/exploring-word-embedding-for-bangla-sentiment-analysis-4vtx54955v
- Hoque et al. 2026, Multi-class Bangla Public Sentiment with embeddings — https://scispace.com/papers/performance-evaluation-of-multi-class-bangla-public-qarwlj8ov007
- Akter et al. 2023, On Evaluation of Bangla Word Analogies — https://scispace.com/papers/on-evaluation-of-bangla-word-analogies-3aotqjyj
- Sadman et al. 2019, Intrinsic Evaluation of Bangla Word Embeddings — https://scispace.com/papers/intrinsic-evaluation-of-bangla-word-embeddings-2zltvvyl77
- Seyler & Zhai 2020, Methods for Domain-Aware Word Embeddings — https://scispace.com/papers/a-study-of-methods-for-the-generation-of-domain-aware-word-1k8dc2ect0
- Sarma et al. 2018, Domain Adapted Word Embeddings — https://scispace.com/papers/domain-adapted-word-embeddings-for-improved-sentiment-56fsw2ruzt
- El Boukkouri et al. 2019, Embedding Strategies for Specialized Domains — https://scispace.com/papers/embedding-strategies-for-specialized-domains-application-to-311zc669qq
- Mandal et al. 2024, Medical nearest-word embedding for Bengali — https://scispace.com/papers/medical-nearest-word-embedding-technique-implemented-using-49cm7yydwj
- Abdullah & Maruf 2026, Benchmarking Transformers for Bangla Medical NER — https://scispace.com/papers/language-specificity-vs-domain-diversity-benchmarking-m03w24thuurk
- FastMedBangla 2026 — https://scispace.com/papers/fastmedbangla-a-multi-task-deep-learning-framework-for-03nd80ryqq3p
- Dhrubo et al. 2026, KAN-BiGRU for legal documents — https://scispace.com/papers/enhancing-bigru-with-a-kan-block-for-legal-document-17rqkc3wvy17
- Sultana et al. 2026, Bangla regional text classification — https://scispace.com/papers/a-hybrid-approach-for-bangla-regional-text-classification-2dqnd8kn2dha
- Abbas et al. 2025, Benchmarking Bengali Dialect Identification — https://scispace.com/papers/benchmarking-bengali-dialect-identification-a-comparative-7eecu2a1v5qs
- BRRD-Net 2026, Romanized Bangla dialect detection — https://scispace.com/papers/brrd-net-a-transformer-based-framework-for-region-specific-36jir0u1owi6
- Data-Augmentation for Bangla-English Code-Mixed Sentiment 2023 — https://scispace.com/papers/data-augmentation-for-bangla-english-code-mixed-sentiment-3a3jaukj
