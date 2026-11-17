# Steps 6–7 – one-time confirmation test on the CONF half

- Run: Kaggle kernel `emon3469/bangla-w2v-conf` version 1 (private), 5.5 minutes, no errors.
- Source of every number: `conf_output/conf_results/` (`conf_results.csv`, `conf_tests.csv`, `conf_bootstrap.npz`, `conf_predictions.pkl`).
- Verification: `python verify_conf.py` → ALL CHECKS PASSED. It recomputes every macro-F1 from the saved predictions with sklearn and re-derives the CIs, p-values and Holm adjustment from the saved bootstrap samples.
- Model under test: **T: CONC** (News-W2V + Informal-W2V + Subs2vec-w2, 900-d). This is the frozen Step 4 selection, loaded from the Step 4 output and checked by MD5 (`10aa206f…`). Nothing was retrained or tuned.
- Data: CONF half, never used before for any choice. News 5,950 texts; reviews 2,250; social 1,500; hate 2,250.
- Protocol: identical to DEV (same probe, 5 stratified folds inside CONF, seed 42). 10,000 paired bootstrap samples; one-sided p; Holm correction at α = 0.05.

## 1. Pre-registered confirmatory tests (Overall macro-F1)

| Hypothesis | Δ Overall | 95% CI | Holm-adjusted p | Result |
|---|---|---|---|---|
| H1: T: CONC > fastText-cc.bn | +0.0073 | [+0.0017, +0.0128] | 0.0044 | **confirmed** |
| H2: T: CONC > News-W2V | +0.0241 | [+0.0186, +0.0295] | 0.0003 | **confirmed** |
| H3: T: CONC > Informal-W2V | +0.0391 | [+0.0335, +0.0448] | 0.0003 | **confirmed** |

All three Holm-adjusted p-values are below 0.05, so the pre-registered claim holds: **the merged model is better than each of its sources and than fastText on held-out data.**

(p = 0.0001 before adjustment is the smallest value 10,000 bootstrap samples can give; no sample had Δ ≤ 0.)

## 2. All models on CONF

| Model | Dim | News (D1) | Informal (D2) | Overall [95% CI] | Overall on DEV | DEV − CONF |
|---|---|---|---|---|---|---|
| X: CONC (extended, includes fastText) | 1200 | 0.8923 | 0.6644 | 0.7784 [0.7711, 0.7853] | 0.7733 | −0.0050 |
| **T: CONC (main model)** | 900 | **0.8840** | **0.6595** | **0.7717 [0.7645, 0.7788]** | 0.7712 | −0.0005 |
| Q: CONC (two-model, exploratory) | 600 | 0.8797 | 0.6585 | 0.7691 [0.7618, 0.7762] | 0.7697 | +0.0005 |
| fastText-cc.bn | 300 | 0.8815 | 0.6474 | 0.7644 [0.7571, 0.7714] | 0.7620 | −0.0024 |
| News-W2V | 300 | 0.8791 | 0.6162 | 0.7477 [0.7403, 0.7549] | 0.7485 | +0.0008 |
| Subs2vec-w2 | 300 | 0.8656 | 0.6206 | 0.7431 [0.7355, 0.7503] | 0.7420 | −0.0011 |
| Informal-W2V | 300 | 0.8073 | 0.6579 | 0.7326 [0.7247, 0.7402] | 0.7304 | −0.0022 |

Per task for T: CONC: news 0.8840; reviews 0.5969; social 0.5896; hate 0.7919.

## 3. Secondary and exploratory comparisons (not part of the claim)

| Comparison | Δ | 95% CI | Reading |
|---|---|---|---|
| T vs fastText, news only | +0.0025 | [−0.0027, +0.0079] | equal on news |
| T vs Informal-W2V, informal only | +0.0015 | [−0.0066, +0.0097] | equal on informal |
| T vs Subs2vec-w2, Overall | +0.0286 | [+0.0230, +0.0342] | T better |
| T vs Q (adding Subs2vec as third model) | +0.0026 | [−0.0010, +0.0062] | not significant |
| X vs T (adding fastText as fourth model) | +0.0066 | [+0.0030, +0.0102] | X better |
| X vs fastText | +0.0139 | [+0.0092, +0.0187] | X better |

## 4. What the results mean

1. **The merge is confirmed.** On data never used for any choice, the three-model Word2Vec merge beats fastText, News-W2V and Informal-W2V. All three Holm-corrected tests pass.
2. **Why it wins.** Each source is strong on one domain only. The merge matches the best model on news (fastText) and the best on informal text (Informal-W2V) *at the same time*. No single model does both, so the Overall score is higher. This is Theorem 2 of `MERGE_PROOF.md` seen in data: the merge reaches at least the better source on every domain.
3. **No winner's curse.** The DEV score (0.7712) and the CONF score (0.7717) are almost identical. Selecting among 28 candidates did not inflate the result.
4. **The third model is optional.** Adding Subs2vec-w2 to the two-model merge gives +0.26 points, not significant. This matches the Step 4 information check (P2 for T was not certified).
5. **Adding fastText helps further.** X: CONC (1200-d) is the best model measured: +0.66 points over T, significant. It is not a pure Word2Vec model, so it is reported as the extended model.

## 5. About "top 1%"

- **Measured:** among all 33 models built in this thesis (28 merges, 5 single models), T: CONC is the best pure Word2Vec model, and its gain is confirmed on held-out data.
- **Not measured:** a ranking against all published Bangla embeddings, or against transformer models (e.g. BanglaBERT, multilingual E5). Those were not evaluated on this benchmark, so a "top 1%" claim cannot be made from these results.
- A defensible thesis sentence: *"The merged Word2Vec model outperforms each of its source models and the fastText Common Crawl baseline on a held-out set of 11,950 Bangla texts (paired bootstrap, Holm-corrected p < 0.005 for all three comparisons)."*

## 6. Limits

- The gain over fastText is small (+0.7 points Overall) though significant; it comes from informal text, not news.
- The test uses a linear probe on averaged static vectors. It measures embedding quality for classification, not the best accuracy reachable with fine-tuned models.
- Word similarity and analogy did not improve with merging (see `MERGE_RESULTS.md` §6); the confirmation is for classification only.
- 20 texts of the CONF half were used earlier for the illustrative mini test (pre-evaluation); they were not used for any choice.
