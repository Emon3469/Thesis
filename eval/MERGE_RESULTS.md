# Step 4 – merging experiments on the DEV half: results

- Run: Kaggle kernel `emon3469/bangla-w2v-merge` version 1 (private), 129.8 minutes, no errors.
- Source of every number: `merge_output/merge_results/merge_dev.csv`, `premerge_checks.csv`, `manifest.json`.
- Verification: `python verify_merge.py` → ALL CHECKS PASSED (scores, selection, deltas, P-check flags and saved models re-derived independently).
- Data: DEV half only (news 5,950 texts; reviews 2,250; social 1,500; hate 2,250). The CONF half was not touched.
- Metric: macro-F1 of a linear probe, 5 folds. D1 = news. D2 = mean of the three informal tasks. Overall = mean of D1 and D2. Intervals are 95% bootstrap (1,000 samples).

## 1. Source models (re-trained in this run)

| Model | Vocabulary | D1 news | D2 informal | Overall | Analogy | Similarity AUC |
|---|---|---|---|---|---|---|
| fastText-cc.bn (baseline, not Word2Vec) | 1,468,578 | 0.879 | 0.645 | 0.762 | 0.076 | 0.867 |
| News-W2V | 187,686 | 0.874 | 0.623 | 0.749 | 0.063 | 0.881 |
| Subs2vec-w2 | 177,376 | 0.859 | 0.626 | 0.742 | 0.059 | 0.858 |
| Informal-W2V | 32,910 | 0.815 | 0.645 | 0.730 | 0.008 | 0.692 |

- Best single model on news: fastText (0.8793). Best on informal: Informal-W2V (0.6454). Best overall: fastText.
- Eligibility thresholds (specialist − 0.01): D1 ≥ 0.8693 and D2 ≥ 0.6354.
- News-W2V scores differ slightly from the pre-evaluation run (0.876 → 0.874) because Word2Vec training with 4 workers is not bit-reproducible.

## 2. All merged candidates (sorted by Overall)

Source sets: P = News + Subs2vec; Q = News + Informal; T = News + Informal + Subs2vec; S = fastText + Informal; X = all four.

| Candidate | Dim | D1 | D2 | Overall | Δ Overall vs fastText [95% CI] | Eligible |
|---|---|---|---|---|---|---|
| X: CONC (extended) | 1200 | 0.8885 | 0.6582 | 0.7733 | +0.0113 [+0.0065, +0.0163] | yes |
| **T: CONC (selected, pure Word2Vec)** | 900 | 0.8823 | 0.6601 | 0.7712 | +0.0092 [+0.0037, +0.0151] | yes |
| Q: CAEME | 600 | 0.8817 | 0.6599 | 0.7708 | +0.0088 [+0.0028, +0.0145] | yes |
| S: CONC (extended) | 600 | 0.8833 | 0.6581 | 0.7707 | +0.0087 [+0.0043, +0.0133] | yes |
| Q: CONC | 600 | 0.8799 | 0.6594 | 0.7697 | +0.0077 [+0.0018, +0.0136] | yes |
| Q: weighted CONC (0.3 and 0.7) | 600 | 0.8799 | 0.6594 | 0.7697 | +0.0077 [+0.0018, +0.0136] | yes |
| Q: CONC-zero | 600 | 0.8808 | 0.6580 | 0.7694 | +0.0074 [+0.0017, +0.0134] | yes |
| Q: AAEME | 300 | 0.8752 | 0.6579 | 0.7666 | +0.0046 [−0.0012, +0.0104] | yes |
| Joint-W2V | 300 | 0.8752 | 0.6570 | 0.7661 | +0.0041 [−0.0015, +0.0098] | yes |
| Continued-W2V | 300 | 0.8734 | 0.6567 | 0.7651 | +0.0031 [−0.0027, +0.0088] | yes |
| Q: CONC+SVD300 | 300 | 0.8787 | 0.6485 | 0.7636 | +0.0016 [−0.0046, +0.0076] | yes |
| T: CONC+SVD300 | 300 | 0.8757 | 0.6504 | 0.7631 | +0.0011 [−0.0049, +0.0070] | yes |
| Q: Procrustes+AVG | 300 | 0.8717 | 0.6538 | 0.7628 | +0.0008 [−0.0051, +0.0067] | yes |
| T: CONC+SVD200 | 200 | 0.8705 | 0.6494 | 0.7600 | −0.0020 [−0.0079, +0.0041] | yes |
| P: CAEME | 600 | 0.8826 | 0.6367 | 0.7596 | −0.0024 [−0.0080, +0.0033] | yes |
| P: CONC | 600 | 0.8793 | 0.6384 | 0.7589 | −0.0031 [−0.0088, +0.0026] | yes |
| P: weighted CONC (0.3 and 0.7) | 600 | 0.8793 | 0.6384 | 0.7589 | −0.0031 [−0.0088, +0.0026] | yes |
| P: CONC-zero | 600 | 0.8804 | 0.6360 | 0.7582 | −0.0038 [−0.0094, +0.0019] | yes |
| Q: CONC+SVD200 | 200 | 0.8671 | 0.6470 | 0.7571 | −0.0049 [−0.0114, +0.0012] | no |
| T: CONC+SVD100 | 100 | 0.8593 | 0.6524 | 0.7559 | −0.0061 [−0.0119, −0.0002] | no |
| P: Procrustes+AVG | 300 | 0.8777 | 0.6335 | 0.7556 | −0.0064 [−0.0122, −0.0006] | no |
| Q: CONC+SVD100 | 100 | 0.8596 | 0.6506 | 0.7551 | −0.0069 [−0.0128, −0.0009] | no |
| P: CONC+SVD300 | 300 | 0.8774 | 0.6265 | 0.7519 | −0.0101 [−0.0165, −0.0042] | no |
| P: AAEME | 300 | 0.8743 | 0.6260 | 0.7502 | −0.0118 [−0.0177, −0.0062] | no |
| P: CONC+SVD200 | 200 | 0.8733 | 0.6254 | 0.7493 | −0.0127 [−0.0190, −0.0068] | no |
| P: CONC+SVD100 | 100 | 0.8638 | 0.6250 | 0.7444 | −0.0176 [−0.0240, −0.0113] | no |

## 3. Selected models

**Pure Word2Vec (thesis main model): T: CONC** — News-W2V + Informal-W2V + Subs2vec-w2, equal-energy concatenation over the imputed union vocabulary. 900 dimensions, 273,531 words. Saved as `models/selected_pure_w2v.f16.npy`.

| Comparison | Difference | 95% CI | Reading |
|---|---|---|---|
| Overall vs fastText (best single) | +0.0092 | [+0.0037, +0.0151] | above zero on DEV |
| D1 news vs fastText (news specialist) | +0.0030 | [−0.0018, +0.0085] | not different |
| D2 informal vs Informal-W2V (informal specialist) | +0.0147 | [+0.0065, +0.0238] | above zero on DEV |

Per task: news 0.8823; reviews 0.5893; social 0.5788; hate 0.8123. Token coverage: 98.7% news, 94.9% informal.

**Extended (includes fastText): X: CONC** — 1200 dimensions, 354,843 words, Overall 0.7733. Saved as `models/selected_extended.f16.npy`.

## 4. Pre-merge checks for the three Word2Vec source sets

| Check | P: News + Subs2vec | Q: News + Informal | T: (News+Informal) + Subs2vec |
|---|---|---|---|
| P1 pooling identity (max error) | 0.0, pass | 0.0, pass | 0.0, pass |
| P2 what B adds on informal, nats [CI] | 0.029 [0.022, 0.036] | 0.068 [0.060, 0.076] | 0.003 [−0.001, 0.006] |
| P2 what A adds on news, nats [CI] | 0.046 [0.040, 0.051] | 0.198 [0.186, 0.212] | 0.046 [0.039, 0.053] |
| P2 result | pass | pass | **fail** |
| P3 SVD guaranteed safe | no | no | no |
| P4 energy balance | pass | pass | pass |
| P5 imputation better than chance | pass | pass | pass |
| P6 alignment P@1 ≥ 0.5 | 0.441, fail | 0.079, fail | not applicable |

## 5. What the results show

1. **Merging works on DEV.** Concatenating the news and informal Word2Vec models (Q) raises Overall from 0.749 / 0.730 to 0.770 and beats fastText by 0.8 points, interval above zero.
2. **The informal gain comes from Informal-W2V.** The professor's pair P (News + Subs2vec) beats both of its sources but not fastText (−0.3 points, interval includes zero).
3. **The third model adds almost nothing.** T beats Q by 0.0015 Overall, far inside the noise (Theorem 10 margin ≈ 0.024). P2 for T confirms it: Subs2vec adds no measurable label information on informal text once News and Informal are merged, and only 0.010 nats on news. The rule selected T, but Q: CONC (two models, 600 dimensions) is statistically the same and matches the "two models" instruction.
4. **Theorem 4 held.** In every set, plain concatenation scored at or above SVD, averaging and the 300-d autoencoder. CAEME (600-d, also a concatenation) is equal to CONC within noise.
5. **Theorem 5 held exactly.** Weighted concatenation gave the same macro-F1 as equal weights on all four tasks, in both sets.
6. **Joint and continued training are weaker than concatenation**, but not significantly different from fastText.

## 6. Limits that must be stated

- **Intrinsic scores did not improve.** T: CONC has similarity AUC 0.819 (News-W2V alone: 0.881) and analogy accuracy 0.060 (News-W2V 0.063, fastText 0.076). The noisy Informal-W2V block pulls similarity down, which is the limit case stated under Theorem 6. The merged model is better for classification, not for similarity or analogy.
- **The news gain is not significant.** The merged model matches fastText on news; the gain is on informal text.
- **These are DEV numbers used for selection.** 28 candidates were compared, so the selected score is optimistic. Nothing here proves "top 1%" or "best". The confirmation is the one-time CONF test of the frozen model against fastText, News-W2V and Informal-W2V.
- **Absolute level.** Overall 0.771 is from a linear probe on averaged static vectors; it is not comparable to fine-tuned transformer scores.
