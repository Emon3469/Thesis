# The merge formula and its mathematical justification

This document states the merge formula, proves what can be proved about it, and says plainly what cannot be proved.
Every measured number quoted here comes from `w2v_output/w2v_results/` (pre-evaluation) or `merge_output/merge_results/` (Step 4).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| A, B | The two source word-embedding tables (Domain A = news, Domain B = conversational/informal), each d = 300 |
| V_A, V_B | Their vocabularies; V = V_A ∪ V_B is the union |
| a(w), b(w) | Row of word w in A and in B, L2-normalised to length 1 |
| x(s) | Sentence vector: the mean of the word rows of the tokens of text s |
| Y | The label of a text |
| probe | Linear classifier on standardised features (ridge, strength chosen by cross-validation) |
| R(H) | Lowest expected loss reachable by any classifier in the class H |

---

## 2. The merge formula

**Step 1 – normalise.** Every source row is scaled to unit length.

**Step 2 – complete the vocabulary.** For a word that only B knows, its A-row is predicted with a linear map fitted on the shared words:

> M_{B→A} = argmin_M Σ_{w ∈ V_A ∩ V_B} ‖b(w)·M − a(w)‖² + λ‖M‖²_F = (BᵀB + λI)⁻¹ BᵀA,  λ = 1
>
> ã(w) = a(w) if w ∈ V_A, otherwise b(w)·M_{B→A} / ‖b(w)·M_{B→A}‖

b̃(w) is defined the same way with M_{A→B}.

**Step 3 – concatenate with equal energy.**

> **m(w) = (1/√2) · [ ã(w) ; b̃(w) ] ∈ ℝ⁶⁰⁰,  for every w ∈ V_A ∪ V_B**

Every merged row has length exactly 1, and each source contributes exactly half of it.

**Optional Step 4 – reduce.** z(w) = V_kᵀ m(w), where V_k holds the top-k right singular vectors of the merged matrix. This step is used only if the experiments show no loss (Theorem 7 gives its risk bound).

For K sources the same formula is used with K blocks and the factor 1/√K.

---

## 3. Theorems

### Theorem 1 – Pooling identity (the merged sentence vector is the concatenation of the source sentence vectors)
**Claim.** x_m(s) = (1/√2)·[x_ã(s) ; x_b̃(s)].
**Proof.** The mean is a linear operation, and concatenation and scaling are linear, so they commute. ∎
**Check.** P1: maximum absolute error 0.0 on 500 texts, for both pairs.

### Theorem 2 – Merging cannot lose (population level)
**Claim.** Let H_A be all linear classifiers on x_A, H_B on x_B, and H_AB on [x_A ; x_B]. Then
R(H_AB) ≤ min( R(H_A), R(H_B) ).
**Proof.** Any classifier W_A·x_A in H_A equals [W_A, 0]·[x_A ; x_B], which is in H_AB. So H_A ⊆ H_AB, and likewise H_B ⊆ H_AB. A minimum over a larger set is never larger. ∎
**Meaning.** With enough data, the concatenation is at least as good as the better source on *every* task. No other property is needed.

### Theorem 3 – The size of the gain is the conditional information
**Claim.** For log-loss and Bayes-optimal predictors,
R*(A) − R*(A,B) = I(Y ; X_B | X_A) ≥ 0, with equality only if B carries no label information beyond A.
**Proof.** The optimal log-loss given features X is the conditional entropy H(Y|X). So the difference is H(Y|X_A) − H(Y|X_A,X_B), which is the definition of conditional mutual information, and that is never negative. ∎
**Check (P2, linear-probe estimate, nats per text, 95% CI).**

| Pair | What B adds on informal tasks | What A adds on news |
|---|---|---|
| News-W2V + Subs2vec-w2 | 0.026 [0.019, 0.033] | 0.045 [0.039, 0.051] |
| fastText + Informal-W2V | 0.021 [0.014, 0.027] | 0.210 [0.197, 0.223] |

Both intervals exclude 0, so the gain is strictly positive for these pairs.

Step 4 repeated the check: News-W2V + Informal-W2V gives 0.068 [0.060, 0.076] on informal tasks and 0.198 [0.186, 0.212] on news, so it passes. Adding Subs2vec-w2 as a third model on top of that pair gives 0.003 [−0.001, 0.006] on informal tasks, so the third model is not certified to add information.

### Theorem 4 – Concatenation dominates every linear merging method
**Claim.** Let z(w) = L·m(w) for any fixed matrix L. This covers:
- averaging: L = [I, I];
- alignment + averaging: L = [I, Rᵀ] with R a rotation;
- SVD/PCA reduction: L = V_kᵀ;
- weighted concatenation: L diagonal;
- a linear autoencoder.

Then R(H_m) ≤ R(H_z), with equality only if the best classifier on m already lies in the row space of L.
**Proof.** By Theorem 1 the sentence vector of z is L·x_m. A linear classifier on it is W·L·x_m, which is a linear classifier on x_m. So H_z ⊆ H_m. ∎
**Meaning.** Among all linear ways to merge two tables, concatenation has the lowest reachable loss. This is the formal reason the formula in §2 uses concatenation and treats SVD and averaging as optional compressions.
**Limit.** The implementation re-normalises rows after SVD or averaging (standard practice), which is not linear. The theorem is exact without that re-normalisation; with it, the ranking is confirmed by experiment.

### Theorem 5 – Block weights do not matter for the probe
**Claim.** Replacing m(w) by [√α·ã(w) ; √(1−α)·b̃(w)] for any 0 < α < 1 leaves the predictions of a probe on standardised features unchanged.
**Proof.** Standardisation divides each coordinate by its own standard deviation. Multiplying a coordinate by a constant c > 0 multiplies its standard deviation by c, so the standardised coordinate is identical. ∎
**Meaning.** The equal-energy choice α = ½ needs no tuning for classification. (α does change cosine similarity, see Theorem 6.)
**Check.** Step 4 ran α = 0.3, 0.5, 0.7 for two source sets. The macro-F1 values are identical on all four tasks (see `MERGE_RESULTS.md`).

### Theorem 6 – Merged similarity is the average of the source similarities, with lower noise
**Claim.** cos(m(u), m(v)) = ½·cos(ã(u), ã(v)) + ½·cos(b̃(u), b̃(v)).
**Proof.** Both rows have length 1, so the cosine is the dot product, and the dot product of concatenated vectors is the sum of the block dot products, each scaled by (1/√2)² = ½. ∎
**Corollary (noise reduction).** Write each source similarity as the true similarity plus an error with variance σ_A², σ_B² and correlation ρ. The merged error variance is (σ_A² + σ_B² + 2ρσ_Aσ_B)/4. For equal variances this is σ²(1+ρ)/2, which is below σ² whenever ρ < 1. Models trained on different corpora make different errors (ρ < 1), so the merged similarity is less noisy than either source.
**Limit.** If one source is much noisier than the other (σ_B ≫ σ_A), the average can be worse than the better source. This is why similarity and analogy are measured, not assumed.

### Theorem 7 – SVD reduction: how much can be lost
**Claim.** Let P_k be the projection onto the top-k singular directions. For a loss that is L-Lipschitz in the scores and a probe with ‖W‖₂ ≤ Λ,
R(after SVD) − R(before) ≤ L·Λ·E‖x − P_k x‖.
Summed over words, the squared residual Σ‖m(w) − P_k m(w)‖² equals Σ_{i>k} σ_i², the smallest possible for any rank-k map (Eckart–Young).
**Proof.** |ℓ(Wx) − ℓ(WP_k x)| ≤ L‖W(x − P_k x)‖ ≤ L·Λ·‖x − P_k x‖; take expectations. ∎
**Check (P3).** The bound is 9–17 nats at k = 100–300, against a gain of about 0.02 nats. The bound is worst-case and too loose to *guarantee* that SVD keeps the gain. So SVD is decided by experiment, not by proof.

### Theorem 8 – Imputing missing words adds only a small, bounded error
**Claim.** If an imputed row is within ε of the row the source would have had, and a share ρ of a text's tokens are imputed, the sentence vector moves by at most ρ·ε, and the risk by at most L·Λ·ρ·ε.
**Proof.** The sentence vector is a mean; only a share ρ of its terms change, each by at most ε (triangle inequality). ∎
**Check (P5 and coverage).**
- ρ is small: News-W2V already covers 98.5% of news tokens and 91.2% of informal tokens.
- Held-out imputed rows have median cosine 0.63 (B→A) and 0.59 (A→B) with the true rows. Random pairs reach only 0.39 and 0.32 at their 95th percentile.

### Theorem 9 – Doubling the dimension does not by itself hurt generalisation
**Claim.** For linear classifiers with ‖W‖ ≤ Λ on inputs with ‖x‖ ≤ 1, the gap between training and true risk is at most 2LΛ/√n + √(ln(1/δ)/2n) with probability 1 − δ. The dimension does not appear.
**Proof.** Standard Rademacher-complexity bound for norm-bounded linear classes. ∎
**Meaning.** Merged sentence vectors have length ≤ 1 (a mean of unit rows), and the probe's norm is controlled by cross-validated ridge. So going from 300 to 600 dimensions with n ≈ 6,000 texts per domain is safe.

### Theorem 10 – What selection on DEV can and cannot guarantee
**Claim.** If K candidates are compared on n texts, then with probability 1 − δ every measured accuracy is within √(ln(2K/δ)/2n) of its true value.
**Proof.** Hoeffding's inequality plus a union bound over the K candidates. ∎
**Numbers.** Step 4 compares K = 28 candidates on n ≈ 5,950–6,000 texts per domain. With δ = 0.05 the worst-case margin is √(ln(1120)/11,900) ≈ 0.024, i.e. about 2.4 points.
**Meaning.** Differences between candidates smaller than this are not guaranteed to be real. The selected model must be re-tested once on the untouched CONF half, where only a few pre-registered comparisons are made.

---

## 4. What is proved and what is not

**Proved**
- The formula in §2 cannot be worse than either source at the population level (Theorem 2).
- Its gain is strictly positive for the pairs we measured (Theorem 3 + P2).
- No linear merging method can reach a lower loss than it (Theorem 4).
- It needs no weight tuning for classification (Theorem 5).
- Its similarity scores are less noisy than a single source's when the sources' errors are not perfectly correlated (Theorem 6).
- Its vocabulary completion and its higher dimension add only bounded error (Theorems 8 and 9).

**Not provable by mathematics**
- "Top 1%" or "best accuracy" in an absolute sense. No theorem can rank a model against every other model that exists. It is an empirical claim.
- That the finite-sample result follows the population result. Theorems 2–4 are about the best reachable loss; with limited data the outcome must be measured (Theorem 10).

**The empirical claim we can make, and its test**
- The merged model is called better only if, on the CONF half, the lower end of the 95% confidence interval of its difference to every baseline (each specialist and fastText) is above 0, after Holm correction.
- Result (see `CONF_RESULTS.md`): the selected model T: CONC passed all three tests on CONF. Its gains over fastText, News-W2V and Informal-W2V are +0.0073, +0.0241 and +0.0391 Overall macro-F1, with Holm-adjusted p = 0.0044, 0.0003 and 0.0003.

---

## 5. Sources (SciSpace)

- Coates & Bollegala 2018, Frustratingly Easy Meta-Embedding (averaging vs concatenation) — https://scispace.com/papers/frustratingly-easy-meta-embedding-computing-meta-embeddings-1m3jmrdh39
- Bollegala 2022, Learning Meta Word Embeddings by Unsupervised Weighted Concatenation of Source Embeddings — https://scispace.com/papers/learning-meta-word-embeddings-by-unsupervised-weighted-x4edqcl9
- Bollegala & Bao 2018, Learning Word Meta-Embeddings by Autoencoding — https://scispace.com/papers/learning-word-meta-embeddings-by-autoencoding-zj3zydu0iu
- Jawanpuria et al. 2020, Learning Geometric Word Meta-Embeddings (rotation + scaling before averaging) — https://scispace.com/papers/learning-geometric-word-meta-embeddings-1z33jexmcs
- Raunak et al. 2019, Effective Dimensionality Reduction for Word Embeddings — https://scispace.com/papers/effective-dimensionality-reduction-for-word-embeddings-1ht9cklr8o
