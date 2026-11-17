# Mathematical Justification for the Two-Model Meta-Embedding

**Pair:** $A$ = `mE5-base` (768-d, contextual), $B$ = `fastText cc.bn.300` (300-d, static subword).
**Merge:** standardised concatenation (CONC-std).

**What this document proves and what it cannot.**
- It **proves** five statements under stated assumptions. Together they say: the merge can never be worse than its
  better member in the population; it is strictly better exactly when the second model carries label information the
  first lacks; and the finite-sample cost of merging shrinks like $1/\sqrt{n}$.
- It **tests** each assumption on the 2,100-text set (`../proof_checks.py`).
- It **cannot prove** "top 1%" or "best of all possible models". That is a ranking claim about real data, and no
  theorem can establish it. What the mathematics gives is a **falsifiable prediction** for the powered study (§7).

---

## 1. Setup and notation

- Text $X$, label $Y \in \{1,\dots,C\}$, joint distribution $P$.
- Frozen encoders $\phi_A : \mathcal{X} \to \mathbb{R}^{d_A}$ and $\phi_B : \mathcal{X} \to \mathbb{R}^{d_B}$. Write $a = \phi_A(x)$, $b = \phi_B(x)$.
- Per-source standardisation: $\tilde a = D_A^{-1}(a - \mu_A)/\sqrt{d_A}$, with $D_A$ the diagonal of feature standard deviations (same for $b$). Then $\mathbb{E}\|\tilde a\|^2 = \mathbb{E}\|\tilde b\|^2 = 1$.
- **Meta-embedding:** $\phi_M(x) = [\alpha\,\tilde a \,;\, \beta\,\tilde b] \in \mathbb{R}^{d_A + d_B}$, with $\alpha, \beta \neq 0$. CONC-std is $\alpha = \beta = 1$.
- Linear probe on a representation $\phi$: scores $W\phi(x) + c$. For a loss $\ell$:
  $$R_\phi(W,c) = \mathbb{E}\,\ell\big(Y, W\phi(X)+c\big), \qquad R^{\text{lin}}_\phi = \inf_{W,c} R_\phi(W,c).$$
- Norm-bounded version: $R^{\text{lin}}_{\phi}(\Lambda) = \inf_{\|W\|_F \le \Lambda,\, c} R_\phi(W,c)$.

---

## 2. Theorem 1 — Merging never loses (population)

**Statement.** For every loss $\ell$ and all $\alpha,\beta \ne 0$:
$$R^{\text{lin}}_{\phi_M} \;\le\; \min\big(R^{\text{lin}}_{\phi_A},\, R^{\text{lin}}_{\phi_B}\big).$$
The same holds for the Bayes risk: $R^{*}(\phi_M) \le \min(R^{*}(\phi_A), R^{*}(\phi_B))$.

**Proof.**
1. Take any probe $(W_A, c)$ on $\phi_A$. Its scores are $W_A a + c$.
2. Since $a = \sqrt{d_A}\,D_A \tilde a + \mu_A$, define $W_M = \big[\tfrac{\sqrt{d_A}}{\alpha} W_A D_A \;\; 0\big]$ and $c_M = c + W_A\mu_A$.
3. Then $W_M \phi_M(x) + c_M = W_A a + c$ for every $x$. So every probe on $A$ is reproduced exactly by a probe on $M$.
4. Hence the set of score functions reachable from $\phi_M$ contains those reachable from $\phi_A$, and the infimum over a larger set is no larger. The argument for $B$ is symmetric.
5. Bayes risk: $\phi_A(x)$ is a deterministic function of $\phi_M(x)$ (drop coordinates, invert the affine map), so any classifier on $\phi_A$ is a classifier on $\phi_M$. $\blacksquare$

**Scope.** This holds for *any* pair, so it does not single out this one. What singles out a pair is the *size* of the gain (Theorem 2).

---

## 3. Theorem 2 — The exact size of the gain, and when it is strictly positive

**Statement.** Under log-loss, with unrestricted probes, the Bayes risk of a representation $\phi$ is $H(Y \mid \phi(X))$. Therefore
$$R^{*}(\phi_A) - R^{*}(\phi_M) \;=\; H(Y\mid A) - H(Y \mid A, B) \;=\; I(Y; B \mid A) \;\ge\; 0,$$
with equality **if and only if** $Y \perp B \mid A$.

**Proof.**
1. For log-loss the risk of a predictor $q(\cdot \mid \phi)$ is $\mathbb{E}[-\log q(Y\mid\phi)] = H(Y\mid\phi) + \mathbb{E}\,\mathrm{KL}\big(p(\cdot\mid\phi)\,\|\,q(\cdot\mid\phi)\big)$. It is minimised at $q = p$, giving $H(Y\mid\phi)$.
2. $\phi_M$ is a bijective function of $(A,B)$, so $H(Y\mid\phi_M) = H(Y\mid A,B)$.
3. The difference is the definition of conditional mutual information, which is a (conditional) KL divergence, hence $\ge 0$, and zero iff the conditional independence holds. $\blacksquare$

**Corollary 2.1 (beating *both* members).** The merge is strictly better than each member iff
$$I(Y;B\mid A) > 0 \quad\text{and}\quad I(Y;A\mid B) > 0,$$
that is, each model knows something about the label that the other does not.

**Empirical test.** Out-of-fold log-loss of the probe on $A$ versus on $[A;B]$ estimates the linear-probe version of $I(Y;B\mid A)$ (in nats per text; 95% bootstrap CI):

| Task | $n$ | log-loss $A$ | log-loss $[A;B]$ | Gain (nats) | 95% CI | only $A$ right / only $B$ right |
|---|---|---|---|---|---|---|
| News topic | 1,050 | 0.436 | 0.391 | **0.045** | [0.024, 0.067] | 75 / 59 |
| Hate speech | 330 | 0.471 | 0.426 | **0.045** | [0.013, 0.074] | 32 / 43 |
| Review sentiment | 360 | 0.879 | 0.875 | 0.005 | [−0.008, 0.016] | 58 / 33 |
| Social sentiment | 360 | 0.956 | 0.949 | 0.007 | [−0.014, 0.027] | 62 / 50 |

- The condition of Theorem 2 is **confirmed on news and hate speech** (CI excludes 0) and **not confirmed on the two sentiment tasks**.
- Both "only-right" counts are non-zero on every task, which is what Corollary 2.1 requires; on hate speech $B$ alone is the stronger model.
- This check uses all 2,100 texts (both halves) as a diagnostic. It was not used to select anything.

---

## 4. Theorem 3 — What the merge does to similarity

**Statement.** Let $u_A, u_B$ be the unit-normalised source vectors and $\phi_M = [\alpha u_A ; \beta u_B]$. Then for any two texts
$$\cos_M(x,x') \;=\; \frac{\alpha^2 \cos_A(x,x') + \beta^2 \cos_B(x,x')}{\alpha^2+\beta^2}.$$
For $\alpha=\beta$ this is the plain mean of the two cosines. Equivalently, the kernel of the merge is $k_M = \alpha^2 k_A + \beta^2 k_B$.

**Proof.** $\langle \phi_M(x), \phi_M(x')\rangle = \alpha^2\langle u_A,u_A'\rangle + \beta^2\langle u_B,u_B'\rangle$ because the blocks are disjoint, and $\|\phi_M(x)\|^2 = \alpha^2+\beta^2$ for all $x$. Divide. $\blacksquare$

**Checked numerically:** maximum deviation over 20,000 random text pairs is $4.4\times10^{-16}$.

**Corollary 3.1 (noise reduction).** Assume each source measures a common "true" similarity $s$ with additive error: $\cos_A = s + \varepsilon_A$, $\cos_B = s + \varepsilon_B$, with $\mathbb{E}\varepsilon=0$, variances $\sigma_A^2,\sigma_B^2$, correlation $\rho$. Then the weighted mean with weight $w$ on $A$ has error variance
$$V(w) = w^2\sigma_A^2 + (1-w)^2\sigma_B^2 + 2w(1-w)\rho\,\sigma_A\sigma_B,$$
minimised at $w^\star = \dfrac{\sigma_B^2 - \rho\sigma_A\sigma_B}{\sigma_A^2+\sigma_B^2-2\rho\sigma_A\sigma_B}$, with
$$V(w^\star) = \frac{\sigma_A^2\sigma_B^2(1-\rho^2)}{\sigma_A^2+\sigma_B^2-2\rho\sigma_A\sigma_B} \;\le\; \min(\sigma_A^2,\sigma_B^2).$$

**Proof.** $V$ is a convex quadratic in $w$; set $V'(w)=0$ and substitute. The inequality holds because $w=1$ and $w=0$ are feasible and give $\sigma_A^2$ and $\sigma_B^2$. Equality requires $\rho = \sigma_B/\sigma_A$ or $\rho=\sigma_A/\sigma_B$. $\blacksquare$

- **Reading:** the lower the error correlation $\rho$ between the two models, the larger the reduction. This is the formal version of "merge complementary models".
- **Assumption status:** the additive-error model is an assumption, not a measured fact. Our proxy is linear CKA between the two spaces (0.57 on news, 0.36 on informal text: the lowest among the strong pairs). CKA is not $\rho$.
- **Why $\alpha,\beta$ matter here and not in Theorem 1:** the probe re-scales features, so weights cancel (measured: CONC-α = CONC-std for the probe). For similarity use (kNN, retrieval) the weight $w^\star$ changes the result.

---

## 5. Theorem 4 — Concatenation dominates averaging

Let $\bar b$ be $b$ zero-padded to $d_A$ dimensions and $\phi_{\text{AVG}} = (u_A + \bar u_B)/2$.

**Statement (a), geometry.**
$$\langle \phi_{\text{AVG}}(x), \phi_{\text{AVG}}(x')\rangle = \tfrac14\Big(k_A + k_B + \underbrace{\langle u_A,\bar u_B'\rangle + \langle \bar u_B, u_A'\rangle}_{\text{cross term } \xi}\Big).$$
So averaging equals concatenation plus a cross term $\xi$ between unrelated coordinate systems.

**Statement (b), probes.** $R^{\text{lin}}_{\text{CONC}} \le R^{\text{lin}}_{\text{AVG}}$.

**Proof.**
- (a) Expand the inner product.
- (b) A probe $W$ on $\phi_{\text{AVG}}$ scores $\tfrac12 W u_A + \tfrac12 W_{[1:d_B]} u_B$. The concatenation probe $[\tfrac12 W \;\; \tfrac12 W_{[1:d_B]}]$ gives the same scores. So averaging is concatenation with the weights on $B$ **tied** to the first $d_B$ weights on $A$: a strict subset of the concatenation's probes. $\blacksquare$

**Measured on our pair** (20,000 text pairs):
- identity (a) holds to $1.7\times10^{-16}$;
- $\mathrm{Var}(\xi)/\mathrm{Var}(k_A+k_B) = 0.033$, mean of $\xi$ = 0.025, correlation with the signal −0.10.

**Reading.**
- The cross term is small (3.3% of the signal variance), consistent with the near-orthogonality argument of Coates & Bollegala (2018).
- So the cross term does **not** explain why averaging fell from 76.7 (DEV) to 73.5 (CONF). The better explanation is selection noise: it was the best of 36 configurations.
- (b) is the reason to carry **concatenation**, not averaging, into the main study.

---

## 6. Theorem 5 — The finite-sample price, and why it vanishes

**Setting.** Probe weights bounded by $\|W\|_F \le \Lambda$; loss $L$-Lipschitz in the scores and bounded by $c_0$; $\|\phi(x)\|\le r_\phi$. Let $\hat W$ be the empirical risk minimiser over $n$ training texts.

**Known bound** (Rademacher complexity of norm-bounded linear classes; Kakade, Sridharan & Tewari, 2008). With probability $\ge 1-\delta$,
$$R_\phi(\hat W) \;\le\; R^{\text{lin}}_\phi(\Lambda) \;+\; \frac{\kappa\, L\, \Lambda\, r_\phi}{\sqrt n} \;+\; c_0\sqrt{\frac{2\log(2/\delta)}{n}},$$
where $\kappa$ is a constant depending only on the number of classes $C$.

**Statement.** Define the norm-bounded approximation gain $\Delta_\Lambda = R^{\text{lin}}_{\phi_A}(\Lambda) - R^{\text{lin}}_{\phi_M}(\Lambda)$. Then:
1. $\Delta_\Lambda \ge 0$.
2. $r_{\phi_M}^2 = r_{\phi_A}^2 + r_{\phi_B}^2$, so with equal-energy sources the radius grows by exactly $\sqrt2$.
3. **Asymptotic dominance.** If $\Delta_\Lambda > 0$, then for every $\delta$ there is an $n_0$ such that for all $n \ge n_0$, with probability $\ge 1-\delta$,
$$R_{\phi_M}(\hat W_M) \;<\; R^{\text{lin}}_{\phi_A}(\Lambda) \;\le\; R_{\phi_A}(W) \quad \text{for every admissible probe } W \text{ on } A.$$
A sufficient size is
$$n_0 = \left(\frac{\kappa L \Lambda\sqrt2\, r + c_0\sqrt{2\log(2/\delta)}}{\Delta_\Lambda}\right)^{2}.$$

**Proof.**
1. Pad any $W_A$ with zeros: same scores, same Frobenius norm. So the bounded class on $M$ contains the bounded class on $A$ (with $\alpha = 1$ and the same scaling).
2. The blocks are orthogonal coordinates, so squared norms add.
3. Apply the bound to $\phi_M$: $R_{\phi_M}(\hat W_M) \le R^{\text{lin}}_{\phi_A}(\Lambda) - \Delta_\Lambda + \epsilon_n$ with $\epsilon_n = (\kappa L\Lambda\sqrt2 r + c_0\sqrt{2\log(2/\delta)})/\sqrt n$. This is below $R^{\text{lin}}_{\phi_A}(\Lambda)$ as soon as $\epsilon_n < \Delta_\Lambda$, i.e. $n > n_0$. $\blacksquare$

**Measured:** $\mathbb{E}\|\tilde a\|^2 = 1.000$, $\mathbb{E}\|[\tilde a;\tilde b]\|^2 = 2.000$, ratio $\sqrt2 = 1.414$.

**Reading.**
- The gain $\Delta_\Lambda$ is a fixed property of the two models. The price $\epsilon_n$ falls like $1/\sqrt n$.
- At small $n$ the price can exceed the gain. Our probes were trained on only 144–420 texts per fold in the DEV/CONF halves.
- This is a bound-based (sufficient) condition. Constants such as $\kappa$ and $\Lambda$ are not tight, so $n_0$ should be read as a scaling law, not a number to plug in.

---

## 7. What the theorems jointly establish, and the testable prediction

| # | Claim | Status |
|---|---|---|
| T1 | Merge is never worse than its better member (population, any loss) | **Proved**, no assumptions on the data |
| T2 | Gain $= I(Y;B\mid A)$; strictly positive iff $B$ adds label information | **Proved**; condition **confirmed on 2 of 4 tasks** |
| T3 | Merge cosine = weighted mean of cosines; error variance $\le$ the better source | Identity **proved and verified**; variance claim holds **under the additive-error assumption** |
| T4 | Concatenation's probes contain averaging's probes | **Proved**; cross term measured at 3.3% |
| T5 | Finite-sample price shrinks as $1/\sqrt n$; merge wins for $n \ge n_0$ when the gain is positive | **Proved** (bound-based) |

**Chain of reasoning for this pair.**
1. By T1, CONC-std(mE5-base, fastText) cannot lose to either model given enough data.
2. By T2 and the measured gains (0.045 nats on news and on hate speech), the gain is strictly positive on at least those tasks.
3. By T5, the remaining obstacle is sample size, and it shrinks as $1/\sqrt n$.
4. By T4, concatenation is the right form to carry forward.

**Prediction for the powered study (~12,000 texts per half).** The gain of CONC-std over the best single model should be positive on news and hate speech, and its confidence interval should exclude zero there. If it does not, step 2 was a small-sample artefact and the merge should be rejected for those tasks.

**Not established.**
- "Top 1%" or "best possible model": not provable; it is an empirical ranking.
- A gain on the two sentiment tasks: the condition of T2 is not confirmed there.
- That this is the best *pair*: T1 holds for every pair. This pair is preferred because it had the largest DEV gain among strong pairs and the lowest CKA, which is evidence, not proof.
- Observed so far: on the confirmation half CONC-std scored 74.6 Overall versus 74.2 (mE5-base) and 75.2 (mE5-large). These differences are within the confidence intervals.

**A consequence for the Q1 design.** T1 extends to any number of sources by induction: adding a niche-domain encoder cannot raise the population risk on the general domains. T5 says the price is the larger radius ($\sqrt{k}$ for $k$ equal-energy sources), which is why the niche study needs enough labelled data per domain or a dimension-reducing fusion.

---

## 8. Sources (found through SciSpace)

- Coates & Bollegala (2018), *Frustratingly Easy Meta-Embedding*: averaging approximates concatenation under near-orthogonality. https://scispace.com/papers/frustratingly-easy-meta-embedding-computing-meta-embeddings-1m3jmrdh39
- Bollegala (2022), *Learning Meta Word Embeddings by Unsupervised Weighted Concatenation*: weighted concatenation as spectrum matching; optimal source weights. https://scispace.com/papers/learning-meta-word-embeddings-by-unsupervised-weighted-x4edqcl9
- Kakade, Sridharan & Tewari (2008), *On the Complexity of Linear Prediction*: Rademacher bounds for norm-bounded linear classes (Theorem 5). https://scispace.com/papers/on-the-complexity-of-linear-prediction-risk-bounds-margin-1p9pnzet6f
- Audhkhasi et al. (2013), *Generalized Ambiguity Decomposition*: ensemble performance ≈ average member performance − diversity (supports Corollary 3.1's reading). https://scispace.com/papers/generalized-ambiguity-decomposition-for-understanding-3bocy1temx
- Jiang et al. (2017), *Generalized Ambiguity Decompositions for Classification*: the same decomposition for classification losses. https://scispace.com/papers/generalized-ambiguity-decompositions-for-classification-with-3lro4wph40

Theorems 1–2 use only standard definitions (function-class containment, conditional entropy and mutual information).

**Reproduce the checks:** `python ../proof_checks.py` → `proof_checks.json`, `proof_checks_information_gain.csv`.
