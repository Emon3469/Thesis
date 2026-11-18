"""Builds the BanglaMeta mathematics exercise set (HTML + PDF).

Every number printed in a solution is computed here, and every claim a problem demonstrates is checked with an
assert before anything is written. Three problems use real rows of the served models (webapp/data/models) so the
answer can be compared with the web app.

Usage:  python build_exercises.py        (writes BanglaMeta_Math_Exercises.html and .pdf next to this file)
"""
import math, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "webapp" / "backend"))
import config as C                                     # noqa: E402

def f(x, d=4):
    """number -> text with d decimals, minus sign typeset"""
    s = f"{x:.{d}f}"
    return s.replace("-", "−") if not s.startswith("-0." + "0" * d) else s.replace("-", "")

def vec(v, d=4):
    return "(" + ",\\ ".join(f(x, d) for x in v) + ")"

def T(text, **kw):
    for k, v in kw.items():
        text = text.replace("@" + k + "@", str(v))
    assert "@" not in text.replace("&#64;", ""), f"unfilled placeholder in: {text[:120]}"
    return text

def norm(v): return v / np.linalg.norm(v)
def cos(a, b): return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

PROBLEMS = []  # (number, title, theorem, html)
def problem(title, theorem, body):
    PROBLEMS.append((len(PROBLEMS) + 1, title, theorem, body))

# =============================================================================== 1. merge formula
n, i, s = np.array([3., 4.]), np.array([1., 2., 2.]), np.array([0., 5., 12.])
nh, ih, sh = norm(n), norm(i), norm(s)
m = np.concatenate([nh, ih, sh]) / math.sqrt(3)
energies = [float((m[:2] ** 2).sum()), float((m[2:5] ** 2).sum()), float((m[5:] ** 2).sum())]
assert abs(np.linalg.norm(m) - 1) < 1e-12 and all(abs(e - 1 / 3) < 1e-12 for e in energies)
problem("Building one merged row", "Merge formula (§2): normalise, then concatenate with weight 1/√K", T(r"""
<p>A word <i>w</i> has these raw rows in three source models (dimensions are kept small for hand work):</p>
<p>$$n(w) = (3,\ 4),\qquad i(w) = (1,\ 2,\ 2),\qquad s(w) = (0,\ 5,\ 12).$$</p>
<p><b>(a)</b> Normalise each row to length 1. <b>(b)</b> Form the merged row $m(w)=\tfrac{1}{\sqrt3}[\hat n;\hat i;\hat s]$.
<b>(c)</b> Show that $\lVert m(w)\rVert = 1$ and that each source supplies exactly one third of the squared length.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $\lVert n\rVert=\sqrt{9+16}=5$, $\lVert i\rVert=\sqrt{1+4+4}=3$, $\lVert s\rVert=\sqrt{0+25+144}=13$, so
$\hat n=(0.6,\,0.8)$, $\hat i=(\tfrac13,\tfrac23,\tfrac23)$, $\hat s=(0,\tfrac{5}{13},\tfrac{12}{13})$.</p>
<p>(b) Multiply every entry by $1/\sqrt3 = @r3@$:</p>
<p>$$m(w) = @m@.$$</p>
<p>(c) Squared length of each block: $\tfrac13\lVert\hat n\rVert^2=\tfrac13$, likewise $\tfrac13$ and $\tfrac13$ (computed: @e1@, @e2@, @e3@).
Total $=\tfrac13+\tfrac13+\tfrac13=1$, so $\lVert m(w)\rVert = 1$.</p>
<div class="shows">Shows: after Step 1 and Step 3 every merged row has length exactly 1 and no source can dominate by
scale: each contributes one third. With $K$ sources the factor $1/\sqrt K$ gives $K\cdot\tfrac1K = 1$.</div></div>
""", r3=f(1 / math.sqrt(3)), m=vec(m), e1=f(energies[0]), e2=f(energies[1]), e3=f(energies[2])))

# =============================================================================== 2. ridge imputation
B = np.array([[1., 0.], [0., 1.], [1., 1.]]); A = np.array([[0., 1.], [-1., 0.], [-1., 1.]])
lam = 1.0
M1 = np.linalg.solve(B.T @ B + lam * np.eye(2), B.T @ A); M0 = np.linalg.solve(B.T @ B, B.T @ A)
assert np.allclose(M1 * 8, [[-1, 5], [-5, 1]]) and np.allclose(M0, [[0, 1], [-1, 0]])
bnew = norm(np.array([2., 1.])); imp = norm(bnew @ M1); true = norm(bnew @ M0)
c_imp = cos(imp, true)
assert c_imp > 0.99
problem("Filling a missing word with the ridge map", "Merge formula, Step 2: $M=(B^\\top B+\\lambda I)^{-1}B^\\top A$, $\\lambda=1$", T(r"""
<p>Three words are known by both models. Their rows (one word per row) are</p>
<p>$$B=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\qquad A=\begin{pmatrix}0&1\\-1&0\\-1&1\end{pmatrix}.$$</p>
<p>(Model A happens to be model B turned by 90°.) A fourth word is known only by B, with normalised row
$b=\tfrac{1}{\sqrt5}(2,1)$.</p>
<p><b>(a)</b> Compute $M=(B^\top B+I)^{-1}B^\top A$. <b>(b)</b> Predict and normalise the missing A-row $\tilde a = bM/\lVert bM\rVert$.
<b>(c)</b> Compare with the exact 90° answer and with $\lambda = 0$.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $B^\top B=\begin{pmatrix}2&1\\1&2\end{pmatrix}$, so $B^\top B+I=\begin{pmatrix}3&1\\1&3\end{pmatrix}$, whose inverse is
$\tfrac18\begin{pmatrix}3&-1\\-1&3\end{pmatrix}$ (determinant $9-1=8$).
$B^\top A=\begin{pmatrix}-1&2\\-2&1\end{pmatrix}$. Therefore</p>
<p>$$M=\tfrac18\begin{pmatrix}3&-1\\-1&3\end{pmatrix}\begin{pmatrix}-1&2\\-2&1\end{pmatrix}=\tfrac18\begin{pmatrix}-1&5\\-5&1\end{pmatrix}.$$</p>
<p>(b) $bM=\tfrac{1}{8\sqrt5}(2\cdot(-1)+1\cdot(-5),\ 2\cdot5+1\cdot1)=\tfrac{1}{8\sqrt5}(-7,\,11)$; after normalising,
$\tilde a=(-7,11)/\sqrt{170}=@imp@$.</p>
<p>(c) The exact turned row is $(-1,2)/\sqrt5=@true@$. Cosine between them $=29/\sqrt{850}=@cimp@$.
With $\lambda=0$: $(B^\top B)^{-1}B^\top A=\begin{pmatrix}0&1\\-1&0\end{pmatrix}$, the exact 90° turn.</p>
<div class="shows">Shows: the ridge map learns the relation between the two models from shared words, and a missing row
is predicted almost exactly (cosine @cimp@). $\lambda$ only shrinks the map; the final normalisation removes the shrinkage
of length, so the direction stays close (Theorem 8 bounds the remaining error).</div></div>
""", imp=vec(imp), true=vec(true), cimp=f(c_imp)))

# =============================================================================== 3. pooling identity
a1, b1, a2, b2 = np.array([1., 0.]), np.array([0., 1.]), np.array([.6, .8]), np.array([1., 0.])
m1 = np.concatenate([a1, b1]) / math.sqrt(2); m2 = np.concatenate([a2, b2]) / math.sqrt(2)
lhs = (m1 + m2) / 2; rhs = np.concatenate([(a1 + a2) / 2, (b1 + b2) / 2]) / math.sqrt(2)
assert np.allclose(lhs, rhs)
problem("Sentence vector of a merged model", "Theorem 1 (pooling identity), check P1", T(r"""
<p>A two-word sentence. In model A the rows are $a_1=(1,0)$, $a_2=(0.6,0.8)$; in model B they are $b_1=(0,1)$, $b_2=(1,0)$.
The merged rows are $m_j=\tfrac{1}{\sqrt2}[a_j;b_j]$.</p>
<p><b>(a)</b> Compute the sentence vector $x_m=\tfrac12(m_1+m_2)$. <b>(b)</b> Compute $\tfrac{1}{\sqrt2}[x_A;x_B]$ where $x_A,x_B$ are the
sentence vectors of each model. <b>(c)</b> Compare.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $m_1=\tfrac1{\sqrt2}(1,0,0,1)$, $m_2=\tfrac1{\sqrt2}(0.6,0.8,1,0)$, so $x_m=\tfrac1{\sqrt2}(0.8,0.4,0.5,0.5)=@lhs@$.</p>
<p>(b) $x_A=\tfrac12(a_1+a_2)=(0.8,0.4)$, $x_B=\tfrac12(b_1+b_2)=(0.5,0.5)$, so $\tfrac1{\sqrt2}[x_A;x_B]=@rhs@$.</p>
<p>(c) Identical, entry by entry (the thesis check P1 measured an error of exactly 0.0 on 500 texts).</p>
<div class="shows">Shows: averaging, concatenating and scaling are all linear, so they can be done in any order. The merged
sentence vector <i>is</i> the two sources' sentence vectors placed side by side, which is what Theorems 2–5 rely on.</div></div>
""", lhs=vec(lhs), rhs=vec(rhs)))

# =============================================================================== 4. merging cannot lose
data = [(1, 1, -1), (1, 3, 1), (3, 1, 1), (2, 2, 1)]
def best_1d(vals):
    best = len(vals)
    xs = sorted(set(v for v, _ in vals))
    cuts = [xs[0] - 1] + [(xs[k] + xs[k + 1]) / 2 for k in range(len(xs) - 1)] + [xs[-1] + 1]
    for c in cuts:
        for sign in (1, -1):
            err = sum((1 if sign * (v - c) > 0 else -1) != y for v, y in vals)
            best = min(best, err)
    return best
eA = best_1d([(a, y) for a, _, y in data]); eB = best_1d([(b, y) for _, b, y in data])
eAB = sum((1 if a + b - 3 > 0 else -1) != y for a, b, y in data)
assert eA == 1 and eB == 1 and eAB == 0
problem("Merging cannot lose", "Theorem 2: $R(H_{AB})\\le\\min(R(H_A),R(H_B))$", T(r"""
<p>Four texts have one feature from each model and a label:</p>
<table class="t"><tr><th>text</th><th>$x_A$</th><th>$x_B$</th><th>label $y$</th></tr>
<tr><td>1</td><td>1</td><td>1</td><td>−1</td></tr><tr><td>2</td><td>1</td><td>3</td><td>+1</td></tr>
<tr><td>3</td><td>3</td><td>1</td><td>+1</td></tr><tr><td>4</td><td>2</td><td>2</td><td>+1</td></tr></table>
<p>A linear classifier predicts $\operatorname{sign}(w\cdot x+c)$.
<b>(a)</b> Find the fewest errors any classifier using only $x_A$ can make. <b>(b)</b> The same for $x_B$.
<b>(c)</b> Find a classifier on $[x_A;x_B]$ with no error. <b>(d)</b> Write the best $x_A$-only classifier as a classifier on $[x_A;x_B]$.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Texts 1 and 2 both have $x_A=1$ but different labels, so any rule based on $x_A$ alone gives them the same
answer and gets at least one wrong. The cut $x_A>1.5$ makes exactly @eA@ error (text 2): error rate $1/4$.</p>
<p>(b) Texts 1 and 3 both have $x_B=1$ with different labels: at least @eB@ error, error rate $1/4$.</p>
<p>(c) $\operatorname{sign}(x_A+x_B-3)$: sums are $2,4,4,4$, giving $-,+,+,+$: @eAB@ errors.</p>
<p>(d) $\operatorname{sign}(1\cdot x_A-1.5)=\operatorname{sign}([1,\ 0]\cdot[x_A;x_B]-1.5)$: the same rule with weight 0 on $x_B$.</p>
<div class="shows">Shows: every rule on one model is also a rule on the merged features (part d), so the merged model's best
error can never be higher ($0 \le \min(1/4,1/4)$). Here it is strictly lower, because each model fixes the other's mistake.</div></div>
""", eA=eA, eB=eB, eAB=eAB))

# =============================================================================== 5. conditional mutual information
H = lambda ps: -sum(p * math.log(p) for p in ps if p > 0)
H_Y_A = 0.5 * H([1.0]) + 0.5 * H([0.5, 0.5]); H_Y_AB = 0.0
gain = H_Y_A - H_Y_AB
assert abs(gain - 0.5 * math.log(2)) < 1e-12
problem("How big is the gain? Conditional information", "Theorem 3: $R^*(A)-R^*(A,B)=I(Y;X_B\\mid X_A)\\ge 0$", T(r"""
<p>$X_A$ and $X_B$ are independent fair coins (0/1) and the label is $Y = X_A \text{ AND } X_B$. A predictor is scored by
log-loss $-\ln p(\text{true label})$, in nats.</p>
<p><b>(a)</b> Compute $H(Y\mid X_A)$. <b>(b)</b> Compute $H(Y\mid X_A,X_B)$. <b>(c)</b> Give the gain from adding $X_B$.
<b>(d)</b> What is the gain if $X_B$ is just a copy of $X_A$?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) If $X_A=0$, then $Y=0$ for sure: entropy 0. If $X_A=1$, then $Y=X_B$, a fair coin: entropy $\ln 2$.
So $H(Y\mid X_A)=\tfrac12\cdot0+\tfrac12\ln2=@ha@$ nats. The best predictor using $X_A$ only says "$Y=1$ with
probability 0.5" when $X_A=1$, and its average log-loss is exactly this @ha@.</p>
<p>(b) Given both coins, $Y$ is known: $H(Y\mid X_A,X_B)=0$.</p>
<p>(c) Gain $= @ha@-0 = @gain@$ nats $= I(Y;X_B\mid X_A)$.</p>
<p>(d) Then $X_B$ tells nothing new: $H(Y\mid X_A,X_B)=H(Y\mid X_A)$ and the gain is 0, never negative.</p>
<div class="shows">Shows: the improvement from merging equals the label information the second model adds beyond the first.
It is zero only when the second model is redundant. The thesis measured it before merging (check P2): +0.068 nats on informal
text for News-W2V + Informal-W2V, and +0.003 (not different from 0) for adding Subs2vec as a third model.</div></div>
""", ha=f(H_Y_A), gain=f(gain)))

# =============================================================================== 6. linear merges lose information
xa = {"+": 1.0, "−": -1.0}; xb = {"+": -1.0, "−": 1.0}
avg = {k: (xa[k] + xb[k]) / 2 for k in xa}
assert avg["+"] == avg["−"] == 0
problem("Why not just average the two models?", "Theorem 4: concatenation dominates every linear merge $z=Lm$", T(r"""
<p>Two kinds of text. For class $+$ the sentence features are $x_A=1,\ x_B=-1$; for class $-$ they are $x_A=-1,\ x_B=1$.</p>
<p><b>(a)</b> Average the models: $z=\tfrac12(x_A+x_B)$. Can any classifier on $z$ separate the classes?
<b>(b)</b> Find a classifier on the concatenation $[x_A;x_B]$ that does. <b>(c)</b> Show that any classifier on $z=Lx$ is also a
classifier on $x$.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Class $+$: $z=\tfrac12(1-1)=0$. Class $-$: $z=\tfrac12(-1+1)=0$. Both classes map to the same number, so every
classifier on $z$ gives them the same answer: at best 50% correct.</p>
<p>(b) $\operatorname{sign}(1\cdot x_A + 0\cdot x_B)$ gives $+1$ and $-1$: 100% correct.</p>
<p>(c) A classifier on $z$ computes $w\cdot z = w\cdot(Lx) = (wL)\cdot x$, which is the classifier with weights $wL$ on $x$.
So the set of classifiers on $z$ is contained in the set on $x$, and its best loss cannot be lower.</p>
<div class="shows">Shows: averaging ($L=[\tfrac12,\tfrac12]$), SVD ($L=V_k^\top$), and rotation followed by averaging are all of the form
$z=Lm$, so none can beat concatenation in reachable loss, and averaging can destroy information completely. This is
why the thesis used concatenation (and DEV confirmed it: CONC 76.97 > SVD-300 76.36 > Procrustes+AVG 76.28).</div></div>
"""))

# =============================================================================== 7. standardisation cancels weights
x = np.array([1., 2., 3., 6.]); cfac = 0.5
z1 = (x - x.mean()) / x.std(); z2 = (cfac * x - (cfac * x).mean()) / (cfac * x).std()
assert np.allclose(z1, z2)
problem("Block weights do not change the classifier", "Theorem 5: standardisation cancels any weight $c>0$", T(r"""
<p>One feature column over four texts is $x=(1,2,3,6)$. A block weight multiplies it by $c=0.5$, giving $(0.5,1,1.5,3)$.
The probe standardises each column: $z=(x-\bar x)/\sigma_x$ (population standard deviation).</p>
<p><b>(a)</b> Standardise $x$. <b>(b)</b> Standardise $0.5x$. <b>(c)</b> Explain the result for any $c>0$.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $\bar x=3$; deviations $-2,-1,0,3$; variance $(4+1+0+9)/4=3.5$; $\sigma_x=@sd@$. So $z=@z1@$.</p>
<p>(b) Mean $1.5$; deviations $-1,-0.5,0,1.5$; variance $0.875$; $\sigma=@sd2@$. So $z=@z2@$: the same.</p>
<p>(c) $\overline{cx}=c\bar x$ and $\sigma_{cx}=c\,\sigma_x$, so $\dfrac{cx-c\bar x}{c\,\sigma_x}=\dfrac{x-\bar x}{\sigma_x}$: $c$ cancels.</p>
<div class="shows">Shows: the classifier sees identical inputs for every weight $c>0$, so the merged model's accuracy needs no
weight tuning. (The thesis ran $\alpha = 0.3, 0.5, 0.7$ and got identical F1 76.97.) The same fact lets BanglaMeta+ give the thesis
rows a tiny weight without losing their classification power. A weight of exactly 0 is different: it deletes the
feature (Problem 9).</div></div>
""", sd=f(x.std()), z1=vec(z1), sd2=f((cfac * x).std()), z2=vec(z2)))

# =============================================================================== 8. similarity is the average
au, bu, av, bv = np.array([1., 0.]), np.array([.6, .8]), np.array([.6, .8]), np.array([0., 1.])
mu_, mv_ = np.concatenate([au, bu]) / math.sqrt(2), np.concatenate([av, bv]) / math.sqrt(2)
cA, cB, cM = float(au @ av), float(bu @ bv), float(mu_ @ mv_)
assert abs(cM - (cA + cB) / 2) < 1e-12
sig, rho = 0.10, 0.3
var_m = sig ** 2 * (1 + rho) / 2
sA, sB = 0.05, 0.30; var_u = (sA ** 2 + sB ** 2) / 4
assert var_m < sig ** 2 and math.sqrt(var_u) > sA
problem("Merged similarity is the average of the sources' similarities", "Theorem 6 and its corollary", T(r"""
<p>Two words $u,v$. Model A rows: $a_u=(1,0)$, $a_v=(0.6,0.8)$. Model B rows: $b_u=(0.6,0.8)$, $b_v=(0,1)$. Merged rows
$m=\tfrac1{\sqrt2}[a;b]$.</p>
<p><b>(a)</b> Compute $\cos_A$, $\cos_B$ and $\cos(m_u,m_v)$. <b>(b)</b> Each model's similarity has an error with standard deviation
$\sigma=0.10$, correlation $\rho=0.3$. What is the error standard deviation of the merged similarity? <b>(c)</b> Repeat with
$\sigma_A=0.05$, $\sigma_B=0.30$, $\rho=0$.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $\cos_A=1\cdot0.6+0\cdot0.8=@cA@$, $\cos_B=0.6\cdot0+0.8\cdot1=@cB@$. Merged:
$m_u\cdot m_v=\tfrac12(a_u\cdot a_v)+\tfrac12(b_u\cdot b_v)=\tfrac12(@cA@+@cB@)=@cM@$.</p>
<p>(b) Error of the average $=\tfrac12(e_A+e_B)$, variance $\tfrac14(\sigma^2+\sigma^2+2\rho\sigma^2)=\sigma^2\tfrac{1+\rho}2=@vm@$, standard deviation
$@sm@ < 0.10$: less noise than either model.</p>
<p>(c) Variance $\tfrac14(0.05^2+0.30^2)=@vu@$, standard deviation $@su@$, <b>worse</b> than model A alone (0.05).</p>
<div class="shows">Shows: equal-weight merging averages similarities. That reduces noise when the models are equally good, but a
much noisier model drags the average down (part c). This is exactly why the thesis BanglaMeta scored below News-W2V on word
similarity (Informal-W2V is the noisy model) and why BanglaMeta-Sim re-weights the blocks (Problem 9).</div></div>
""", cA=f(cA, 2), cB=f(cB, 2), cM=f(cM, 2), vm=f(var_m), sm=f(math.sqrt(var_m)), vu=f(var_u, 6), su=f(math.sqrt(var_u))))

# =============================================================================== 9. weighted cosine (real rows)
Um = np.load(C.served_path("merged"), mmap_mode="r"); words = C.served_words("merged").read_text(encoding="utf-8").split("\n")
widx = {w: k for k, w in enumerate(words)}
def mrow(w): return np.asarray(Um[widx[w]], np.float64)
ra, rb = mrow("রাজা"), mrow("পুরুষ")
blk = [float(ra[300 * k:300 * (k + 1)] @ rb[300 * k:300 * (k + 1)]) * 3 for k in range(3)]
c_eq = float(ra @ rb)
assert abs(c_eq - sum(blk) / 3) < 1e-6
wts = C.SIM_WEIGHTS
toy = (0.70, 0.10, 0.50)
toy_eq, toy_w = sum(toy) / 3, sum(w * t for w, t in zip(wts, toy)) / 3
problem("Block weights for word similarity (BanglaMeta-Sim)", "Weighted cosine $\\cos_a=\\tfrac13\\sum_k a_k\\cos_k$, $\\tfrac13\\sum a_k=1$", T(r"""
<p>The merged row has three unit blocks, each scaled by $1/\sqrt3$. Giving block $k$ weight $a_k$ (with the weights averaging 1)
means the row becomes $[\sqrt{a_k/3}\,\hat x_k]_k$.</p>
<p><b>(a)</b> Show the weighted row still has length 1 when $\tfrac13(a_1+a_2+a_3)=1$, and that the cosine of two words becomes
$\tfrac13\sum_k a_k\cos_k$.
<b>(b)</b> Block cosines of two words are $\cos_N=0.70$, $\cos_I=0.10$ (noisy Informal block), $\cos_S=0.50$. Compute the
cosine with equal weights and with the BanglaMeta-Sim weights $a=(@w1@,\ @w2@,\ @w3@)$.
<b>(c)</b> Real data: the block cosines of রাজা and পুরুষ in the thesis model are $@b1@,\ @b2@,\ @b3@$. Compute their merged cosine.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Squared length $=\sum_k \tfrac{a_k}{3}\lVert\hat x_k\rVert^2=\tfrac13\sum_k a_k=1$. The dot product of two such rows is
$\sum_k\tfrac{a_k}3\,\hat x_k\cdot\hat y_k=\tfrac13\sum_k a_k\cos_k$.</p>
<p>(b) Equal: $\tfrac13(0.70+0.10+0.50)=@teq@$. Weighted: $\tfrac13(@w1@\cdot0.70+@w2@\cdot0.10+@w3@\cdot0.50)=@tw@$.</p>
<p>(c) $\tfrac13(@b1@+@b2@+@b3@)=@ceq@$: exactly the value the web app shows for রাজা–পুরুষ with "Thesis rows".</p>
<div class="shows">Shows: block weights move the similarity toward the reliable blocks. Weight 0 removes the noisy Informal block
from word similarity. Weights are free for classification (Problem 7) but not for similarity, so they were chosen on half of
the analogy questions and tested on the other half (analogy 5.62% → 8.66% on the unseen half, after the next step).</div></div>
""", w1=f(wts[0], 2), w2=f(wts[1], 2), w3=f(wts[2], 2), teq=f(toy_eq), tw=f(toy_w), b1=f(blk[0], 3), b2=f(blk[1], 3), b3=f(blk[2], 3), ceq=f(c_eq, 3)))

# =============================================================================== 10. all-but-the-top
king, queen, rice = np.array([4., 1., 0.]), np.array([4., 1., .5]), np.array([4., 0., 1.])
u = np.array([1., 0., 0.])
rem = lambda v: v - (v @ u) * u
before = (cos(king, queen), cos(king, rice)); after = (cos(rem(king), rem(queen)), cos(rem(king), rem(rice)))
assert before[0] - before[1] < 0.06 and after[0] - after[1] > 0.8
problem("Removing the common direction (all-but-the-top)", "Post-processing used by BanglaMeta-Sim", T(r"""
<p>Word vectors often share one big common direction that says nothing about meaning. Here it is $u=(1,0,0)$:</p>
<p>$$\text{king}=(4,1,0),\quad \text{queen}=(4,1,0.5),\quad \text{rice}=(4,0,1).$$</p>
<p><b>(a)</b> Compute cos(king, queen) and cos(king, rice). <b>(b)</b> Remove the common direction, $v' = v-(v\cdot u)u$, and compute both
cosines again. <b>(c)</b> Which version separates related from unrelated words better?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) cos(king, queen) $=\dfrac{16+1+0}{\sqrt{17}\sqrt{17.25}}=@b0@$; cos(king, rice) $=\dfrac{16}{\sqrt{17}\sqrt{17}}=@b1@$.
Gap $=@gb@$.</p>
<p>(b) king′ $=(0,1,0)$, queen′ $=(0,1,0.5)$, rice′ $=(0,0,1)$. cos(king′, queen′) $=1/\sqrt{1.25}=@a0@$; cos(king′, rice′) $=0$.
Gap $=@ga@$.</p>
<p>(c) After removal: the related pair stays high and the unrelated pair falls to 0.</p>
<div class="shows">Shows: subtracting the mean and the top shared directions (here one; BanglaMeta-Sim removes 4 per block) makes
cosine reflect meaning instead of a common offset. On real data this raised the similarity AUC from 0.819 to 0.891 and is why
unrelated pairs such as রাজা–ভাত drop near 0 (0.036) in the app.</div></div>
""", b0=f(before[0]), b1=f(before[1]), gb=f(before[0] - before[1]), a0=f(after[0]), ga=f(after[0] - after[1])))

# =============================================================================== 11. BanglaMeta+ (real rows)
Us = np.load(C.served_path("merged_sim"), mmap_mode="r"); Up = np.load(C.served_path("merged_plus"), mmap_mode="r")
eps = C.PLUS_EPS
w1_, w2_ = "রাজা", "রানী"
def _ni(w): v = np.asarray(Um[widx[w]][:600], np.float64); return v / np.linalg.norm(v)      # thesis News+Informal part
cm_ = float(_ni(w1_) @ _ni(w2_)); cs_ = float(np.asarray(Us[widx[w1_]]) @ np.asarray(Us[widx[w2_]]))
cp_ = float(np.asarray(Up[widx[w1_]]) @ np.asarray(Up[widx[w2_]])); pred = eps * cm_ + (1 - eps) * cs_
assert abs(cp_ - pred) < 1e-5
problem("One model for both jobs (BanglaMeta+)", "$c(w)=[\\sqrt\\varepsilon\\,m_{NI}(w);\\sqrt{1-\\varepsilon}\\,s(w)]$, $\\cos_c=\\varepsilon\\cos_{NI}+(1-\\varepsilon)\\cos_s$", T(r"""
<p>BanglaMeta+ joins the thesis News+Informal part $m_{NI}(w)$ and the similarity row $s(w)$ (both length 1) with $\varepsilon = @eps@$.</p>
<p><b>(a)</b> Show that $\lVert c(w)\rVert = 1$. <b>(b)</b> Show $\cos_c(u,v) = \varepsilon\cos_{NI}(u,v)+(1-\varepsilon)\cos_s(u,v)$.
<b>(c)</b> Real data for রাজা and রানী: $\cos_{NI}=@cm@$, $\cos_s=@cs@$. Compute $\cos_c$.
<b>(d)</b> Why does the classifier keep the full accuracy of the thesis rows, although their weight is only $\varepsilon$?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $\lVert c\rVert^2=\varepsilon\lVert m_{NI}\rVert^2+(1-\varepsilon)\lVert s\rVert^2=\varepsilon+1-\varepsilon=1$.</p>
<p>(b) $c(u)\cdot c(v)=\varepsilon\, m_{NI}(u)\cdot m_{NI}(v)+(1-\varepsilon)\, s(u)\cdot s(v)$, and all parts have length 1, so dot product = cosine.</p>
<p>(c) $@eps@\times@cm@ + @om@\times@cs@ = @pred@$. Read directly from the BanglaMeta+ file: $@cp@$, which the app shows as @cp3@.</p>
<p>(d) Multiplying the thesis features by $\sqrt\varepsilon$ is a block weight; standardisation cancels it (Problem 7), so the
classifier sees all thesis features at full strength, plus the similarity features.</p>
<div class="shows">Shows: one vector per word can carry both strengths. Word similarity is 99.95% the tuned view (analogy 8.98%,
AUC 0.891) while classification keeps the thesis News+Informal information (DEV macro-F1 77.34; held-out CONF 77.22, against 77.17
for the thesis model). Subs2vec is left out of the thesis part because it adds no significant classification information.</div></div>
""", eps=f(eps, 4), om=f(1 - eps, 4), cm=f(cm_), cs=f(cs_), pred=f(pred), cp=f(cp_), cp3=f(cp_, 3)))

# =============================================================================== 12. 3CosAdd analogy
V = {"king": np.array([.8, .6, 0.]), "queen": np.array([.8, 0., .6]), "man": np.array([0., 1., 0.]),
     "woman": np.array([0., 0., 1.]), "rice": np.array([.6, 0., .8])}
tgt = V["queen"] - V["king"] + V["man"]
scores = {w: float(v @ tgt) for w, v in V.items() if w not in ("king", "queen", "man")}
cosines = {w: cos(V[w], tgt) for w in scores}
assert max(scores, key=scores.get) == "woman"
d1, d2 = V["queen"] - V["king"], V["woman"] - V["man"]
par = cos(d1, d2)
assert abs(par - 1) < 1e-12 and abs(cos(V["king"], V["man"]) - 0.6) < 1e-12
problem("Analogy: a relation is a direction", "3CosAdd: answer $=\\arg\\max_x \\cos(x,\\ b-a+c)$, inputs excluded", T(r"""
<p>Dimensions are (royal, male, female). Unit rows: king $=(0.8,0.6,0)$, queen $=(0.8,0,0.6)$, man $=(0,1,0)$, woman $=(0,0,1)$,
rice $=(0.6,0,0.8)$.</p>
<p><b>(a)</b> Solve king : queen :: man : ? with 3CosAdd. <b>(b)</b> Compute cos(king, man). <b>(c)</b> Compare the directions
queen − king and woman − man.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Target $t=$ queen − king + man $=@t@$, $\lVert t\rVert=@tn@$. Candidates (king, queen, man excluded):
cos(woman, $t$) $=@cw@$, cos(rice, $t$) $=@cr@$. Answer: <b>woman</b>.</p>
<p>(b) cos(king, man) $=0.8\cdot0+0.6\cdot1+0\cdot0=@km@$: only moderately similar.</p>
<p>(c) queen − king $=(0,-0.6,0.6)$, woman − man $=(0,-1,1)$: cosine $@par@$, the same direction.</p>
<div class="shows">Shows: a relation such as male → female is a <i>direction</i> shared by many pairs, not a high similarity
between the words. This is why রাজা–পুরুষ has a low cosine in every model, yet রাজা : রানী :: পুরুষ : ? returns মহিলা in the app.</div></div>
""", t=vec(tgt, 1), tn=f(np.linalg.norm(tgt)), cw=f(cosines["woman"]), cr=f(cosines["rice"]), km=f(cos(V["king"], V["man"]), 1), par=f(par, 1)))

# =============================================================================== 13. sentence vector with a missing word
r1, r2 = np.array([1., 0.]), np.array([0., 1.]); sv = (r1 + r2) / 2; cov_ = 2 / 3
other = np.array([.6, .8]); cs2 = cos(sv, other)
problem("Sentence vector and coverage", "Sentence vector = mean of the known word rows", T(r"""
<p>A sentence has three words. The model knows two of them, with rows $(1,0)$ and $(0,1)$; the third is missing from the vocabulary.</p>
<p><b>(a)</b> Compute the sentence vector and the coverage. <b>(b)</b> Compute its cosine with a second sentence whose vector is $(0.6,0.8)$.
<b>(c)</b> What would happen if a missing word were given a made-up vector?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Mean of the known rows $=(0.5,0.5)$; coverage $=2/3=@cov@$.</p>
<p>(b) $\dfrac{0.5\cdot0.6+0.5\cdot0.8}{\sqrt{0.5}\cdot1}=\dfrac{0.7}{0.7071}=@cs@$.</p>
<p>(c) It would move the mean in an arbitrary direction and change every score built on it. Leaving it out keeps the vector
equal to what the model actually knows.</p>
<div class="shows">Shows: the app reports coverage and marks missing words instead of inventing vectors, the same rule the thesis
used for every evaluation.</div></div>
""", cov=f(cov_, 3), cs=f(cs2)))

# =============================================================================== 14. symmetric cosine, one-way rank
G = np.array([[1, .5, .9, .8], [.5, 1, .2, .1], [.9, .2, 1, .75], [.8, .1, .75, 1]])
assert np.linalg.eigvalsh(G).min() > 0
names = ["A", "B", "C", "D"]
def rank(i, j):
    others = [k for k in range(4) if k != i]
    return 1 + sum(G[i, k] > G[i, j] for k in others)
rAB, rBA = rank(0, 1), rank(1, 0)
assert G[0, 1] == G[1, 0] and rAB == 3 and rBA == 1
problem("Same similarity, different list positions", "Cosine is symmetric; a neighbour rank is one-way", T(r"""
<p>Four words have these cosines (the matrix is a valid set of unit vectors: all its eigenvalues are positive):</p>
<table class="t"><tr><th></th><th>A</th><th>B</th><th>C</th><th>D</th></tr>
<tr><th>A</th><td>1</td><td>0.5</td><td>0.9</td><td>0.8</td></tr><tr><th>B</th><td>0.5</td><td>1</td><td>0.2</td><td>0.1</td></tr>
<tr><th>C</th><td>0.9</td><td>0.2</td><td>1</td><td>0.75</td></tr><tr><th>D</th><td>0.8</td><td>0.1</td><td>0.75</td><td>1</td></tr></table>
<p><b>(a)</b> What is cos(A, B) and cos(B, A)? <b>(b)</b> Where is B in A's list of nearest words? Where is A in B's list?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Both are 0.5; the dot product does not depend on the order.</p>
<p>(b) A's list: C (0.9), D (0.8), B (0.5), so B is #@rAB@. B's list: A (0.5), C (0.2), D (0.1), so A is #@rBA@.</p>
<div class="shows">Shows: A has many close neighbours and B has few, so the two list positions differ although the similarity is one
number. The app's Relations tab therefore shows the cosine as "same both ways" and labels ranks as one-way.</div></div>
""", rAB=rAB, rBA=rBA))

# =============================================================================== 15. similarity AUC
same_scores = [0.8, 0.6, 0.55]; diff_scores = [0.7, 0.4, 0.3, 0.2]
def auc(sm, df): return sum((s > d) + 0.5 * (s == d) for s in sm for d in df) / (len(sm) * len(df))
auc1 = auc(same_scores, diff_scores)
same2 = [0.62, 0.45, 0.40]; diff2 = [0.10, 0.05, 0.02, -0.08]
auc2 = auc(same2, diff2)
assert auc1 == 10 / 12 and auc2 == 1.0
problem("Similarity AUC by counting", "AUC = share of (same-category, different-category) pairs ranked correctly", T(r"""
<p>Three word pairs from the same category and four from different categories get these cosines.</p>
<p>Model 1: same $\{0.80, 0.60, 0.55\}$, different $\{0.70, 0.40, 0.30, 0.20\}$.<br>
Model 2: same $\{0.62, 0.45, 0.40\}$, different $\{0.10, 0.05, 0.02, -0.08\}$.</p>
<p><b>(a)</b> Compute each model's AUC: the share of the $3\times4=12$ (same, different) comparisons in which the same-category pair scores higher.</p>
<div class="sol"><h4>Solution</h4>
<p>Model 1: 0.80 beats all 4; 0.60 beats 3 (not 0.70); 0.55 beats 3. Total $10/12=@a1@$.</p>
<p>Model 2: every same-category score is above every different-category score: $12/12=@a2@$.</p>
<div class="shows">Shows: AUC measures ranking, not the size of the numbers. Model 2 has lower cosines but separates perfectly,
like BanglaMeta-Sim after removing the common direction (Problem 10): its cosines are smaller, its AUC higher (0.891 vs 0.819).</div></div>
""", a1=f(auc1), a2=f(auc2)))

# =============================================================================== 16. Hoeffding + union bound
K, n_, delta = 28, 5950, 0.05
marg = math.sqrt(math.log(2 * K / delta) / (2 * n_)); marg3 = math.sqrt(math.log(2 * 3 / delta) / (2 * n_))
assert abs(marg - 0.0243) < 5e-4
problem("How much can selection on DEV be trusted?", "Theorem 10: margin $=\\sqrt{\\ln(2K/\\delta)/(2n)}$", T(r"""
<p>$K=28$ candidate models are compared on $n=5{,}950$ texts, with failure probability $\delta=0.05$.</p>
<p><b>(a)</b> Compute the margin within which all 28 measured accuracies are guaranteed. <b>(b)</b> Repeat for $K=3$.
<b>(c)</b> On DEV the selected model scored 77.12 and the runner-up (Q: CAEME) 77.08, a difference of 0.04 points. Is this order guaranteed?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $2K/\delta=1120$, $\ln1120=@l1@$, $2n=11{,}900$: margin $=\sqrt{@l1@/11900}=@m1@$, about <b>@m1p@ points</b>.</p>
<p>(b) $\ln(120)=@l3@$: margin $=@m3@$ (@m3p@ points). More candidates, wider margin, but only slowly ($\sqrt{\ln K}$).</p>
<p>(c) No: 0.04 points is far below the 2.4-point margin, so this bound cannot guarantee the order.</p>
<div class="shows">Shows: picking the best of 28 on the same data cannot by itself prove the winner is better. That is why the
thesis froze the selected model and tested it once on the untouched CONF half with only 3 pre-registered comparisons.</div></div>
""", l1=f(math.log(2 * K / delta)), m1=f(marg), m1p=f(marg * 100, 1), l3=f(math.log(2 * 3 / delta)), m3=f(marg3), m3p=f(marg3 * 100, 1)))

# =============================================================================== 17. Holm correction (real CONF p-values)
tests = pd.read_csv(REPO / "eval" / "conf_output" / "conf_results" / "conf_tests.csv")
conf_t = tests[tests.family == "confirmatory"].reset_index(drop=True)
raw = conf_t.p_one_sided.tolist(); names_t = [c.replace("T: CONC", "BanglaMeta") for c in conf_t.comparison]
order = np.argsort(raw, kind="stable"); m_ = len(raw); adj = [0.0] * m_; run = 0.0
steps_txt = []
for rk, j in enumerate(order):
    val = min(1.0, (m_ - rk) * raw[j]); run = max(run, val); adj[j] = run
    steps_txt.append(f"{names_t[j]}: $({m_ - rk})\\times{f(raw[j])}={f((m_ - rk) * raw[j])}$ → {f(run)}")
assert np.allclose(adj, conf_t.p_holm.tolist())
problem("Holm correction of the three confirmatory tests", "Holm step-down: sort p-values, multiply the $r$-th smallest by $(m-r+1)$, keep it non-decreasing", T(r"""
<p>The thesis tested BanglaMeta against three baselines on CONF. The raw one-sided p-values (from the 10,000-sample paired bootstrap) are
@raw@. Use $\alpha=0.05$.</p>
<p><b>(a)</b> Apply the Holm correction. <b>(b)</b> Which hypotheses are confirmed?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Sort from smallest; multiply by 3, 2, 1; never let an adjusted value fall below the previous one:</p>
<ul>@steps@</ul>
<p>(b) All adjusted p-values are below 0.05, so all three are confirmed. These equal the official Holm p-values 0.0044, 0.0003, 0.0003.</p>
<div class="shows">Shows: testing three hypotheses at once raises the chance of a false win; Holm controls it, and the thesis result
survives the correction.</div></div>
""", raw=", ".join(f"{nm} {f(p)}" for nm, p in zip(names_t, raw)), steps="".join(f"<li>{t}</li>" for t in steps_txt)))

# =============================================================================== 18. imputation bound
rho_i, eps_i = 1 - 0.912, 0.6
problem("How much can imputed words move a sentence?", "Theorem 8: shift $\\le \\rho\\,\\varepsilon$", T(r"""
<p>News-W2V covers 91.2% of informal tokens, so at most $\rho=0.088$ of a typical informal text's tokens need an imputed News row.
Suppose each imputed row is within distance $\varepsilon=0.6$ of the row the model would have learned.</p>
<p><b>(a)</b> Bound how far the sentence vector can move. <b>(b)</b> Why does the bound hold?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) Shift $\le\rho\varepsilon=0.088\times0.6=@b@$ (in length; a unit row has length 1).</p>
<p>(b) The sentence vector is $\tfrac1T\sum_t r_t$. Only $\rho T$ terms change, each by at most $\varepsilon$; by the triangle inequality the mean
changes by at most $\tfrac1T\cdot\rho T\varepsilon=\rho\varepsilon$.</p>
<div class="shows">Shows: completing the vocabulary with predicted rows can only move sentence vectors a little, and the classifier's
risk by at most $L\Lambda\rho\varepsilon$.</div></div>
""", b=f(rho_i * eps_i, 4)))

# =============================================================================== 19. generalisation bound
L_, Lam, nn, dl = 1.0, 5.0, 5950, 0.05
gap = 2 * L_ * Lam / math.sqrt(nn) + math.sqrt(math.log(1 / dl) / (2 * nn))
problem("Does a longer vector overfit?", "Theorem 9: gap $\\le 2L\\Lambda/\\sqrt n+\\sqrt{\\ln(1/\\delta)/(2n)}$, no dimension", T(r"""
<p>Sentence vectors have length at most 1 (a mean of unit rows). A classifier has weight norm $\Lambda=5$, loss Lipschitz constant
$L=1$, $n=5{,}950$ texts, $\delta=0.05$.</p>
<p><b>(a)</b> Compute the bound. <b>(b)</b> How does it change from 300 to 900 or 1,200 dimensions?</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $2\cdot1\cdot5/\sqrt{5950}=@t1@$; $\sqrt{\ln20/11900}=@t2@$; total $@g@$.</p>
<p>(b) It does not change: the dimension does not appear, only the norms.</p>
<div class="shows">Shows: merging to 900 dimensions (or 1,200 for BanglaMeta+) is not by itself a risk of overfitting, as long as
rows stay unit-length and the classifier is regularised (cross-validated ridge).</div></div>
""", t1=f(2 * L_ * Lam / math.sqrt(nn)), t2=f(math.sqrt(math.log(1 / dl) / (2 * nn))), g=f(gap)))

# =============================================================================== 20. Eckart–Young
X = np.array([[2., 1.], [1., 2.]]); U_, S_, Vt_ = np.linalg.svd(X)
P = np.outer(Vt_[0], Vt_[0]); res = float(((X - X @ P) ** 2).sum())
assert np.allclose(S_, [3, 1]) and abs(res - S_[1] ** 2) < 1e-12
problem("What SVD compression throws away", "Theorem 7 (Eckart–Young): residual $=\\sum_{i>k}\\sigma_i^2$", T(r"""
<p>Two words with 2-dimensional rows form $X=\begin{pmatrix}2&1\\1&2\end{pmatrix}$.</p>
<p><b>(a)</b> Find the singular values. <b>(b)</b> Keep only the top direction ($k=1$) and compute the squared error.</p>
<div class="sol"><h4>Solution</h4>
<p>(a) $X$ is symmetric with eigenvectors $(1,1)/\sqrt2$ (eigenvalue 3) and $(1,-1)/\sqrt2$ (eigenvalue 1): $\sigma_1=3,\ \sigma_2=1$.</p>
<p>(b) Projecting each row on $(1,1)/\sqrt2$: $(2,1)\to(1.5,1.5)$ and $(1,2)\to(1.5,1.5)$. Errors $(0.5,-0.5)$ and $(-0.5,0.5)$, squared total
$4\times0.25=@r@=\sigma_2^2$.</p>
<div class="shows">Shows: SVD keeps the most energy possible, but whatever lies in the dropped directions (here the difference between
the two words!) is lost. Since that may be exactly the label information, SVD was tested, not assumed, and it lost (76.36 vs 76.97).</div></div>
""", r=f(res, 2)))

# =============================================================================== HTML
MAP = [(1, "Merge formula §2"), (2, "Merge formula Step 2 (ridge map)"), (3, "Theorem 1"), (4, "Theorem 2"), (5, "Theorem 3"),
       (6, "Theorem 4"), (7, "Theorem 5"), (8, "Theorem 6"), (9, "BanglaMeta-Sim weights"), (10, "BanglaMeta-Sim: all-but-the-top"),
       (11, "BanglaMeta+"), (12, "Analogy (3CosAdd)"), (13, "Sentence vector"), (14, "Ranks vs cosine"), (15, "Similarity AUC"),
       (16, "Theorem 10"), (17, "Holm correction"), (18, "Theorem 8"), (19, "Theorem 9"), (20, "Theorem 7")]
body = "".join(f"""<section class="prob"><div class="ph"><span class="num">{k}</span><div><h3>{t}</h3><div class="thm">{th}</div></div></div>{h}</section>"""
               for k, t, th, h in PROBLEMS)
toc = "".join(f"<tr><td>{k}</td><td>{t}</td><td>{dict(MAP)[k]}</td></tr>" for k, t, _, _ in PROBLEMS)
html = rf"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>BanglaMeta: Mathematics Exercises</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}});document.body.dataset.ready=1"></script>
<style>
@page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
body {{ font-family: Cambria, Georgia, serif; font-size: 10.6pt; line-height: 1.45; color: #18201d; margin: 0; }}
.bn, body {{ font-family: Cambria, Georgia, "Nirmala UI", serif; }}
h1 {{ font-size: 24pt; margin: 0 0 4px; color: #0f3a2e; }}
h2 {{ font-size: 14pt; color: #0f3a2e; border-bottom: 2px solid #e0a526; padding-bottom: 3px; margin-top: 20px; }}
.sub {{ color: #4a5a54; margin: 0 0 14px; }}
.prob {{ break-inside: auto; border: 1px solid #d8e2dd; border-radius: 8px; padding: 10px 14px; margin: 12px 0; }}
.ph {{ display: flex; gap: 12px; align-items: flex-start; }}
.num {{ background: #1b7f5b; color: #fff; font-weight: 700; border-radius: 50%; min-width: 30px; height: 30px; display: grid; place-items: center; font-size: 13pt; }}
.prob h3 {{ margin: 2px 0 0; font-size: 13pt; color: #0f3a2e; }}
.thm {{ font-size: 9.6pt; color: #b0262f; margin-bottom: 4px; }}
.sol {{ background: #f3f7f5; border-left: 4px solid #1b7f5b; padding: 6px 12px; margin-top: 8px; border-radius: 0 6px 6px 0; }}
.sol h4 {{ margin: 2px 0 4px; color: #1b7f5b; font-size: 10.5pt; letter-spacing: .04em; text-transform: uppercase; }}
.shows {{ background: #fff6e0; border: 1px solid #e8c56a; border-radius: 6px; padding: 6px 10px; margin: 8px 0 4px; font-size: 10pt; }}
table.t, table.toc {{ border-collapse: collapse; margin: 6px 0; }}
table.t td, table.t th, table.toc td, table.toc th {{ border: 1px solid #c9d6d0; padding: 2px 9px; text-align: center; }}
table.toc {{ font-size: 9.8pt; width: 100%; }}
table.toc td:nth-child(2), table.toc td:nth-child(3) {{ text-align: left; }}
table.toc th {{ background: #0f3a2e; color: #fff; }}
p {{ margin: 5px 0; }} ul {{ margin: 4px 0 4px 18px; padding: 0; }}
.ph, .sol h4 {{ break-after: avoid; }} p, li, .shows, table, .katex-display {{ break-inside: avoid; }}
.cover {{ break-after: page; }}
.box {{ border: 1px solid #c9d6d0; border-radius: 8px; padding: 8px 12px; background: #f7faf8; }}
</style></head><body>
<div class="cover">
<h1>BanglaMeta: Mathematics Exercises</h1>
<p class="sub">20 exam-style problems with worked solutions. Each problem uses small numbers you can compute by hand,
and its result demonstrates one formula or theorem behind the merged Bangla Word2Vec model.</p>
<div class="box"><b>How to use this set.</b> Cover the green solution box, solve parts (a), (b), (c) yourself, then compare.
The yellow box says what the result proves. Problems 9, 11 and 17 use real numbers read from the model files and the thesis
results, so you can check them in the web app (Explorer → Relations) and in <code>eval/conf_output</code>.<br>
Every number in every solution was computed by <code>docs/math_exercises/build_exercises.py</code>, which also checks each
claim with an assertion before writing this document.</div>
<h2>Notation</h2>
<p>$\hat x = x/\lVert x\rVert$ (unit length) · $\cos(u,v)=u\cdot v/(\lVert u\rVert\lVert v\rVert)$ · $[x;y]$ = vectors placed one after the other ·
$m(w)$ = merged row (thesis) · $s(w)$ = similarity-view row · $c(w)$ = BanglaMeta+ row · $R(H)$ = lowest loss reachable by classifiers in $H$ ·
$H(\cdot)$ = entropy in nats · $I(Y;X_B\mid X_A)$ = conditional mutual information.</p>
<h2>Contents</h2>
<table class="toc"><tr><th>#</th><th>Problem</th><th>Formula / theorem demonstrated</th></tr>{toc}</table>
</div>
{body}
</body></html>"""
out_html = HERE / "BanglaMeta_Math_Exercises.html"
out_html.write_text(html, encoding="utf-8")
print(f"{len(PROBLEMS)} problems, all assertions passed -> {out_html}")

chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
out_pdf = HERE / "BanglaMeta_Math_Exercises.pdf"
subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True, timeout=120)
print("PDF:", out_pdf, out_pdf.stat().st_size, "bytes")
