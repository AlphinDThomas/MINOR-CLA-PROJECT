# Derivation of ALS for Regularized Matrix Factorization (weighted version included)

*Author: Akhila. Matches the implementation in `als.py`. Equations use LaTeX ($...$); paste into Word with the equation editor or render in any Markdown/LaTeX viewer.*

## 1. Setup and notation

- $U$ users, $I$ items, $\Omega$ = set of **observed** (user, item) pairs; $\Omega_u$ = items rated by $u$, $\Omega_i$ = users who rated $i$. Missing entries are unknown, not zero.
- Rating $r_{ui}$, global mean $\mu$, user bias $b_u$, item bias $b_i$, latent vectors $p_u, q_i \in \mathbb{R}^k$.
- Prediction: $\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i$.
- Weight $w_{ui} \in [0,1]$ per rating (all ones = unweighted). $n_u = |\{i\in\Omega_u : w_{ui}>0\}|$, $m_i = |\{u\in\Omega_i : w_{ui}>0\}|$.

**Objective (weighted-$\lambda$ regularization):**
$$
J = \sum_{(u,i)\in\Omega} w_{ui}\big(r_{ui}-\mu-b_u-b_i-p_u^\top q_i\big)^2
 + \lambda\Big(\sum_u n_u\big(\|p_u\|^2+b_u^2\big)+\sum_i m_i\big(\|q_i\|^2+b_i^2\big)\Big).
$$
Scaling the penalty by the number of ratings ($n_u$, $m_i$) is the "ALS-WR" variant of Zhou, Wilkinson, Schreiber and Pan (2008); it keeps the amount of shrinkage proportional to the amount of data each user/item has, which stabilises users with very few ratings.

**Trick for biases.** Define for user $u$ the unknown $x_u=\begin{bmatrix}p_u\\ b_u\end{bmatrix}\in\mathbb{R}^{k+1}$ and for each item the fixed feature vector $z_i=\begin{bmatrix}q_i\\ 1\end{bmatrix}$, with target $t_{ui}=r_{ui}-\mu-b_i$. Then $r_{ui}-\mu-b_u-b_i-p_u^\top q_i = t_{ui}-z_i^\top x_u$, so the bias is just one more coordinate of the user's vector.

## 2. User step: fix all item quantities, solve for one user

Terms of $J$ that contain $x_u$:
$$
J_u(x_u)=\sum_{i\in\Omega_u} w_{ui}\,(t_{ui}-z_i^\top x_u)^2+\lambda n_u\|x_u\|^2 .
$$
Set the gradient to zero:
$$
\nabla J_u=-2\sum_{i\in\Omega_u} w_{ui}\,z_i\,(t_{ui}-z_i^\top x_u)+2\lambda n_u x_u=0
$$
$$
\Longrightarrow\quad \underbrace{\Big(\sum_{i\in\Omega_u} w_{ui}\,z_iz_i^\top+\lambda n_u I_{k+1}\Big)}_{A_u}\,x_u=\underbrace{\sum_{i\in\Omega_u} w_{ui}\,t_{ui}\,z_i}_{v_u}.
$$
These are the **normal equations**: a $(k+1)\times(k+1)$ matrix times the unknown vector equals a right-hand side. In matrix form, with $Z_u$ the $|\Omega_u|\times(k+1)$ matrix whose rows are $z_i^\top$ and $W_u=\mathrm{diag}(w_{ui})$: $A_u=Z_u^\top W_uZ_u+\lambda n_uI$, $v_u=Z_u^\top W_ut_u$. *(Weighting changes exactly one line of the code: the product $Z^\top W Z$ instead of $Z^\top Z$.)*

## 3. $A_u$ is symmetric positive definite, so the solution is unique

*Symmetric:* each $w_{ui}z_iz_i^\top$ is symmetric, $I$ is symmetric, sums of symmetric matrices are symmetric.

*Positive definite:* for any $y\neq 0$,
$$
y^\top A_u y=\sum_{i\in\Omega_u} w_{ui}(z_i^\top y)^2+\lambda n_u\|y\|^2\;\ge\;\lambda n_u\|y\|^2>0
$$
because $w_{ui}\ge0$, $\lambda>0$, $n_u\ge1$. Hence all eigenvalues are $\ge\lambda n_u>0$, $A_u$ is invertible, and (Cholesky exists). The Hessian of $J_u$ is $2A_u\succ0$, so $J_u$ is **strictly convex** and the stationary point is the **unique global minimiser** of the block. Note that without regularization ($\lambda=0$) $A_u$ can be singular, e.g. when $|\Omega_u|<k+1$ — regularization is what makes the problem well posed for users with few ratings.

## 4. Cost

Forming $A_u$: $|\Omega_u|(k+1)^2$ operations; forming $v_u$: $|\Omega_u|(k+1)$; solving: $O((k+1)^3)$. One full user sweep:
$$
O\big(|\Omega|\,k^2+U\,k^3\big),
$$
and the item sweep costs $O(|\Omega|k^2+I\,k^3)$. Each user (and each item) is independent of the others in the same half-step, so the work is parallelisable.

## 5. Item step (exactly symmetric)

Fix users. For item $i$ let $y_i=\begin{bmatrix}q_i\\ b_i\end{bmatrix}$, user features $s_u=\begin{bmatrix}p_u\\ 1\end{bmatrix}$ and targets $t_{ui}'=r_{ui}-\mu-b_u$. Then
$$
\Big(\sum_{u\in\Omega_i} w_{ui}\,s_us_u^\top+\lambda m_i I_{k+1}\Big)y_i=\sum_{u\in\Omega_i} w_{ui}\,t'_{ui}\,s_u,
$$
with the same symmetric positive-definite argument (replace $n_u$ by $m_i$).

## 6. Why the objective never increases

Let $(P^t,Q^t)$ denote all user and item parameters after $t$ iterations. The user step minimises $J$ over the user block $(P,b_u)$ **exactly**, with the item block fixed, so
$$
J(P^{t+1},Q^t)\le J(P^t,Q^t)
$$
(the old $P^t$ is one feasible choice). The item step likewise gives $J(P^{t+1},Q^{t+1})\le J(P^{t+1},Q^t)$. Hence $J$ is non-increasing at every half-step. Since $J\ge0$ the sequence of objective values is bounded below and therefore **converges**. *Caveat (stated honestly):* $J$ is convex in each block but not jointly convex, so the limit is a stationary point (typically a local minimum), not guaranteed to be the global minimum, and different initialisations can give slightly different solutions. Decreasing training objective also does not guarantee decreasing validation error — see the lambda study.

## 7. Weights and the "zero weight = deletion" property

If $w_{ui}=0$, that rating contributes nothing to $Z^\top WZ$, to $v_u$, or to $n_u$ (since $n_u$ counts only $w>0$), so the solution is **identical** to deleting the row. A small positive weight keeps the rating in $n_u$ (full regularization) but shrinks its pull on the data term, so a suspected-fake rating influences the factors less. This is exactly what the filter modes use: all ones (no filter), $w\in\{0,1\}$ (hard filter), $w=(1-p_{\text{fake}})^\gamma$ (soft weights).

## 8. Link to the code (`als.py`)

| Math | Code |
|---|---|
| $Z_u$ rows $z_i^\top=[q_i^\top,1]$ | `X[:, :k] = F[idx]; X[:, k] = 1` |
| $t_{ui}=r_{ui}-\mu-b_i$ | `t = r[p] - mu - other_bias[idx]` |
| $A_u=Z^\top WZ+\lambda n_uI$ | `A = X.T @ (X*wa[:,None]) + lam*n_a*I` |
| $v_u=Z^\top Wt$ | `v = Xw.T @ t` |
| solve | `x = np.linalg.solve(A, v)` |
| user step uses rows of $R$ (CSR); item step uses columns (CSC = CSR of $R^\top$) | `_positional_csr(u,i,...)`, `_positional_csr(i,u,...)` |

## 9. Separate regularization for the biases (option `reg_bias`)

The bias coordinate may be penalised with its own strength $\lambda_b$ instead of $\lambda$. Replacing $\lambda n_u I_{k+1}$ by
$$
D_u=n_u\,\mathrm{diag}(\underbrace{\lambda,\dots,\lambda}_{k},\;\lambda_b),\qquad A_u=Z_u^\top W_uZ_u+D_u ,
$$
$A_u$ is still symmetric and, for $\lambda,\lambda_b>0$, still positive definite ($y^\top A_uy\ge n_u\min(\lambda,\lambda_b)\|y\|^2$), so every result in Sections 3 to 6 (unique block minimiser, monotone decrease of $J$) carries over unchanged. The objective becomes
$J=\sum w(\cdot)^2+\lambda\sum n_u\|p_u\|^2+\lambda\sum m_i\|q_i\|^2+\lambda_b\sum n_ub_u^2+\lambda_b\sum m_ib_i^2$.
