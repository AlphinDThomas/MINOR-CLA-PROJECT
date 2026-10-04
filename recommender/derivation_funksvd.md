# Derivation of FunkSVD (SGD for regularized matrix factorization), weighted version included

*Author: Akhila. Matches `funksvd.py`. Equations use LaTeX ($...$).*

## 1. Model and objective

Notation as in the ALS derivation: observed pairs $\Omega$, rating $r_{ui}$, global mean $\mu$ (fixed), biases $b_u,b_i$, latent vectors $p_u,q_i\in\mathbb{R}^k$, weight $w_{ui}\in[0,1]$ (all ones = unweighted).
$$
\hat r_{ui}=\mu+b_u+b_i+p_u^\top q_i,\qquad e_{ui}=r_{ui}-\hat r_{ui}.
$$
FunkSVD minimises a sum of **per-rating losses**
$$
\ell_{ui}=\frac{w_{ui}}{2}\Big[e_{ui}^2+\lambda\big(b_u^2+b_i^2+\|p_u\|^2+\|q_i\|^2\big)\Big],\qquad
J_F=\sum_{(u,i)\in\Omega}\ell_{ui}.
$$
Only observed pairs appear. Missing ratings are unknown, never zero. (The name "SVD" is historical; no singular value decomposition is computed.)

## 2. Gradients

With $e=r-\mu-b_u-b_i-p_u^\top q_i$ and $\partial e/\partial p_u=-q_i$, $\partial e/\partial q_i=-p_u$, $\partial e/\partial b_u=\partial e/\partial b_i=-1$:
$$
\frac{\partial\ell}{\partial p_u}=\frac{w}{2}\big[-2e\,q_i+2\lambda p_u\big]=-w\,(e\,q_i-\lambda p_u),\qquad
\frac{\partial\ell}{\partial q_i}=-w\,(e\,p_u-\lambda q_i),
$$
$$
\frac{\partial\ell}{\partial b_u}=-w\,(e-\lambda b_u),\qquad \frac{\partial\ell}{\partial b_i}=-w\,(e-\lambda b_i).
$$

## 3. The SGD update

Visit the observed ratings in random order (reshuffled every epoch). For each one, step against the gradient with learning rate $\eta$, using the parameter values from **before** this step for both $p_u$ and $q_i$:
$$
\begin{aligned}
b_u&\leftarrow b_u+\eta w\,(e-\lambda b_u), &\qquad b_i&\leftarrow b_i+\eta w\,(e-\lambda b_i),\\
p_u&\leftarrow p_u+\eta w\,(e\,q_i-\lambda p_u), & q_i&\leftarrow q_i+\eta w\,(e\,p_u-\lambda q_i).
\end{aligned}
$$
(Using the old $p_u$ in the $q_i$ update is what makes it a true gradient step on $\ell_{ui}$; updating $p_u$ first and then using the new value would be a different, sequential scheme.) Hyperparameters used: $k=10$, $\lambda=0.2$ (chosen on the validation split), $\eta=0.005$, 20 epochs, initial factors $\sim\mathcal N(0,0.1^2)$, biases initialised at 0, early stopping on validation RMSE.

**Cost.** One update costs $O(k)$, so one epoch costs $O(|\Omega|k)$. Memory is $O((U+I)k)$.

## 4. Weights

The weight $w_{ui}$ multiplies the whole step, so a suspected-fake rating moves the parameters less. Two special cases are exact by construction and are tested in `tests/test_weights.py`:

* $w_{ui}=1$ for all ratings reproduces the unweighted run bit for bit.
* $w_{ui}=0$: the rating is skipped and the shuffle runs over the remaining ratings only, so the run is **identical** to deleting those rows.

## 5. How it relates to ALS (an observation, not a theorem we rely on)

For $w_{ui}\in\{0,1\}$ the regularization terms collect per user and per item: $\sum_{i\in\Omega_u}\|p_u\|^2=n_u\|p_u\|^2$. Hence
$$
J_F=\tfrac12\Big[\sum_{\Omega}w\,e^2+\lambda\sum_u n_u(\|p_u\|^2+b_u^2)+\lambda\sum_i m_i(\|q_i\|^2+b_i^2)\Big]=\tfrac12 J_{\text{ALS}} .
$$
So **both models minimise the same objective** (up to the factor $\tfrac12$), and ALS (alternating exact solves, see `derivation_als.md`) is simply a different optimiser. Two caveats keep their results from being identical:

1. FunkSVD is stopped after a fixed number of epochs and is not run to convergence, so it acts as an *early-stopped* solution. This is why its best $\lambda$ (0.2) differs from ALS's (0.5 to 1.0).
2. With **soft** weights ($0<w<1$) the models differ: FunkSVD scales the regularization of each rating by $w_{ui}$, while ALS uses the count $n_u$ of ratings with $w>0$. Soft weights therefore also change the shrinkage of FunkSVD, but not that of ALS.

## 6. Convergence (honest summary)

For a small enough step size SGD decreases the objective on average, but with a constant step it does not converge exactly: it settles into a noisy neighbourhood of a stationary point. $J_F$ is not jointly convex in $(P,Q)$, so the end point depends on the initialisation and the shuffling order, which is why the experiments average over 5 seeds. Decreasing training error does not imply decreasing validation error, which is why validation early stopping is used.
