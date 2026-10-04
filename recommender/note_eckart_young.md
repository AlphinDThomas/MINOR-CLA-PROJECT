# Why truncated SVD is optimal for a full matrix, and why that does not directly apply to ratings

*Short note for the report (Phase 5). Author: Akhila.*

## 1. The Eckart–Young theorem (fully observed matrix)

Let $A\in\mathbb{R}^{m\times n}$ have the SVD $A=\sum_{j=1}^{r}\sigma_j u_jv_j^\top$ with $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$. For any $k<r$ let $A_k=\sum_{j=1}^{k}\sigma_j u_jv_j^\top$ be the truncated SVD. Then
$$
\|A-A_k\|_F=\min_{\operatorname{rank}(B)\le k}\|A-B\|_F=\Big(\sum_{j>k}\sigma_j^2\Big)^{1/2}
\quad\text{(and the same holds for the spectral norm: } \|A-A_k\|_2=\sigma_{k+1}\text{).}
$$
So **if every entry of $A$ is known**, the best rank-$k$ approximation in squared error is obtained by simply keeping the top $k$ singular triplets. The size of the discarded singular values $\sigma_{k+1},\sigma_{k+2},\dots$ tells exactly how much error a rank-$k$ model must make. This is why the singular-value spectrum plot helps to justify a choice of $k$: a sharp drop after index $k$ means a rank-$k$ model captures most of the structure; a slowly decaying spectrum means more factors are needed.

## 2. Why it does not apply directly to a ratings matrix

A ratings matrix $R$ is more than 99.9% **missing** (our sample: density 0.076%). The quantity we want to minimise is the error over **observed** entries only,
$$
\min_{P,Q}\sum_{(u,i)\in\Omega}\big(r_{ui}-p_u^\top q_i\big)^2 ,
$$
and Eckart–Young says nothing about this problem, for two reasons:

1. **To apply the SVD we must fill the missing entries with something** (zero, or the mean). The truncated SVD then fits those invented values just as hard as the real ratings, so the result is optimal for the *filled* matrix, not for the observed ratings. Filling with zero in particular treats "not rated" as "rated 0", which is wrong; filling with the mean biases every prediction toward the mean.
2. **The observed-entries problem is not a SVD problem.** The weight matrix (1 on observed, 0 on missing) couples the entries, and weighted low-rank approximation has no closed-form solution: the problem is non-convex and can have local minima. (Even the rank-$k$ constraint alone is non-convex, though Eckart–Young shows it is still solvable by SVD in the unweighted case.)

## 3. What we do instead

FunkSVD and ALS minimise the squared error **over observed entries only**, with regularization, by SGD or by alternating exact least-squares solves. They use the same low-rank form $p_u^\top q_i$ as the truncated SVD, but they never see the missing entries. We use the SVD of the centred sparse matrix only as a **diagnostic** (the spectrum plot, with missing entries at 0 after centring), not as the recommender itself. When fake profiles are injected, the spectrum of the attacked matrix can show extra large singular values created by the coordinated rows, which is the comparison planned for the attack study.
