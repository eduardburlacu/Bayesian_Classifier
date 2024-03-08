# The need for Bayesian approach

- In high dimensions, there is no unique boundary between classes
- This can cause overfitting
- To mitigate this, we proceed similarly to the case of regression, with using the adequate classification model instead.

# Regularization

We impose a prior $p(\mathbf{w})=\mathcal{N}(\mathbf{w}|\mathbf{0},\frac{1}{\lambda}\mathbf{I}_N)$

The model for our classification is logistic regression:
$p(y_n|\mathbf{x}_n,\mathbf{w})=\sigma(y_n\mathbf{w}^T\tilde{\mathbf{x}}_n)$

The joint distribution:
$p(\mathbf{y}|\mathbf{X},\mathbf{w})=\prod_{i=1}^N p(y_n|\mathbf{x}_n,\mathbf{w})= \prod_{i=1}^N \sigma(y_n \mathbf{w}^T \mathbf{x}_n)$

Therefore we obtain the posterior as:

$p(\mathbf{w}|\{\mathbf{X, y}\}) \pr
opto p(\mathbf{w}) \prod_{n=1}^N p(y_n|\mathbf{x}_n,\mathbf{w})=\mathcal{N}(\mathbf{w}|\mathbf{0},\lambda^{-1}\mathbf{I_N})\prod_{i=1}^N \sigma(y_n \mathbf{w}^T \mathbf{x}_n)$

$\therefore p(\mathbf{w}|\mathcal{D}) = \frac{1}{Z(\mathcal{D})} \mathcal{N}(\mathbf{w}|\mathbf{0},\lambda^{-1}\mathbf{I_N})\prod_{i=1}^N \sigma(y_n \mathbf{w}^T \mathbf{x}_n)$

The predictive distribution is then given by:

$p(y_*|\mathbf{x_*},\mathcal{D})= \int_{\mathbb{R}^D} p(y_*|\mathbf{w},\mathbf{x}_*) p(\mathbf{w}|\mathcal{D}) \mathbf{dw}=\int_{\mathbb{R}^D} \sigma(y_* \mathbf{w}^T \mathbf{x}_*) p(\mathbf{w}|\mathcal{D}) \mathbf{dw}$

## Issue with this

Finding the normalizing constant is not trivial, nor it is the predictive distribution. We thus emply approximate Bayesian regularization.

## Approximate Bayesian inference

1) Monte Carlo methods

We draw samples from the posterior ${p(\mathbf{w}|\mathcal{D}})$

2) Deterministic approximate inference

We approximate the posterior with a simpler function like a Gaussian.

# The Laplace Approximation

- A Gaussian is fitted for the posterior. reconsider the generic Bayesian inference when :

    $p(\mathbf{w}|\mathcal{D})= \frac{p(\mathbf{w},\mathcal{D})}{p(\mathcal{D})}=\frac{p(\mathbf{w},\mathcal{D})}{\int p(\mathbf{w},\mathcal{D}) d\mathbf{w}} = \frac{p(\mathbf{y}|\mathbf{w},\mathbf{X})p(\mathbf{w})}{\int p(\mathbf{w},\mathcal{D}) d\mathbf{w}} $

    If now say $f(\mathbf{w})=p(\mathbf{y}|\mathbf{w},\mathbf{X}) p(\mathbf{w})$ then:

    $p(\mathbf{w}|\mathcal{D})=\frac{f(\mathbf{w})}{\int f(\mathbf{w})\mathbf{dw}}=\frac{f(\mathbf{w})}{Z}$

- We propose a surrogate function $q(\mathbf{w|\mu,\Sigma})=\mathcal{N}(\mathbf{w}|\mu,\Sigma)$ aimed to approximate the posterior.
