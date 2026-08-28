# Multivariate rectangle probabilities

`MultivariateNormal`, `MultivariateStudentT`, `MultivariateCauchy`,
`MultivariateLogNormal`, `Dirichlet`, `MultivariateLaplace`, and
`MultivariatePowerExponential` expose `probability(lower, upper, ...)` for
rectangular events. Their `cumulative` methods are lower-orthant conveniences.
These are randomized numerical calculations except for exact boundary,
unrestricted, one-dimensional, and distribution-specific analytic reductions.

`Multinomial`, `DirichletMultinomial`, and `MultivariateHypergeometric` expose
inclusive integer-bound `probability` and `cumulative` methods. Those three are
deterministic: a shared sequential-conditional dynamic program sums binomial,
beta-binomial, or hypergeometric conditional masses without Monte Carlo error.

Wishart probabilities are not rectangles. `Wishart` provides named exact
directional quadratic-form and standardized-trace CDFs plus numerical
determinant/log-determinant intervals. See `WISHART_PROBABILITIES.md`; there is
deliberately no ambiguous entrywise matrix CDF.

## Distribution-specific transformations

- Dirichlet probabilities use conditional beta stick breaking. At each step,
  future lower and upper sums tighten the admissible interval, so integration
  never samples a point outside the feasible simplex slice.
- Multivariate Laplace probabilities integrate the existing sequential-normal
  rectangle kernel over the law's exponential mixing variable.
- Multivariate power-exponential probabilities condition on a uniform
  direction and integrate the exact gamma radial interval. This is valid for
  every positive shape; the Gaussian shape and centered orthants delegate to
  the multivariate-normal engine. Gamma tail differences are evaluated in the
  safer tail to avoid cancellation.

## Result contract

`MultivariateProbabilityResult.getStatus()` returns:

- `SUCCESS`: the replication-based error indicator met the requested tolerance;
- `MAX_EVALUATIONS_REACHED`: `probability` is a usable estimate, but its error
  indicator did not meet tolerance within the work limit; or
- `INVALID_INPUT`: no estimate is available.

`absoluteError` is 3.5 times the estimated standard error across randomized
replications. It is deliberately conservative as a stopping indicator, but it
is not a rigorous confidence interval or a deterministic error bound.
Convergence uses
`max(absoluteTolerance, relativeTolerance * abs(probability))`.

## Reproducibility and concurrency

The convenience overloads use a fixed seed and are repeatable for a fixed
JDistLib version. Supply both `MultivariateProbabilityOptions` and a fresh,
explicit `RandomEngine` to control a stream. Calls keep integration state local;
separate engines make independent calls safe to run concurrently. Sharing one
mutable engine still requires caller synchronization.

## Cost and difficult cases

There is no artificial dimension cap. Covariance validation/factorization costs
roughly cubically in dimension, and each sequential-conditioning integrand
evaluation costs roughly quadratically. The practical limit therefore depends
on correlation, bounds, tolerance, and the evaluation budget rather than a
single advertised number.

Expect more work, and treat the error indicator cautiously, for high dimension,
nearly singular correlation matrices, very narrow rectangles, rare-event tails,
strong dependence, and low-degree-of-freedom Student t or Cauchy laws. A
covariance or scale matrix must be finite, symmetric, and positive definite.
Increase `maxEvaluations` and replications, compare multiple independent seeded
runs, and use problem-specific analytic reductions when available.

Quantiles are explicitly named `equicoordinateQuantile` and `radialQuantile`;
JDistLib does not pretend that a random vector has one generic scalar inverse
CDF.
