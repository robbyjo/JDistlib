# Multivariate rectangle probabilities

`MultivariateNormal`, `MultivariateStudentT`, `MultivariateCauchy`, and
`MultivariateLogNormal` expose `probability(lower, upper, ...)` for rectangular
events. Their `cumulative` methods are lower-orthant conveniences. These are
randomized numerical calculations except for exact boundary, unrestricted, and
one-dimensional reductions.

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
