# Wishart probability events

There is no `Wishart.cumulative(matrix)` method. An entrywise event and a
positive-semidefinite (Loewner) order event are different mathematical objects,
and treating either as *the* matrix CDF would be ambiguous. JDistlib instead
exposes named scalar events with exact reductions or a controlled numerical
calculation.

For `W ~ Wishart(df, scale)`:

- `quadraticFormCumulative(upper, direction, ...)` evaluates
  `P(direction' W direction <= upper)`. The quadratic form divided by
  `direction' scale direction` is exactly chi-square with `df` degrees of
  freedom. Lower/upper-tail and log-probability flags follow the scalar JDistlib
  convention.
- `standardizedTraceCumulative(upper, ...)` evaluates
  `P(trace(scale^-1 W) <= upper)`. The standardized trace is exactly chi-square
  with `dimension * df` degrees of freedom.
- `determinantProbability(lower, upper, ...)` and `determinantCumulative(...)`
  evaluate determinant intervals and lower tails.
- `logDeterminantProbability(lower, upper, ...)` and
  `logDeterminantCumulative(...)` accept log thresholds directly. Prefer these
  for extreme events so a determinant threshold need not overflow or underflow.

The determinant methods use Bartlett's decomposition:

```text
det(W) / det(scale) = product(i = 0 .. dimension - 1) ChiSquare(df - i),
```

with independent factors. JDistlib conditions on all but one factor, evaluates
the last chi-square interval analytically in its safer tail, and integrates the
remaining smooth conditional probability with randomized antithetic Halton
replications. The result is a `MultivariateProbabilityResult`; always inspect
its status and `absoluteError`. A constant all-zero or all-one randomized tail
sample is reported as `MAX_EVALUATIONS_REACHED`, not false convergence.

## What is deliberately absent

JDistlib does not currently expose:

- an entrywise Wishart matrix CDF;
- a general `P(W <= A)` Loewner-order probability; or
- joint eigenvalue-order probabilities.

Those events require distinct matrix-variate integration algorithms and cannot
be obtained reliably by relabeling a scalar or rectangular CDF. A directional
quadratic form is often the useful rank-one projection event. Determinant and
standardized-trace events cover generalized volume and total standardized
variation without introducing matrix-order ambiguity.

These probabilities are useful for posterior predictive checks, covariance
calibration, and threshold decisions. An MCMC target itself normally uses the
Wishart log density rather than these CDFs.
