# Choosing a JDistLib API

New to JDistlib? Start with the website's
[learning center](https://robbyjo.github.io/JDistlib/learn.html). It links a
beginner distribution tutorial and vignette, a custom-distribution tutorial and
vignette, and copula material clearly marked for JDistlib 0.7.0 and later.

Use the smallest API that expresses the distribution you have.

| Need | API to start with |
| --- | --- |
| One calculation for a built-in scalar law | Static methods such as `Normal.density`, `Normal.cumulative`, and `Normal.quantile` |
| Repeated scalar calculations with fixed parameters | A distribution instance such as `new Normal(mean, standardDeviation)` |
| A nonnegative formula that must be normalized | `NumericalContinuousDistribution.builder()` or `NumericalDiscreteDistribution.builder()` |
| A mixture, truncation, censoring, or monotone transformation | The named factories in `Distributions` |
| Continuous marginals joined by a dependence model | A `Copula` implementation and `CopulaDistribution` |
| Discrete or mixed marginals joined by a copula | `CopulaMarginal` declarations and `MixedCopulaDistribution` |
| Flexible multivariate dependence or family fitting | `CVineCopula`/`DVineCopula`, `CopulaFitter`, or `CopulaSelector` |
| Adjust many p-values or calculate Storey q-values | `jdistlib.disttest.MultipleTesting` |
| A finite, semi-infinite, or whole-line integral | `Integrate.integrate` with `IntegrationOptions` when defaults are insufficient |
| A multivariate normal/t/Cauchy/log-normal rectangle probability | The law's `probability` method and `MultivariateProbabilityResult` |

Scalar static methods follow R conventions. `lowerTail=true` requests the lower
tail, and `logP=true` returns a logarithm. Prefer log probabilities for extreme
tails. Methods return `NaN` for invalid distribution parameters and use zero,
one, and infinities for mathematical boundary limits where appropriate.

Random generation should use an explicit `RandomEngine` when reproducibility,
parallelism, or stream ownership matters. Cached binomial, hypergeometric, and
Poisson algorithms can also take an explicit per-stream `RandomState`. Do not
share one mutable engine or state across threads without external locking.

Numerical APIs return result objects when convergence matters. Check the typed
status and inspect the reported error or diagnostics; a finite estimate is not
the same promise as convergence. Builder diagnostics identify observed risks,
not a proof that an arbitrary caller-supplied formula is a probability law.

See `NUMERICAL_DISTRIBUTIONS.md` for integration and custom-law guarantees and
`MULTIVARIATE_PROBABILITIES.md` for randomized rectangle probabilities. See
`COPULAS.md` for dependence families and continuous-marginal composition.

## Distribution tests

`jdistlib.disttest.DistributionTest` includes classical location, scale, rank,
count, and Kolmogorov-Smirnov procedures. In 0.7.0 it also provides:

- `cramer_von_mises_test(sample, distribution)` and
  `anderson_darling_test(sample, distribution)` for a fully specified continuous
  reference law, with deterministic parametric-bootstrap p-values and overloads
  accepting the resampling count and `RandomEngine`;
- `cramer_von_mises_test(first, second)` with a deterministic permutation
  p-value and a caller-controlled overload;
- `chi_square_goodness_of_fit_test` for categorical counts and
  `chi_square_independence_test` for contingency tables.

Each test returns the statistic followed by its p-value. Chi-square methods also
return degrees of freedom. If reference parameters were estimated from the same
sample, the default fully-specified-law bootstrap is not sufficient: refit inside
each replicate or use a model-specific correction.

## Multiple testing and FDR

Version 0.7.0 adds the stateless `jdistlib.disttest.MultipleTesting` facade.
`adjust(pValues, method)` covers pass-through, Bonferroni, Holm, Hochberg,
Hommel, Šidák, Holm–Šidák, Benjamini–Hochberg, and Benjamini–Yekutieli
procedures while preserving input order and missing `NaN` positions. `reject`,
`countRejected`, and `threshold` turn adjusted values into decisions.

`benjaminiKriegerYekutieli(pValues, level)` exposes the adaptive two-stage BKY
procedure as a level-dependent result. `adjustLog` performs every adjustment
directly on natural-log p-values, and `testRightCensored` conservatively handles
a recorded lower tail when both the censoring limit and full family size are
known.

`qValues` implements Storey q-values with a smoothing-spline π₀ estimate by
default. Callers can instead provide π₀ directly or use the quantile estimator.
See the [multiple-testing guide](https://robbyjo.github.io/JDistlib/multiple-testing.html)
for assumptions and examples.
