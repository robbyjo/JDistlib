# Choosing a JDistLib API

Use the smallest API that expresses the distribution you have.

| Need | API to start with |
| --- | --- |
| One calculation for a built-in scalar law | Static methods such as `Normal.density`, `Normal.cumulative`, and `Normal.quantile` |
| Repeated scalar calculations with fixed parameters | A distribution instance such as `new Normal(mean, standardDeviation)` |
| A nonnegative formula that must be normalized | `NumericalContinuousDistribution.builder()` or `NumericalDiscreteDistribution.builder()` |
| A mixture, truncation, censoring, or monotone transformation | The named factories in `Distributions` |
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
`MULTIVARIATE_PROBABILITIES.md` for randomized rectangle probabilities.
