# Choosing a JDistLib API

New to JDistlib? Start with the website's
[learning center](https://robbyjo.github.io/JDistlib/learn.html). It links a
beginner distribution tutorial and vignette, a custom-distribution tutorial and
vignette, copula material clearly marked for JDistlib 0.7.0 and later, and an
0.8.0 inference path with tutorials, worked vignettes, and executable examples.
The [composition tutorial](composition-tutorial.html) shows how the scalar
factories fit together and how to define a general monotone transformation.

Use the smallest API that expresses the distribution you have.

## Finance, tail risk, and option-implied distributions (0.9.0+)

Use `jdistlib.finance.FinancialRisk` when the input is already a distribution
and the output is VaR, atom-aware expected shortfall, a partial moment,
downside/shortfall measure, expectile, stop-loss, or European terminal payoff.
Always pass `RiskConvention.LOSS` or `RiskConvention.RETURN`; approximate
outputs are `NumericalEstimate` values and must be checked.

Use `TransformDistribution` and `DistributionTransforms` for log
characteristic/MGF values, domains, cumulants, Fourier inversion, and Esscher
tilting. Use `GeneralizedHyperbolicDistribution`,
`NormalInverseGaussianDistribution`, `VarianceGammaDistribution`, or
`StableDistribution` when those parameterizations match the model. Use
`DistributionAggregation` for seeded independent sums, compound sums,
products, ratios, and scenarios; its `DistributionApproximation` states the
strategy, error, and seed.

Use `CopulaTailAnalysis` with `RotatedCopula`, `JoeCopula`, `BB1Copula`, or an
existing pair copula for finite/asymptotic concentration and stress regions.
Use `DistributionFit` for bounded exact/censored/interval MLE or MAP and
`ExtremeValueInference` for GEV, GPD, tail-index, return-level, bootstrap, and
threshold diagnostics.

For European quotes, use `ReferenceOptions` for checked Black-Scholes or
Bachelier inversion and `OptionCurve` to enforce parity, price bounds,
monotonicity, convexity, and boundary conditions before extracting an
`OptionImpliedDistribution`. Use `OptionCalibration` for parametric fit and
`OptionInference` for model factors or labeled posterior ensembles. See the
[finance tutorial](finance-tutorial.html) and
[worked options analysis](options-trading-worked-example.html).

| Need | API to start with |
| --- | --- |
| One calculation for a built-in scalar law | Static methods such as `Normal.density`, `Normal.cumulative`, and `Normal.quantile` |
| Repeated scalar calculations with fixed parameters | A distribution instance such as `new Normal(mean, standardDeviation)` |
| A nonnegative formula that must be normalized | `NumericalContinuousDistribution.builder()` or `NumericalDiscreteDistribution.builder()` |
| A mixture, truncation, censoring, or monotone transformation | The named factories in `Distributions` |
| Continuous marginals joined by a dependence model | A `Copula` implementation and `CopulaDistribution` |
| Discrete or mixed marginals joined by a copula | `CopulaMarginal` declarations and `MixedCopulaDistribution` |
| Flexible multivariate dependence or family fitting | `CVineCopula`/`DVineCopula`, `CopulaFitter`, or `CopulaSelector` |
| Audit copula likelihood rows and boundary proximity | `CopulaLikelihoodDiagnostics` or `MixedCopulaDistribution.logLikelihoodResult` |
| Adjust many p-values or calculate Storey q-values | `jdistlib.disttest.MultipleTesting` |
| Test prespecified groups of hypotheses | `MultipleTesting.selectiveGroupedBenjaminiHochberg` |
| Test hypotheses arriving sequentially | `jdistlib.disttest.online.LordPlusPlus` or `Saffron` |
| Exploit known finite discrete null CDFs | `jdistlib.disttest.DiscreteFdr` |
| A finite, semi-infinite, or whole-line integral | `Integrate.integrate` with `IntegrationOptions` when defaults are insufficient |
| A continuous multivariate rectangle probability | The law's `probability` method and `MultivariateProbabilityResult` |
| An exact multinomial/Dirichlet-multinomial/multivariate-hypergeometric rectangle | The law's inclusive-bound `probability` method |
| A Wishart directional, standardized-trace, or determinant event | The named methods in `Wishart` and `WISHART_PROBABILITIES.md` |
| A programmatic Bayesian posterior | `jdistlib.inference.ModelBuilder` and `BayesianModel` |
| LOO, WAIC, model comparison, or stacking | `PointwiseLogLikelihoodDraws`, `PsisLoo`, `Waic`, and `PredictiveStacking` |
| Predictive or shrinkage variable selection | `ProjectionPredictiveSelection` or `ShrinkageSelection` |
| A differentiable continuous posterior | `NoUTurnSampler` with multiple explicit-seed chains |
| A posterior without gradients | Metropolis, `SliceSampler`, or composable `GibbsSampler` blocks |
| A fixed-dimensional mixed continuous/discrete posterior | `MixedStateSpace` and a scheduled `HybridSampler` |
| A genuinely variable-dimensional Java model | `ReversibleJumpSampler`, `SubsetSelectionRj` for up to 62 candidates, or `SparseSubsetRjSampler` for a large candidate universe with a small active cap |
| MCMC convergence and sampler health | `McmcDiagnostics` and `McmcDiagnosticReport` |
| Trace, rank, ACF, energy, pair, or model graphs | `DiagnosticGraphs`, `ModelGraphExport`, and `InferenceHtmlReport` |
| A Stan-inspired model script or generated Java wrapper | `jdistlib.inference.lang.ModelScript` or `ModelSourceGenerator` |
| Ordinary Stan syntax supported by the Java-native core | `ModelScript.compileStan` and `STAN_SOURCE_COMPATIBILITY.md` |
| Allocation-conscious reverse gradients for an HMC/NUTS target | `jdistlib.inference.autodiff.ReverseModeLogDensity` and `ReverseTape` |

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

The additive 0.7.2 copula conveniences preserve the 0.7.0 signatures. Use an
overload without a random engine for a deterministic midpoint transform, a
`long` seed overload for repeatable randomized transforms, or an explicit
`RandomEngine` when the caller owns the stream.

See `NUMERICAL_DISTRIBUTIONS.md` for integration and custom-law guarantees and
`MULTIVARIATE_PROBABILITIES.md` for randomized rectangle probabilities. See
`WISHART_PROBABILITIES.md` for the deliberately named Wishart scalar events and
the matrix-order boundary. See `COPULAS.md` for dependence families and
continuous-marginal composition. See
`INFERENCE.md`, `MODELING_LANGUAGE.md`, and `INFERENCE_COMPATIBILITY.md` for
Bayesian modeling, MCMC, diagnostics, graphing, scripting, and schema guarantees.
For a guided sequence, start at `inference-tutorial.html`, continue with
`inference-guide.html`, and use `inference-examples.html` as a tested catalog.
The website's `examples.html` index links fifty-one JDistlib model scripts, forty-one
ordinary `.stan` compatibility fixtures, and
feature-focused Java integrations for copulas, composition, FDR, custom laws,
numerical integration, and MCMC.

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

Version 0.7.1 adds weighted BH, BY, Bonferroni, and Holm with automatic
mean-one normalization of positive prespecified weights and log-domain
variants, plus `gavrilovBenjaminiSarkar` for adaptive step-down FDR decisions
under independence. It also adds Benjamini–Bogomolov two-level grouped testing,
separate stateful LORD++ and SAFFRON controllers for online streams, and proven
DBH step-up/down procedures for independent heterogeneous discrete p-values.

`qValues` implements Storey q-values with a smoothing-spline π₀ estimate by
default. Callers can instead provide π₀ directly or use the quantile estimator.
See the [multiple-testing guide](https://robbyjo.github.io/JDistlib/multiple-testing.html)
for assumptions and examples.
