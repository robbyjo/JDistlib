# JDistlib

[![CI](https://github.com/robbyjo/JDistlib/actions/workflows/ci.yml/badge.svg)](https://github.com/robbyjo/JDistlib/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/robbyjo/JDistlib)](https://github.com/robbyjo/JDistlib/releases/latest)
[![Documentation](https://img.shields.io/badge/docs-distributions-315b7d)](https://robbyjo.github.io/JDistlib/distributions.html)
[![License: GPL v2+](https://img.shields.io/badge/license-GPL--2.0%2B-315b7d)](LICENSE)

JDistlib is a Java library for probability distributions and related numerical
methods. Its core is a manual Java translation of R's `src/nmath`, designed to
retain R-compatible results without inheriting the process-global caches that
made older native implementations awkward to call concurrently.

The library also contains distributions and utilities that are not part of R,
including arcsine, asymmetric Laplace, beta-binomial, beta-negative-binomial, beta-prime,
Birnbaum-Saunders, categorical, discrete Laplace/Weibull, empirical, Feller-Pareto,
exponentially modified Gaussian, generalized beta/gamma/F, Gompertz, Huber,
hurdle and zero-modified counts,
half-Cauchy/normal/t, inverse gamma, inverse normal, Lindley, logit-normal,
Makeham, Kumaraswamy, Laplace, Levy, log-logistic, logarithmic,
Maxwell/Maxwell-Boltzmann, multinomial, Nakagami, negative hypergeometric,
phase-type, Poisson-binomial, Poisson-inverse-Gaussian, Rice, sinh-arcsinh,
Skellam, skewed t, slash, triangular, Tukey lambda, Tweedie, Wishart, Zipf,
and the `jdistlib.evd` package. Vector APIs
cover Dirichlet and Dirichlet-multinomial, multivariate hypergeometric,
bivariate Poisson/logistic, and multivariate normal, Student t, Cauchy,
lognormal, Laplace, and power-exponential laws. These are first-class JDistlib features
and are not removed during upstream synchronization.

## Project status

Version 0.7.2 is the current stable release. It adds independent boundary-heavy
mixed/vine reference validation, fitted-model diagnostics, analytic built-in
pair conditionals, and additive fitting conveniences while retaining the 0.7.0
copula API. Version 0.7.1 expanded multiple testing, added CQUAD integration,
and introduced general monotone-transformation composition. The R `src/nmath`
file-by-file audit from the historical R 3.3.2 baseline to R 4.6.1 is complete.
[UPSTREAM.md](UPSTREAM.md) is the source-of-truth checklist and
[NMATH_AUDIT.md](NMATH_AUDIT.md) records the source disposition and reproducible
R 4.6.1 reference corpus. JDistlib-specific APIs remain separately documented
and tested.

## Download

The [latest GitHub release](https://github.com/robbyjo/JDistlib/releases/latest)
contains the Java library, source archive, JavaDoc archive, and SHA-256
checksums. Version 0.7.1 produces Java 8-compatible bytecode.

## Building

JDK 17 or newer is recommended for building. Produced class files remain Java 8
compatible.

```text
./gradlew test
./gradlew build
```

On Windows, use `gradlew.bat` instead.

## Documentation

The project website is <https://robbyjo.github.io/JDistlib/> and the generated
JavaDoc is published at <https://robbyjo.github.io/JDistlib/api/>. Both are
rebuilt from `master` by the GitHub Pages workflow.

The [distribution reference](https://robbyjo.github.io/JDistlib/distributions.html)
lists every available scalar, multivariate, matrix, and exact-statistic law with
its parameterization, defining formula, and density/mass, CDF, quantile, random,
and auxiliary method coverage. The [API selection guide](docs/API_GUIDE.md)
provides a short path from a use case to the appropriate API.

### Start learning

The website now puts beginner material first:

* [Using distributions](https://robbyjo.github.io/JDistlib/getting-started.html) —
  density/mass, CDF, quantile, tails, and reproducible simulation.
* [Response-time vignette](https://robbyjo.github.io/JDistlib/distribution-vignette.html) —
  an applied univariate analysis with goodness-of-fit checks.
* [Compose and transform distributions](https://robbyjo.github.io/JDistlib/composition-tutorial.html) —
  a beginner guide to mixtures, truncation, censoring, changes of units, and
  general monotone transformations.
* [Multiple testing and FDR](https://robbyjo.github.io/JDistlib/multiple-testing.html) —
  adjusted and log p-values, adaptive BKY, censored families, rejection
  thresholds, and Storey q-values (JDistlib 0.7.0+).
* [Building a custom distribution](https://robbyjo.github.io/JDistlib/custom-distributions.html#beginner-path)
  and the [sensor-error vignette](https://robbyjo.github.io/JDistlib/custom-distribution-vignette.html)
  (JDistlib 0.6.0+).
* [Using copulas](https://robbyjo.github.io/JDistlib/copula-tutorial.html) and the
  [mixed-claims vignette](https://robbyjo.github.io/JDistlib/copula-vignette.html)
  (**copula features require JDistlib 0.7.0+**).

## Using the distribution APIs

Distribution classes expose static density, cumulative, quantile, and random
methods. For example:

```java
double p = Normal.cumulative(1.96, 0.0, 1.0, true, false);
double x = Normal.quantile(0.975, 0.0, 1.0, true, false);
```

Boolean arguments follow R's `lower.tail` and `log.p` conventions.

For multivariate normal, Student t, Cauchy, and log-normal laws, `probability`
evaluates an arbitrary rectangular region and `cumulative` uses a lower bound of
negative infinity in every coordinate. `MultivariateProbabilityResult` reports
the estimated error, work count, and convergence status. The overloads accepting
`MultivariateProbabilityOptions` and `RandomEngine` provide reproducible,
thread-safe control of the randomized integration. Multivariate quantiles are
named `equicoordinateQuantile` or `radialQuantile` because a random vector has no
unique scalar inverse CDF.

```java
MultivariateProbabilityResult region = MultivariateNormal.probability(
    new double[] {-1.0, -1.0}, new double[] {1.0, 1.0},
    new double[] {0.0, 0.0}, new double[][] {{1.0, 0.5}, {0.5, 1.0}});
if (!region.isSuccess()) {
    System.err.println(region.message());
}
```

The error field is a replication-based stopping indicator, not a rigorous
confidence bound. See the [multivariate probability contract](docs/MULTIVARIATE_PROBABILITIES.md)
for statuses, reproducibility, scaling, and difficult cases.

## Copulas and composed joint distributions

> **Version requirement:** the copula APIs in this section are available only
> in JDistlib 0.7.0 and later.

The `Copula` interface separates marginal laws from dependence. Implementations
cover independence, Gaussian, Student-t, Clayton, Gumbel, and Frank copulas.
Every family provides a CDF, density and log-density on the open unit cube,
explicit-stream and seeded sampling, and pairwise Kendall's tau. Gaussian and
Student-t copulas accept positive-definite correlation matrices; the
Archimedean families provide parameter-from-tau factories.

```java
Copula copula = GaussianCopula.fromKendallsTau(
    new double[][] {{1.0, 0.45}, {0.45, 1.0}});
CopulaDistribution joint = new CopulaDistribution(
    copula, new Normal(10.0, 2.0), new Exponential(3.0));

double logDensity = joint.logDensity(new double[] {11.0, 2.0});
double[] draw = joint.random(20260826L);
```

`CopulaDistribution` requires continuous marginals. Exact unit-cube boundary points can
have singular or path-dependent density limits; `diagnose` classifies them
before evaluation. `MixedCopulaDistribution` adds declared continuous and
discrete marginals, exact CDF rectangle differences for masses, and typed
numerical results for mixed-measure derivatives. `CVineCopula` and
`DVineCopula` assemble simplified vines from `PairCopula` objects, while
`CopulaFitter`, `CopulaSelector`, and `VineFitter` provide rank or
distributional transforms, dependence estimation, and AIC/BIC family
selection. The additive 0.7.2 API exposes row-level
`CopulaLikelihoodDiagnostics`, retained mixed-measure contributions through
`logLikelihoodResult`, vine AIC/BIC, and explicit seed overloads; all 0.7.0
entry points remain available. See the [copula guide](docs/COPULAS.md) for the
full contract and independent high-precision reference cases.

## Multiple testing and false discovery rates

JDistlib 0.7.0 adds a stateless `MultipleTesting` facade in
`jdistlib.disttest`. It provides Bonferroni, Holm, Hochberg, Hommel, Šidák,
Holm–Šidák, Benjamini–Hochberg, and Benjamini–Yekutieli adjusted p-values,
along with rejection flags, counts, and raw p-value thresholds. `NaN` is treated
as missing and preserved in place.

```java
double[] adjusted = MultipleTesting.adjust(
    pValues, MultipleTesting.Method.BENJAMINI_HOCHBERG);
boolean[] rejected = MultipleTesting.reject(
    pValues, 0.05, MultipleTesting.Method.BENJAMINI_HOCHBERG);
```

Adaptive two-stage BKY is available through
`benjaminiKriegerYekutieli(pValues, level)`. Extremely small probabilities can
remain in natural-log form with `adjustLog`, and `testRightCensored` supports a
known p-value recording limit when the original family size is also known.
Version 0.7.1 expands prespecified weighting to BH, BY, Bonferroni, and Holm
(all with log-domain variants), and adds the independence-based adaptive GBS
step-down procedure. Structured families can use the two-level
Benjamini–Bogomolov grouped API. Hypotheses arriving over time use the separate,
stateful `LordPlusPlus` or `Saffron` controllers, while heterogeneous discrete
nulls use the proven DBH step-up or step-down procedures in `DiscreteFdr`.

Storey q-values can use the default smoothing-spline estimate of π₀, the
quantile estimator inherited from QGeneric, or a caller-supplied π₀. Setting π₀
to one reproduces Benjamini–Hochberg adjusted values. See the
[multiple-testing guide](docs/multiple-testing.html) for assumptions and the
complete API.

## Numerical integration

`Integrate.integrate` supports finite, semi-infinite, and doubly-infinite
intervals. The default tolerances match R's `integrate()` default of
`.Machine$double.eps^0.25`.

```java
IntegrationResult result = Integrate.integrate(
    x -> Math.exp(-x * x),
    Double.NEGATIVE_INFINITY,
    Double.POSITIVE_INFINITY
);

if (!result.isSuccess()) {
    throw new ArithmeticException(result.message());
}
```

The mutable `IntegrationResult` remains available for source compatibility.
New code can call `Integrate.integrateImmutable` for an immutable snapshot with
a typed `IntegrationStatus`, error estimate, callback cost profile, and no
retained reference to the user callback.

For hostile or difficult callbacks, the additive `IntegrationOptions` API
supports evaluation budgets, cancellation checks, declared discontinuity or
singularity points, caught callback diagnostics, stability assessment, and
CQUAD or double-exponential methods for difficult finite integrals, plus
double-exponential methods for infinite intervals. It also supports
total/per-callback wall-clock limits and opt-in isolated daemon execution for a
callback that may not return. The historical overloads retain their R/QUADPACK
behavior.

```java
IntegrationOptions options = IntegrationOptions.builder()
    .tolerances(1e-10, 1e-10)
    .subdivisions(300)
    .maxEvaluations(250_000)
    .maxCallbackTime(250, TimeUnit.MILLISECONDS)
    .maxTotalTime(5, TimeUnit.SECONDS)
    .callbackExecution(
        IntegrationOptions.CallbackExecution.ISOLATED_DAEMON)
    .breakpoints(0.5)
    .method(IntegrationOptions.Method.AUTO)
    .build();

IntegrationStabilityResult stability =
    Integrate.assessStability(kernel, 0.0, 1.0, options);

String machineReadable = stability.toJson();
```

`AUTO` retains QUADPACK as its first choice, tries CQUAD next for a finite
interval, and then uses the applicable double-exponential rule. Select
`IntegrationOptions.Method.CQUAD` directly when you want doubly adaptive
Clenshaw-Curtis integration on finite bounds.

Isolation releases the integrating thread when the deadline expires and uses a
daemon worker so a permanently blocked callback cannot keep the JVM alive. Java
cannot forcibly terminate arbitrary user code, so the abandoned daemon may live
until that callback returns; process isolation is still required for untrusted
code or strict resource containment.

## User-defined numerical distributions

> Available from JDistlib 0.6.0. See the
> [custom-distribution guide](https://robbyjo.github.io/JDistlib/custom-distributions.html)
> for continuous and
> discrete quick starts, mathematical definitions, diagnostics, and
> troubleshooting.

`NumericalContinuousDistribution` turns a nonnegative kernel into a complete
continuous distribution. The normalization constant is integrated and cached;
density, lower and upper CDF tails, numerical quantiles, and inverse-transform
random generation follow the usual `GenericDistribution` API.

```java
NumericalContinuousDistribution quartic =
    new NumericalContinuousDistribution(
        x -> Math.exp(-x * x * x * x),
        Double.NEGATIVE_INFINITY,
        Double.POSITIVE_INFINITY
    );

double logZ = quartic.getLogNormalizationConstant();
double p = quartic.cumulative(1.0, true, false);
double draw = quartic.random();
```

`NumericalDiscreteDistribution` provides the analogous operation over a finite
set of outcomes. It evaluates the weight formula once, uses scaled compensated
summation, caches the resulting probability and cumulative-mass tables, and
builds a Walker alias table for constant-time repeated sampling.

```java
NumericalDiscreteDistribution customCount =
    new NumericalDiscreteDistribution(
        k -> 1.0 / (1.0 + k * k),
        0, 100 // inclusive integer support
    );
```

An explicit `double[]` may be used for an irregular finite support. Constructors
reject sampled negative/non-finite values and zero normalization. For continuous
kernels, `getNormalizationResult()` exposes the integration error, evaluation
count, subdivisions, and status. These checks cannot mathematically prove that
an arbitrary function is nonnegative and integrable at every unsampled point.

Kernels can be inspected before construction. The report checks sampled signs,
finite values, repeatability, sharp changes, oscillation, dynamic range, tail
decay, and normalization stability. A build result retains this report even
when construction fails. Seeded randomized probes have an explicit budget and
adapt toward observed sharp changes. `STRICT`, `WARNING`, and `PERMISSIVE`
construction policies decide whether advisory findings prevent an attempt.
Use `analyzeLogKernel` for a log-density formula; fluent builders select the
ordinary or log-space analyzer automatically. Log-space probes accept
`Double.NEGATIVE_INFINITY` as zero mass and reject NaN or positive infinity.

```java
FunctionAnalysisOptions checks = FunctionAnalysisOptions.builder()
    .randomizedProbeBudget(256)
    .randomSeed(42L)
    .constructionPolicy(ConstructionPolicy.WARNING)
    .build();
NumericalDistributionBuildResult candidate =
    NumericalContinuousDistribution.analyze(kernel, lower, upper, checks);

for (DiagnosticFinding finding : candidate.getAnalysis().getFindings()) {
    System.out.println(finding);
}

if (candidate.canBuild()) {
    NumericalContinuousDistribution distribution = candidate.build();
    DistributionAnalysis checks = distribution.analyzeDistribution();
}
```

Fluent builders gather support, singularities, numerical settings, diagnostics,
and sampling configuration. Named presets provide useful starting points:

```java
NumericalContinuousDistribution custom =
    NumericalContinuousDistribution.builder()
        .logKernel(x -> -0.5 * x * x)
        .support(-8.0, 8.0)
        .diagnosticPreset(DiagnosticPreset.THOROUGH)
        .adaptiveRejectionSampling(x -> -x, -2.0, 0.0, 2.0)
        .build();
```

Distribution objects can be composed without rewriting kernels:

```java
MixtureDistribution mixture = Distributions.mixture(
    new double[] {0.25, 0.75}, first, second);
TruncatedContinuousDistribution positive =
    Distributions.truncate(mixture, 0.0, Double.POSITIVE_INFINITY);
MonotoneTransformDistribution rescaled =
    Distributions.affine(positive, 10.0, 2.0);
```

General monotone changes of variable use `Distributions.transform`, which asks
for the forward function, inverse function, log absolute inverse derivative,
direction, and transformed support. The beginner-friendly
[composition tutorial](docs/composition-tutorial.html) walks through mixtures,
truncation, censoring, affine transformations, and a complete nonlinear
example. Its worked response-time vignette transforms a normal mixture and then
uses the resulting distribution for density, CDF, quantile, and reproducible
random-sample calculations. [`examples/CompositionExamples.java`](examples/CompositionExamples.java)
is compiled by every `check` build.

Numerical distributions also expose expectations, raw and central moments,
entropy, modes, and equal-tail probability intervals. Array APIs now include
allocation-free `densityInto`, `cumulativeInto`, `quantileInto`, and
`randomInto`; continuous batch CDF calls reuse the monotone cache.

The complete snippets in
[`examples/CustomDistributionExamples.java`](examples/CustomDistributionExamples.java)
are compiled against the packaged JAR during every `check` build.

Moment diagnostics integrate `abs(x)^k * density(x)` for user-selected orders
and report convergence separately on either side of a chosen split. This
prevents symmetric cancellation from making a nonexistent signed moment look
convergent.

For suitable finite-support densities, callers can install a certified uniform
or custom `RejectionEnvelope`; `random()` then uses rejection sampling instead
of repeated inverse-CDF calculations. Encountered envelope violations fail
explicitly, but the caller remains responsible for the global bound.

Log-kernel and log-weight factories avoid overflow or underflow when the
unnormalized formula cannot be represented on the ordinary scale:

```java
NumericalContinuousDistribution shiftedNormal =
    NumericalContinuousDistribution.fromLogKernel(
        x -> 1000.0 - 0.5 * x * x,
        Double.NEGATIVE_INFINITY,
        Double.POSITIVE_INFINITY
    );
```

Automatic log-kernel construction searches for multiple modes, normalizes up to
eight regions with independent scales, and combines them with log-sum-exp. A
reusable adaptive monotone CDF table accelerates central quantiles and sampling;
the ordinary CDF remains direct for compatibility, while `cumulativeCached`
explicitly selects the approximation.

Piecewise and mixed distributions use `NumericalSupport`:

```java
NumericalSupport support = NumericalSupport.builder()
    .interval(-10.0, 10.0)
    .hole(-1.0, 1.0)
    .atom(0.0)
    .singularity(4.0)
    .build();

NumericalPiecewiseDistribution mixed =
    new NumericalPiecewiseDistribution(kernel, support, atomWeights, options);
```

Infinite integer supports require a user-supplied remainder certificate. The
result records an upper bound on omitted probability rather than inferring
convergence from several small terms.

```java
CertifiedInfiniteDiscreteDistribution geometric =
    CertifiedInfiniteDiscreteDistribution.rightInfinite(
        k -> Math.pow(r, k),
        0,
        DiscreteTailBounds.geometricRatio(r),
        CertifiedDiscreteOptions.defaults()
    );
```

See [the numerical-distribution guide](docs/NUMERICAL_DISTRIBUTIONS.md) for the
diagnostic model, limitations, and integration-method guidance.

## Tweedie distributions

`Tweedie` now provides the same density, cumulative, quantile, random, and
log-likelihood surface as the other distribution classes. It uses exact normal,
scaled-Poisson, gamma, and inverse-normal identities at powers 0, 1, 2, and 3;
the compound Poisson-gamma representation between 1 and 2; and a stabilized
series plus adaptive integration above 2.

## Contributed R-package distributions

JDistlib also audits GPL-compatible CRAN packages for useful distributions that
are absent from R's `src/nmath`. Modern contributed batches add complete APIs
from `distributions3`, VGAM, `actuar`, `extraDistr`, and `flexsurv`, including
modified count models, Poisson-binomial, sinh-arcsinh, generalized gamma/GB2/F,
phase-type and Feller-Pareto lifetime models, heavy-tailed half/slash families,
and Poisson-inverse-Gaussian. Sources, versions, parameterizations,
verification, duplicate exclusions, and screened dispositions are recorded in
[DISTRIBUTIONS.md](DISTRIBUTIONS.md).

Multivariate distributions use arrays for observations and parameters. They
provide joint density or mass and explicit-engine random generation; exact
joint CDFs are also available for bivariate Poisson and bivariate logistic.
There is intentionally no generic scalar quantile for a random vector.

## Thread safety

Pure static density, cumulative, and quantile calls use call-local state. Cached random
algorithms for binomial, hypergeometric, and Poisson sampling accept an explicit
`RandomState`; use one state and one `RandomEngine` per random stream. `SignRank`
and `Wilcoxon` intentionally keep their work tables in instances, so do not share
one mutable instance across concurrent callers.

`GenericDistribution` instances also contain a mutable random engine (changed by
`setRandomEngine`), so instance `random()` calls are not safe to share across
threads unless the caller supplies synchronization. Prefer one distribution
instance and one engine per thread or stream. Read-only density/CDF/quantile
calls on ordinary fixed-parameter instances do not mutate that engine.

Corrected BTPE sampling is the binomial default. To reproduce the historical R
4.6-and-earlier BTPE stream, create the per-stream state with
`Binomial.create_random_state(Binomial.BinomialKind.BUGGY_BTPE)`. The
`DistributionTest` Wilcoxon helpers default to R-devel's seven significant
digits and provide overloads for explicit `digitsRank` and `digitsZap` values;
pass positive infinity to disable either operation.

Version 0.7.0 also adds general Anderson-Darling and Cramer-von Mises
goodness-of-fit tests with reproducible parametric-bootstrap or permutation
p-values, plus Pearson chi-square tests for categorical fit and contingency
table independence. The one-sample resampling defaults assume the reference
distribution was fully specified independently of the tested sample.

## Upstream and license

The nmath-derived code is synchronized against the official R sources and is
distributed under the GNU General Public License, version 2 or later. See
[LICENSE](LICENSE), [UPSTREAM.md](UPSTREAM.md),
[DISTRIBUTIONS.md](DISTRIBUTIONS.md), and the historical
[JDistlib website](https://jdistlib.sourceforge.net/).
