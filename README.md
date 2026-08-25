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
including arcsine, beta-binomial, beta-prime, Birnbaum-Saunders, categorical,
empirical, generalized beta/gamma, Gompertz, hurdle and zero-modified counts,
half-normal, inverse gamma, inverse normal, Lindley, Makeham, Kumaraswamy,
Laplace, Levy, log-logistic, logarithmic, Maxwell/Maxwell-Boltzmann, multinomial, Nakagami,
Poisson-binomial, Poisson-inverse-Gaussian, Rice, sinh-arcsinh, skewed t,
triangular, Tweedie, Wishart, Zipf, and the `jdistlib.evd` package. Vector APIs
cover Dirichlet and Dirichlet-multinomial, multivariate hypergeometric,
bivariate Poisson/logistic, and multivariate normal, Student t, Cauchy,
lognormal, Laplace, and power-exponential laws. These are first-class JDistlib features
and are not removed during upstream synchronization.

## Project status

Version 0.5.0 is the current stable release; `master` is the 0.5.1 development
line. The R `src/nmath` file-by-file audit from the historical R 3.3.2 baseline
to R 4.6.1 is complete.
[UPSTREAM.md](UPSTREAM.md) is the source-of-truth checklist and
[NMATH_AUDIT.md](NMATH_AUDIT.md) records the source disposition and reproducible
R 4.6.1 reference corpus. JDistlib-specific APIs remain separately documented
and tested.

## Download

The [latest GitHub release](https://github.com/robbyjo/JDistlib/releases/latest)
contains the Java library, source archive, JavaDoc archive, and SHA-256
checksums. Version 0.5.0 produces Java 8-compatible bytecode.

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
and auxiliary method coverage.

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

The result includes the estimated integral, absolute error, number of
subdivisions, and a QUADPACK-compatible status code.

For hostile or difficult callbacks, the additive `IntegrationOptions` API
supports evaluation budgets, cancellation checks, declared discontinuity or
singularity points, caught callback diagnostics, stability assessment, and
double-exponential methods for finite or infinite intervals. The historical overloads retain
their R/QUADPACK behavior.

```java
IntegrationOptions options = IntegrationOptions.builder()
    .tolerances(1e-10, 1e-10)
    .subdivisions(300)
    .maxEvaluations(250_000)
    .breakpoints(0.5)
    .method(IntegrationOptions.Method.AUTO)
    .build();

IntegrationStabilityResult stability =
    Integrate.assessStability(kernel, 0.0, 1.0, options);
```

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
summation, and caches the resulting probability and cumulative-mass tables.

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
        (first, firstWeight) -> firstWeight / (1.0 - r),
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
modified count models, Poisson-binomial, sinh-arcsinh, generalized gamma/GB2,
lifetime models, and Poisson-inverse-Gaussian. Sources, versions,
parameterizations, verification, duplicate exclusions, and the screened backlog
are recorded in [DISTRIBUTIONS.md](DISTRIBUTIONS.md).

Multivariate distributions use arrays for observations and parameters. They
provide joint density or mass and explicit-engine random generation; exact
joint CDFs are also available for bivariate Poisson and bivariate logistic.
There is intentionally no generic scalar quantile for a random vector.

## Thread safety

Pure density, cumulative, and quantile calls use call-local state. Cached random
algorithms for binomial, hypergeometric, and Poisson sampling accept an explicit
`RandomState`; use one state and one `RandomEngine` per random stream. `SignRank`
and `Wilcoxon` intentionally keep their work tables in instances, so do not share
one mutable instance across concurrent callers.

Corrected BTPE sampling is the binomial default. To reproduce the historical R
4.6-and-earlier BTPE stream, create the per-stream state with
`Binomial.create_random_state(Binomial.BinomialKind.BUGGY_BTPE)`. The
`DistributionTest` Wilcoxon helpers default to R-devel's seven significant
digits and provide overloads for explicit `digitsRank` and `digitsZap` values;
pass positive infinity to disable either operation.

## Upstream and license

The nmath-derived code is synchronized against the official R sources and is
distributed under the GNU General Public License, version 2 or later. See
[LICENSE](LICENSE), [UPSTREAM.md](UPSTREAM.md),
[DISTRIBUTIONS.md](DISTRIBUTIONS.md), and the historical
[JDistlib website](https://jdistlib.sourceforge.net/).
