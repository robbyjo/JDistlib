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
phase-type, Poisson-binomial, Poisson-inverse-Gaussian, Delaporte, Polya-Aeppli,
generalized-hyperbolic/NIG, variance-gamma, alpha-stable, CGMY/KoBoL,
normal-tempered-stable, Meixner, Rice, sinh-arcsinh,
Skellam, skewed t, slash, triangular, Tukey lambda, Tweedie, Wishart, Zipf,
Wiener first-passage, and the `jdistlib.evd` package. Vector APIs
cover Dirichlet and Dirichlet-multinomial, multivariate hypergeometric,
bivariate Poisson/logistic, and multivariate normal, Student t, Cauchy,
lognormal, Laplace, power-exponential, and financial GH/NIG/VG/stable/NTS
constructions. These are first-class JDistlib features
and are not removed during upstream synchronization.

## Project status

Version 0.9.1 is the current stable release. It adds sparse subset RJMCMC for
thousands of candidate features, exact restart during warmup or sampling,
crash-safe segmented output, CUDA/OpenCL-assisted residual proposals, and a
public GSE93272 mixed-model worked example. Version 0.9.0 added
probability-first finance:
tail and payoff functionals, transform-domain and heavy-tailed laws,
aggregation, tail-sensitive copulas, EVT fitting, checked implied volatility,
arbitrage-repaired option curves, and risk-neutral or posterior-predictive
distribution outputs. Version 0.8.5 added Java-only reversible-jump MCMC,
fixed-dimensional mixed-state inference, PSIS-LOO/WAIC assessment, predictive
stacking and variable-selection tools, multivariate rectangle and Wishart event
probabilities, and the Wiener first-passage model. Version 0.8.4
added the tested Vulkan FP64 provider and the self-contained cross-platform
x86-64 JAR that bundles core JDistlib, CUDA, OpenCL, Vulkan, and their Java/JNI
runtime dependencies. The modular artifacts remain available for dependency-
managed builds. Version 0.8.3 added
Stan complex and tuple values, sparse kernels, external and higher-order
functions, reusable reverse-tape script execution, sensitivity-aware algebraic/
ODE/DAE solvers, a stiff BDF path, holonomic index-3 DAE projection, and a much
larger ordinary-Stan compatibility catalog. It also adds multi-path Pathfinder,
many-short-chain and adaptive-static-HMC workflows, automatic adjusted-MCLMC
tuning, production diagnostics/storage QoL, and optional CUDA/OpenCL compute
backends. The consolidated [inference and acceleration result](docs/INFERENCE_ACCELERATION_RESULT.md)
records the delivered surface, validation, and measured accelerator boundary.
Version 0.8.2 expanded the language
with scoped scalar locals, control flow, stable scalar math, more than thirty
probability families and RNGs, file-backed data examples, and complete
in-memory/cached/ahead-of-time compilation guidance. It retains the 0.8.1
numerical corrections and the Bayesian model composition, reproducible MCMC,
diagnostics, and graph exports introduced in 0.8.0. Version 0.7.x
introduced copula composition, dependence fitting,
multiple-testing/FDR, CQUAD integration, and distribution composition. The R `src/nmath`
file-by-file audit from the historical R 3.3.2 baseline to R 4.6.1 is complete.
[UPSTREAM.md](UPSTREAM.md) is the source-of-truth checklist and
[NMATH_AUDIT.md](NMATH_AUDIT.md) records the source disposition and reproducible
R 4.6.1 reference corpus. JDistlib-specific APIs remain separately documented
and tested.

The executable catalog contains fifty-one JDistlib model scripts and forty-one
ordinary Stan fixtures, including focused
examples for multidimensional indexing, matrix probability, container-valued
functions, complex/tuple values, sparse and external functions, solver
sensitivities, structured constraints, robust/logit/count/lifetime models, and
control flow. Java-native algebraic, adaptive RK45, stiff BDF, index-1 DAE, and
holonomic index-3 DAE solvers are available under `jdistlib.inference.solver`;
the algebraic, ODE, DAE, and 1-D quadrature paths also have differentiable
modeling-language bindings.

The library also provides a Java-native
[Stan source-compatibility core](docs/STAN_SOURCE_COMPATIBILITY.md). Ordinary
`.stan` fixtures exercise arbitrary-rank arrays and slices, complex/tuple
values, sparse and typed matrix algebra, container/external/higher-order user
functions, broadcasting, structured constraints, and transformed containers.
JDistlib preserves the model meaning of supported source while intentionally
retaining its own Java reverse-mode implementation, RNG, samplers,
diagnostics, floating-point behavior, and output formats.

The completed [0.9.0 finance and options roadmap](docs/FINANCE_ROADMAP.md) and
[finance tutorial](docs/finance-tutorial.html) cover tail risk, transform-domain
and heavy-tailed families, aggregation, tail-sensitive dependence,
distribution/EVT fitting, and arbitrage-constrained option-implied and
posterior-predictive distributions. A separate
[worked options analysis](docs/options-trading-worked-example.html) keeps
risk-neutral and physical probabilities explicit.
Version 0.9.0 additionally completed the finance follow-ups with
multivariate GH/NIG/VG/stable/NTS constructions, exact/Panjer/FFT/COS and
saddlepoint aggregation, adaptive transform inversion, stable tail paths,
smooth option-implied recovery, spectral/distortion/entropic risk, drawdown
laws, and Lévy-process increment composition. See the compiled
[`AdvancedFinanceExamples`](examples/AdvancedFinanceExamples.java).

## Download

The [latest JDistlib release](https://github.com/robbyjo/JDistlib/releases/latest)
provides one recommended, self-contained `jdistlib-all` JAR with SHA-256
checksums. It contains core JDistlib, the optional oneMKL/OpenBLAS adapter,
CUDA, OpenCL, Vulkan, and the required
Java/JNI libraries for Windows, Linux, and macOS x86-64. It produces Java
8-compatible bytecode. Download the
[all-in-one JAR directly](https://github.com/robbyjo/JDistlib/releases/latest/download/jdistlib-all.jar).

GPU vendor runtimes remain system components: CUDA requires a compatible
NVIDIA driver and NVRTC, OpenCL requires an installed OpenCL implementation,
and Vulkan requires a Vulkan driver. With none present, the same JAR uses the
CPU backend. Gradle/Maven users who prefer small dependency-managed artifacts
can use the core, `jdistlib-nativecpu`, `jdistlib-cuda`, `jdistlib-opencl`, or `jdistlib-vulkan`
modules instead.

## Building

JDK 17 or newer is recommended for building. Produced class files remain Java 8
compatible.

```text
./gradlew test
./gradlew build
./gradlew validateModelScripts
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
* [Probability-first finance](https://robbyjo.github.io/JDistlib/finance-tutorial.html) —
  tail risk, heavy-tailed laws, aggregation, EVT, option-implied distributions,
  calibration, and predictive outputs (JDistlib 0.9.0+).
* [Worked options analysis](https://robbyjo.github.io/JDistlib/options-trading-worked-example.html) —
  checked IV, arbitrage repair, risk-neutral probabilities, a terminal covered-
  call payoff, and an explicitly separate physical scenario.
* [Multiple testing and FDR](https://robbyjo.github.io/JDistlib/multiple-testing.html) —
  adjusted and log p-values, adaptive BKY, censored families, rejection
  thresholds, and Storey q-values (JDistlib 0.7.0+).
* [Building a custom distribution](https://robbyjo.github.io/JDistlib/custom-distributions.html#beginner-path)
  and the [sensor-error vignette](https://robbyjo.github.io/JDistlib/custom-distribution-vignette.html)
  (JDistlib 0.6.0+).
* [Using copulas](https://robbyjo.github.io/JDistlib/copula-tutorial.html) and the
  [mixed-claims vignette](https://robbyjo.github.io/JDistlib/copula-vignette.html)
  (**copula features require JDistlib 0.7.0+**).
* [Bayesian modeling and MCMC](docs/INFERENCE.md) — model composition,
  samplers, diagnostics, graphing, and reproducibility (JDistlib 0.8.0+).
* [Modern MCMC API](docs/MCMC_MODERNIZATION.md) — Stan-style warmup, exact
  checkpoints, metrics, adjusted MCLMC, additional samplers, initialization,
  streaming, diagnostics, and SBC.
* [GPU acceleration and measured smoke](docs/gpu-acceleration.html) — CUDA,
  OpenCL, and Vulkan backends, automatic CPU fallback, the unified download,
  reproducible RTX 2080 likelihood numbers, and the still-provisional
  whole-NUTS decision.
* [Unified dense and sparse linear algebra](docs/LINEAR_ALGEBRA_ACCELERATION.md) —
  parallel FP64/FP32 BLAS operations, CSR kernels, and reusable Cholesky,
  symmetric-eigen, pivoted-QR, and thin-SVD decompositions,
  backend capabilities, routing, and numerical semantics.
* [Inference acceleration result](docs/INFERENCE_ACCELERATION_RESULT.md) — the
  consolidated feature inventory, validation record, benchmark interpretation,
  compatibility boundary, and v0.8.4 packaging/provider status.
* [Post-Stan MCMC roadmap](docs/MCMC_FUTURE_ROADMAP.md) — the design record for
  full Pathfinder, nested R-hat, adaptive static HMC, adjusted-MCLMC tuning,
  workflow QoL, and explicitly experimental algorithms.
* [Fully worked CSV-to-MCMC tutorial](docs/inference-tutorial.html#worked) — a
  line-by-line-commented Java example that loads a CSV and JDM file, compiles
  the model, tunes and runs NUTS, diagnoses and plots the chains, summarizes the
  posterior, and prints a conclusion.
* [Modeling language](docs/MODELING_LANGUAGE.md) — the versioned,
  Stan-inspired script frontend and ahead-of-time Java workflow.
* [Data ingestion and script compilation](docs/modeling-language-tutorial.html) —
  read CSV data, fit the same model through Java or a `.jdm` script, and choose
  in-memory, cached, or ahead-of-time compilation.
* [JDistlib for Stan users](docs/stan-users.html) — concept mapping, model
  migration, data binding, sampling, output, and compatibility boundaries.
* [Stan source compatibility](docs/STAN_SOURCE_COMPATIBILITY.md) — supported
  ordinary Stan syntax, Java execution differences, conformance tests, and the
  remaining boundary.
* [Reverse-mode autodiff](docs/REVERSE_AUTODIFF.md) — reusable tape lifecycle,
  automatic script lowering, atomic kernels, and the sampler-facing API.
* [Stan containers and matrices](docs/stan-containers-tutorial.html) — literals,
  arrays of vectors/matrices, slicing, assignment, and matrix pipelines.
* [Stan user functions](docs/stan-functions-tutorial.html) — forward declarations,
  overloads, data-qualified arguments, recursion, and probability suffixes.
* [Algebraic, ODE, and DAE solvers](docs/stan-solvers-tutorial.html) — Java and
  script callbacks, sensitivities, stiff integration, higher-index projection,
  tolerances, and work guards.
* [Inference tutorial](docs/inference-tutorial.html), [complete guide](docs/inference-guide.html),
  [worked vignette](docs/inference-vignette.html), and
  [diagnostics vignette](docs/inference-diagnostics-vignette.html) — the full
  0.8.0 learning path, including fifteen executable reference models.
* [Browse all examples](https://robbyjo.github.io/JDistlib/examples.html) —
  fifty-one JDistlib scripts, forty-one ordinary `.stan` compatibility fixtures, and
  compilable Java workflows for
  copulas, mixtures, transformations, FDR, custom distributions, finance,
  options, MCMC, and numerical integration.

## Using the distribution APIs

Distribution classes expose static density, cumulative, quantile, and random
methods. For example:

```java
double p = Normal.cumulative(1.96, 0.0, 1.0, true, false);
double x = Normal.quantile(0.975, 0.0, 1.0, true, false);
```

Boolean arguments follow R's `lower.tail` and `log.p` conventions.

For multivariate normal, Student t, Cauchy, log-normal, Dirichlet, Laplace, and
power-exponential laws, `probability`
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

Multinomial, Dirichlet-multinomial, and multivariate-hypergeometric rectangles
instead use inclusive integer bounds and an exact sequential-conditional
dynamic program, so they return a `double` without numerical-error metadata.
Wishart exposes named directional quadratic-form and standardized-trace CDFs,
and error-reporting determinant/log-determinant probabilities; it deliberately
does not label an entrywise matrix event as a generic CDF. See the
[Wishart probability contract](docs/WISHART_PROBABILITIES.md).

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

## Bayesian modeling, MCMC, and diagnostics

The `jdistlib.inference` package composes observed data, constrained parameters,
priors, and likelihood factors into an unnormalized multivariate target.
Programmatic Java models and the Stan-inspired 0.8 script language lower into
the same model representation. Scripts support shaped real/complex/sparse
containers, tuples, external and solver callbacks, control flow, a broad Stan
math surface, and more than thirty scalar probability families. HMC and
multinomial NUTS use script gradients lowered onto reusable reverse tapes or
Java-authored `ReverseModeLogDensity` targets,
dual-averaging warmup, and diagonal or dense metric
adaptation; Metropolis, slice, Gibbs, adaptive-rejection, and mixed block updates
cover targets without a single continuous gradient.

Multiple explicitly seeded chains produce immutable results. Diagnostics include
rank-normalized split/folded R-hat, bulk/tail ESS, MCSE, divergences, tree-depth
saturation, and E-BFMI. Trace, rank, autocorrelation, energy, pair, and model
graphs export to JSON, CSV, SVG, HTML, or Graphviz without a UI dependency. See
the [inference guide](docs/INFERENCE.md), [language guide](docs/MODELING_LANGUAGE.md),
[Stan-user guide](docs/stan-users.html), and
[compatibility contract](docs/INFERENCE_COMPATIBILITY.md). The fully commented
[worked CSV/JDM analysis](examples/WorkedMcmcCsvJdmExample.java) proceeds from
file loading through compilation, tuned parallel NUTS, diagnostics, plots,
exports, posterior summaries, and a conclusion. The shorter
[data-ingestion comparison](examples/McmcDataIngestionExamples.java) feeds the
same observations into both the Java builder and script compiler.

The 0.8.3 API adds full multi-path Pathfinder with PSIS, many-short-
chain superchains and nested R-hat, ChEES/SNAPER adaptive static HMC, automatic
adjusted-MCLMC step/decorrelation/mass-scale tuning, checksummed portable
checkpoints, MCSE-driven continuation, factor profiling, safe warmup reuse,
compressed selected-column draw storage, and machine-readable health advice.

The 0.8.5 API adds pointwise likelihood extraction for Java and compiled Stan
models, PSIS-LOO/WAIC comparison and predictive stacking, projection-predictive
and shrinkage variable selection, typed fixed-dimensional hybrid inference,
and Java-only reversible-jump MCMC. The RJ layer includes dimension-matching
maps, general reversible moves, subset add/drop/swap selection, model-specific
within-model kernels, frozen adaptation, ragged diagnostics/exports, portable
checkpoints, and a [worked covariate-selection analysis](docs/rjmcmc-example.html).

The additive sparse RJMCMC path removes the 62-candidate bit-mask limit for
large screening problems. It stores sorted active indices, caps model size
independently (for example, 20 active genes among 17,000), supports exact
restart during warmup or sampling, and offers residual-informed proposals using
a prepared `X'v` product. The [GSE93272 public expression-array
tutorial](docs/sparse-transcriptome-rjmcmc-example.html) combines this state with
a marginalized subject random-intercept model, bounded crash-safe draw
segments, and optional CUDA/OpenCL acceleration.

### Optional CUDA, OpenCL, and Vulkan acceleration

The dependency-managed core remains native-free. The `jdistlib-cuda` module
uses JCuda/JNvrtc, `jdistlib-opencl` uses JOCL, and `jdistlib-vulkan` uses LWJGL
Vulkan plus shaderc; all are discovered through
`ComputeBackend`, require FP64, and fall back to the deterministic CPU backend
when absent under `Compute.AUTO`. Automatic routing keeps small heap-backed work
on CPU and sends sufficiently large vector math, dense linear algebra, and
batched likelihood/gradient evaluation to CUDA, OpenCL, or Vulkan. Java callers
select through `SamplingOptions.backend(Compute...)` and
`nutsBackend(ComputeNuts...)`; embedding CLIs can accept `--compute`,
`--nuts-offload`, or the strict `--gpu-nuts` alias. Explicit GPU/provider
requests fail rather than silently falling back. See the
[GPU acceleration webpage](docs/gpu-acceleration.html) for Java and command-line
examples, measured guidance, logging/provenance, and the warning that forced
NUTS target offload may be slower. Direct-download users should use
`jdistlib-all`; modular artifacts are intended for dependency-managed builds.
The same providers expose public row-major FP64 and FP32 linear-algebra surfaces
with standard `d*` and `s*` BLAS/LAPACK names, CPU/GPU CSR matrix-vector and
matrix-matrix products, and FP64/FP32 Cholesky, symmetric-eigen, pivoted-QR,
and thin-SVD decompositions on CPU, CUDA, OpenCL, and Vulkan. See the
[linear-algebra contract](docs/LINEAR_ALGEBRA_ACCELERATION.md).
That contract also includes symmetric rank-k updates, triangular vector and
multi-RHS solves, prepared Cholesky handles, execution/device identification,
and strict optional `Compute.ONEMKL` and `Compute.OPENBLAS` CPU choices.
CUDA and OpenCL additionally implement a resident prepared transpose product
for repeated high-dimensional score calculations; Vulkan currently uses its
CPU fallback for that primitive.

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
