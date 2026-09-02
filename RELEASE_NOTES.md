# JDistlib 0.10.1

JDistlib 0.10.1 completes native reusable sparse-Cholesky support across the
optional compute providers. CUDA, OpenCL, and Vulkan now perform FP64 and FP32
numeric factorization, refactorization, log-determinant evaluation, and
multi-right-side solves in retained provider buffers. A shared host-side
symbolic plan determines the fill pattern and permutation without densifying
the matrix.

GPU refactorization is transactional: rejected non-positive-definite numeric
updates leave the previous factor usable. Execution plans distinguish the
host-side symbolic analysis from provider-resident numeric work, and immutable
`dcsrpotrf`/`scsrpotrf` results retain the deterministic Java representation.

The existing oneMKL PARDISO path now uses persistent ABI-safe handle and
permutation storage, enables CSR matrix checking, and has broader regression
coverage for factor reuse and rollback. OpenBLAS itself has no sparse-direct
solver; when SuiteSparse CHOLMOD 5 or later is installed, JDistlib pairs it
with the OpenBLAS provider for minimum-degree FP64/FP32 prepared factors.
Natural ordering and installations without CHOLMOD retain the portable path.

The release gates exercise symbolic fill, FP64/FP32 solves, multiple right
sides, successful refactorization, rejected-refactor rollback, capability and
execution reporting, and a nontrivial oneMKL PARDISO sparse benchmark. All
changes preserve the 0.10.0 API and Java 8-compatible bytecode.

## Previous release: JDistlib 0.10.0

JDistlib 0.10.0 turns the optional accelerator layer into a reusable
backend-neutral linear-algebra engine for downstream statistical, graphical-
model, and scientific libraries. Matching FP64 and FP32 contracts cover
strided Level-1 operations; row-major GEMV, GEMM, symmetric updates and
triangular solves; no-copy matrix regions with explicit offsets and leading
dimensions; and batched GEMM, Cholesky, and LU entry points.

Reusable dense factors now include Cholesky, partial-pivoted LU,
symmetric-indefinite LDL', column-pivoted QR, ordinary and generalized
symmetric eigendecomposition, and thin SVD. Factor objects expose solves,
rank or pivot metadata, and determinant information where applicable. The
generalized eigen path uses an SPD metric, while portable reference
implementations preserve the API when a provider does not export a native
routine.

Sparse support now spans CSR matrix-vector, matrix-dense, and canonical
CSR-by-CSR multiplication; transposed and unit-diagonal sparse triangular
solves; and ordered sparse Cholesky. Prepared sparse factorizations separate
symbolic analysis from repeatable numeric refactorization. The oneMKL provider
can use PARDISO for this reusable path, while every installation retains a
checked portable implementation.

Prepared FP64 and FP32 dense handles retain a fixed operand for repeated or
batched products. Native CPU handles use the selected CBLAS runtime, and CUDA,
OpenCL, and Vulkan handles retain provider buffers instead of retransferring
the fixed matrix for each call. Expanded capability and execution-plan records
identify the selected backend, hardware, native support, and portable
fallbacks rather than implying that every operation executes on the device.

The native CPU module dynamically supports either oneMKL or OpenBLAS without
making either library mandatory. LAPACKE exports are used for supported dense
factorizations when available. The native-free Java CPU implementation remains
the deterministic reference and the all-in-one JAR continues to operate when
no native CPU or GPU runtime is installed.

Provider parity, mixed 1x1/2x2 LDL' pivots, matrix regions, batching, prepared
operands, sparse multiplication/solves, factor reuse, capability reporting,
packaging, JavaDoc, and the complete 92-model script catalog are release-gated.
All additions preserve existing APIs and Java 8-compatible bytecode.

## Previous release: JDistlib 0.9.1

JDistlib 0.9.1 adds a production-oriented sparse subset RJMCMC path for a large
candidate universe with a small active model. `SparseSubsetState` stores sorted
integer candidate indices instead of a fixed-width model bit mask, while
`SparseSubsetTarget` enforces an independent maximum-active bound. Exact
add/drop/swap calculations retain state-dependent forward and reverse candidate
probabilities, coefficient birth densities, and boundary-dependent move
selection probabilities.

Extremely long chains can run as bounded segments and resume during either
warmup or retained sampling. The checksummed portable checkpoint contains the
sparse state, current normalized log joint, exact random stream, global warmup
target and adaptation, 64-bit counters, move statistics, and online model-size,
inclusion, coefficient, and common-parameter moments. Draw segments and
checkpoints are flushed, forced, and atomically installed in a deterministic
order so interruption recovery can replay the same segment safely.

The new residual-informed candidate proposal uses repeated `X'v` scores with a
uniform mixture that preserves support for every inactive candidate. CUDA and
OpenCL providers can keep the prepared feature matrix resident on device; CPU
remains the reference implementation and Vulkan uses the CPU fallback for this
primitive. Accelerator parity and exact split/resume tests cover interrupted
warmup, corrupt or incompatible checkpoints, and 17,000-candidate states.

A complete public-data analysis downloads the processed GSE93272 Affymetrix
study from GEO, applies an outcome-blind 17,000-gene preparation, and fits a
normalized Gaussian subject random-intercept model with age, sex, batch, and RIN
covariates. The compiled Java example, Bioconductor preparation script, and
website tutorial cover priors, GPU measurement, multiple-chain review,
reboot-safe operation, and scientific limitations. The MCMC learning center
links the example directly.

Release downloads now include both the versioned all-in-one JAR and a stable
`jdistlib-all.jar` alias. Website download buttons use GitHub's latest-release
redirect, and runnable source-tree documentation uses version-independent
classpath wildcards, preventing the version drift corrected in this release.

All additions preserve the Java 8 bytecode target and existing APIs.

## Previous release: JDistlib 0.9.0

JDistlib 0.9.0 completes the probability-first finance and options roadmap
without turning the library into a market-data, instrument-management, pricing,
or execution system. The development surface after 0.9.0 also completes every
additive finance follow-up recorded in `TODO.md`: tempered-stable and Meixner
laws, multivariate financial families, accelerated checked aggregation,
adaptive transforms and stable tail paths, smooth option-implied recovery,
advanced risk measures, path functionals, and Lévy increment composition.
`AdvancedFinanceExamples` demonstrates the combined workflow.

The new `jdistlib.finance` package starts with explicit
loss/return and upper/lower-tail conventions. It supplies atom-aware VaR and
expected shortfall, partial moments, downside and shortfall measures, expectiles,
stop-loss, call, and put expectations, including caller-owned VaR output arrays
and immutable numerical diagnostics.

Normal, gamma, Poisson, and Student-t objects now expose log characteristic and
cumulant-generating transforms with existence domains. Numerical transform,
cumulant, Fourier-inversion, and normalized Esscher-tilt helpers sit beside a
canonical generalized-hyperbolic/NIG family, variance-gamma mixture law, and a
documented S1 alpha-stable implementation with named S0 conversion, exact normal
and Cauchy reductions, and explicit-stream generation.

Independent convolution, weighted sums, compound sums, products, ratios, and
scenario transformations use a reproducible, strategy-reporting Monte Carlo
fallback. Conditional intervals and iid extrema retain exact distribution
semantics. Delaporte and Polya-Aeppli provide robust compound-count APIs. Copula
coverage adds 90/180/270-degree rotation, survival forms, Joe and BB1 families,
finite-level and asymptotic tail concentration, mixed stress regions, and
tail-weighted likelihoods; ordinary pair fitting and selection recognize the
new families.

The reusable fitting layer supports bounded MLE/MAP objectives, weighted exact,
left/right-censored, and interval observations, numerical covariance estimates,
and caller-defined calibration losses. EVT helpers add GEV MLE/PWM, GPD peaks
over threshold, Hill and Pickands estimators, return levels, bootstrap
uncertainty, mean residual life, and threshold-stability summaries.

Reference Black-Scholes and Bachelier transformations now have checked implied-
volatility inversion with no-arbitrage bounds, bracket, iterations, residual,
and status. `OptionCurve` converts puts through parity, repairs call strips to
bounds, monotonicity, convexity, and terminal conditions, reports every repair,
and returns a normalized atom-aware risk-neutral distribution. Calibration and
MCMC likelihood adapters accept point or bid/ask observations, while posterior
ensembles produce terminal-price, payoff, strike-event, VaR, and expected-
shortfall summaries with MCSE, seed/chain provenance, and an explicit
risk-neutral or physical/predictive label.

Two compiled Java catalogs and two website tutorials cover the new surface.
`FinanceFeatureExamples` demonstrates tails, transforms, heavy-tailed families,
aggregation, dependence, and EVT. `WorkedOptionsTradingExample` validates a call
strip, recovers probabilities, works a covered-call terminal payoff, and keeps a
physical scenario separate from the option-implied measure. Regression coverage
includes reductions, transform domains, seeded reproducibility, curve repair,
normalization, count-law degeneracies, and finance boundary conventions. All
public additions preserve Java 8 bytecode compatibility.

# JDistlib 0.8.5

JDistlib 0.8.5 adds two complementary model-selection paths without changing
the existing sampler contracts or Java 8 bytecode target. The new Java-only
reversible-jump layer represents each model with a named variable-dimensional
state and evaluates a complete normalized log joint. General moves declare
their reverse, forward and reverse auxiliary densities, and log absolute
Jacobian; the sampler adds the boundary-dependent probabilities of selecting
each applicable move. Dimension-matching utilities validate inverse round
trips and reciprocal Jacobians.

The first production RJMCMC workflow targets covariate, feature, and genotype-
locus selection. It supplies add, drop, and swap moves for up to 62 candidates,
prior-matched or warmup-adaptive Gaussian coefficient births, per-model
continuous random walks, and adapters for existing fixed-dimensional JDistlib
samplers. Move schedules and proposal adaptation freeze after discarded
warmup. Ragged draws support tidy CSV export, and checksummed portable
checkpoints retain the exact random stream, model state, move weights, and
component adaptation for deterministic continuation.

RJMCMC diagnostics report posterior model and inclusion probabilities with
ESS, MCSE, and split R-hat, model-transition counts, top-model round trips,
per-direction acceptance and invalid proposals, and coefficient summaries
conditional on presence. Analytical tests verify exact posterior odds,
birth/death and swap reciprocity, nonzero Jacobians, dimension-map inverses,
schedule-independent parallel chains, and exact restart. A complete four-chain
linear-regression example is linked directly from the MCMC learning center and
walks through normalization, add/drop/swap sampling, review, export, and
checkpoint resume.

For models that fit a fixed padded representation, 0.8.5 also adds typed mixed
state spaces, exact finite-discrete Gibbs updates, discrete and continuous-
block Metropolis kernels, hybrid scheduling, and per-kernel diagnostics. Model
assessment now includes pointwise log-likelihood extraction for Java and
compiled scripts, PSIS-LOO with Pareto-k diagnostics and refit fallbacks, WAIC,
paired comparison, predictive stacking, Gaussian projection-predictive forward
selection, and shrinkage ranking by practical-significance probability.

The probability layer adds exact multinomial, Dirichlet-multinomial, and
multivariate-hypergeometric rectangle calculations; error-reporting Dirichlet,
multivariate Laplace, and multivariate power-exponential rectangles; and named
Wishart directional quadratic-form, standardized-trace, determinant, and log-
determinant events. The four-parameter Wiener first-passage density and exact
bias-conditioned inversion RNG are available in Java and compiled model
scripts, with a checked reaction-time model in the executable catalog.

The recommended GitHub artifact remains the self-contained
`jdistlib-all-0.8.5.jar`, containing core JDistlib and the CUDA, OpenCL, and
Vulkan providers with their cross-platform x86-64 Java/JNI dependencies. The
native-free core and small modular accelerator artifacts remain available for
dependency-managed builds. This release preserves existing 0.8 APIs,
deterministic caller-owned random streams, CPU fallback, and Java 8-compatible
class files.

## Previous release: JDistlib 0.8.4

JDistlib 0.8.4 makes GPU-enabled installation a one-download operation and
adds the Vulkan provider that was intentionally deferred from 0.8.3. The
recommended release artifact is now `jdistlib-all-0.8.4.jar`, a standalone
x86-64 JAR containing core JDistlib, CUDA, OpenCL, Vulkan, JTransforms, and all
required Java/JNI binding libraries for Windows, Linux, and macOS. GPU vendor
drivers remain system components. When no compatible GPU runtime is present,
the same artifact retains deterministic CPU operation.

The new Java 8-compatible `jdistlib-vulkan` module uses LWJGL Vulkan 3.3.6 and
shaderc. It selects a compute-capable device with `shaderFloat64`, records the
device and memory capabilities, and implements the same unary vector, AXPY,
dot-product, dense-GEMM, and batched logistic likelihood/gradient contract as
the CUDA and OpenCL providers. Vulkan core GLSL does not guarantee FP64
transcendental overloads, so the provider supplies deterministic range-reduced
double-precision exponential, logarithm, `log1p`, logistic, and hyperbolic
tangent shader routines instead of reducing those operations to float.

Real-device parity tests compare every Vulkan primitive with the CPU reference.
The release candidate passed on an NVIDIA GeForce RTX 2080 through Vulkan 1.4,
including direct provider selection and a standalone hardware smoke. CI hosts
without FP64 Vulkan hardware skip only the device-dependent parity assertions;
CPU fallback and unified-JAR discovery remain mandatory release checks.

The all-in-one build merges the CUDA, OpenCL, and Vulkan service descriptors,
bundles Windows/Linux JCuda JNI libraries and Windows/Linux/macOS x86-64 LWJGL
and shaderc resources, removes conflicting signatures/module descriptors, and
runs a classpath-isolated smoke using no external JARs. On supported hardware
that smoke exercises every detected backend. Its approximately 41 MB size is
the cost of carrying multiple operating-system native runtimes in one file.

Small modular artifacts remain available for Gradle/Maven users and for
applications that require only CPU or one GPU API. The native-free core and all
existing 0.8 APIs remain source- and binary-compatible, deterministic caller-
owned random streams are unchanged, and JDistlib-produced classes remain Java
8 bytecode. Accelerator lint flags are now JDK-version-aware, restoring the
Java 17 CI lane and allowing the downstream Java 8 smoke to exercise both core
and unified artifacts. The GitHub release deliberately exposes the single all-in-one
runtime plus `SHA256SUMS`; source is available from the signed release tag and
JavaDoc remains on the project website.

## Previous release: JDistlib 0.8.3

JDistlib 0.8.3 is a broad Stan-compatibility, numerical-modeling, and modern
inference release. It preserves the complete 0.8 API, deterministic caller-owned
random streams, the native-free core artifact, and Java 8 bytecode.

The source frontend now accepts arbitrary-rank arrays and slicing, vectors,
row-vectors, rectangular matrices, complex values, procedural tuples, immutable
CSR matrices, structured covariance/correlation constraints, and typed
orientation-aware linear algebra. Forward-declared, overloaded, recursive,
container-valued, tuple-valued, external Java, and higher-order functions are
supported within the documented compatibility boundary. Compiled script factors
lower onto reusable thread-local reverse tapes with specialized probability,
matrix, distance, external, and numerical-sensitivity kernels.

Java-native damped-Newton algebraic solving, adaptive Dormand-Prince ODEs,
stiff BDF integration, implicit index-1 DAEs, projected holonomic index-3
mechanics, and one-dimensional quadrature have differentiable script bindings.
Parameter sensitivities propagate through the reusable reverse-mode runtime.
The checked catalog now contains fifty JDistlib model scripts and forty-one
ordinary `.stan` fixtures; every release build validates all ninety-one models.

The inference layer adds full multi-path Pathfinder with retained L-BFGS
curvature, ELBO selection, mixture scoring, Pareto-smoothed importance sampling,
Pareto-k diagnostics, and systematic resampling. Many-short-chain superchains
include basic, rank-normalized, and folded nested R-hat. Coordinated ChEES or
SNAPER adaptive static HMC and auditable adjusted-MCLMC pilot tuning add regular
many-chain and microcanonical alternatives while retaining NUTS as the
conservative default.

Workflow additions include checksummed portable checkpoints, fingerprinted
warmup reuse, arbitrary-function and indicator ESS/MCSE, precision-driven
continuation, factor cost/nonfinite/heap profiling, geometry advice,
machine-readable health policies, compressed selected-column storage,
memory-mapped draws, and generated-quantity sinks.

Optional `jdistlib-cuda` and `jdistlib-opencl` modules provide FP64 vector math,
AXPY, dot products, dense GEMM, and prepared batched likelihood/gradient
evaluation behind a Java 8 service-provider interface. `Compute.AUTO` uses
conservative workload thresholds and deterministic CPU fallback; explicit
GPU/CUDA/OpenCL requests are strict. Java applications configure
`SamplingOptions.backend(...)` and `nutsBackend(...)`; embedding command-line
applications can expose `--compute`, `--nuts-offload`, and `--gpu-nuts`.
Selection is logged and retained with device provenance in `RunManifest`.

The measured RTX 2080 logistic-regression smoke shows why automatic routing is
thresholded: one state was slower than CPU, while resident batches of 4, 16, and
64 reached 3.259x, 7.406x, and 16.665x speedups with maximum likelihood/gradient
error below 3.1e-11. NUTS tree control remains on CPU; forced target offload is
available but explicitly warns that it may be slower. Vulkan is recognized as a
future provider identifier but is not shipped in 0.8.3.

The learning center includes focused container, function, solver, reverse-mode,
complete CSV/JDM MCMC, inference modernization, and GPU acceleration guides.
Release assets include binary, source, and JavaDoc JARs for the core, CUDA, and
OpenCL modules, with SHA-256 checksums.

## Previous release: JDistlib 0.8.2

JDistlib 0.8.2 expanded the versioned Stan-inspired model-script frontend with
scoped scalar locals, control flow, stable scalar mathematics, more than thirty
probability families and RNGs, file-backed data examples, and complete
in-memory, cached, ahead-of-time, and CLI compilation guidance. It added nine
focused models and brought the JDistlib script catalog to fifty programs while
preserving the numerical corrections from 0.8.1 and the complete 0.8.0
inference API.

## Previous release: JDistlib 0.8.1

JDistlib 0.8.1 is a focused correctness release following the 0.8.0 inference
release. It addresses five reports found by cross-checking R Bugzilla against
JDistlib's numerical and `jdistlib.disttest` implementations:

- tied Mood and Ansari-Bradley tests now use the corrected exchangeable-score
  moments from R PR#19013, including the proper one-sided Ansari tails;
- negative half-integer Bessel J and Y orders retain their connection-formula
  terms instead of incorrectly returning zero;
- noncentral-t densities use a direct log-domain hypergeometric representation
  in the cancellation-prone central region identified by R PR#17519;
- Fligner-Killeen tests round centered absolute deviations to seven significant
  digits before ranking by default, with an overload accepting a caller-chosen
  value and `Double.POSITIVE_INFINITY` preserving raw binary64 ranking; and
- fixed-probability negative binomials have explicit infinite-size limits and
  stable asymptotic behavior when finite size is near `Double.MAX_VALUE`.

Regression coverage contains the original Bugzilla examples, corrected R
reference values, affine-invariance checks, log/tail boundaries, and extreme
finite-size sweeps. The complete project check, documentation examples, and 40
model-script validations pass, and produced classes remain Java 8 bytecode.

## Previous release: JDistlib 0.8.0

JDistlib 0.8.0 introduces a separate Bayesian inference layer without changing
the normalized distribution APIs. Named Java models and a versioned,
Stan-inspired script frontend lower into the same constraint- and
Jacobian-aware representation. The release includes Metropolis, slice, Gibbs,
adaptive-rejection conditional updates, HMC, and adaptive multinomial NUTS with
deterministic multi-chain execution.

Inference results are immutable and include warmup, retained sampler
statistics, cancellation status, warnings, and in-memory state-and-stream restart points.
Diagnostics cover rank-normalized split and folded R-hat, bulk/tail ESS, MCSE,
divergences, tree-depth saturation, acceptance, and E-BFMI. Chart-neutral trace,
rank, ACF, energy, pair, and model graph data export to JSON, CSV, SVG,
self-contained HTML, and Graphviz DOT without requiring a UI toolkit.

The 0.8 modeling language supports validated data and constrained parameter
declarations, transformed blocks, vectorized sampling statements, `target +=`,
forward-mode derivatives, generated quantities/RNGs, ahead-of-time Java source,
SHA-256 compilation caching, isolated class loading, and a CLI workflow. It is
Stan-inspired rather than a claim of Stan compatibility.

Reference and regression coverage includes conjugate beta-binomial and normal
posteriors, gradients/Jacobians, ordered and simplex transforms, heavy-tailed
and mixed discrete/continuous sampling, checkpoints, deterministic chains,
diagnostic/export schemas, model graphs, generated source, and strict Java 8
compilation. See `docs/INFERENCE.md`, `docs/MODELING_LANGUAGE.md`, and
`docs/INFERENCE_COMPATIBILITY.md` for the complete contracts.

The learning center now includes an end-to-end inference tutorial, a sampler
and diagnostics guide, posterior-predictive and difficult-geometry vignettes, a
second complete Stan-inspired script lesson, and a catalog backed by fifteen
named executable models. Sampling and post-processing also reuse chain-local
model buffers, Metropolis proposals, covariance scratch storage, and
allocation-free chain/metric read paths.

The release also ships forty standalone Stan-inspired `.jdm` examples and a
semantic catalog validator run by every `check` build. Feature-focused Java
examples cover copulas, mixtures, transformations, FDR, custom laws, numerical
integration, and both programmatic and scripted MCMC. The website’s **Browse
examples** center links every catalog and source file.

## Previous release: JDistlib 0.7.2

JDistlib 0.7.2 hardens the copula framework introduced in 0.7.0. It expands
independent validation for mixed and simplified-vine likelihoods, adds
auditable fitted-model diagnostics, and incorporates API feedback through
additive conveniences while preserving the existing public API, deterministic
seeded behavior, and Java 8 bytecode.

Validation and numerical improvements include:

- checked-in 90-digit Decimal reference corpora for central, rare-event, and
  boundary-heavy Clayton mixed measures and three-dimensional C-/D-vine log
  densities, with an implementation-independent generator;
- analytic conditional CDFs for Clayton, Gumbel, and Frank pair copulas, direct
  analytic inverses for Clayton and Frank, and the existing numerical fallback
  for custom copulas; and
- a scaled/log-domain Clayton conditional calculation that remains stable near
  unit-cube boundaries.

API and diagnostic improvements include:

- row-level `CopulaLikelihoodDiagnostics`, including individual log-density
  contributions, boundary distances, summary statistics, and the first
  non-finite observation;
- `CopulaLogLikelihoodResult`, which retains mixed-measure contributions,
  numerical warnings, CDF evaluation cost, maximum reported error, and the
  first failing row;
- fitted pair and vine diagnostics, observation counts, and vine AIC/BIC;
- deterministic midpoint and explicit-seed overloads for mixed fitting,
  selection, and vine fitting; and
- defensive marginal-array access and conventional getters on existing result
  objects without removing their original fields or methods.

The beginner copula tutorial, complete copula guide, compiled examples, API
guide, website, changelog, and TODO ledger describe the additions. Regression
coverage includes independent mixed/vine references, boundary conditionals,
diagnostic aggregation, fitted-model information criteria, defensive copies,
and exact equivalence between seed overloads and caller-owned
`MersenneTwister` streams.

Release assets include the binary library, sources, JavaDoc, and SHA-256
checksums. See `CHANGELOG.md` for the detailed change list and `PUBLISHING.md`
for the separate Maven Central maintainer step.
