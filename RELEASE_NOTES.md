# JDistlib 0.8.4

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
