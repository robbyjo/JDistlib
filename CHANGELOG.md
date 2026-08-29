What's new:

Version 0.9.1 (August 28, 2026):
* Replaced prose-only and malformed formulas in the distribution catalog with
  rendered definitions for the finance, copula, Wiener, and related laws, and
  added a build-time catalog validator for formula delimiters and summary counts.
* Added an arbitrary-candidate sparse subset RJMCMC engine with sorted active
  indices, an explicit active-size cap, exact add/drop/swap proposal accounting,
  residual-informed proposals, online inclusion summaries, and crash-safe tidy
  draw segments plus checksummed atomic continuation checkpoints. CUDA and
  OpenCL can keep prepared transcriptome matrices resident for repeated
  transpose products; CPU remains the reference and Vulkan falls back for this
  primitive.
* Added a compiled public-data GSE93272 expression-array mixed-model analysis,
  reproducible Bioconductor preparation script, and troubleshooting tutorial
  linked from the MCMC learning center. The regression suite verifies 17,000
  candidate indices, accelerator parity, corrupt/incompatible checkpoint
  rejection, and exact split/resume trajectories including interrupted warmup.

Version 0.9.0 (August 28, 2026):
* Completed the six additive finance follow-ups: tempered-stable and Meixner
  laws, multivariate normal-mixture and elliptical families, exact/Panjer/FFT/
  COS/saddlepoint aggregation, stable analytical tails and adaptive inversion,
  smooth option-implied density recovery, advanced risk measures, path
  functionals, and Lévy increment composition.
* Added checked diagnostics, numerical and reduction regressions, a compiled
  advanced-finance example, and expanded tutorial and distribution pages.
* Completed the probability-first finance roadmap with atom-aware tail/payoff
  analysis, transform domains and Esscher tilts, generalized-hyperbolic/NIG,
  variance-gamma and stable laws, reproducible aggregation, compound counts,
  rotated/Joe/BB1 copulas, tail stress diagnostics, generic censored fitting,
  EVT inference, checked implied volatility, arbitrage-repaired option curves,
  parametric option calibration, and labeled posterior-predictive adapters.
* Added `FinanceFeatureExamples`, a complete worked European-options analysis,
  two website tutorials, API/website navigation, and focused Java 8 regressions.

Version 0.8.5 (August 28, 2026):
* Added a Java-only reversible-jump MCMC framework with ragged model states,
  complete normalized targets, boundary-aware move scheduling, forward/reverse
  proposal terms, dimension-matching maps, Jacobians, model-specific
  within-model kernels, deterministic parallel chains, and frozen exact
  checkpoint/resume state.
* Added a production subset-selection workflow with add/drop/swap moves,
  adaptive or prior-matched coefficient births, model and inclusion
  ESS/MCSE/R-hat, transition and round-trip diagnostics, tidy ragged export,
  analytical detailed-balance tests, and a worked covariate/locus-selection
  website example linked from the MCMC learning center.
* Added pointwise log-likelihood extraction for Java and compiled model scripts,
  PSIS-LOO and WAIC assessment, paired model comparison, predictive stacking,
  Gaussian projection-predictive forward selection, and posterior shrinkage
  ranking.
* Added typed fixed-dimensional mixed state spaces with finite-discrete Gibbs
  and Metropolis kernels, conditional continuous-block updates, hybrid
  scheduling, and per-kernel diagnostics as a simpler indicator-model
  alternative to RJMCMC.
* Added exact multinomial, Dirichlet-multinomial, and multivariate-
  hypergeometric rectangle probabilities; error-reporting Dirichlet,
  multivariate Laplace, and multivariate power-exponential rectangle
  probabilities; and named Wishart directional, trace, determinant, and
  log-determinant probability APIs.
* Added the four-parameter Wiener first-passage density and exact
  bias-conditioned inversion RNG, including compiled-script likelihood/RNG
  bindings and a checked reaction-time model.
* Preserved the Java 8 bytecode target, native-free core, modular accelerator
  artifacts, and the self-contained CPU/CUDA/OpenCL/Vulkan distribution while
  advancing all release documentation and artifact links to 0.8.5.

Version 0.8.4 (August 28, 2026):
* Added the Java 8-compatible `jdistlib-vulkan` provider using LWJGL Vulkan and
  shaderc. It discovers a compute queue with FP64 shader support and implements
  unary vector math, AXPY, dot products, dense matrix multiplication, and
  batched logistic likelihoods/gradients against the CPU reference contract.
* Added deterministic FP64 shader implementations for exponential, logarithm,
  `log1p`, logistic, and hyperbolic tangent operations because Vulkan's core
  GLSL profile does not guarantee double-precision transcendental overloads.
* Added real-device Vulkan parity tests and an explicit hardware smoke; the
  release candidate passed on an NVIDIA GeForce RTX 2080 exposed through
  Vulkan 1.4.
* Added `jdistlib-all`, a standalone x86-64 JAR containing core JDistlib, all
  three GPU providers, JTransforms, JCuda, JOCL, LWJGL Vulkan/shaderc, and the
  Windows/Linux/macOS JNI resources needed by those bindings. CPU fallback
  remains available when no compatible GPU runtime is installed.
* Merged all compute service descriptors and added a classpath-isolated smoke
  that exercises every detected backend using only the unified JAR.
* Retained the small native-free core and modular CUDA/OpenCL/Vulkan artifacts
  as signed Maven publications while making the all-in-one JAR the sole
  recommended GitHub runtime download. Portable provider POMs leave platform-
  specific JCuda/LWJGL native classifier selection to the consuming build.
* Updated release automation, checksums, the prominent website download,
  README, publishing guide, GPU guide, and release notes for the unified
  distribution. All produced JDistlib classes remain Java 8 bytecode.
* Fixed accelerator-module lint configuration under JDK 17 by disabling the
  JDK-20-only `this-escape` category conditionally; restored the full CI matrix
  and extended its Java 8 runtime smoke to the unified JAR.

Version 0.8.3 (August 28, 2026):
* Added full multi-path Pathfinder with retained L-BFGS curvature, ELBO
  selection, mixture scoring, PSIS/Pareto-k diagnostics, and systematic
  resampling; added many-short-chain superchains with robust nested R-hat.
* Added coordinated ChEES/SNAPER adaptive static HMC and auditable automatic
  step-size, decorrelation-length, leapfrog-count, and diagonal-mass tuning for
  adjusted MCLMC.
* Added optional JCuda/JNvrtc and JOCL accelerator modules behind a Java 8
  service-provider API, with automatic availability detection, deterministic
  CPU fallback, vector/linear-algebra primitives, prepared batched likelihoods,
  and measured RTX 2080 CPU-reference results.
* Added thresholded `Compute.AUTO`, strict CPU/any-GPU/CUDA/OpenCL selection,
  programmatic `SamplingOptions` compute and NUTS-offload policies, reusable
  `--compute`/`--nuts-offload`/`--gpu-nuts` CLI parsing, selection logging, and
  backend/device provenance in `RunManifest`. Added a complete GPU acceleration
  webpage; Vulkan is explicitly recognized but not shipped.
* Added checksummed portable checkpoints, fingerprinted warmup reuse,
  arbitrary-function ESS/MCSE and precision continuation, factor profiling,
  geometry and health advice, mapped/compressed draw storage, and generated-
  quantity sinks. The complete result and release-label rationale are recorded
  in `docs/INFERENCE_ACCELERATION_RESULT.md`.
* Removed the homepage's duplicate Start Learning grid, folded distribution,
  custom-law, multiple-testing, and copula learning links into their matching
  feature cards, and retained one dedicated MCMC gateway organized around
  quick-start, pure-Java, example, JDM/Stan, beginner, and diagnostic routes.
* Added a fully commented, executable CSV-to-JDM MCMC analysis covering model
  compilation, configurable parallel NUTS, comprehensive diagnostics, trace/
  rank/ACF/energy exports, posterior summaries, predictive generation, and a
  convergence-aware conclusion; reorganized the MCMC tutorials around a
  task-oriented learning map.
* Added differentiable complex scalars/vectors/matrices and elementary
  overloads, nested procedural tuples and tuple-valued functions, immutable CSR
  matrices and Stan sparse kernels, and Java external-function declarations
  with explicit value/shape/Jacobian results.
* Lowered compiled script factors onto thread-local reusable reverse tapes;
  added reusable n-ary atomic edges and specialized normal, Student-t,
  dot/distance, matrix-normal, external, and solver-sensitivity reverse kernels.
* Added algebraic implicit sensitivities, ODE/DAE sensitivities, script
  higher-order bindings for `integrate_1d`, algebraic, RK45, BDF, and DAE
  callbacks, adaptive stiff BDF1 integration, and projected velocity-Verlet for
  holonomic index-3 DAEs with separate reference corpora.
* Added reduction, distance, SPD solve/inverse/log-determinant, triangular,
  symmetrization, and trace-quadratic standard-library overloads; expanded the
  ordinary Stan catalog from thirty to forty-one fixtures and the combined
  semantic gate from eighty to ninety-one scripts.
* Added Stan-compatible array/vector/row-vector/matrix literal expressions and
  forward function declarations, plus append, slicing, sequence-transform,
  diagonal, quadratic-form, cross-product, and matrix dot-product operations.
* Added Java-native damped-Newton algebraic, adaptive Dormand-Prince ODE, and
  implicit-Euler index-1 DAE solvers with explicit controls, work guards,
  numerical tests, and a runnable migration example.
* Expanded the ordinary `.stan` conformance suite from twelve to forty-one models
  and the combined semantic gate to ninety-one scripts; added focused v0.8.3
  tutorials for containers/matrices, functions, and numerical solvers.
* Completed the shaped Stan container core: arbitrary-rank arrays, partial/range/
  all indexes, indexed assignments, arrays of vectors and matrices, typed
  transpose and matrix products, Cholesky/inverse/determinant/SPD solves,
  multivariate-normal kernels, and binding-time shape diagnostics.
* Added container/array user functions, data-qualified arguments, guarded
  recursion and overload resolution, probability-function suffix rules, `_lp`
  target effects, and an explicit tested broadcasting compatibility matrix.
* Added exact unit-vector, orthogonal sum-to-zero, covariance/correlation, and
  Cholesky transforms/Jacobians plus repeated transforms for structured arrays.
* Added the allocation-conscious `ReverseTape`, reusable
  `ReverseModeLogDensity` sampler target, gradient/lifecycle tests, and a
  forward-versus-reverse benchmark.
* Added the Java-native Stan source-compatibility core and explicit
  `compileStan`/`validateStanSyntax` entry points. The compiler now accepts
  modern arrays, row vectors, matrices with scalar multi-indexing, general
  bounds, offset/multiplier transforms, scalar user functions and overloads,
  conditional and container expressions, reductions, symmetric probability
  broadcasting, and initialized transformed containers.
* Added ordinary `.stan` conformance fixtures to the semantic build gate and a
  compatibility contract distinguishing source meaning from Java autodiff,
  floating-point, RNG, sampler, and output behavior.

Version 0.8.2 (August 27, 2026):
* Expanded the Stan-inspired model language with initialized scoped scalar
  locals, assignment operators, comparisons and boolean expressions,
  `if`/`else`, integer-range `for`, guarded `while`, and Stan's vertical-bar
  probability-function syntax.
* Added a broad differentiable scalar-math catalog and more than thirty scalar
  probability families and RNGs, including logit count models, Student-t,
  lognormal, double-exponential, logistic, Gumbel, skew-normal,
  exponentially-modified normal, von Mises, inverse-gamma,
  inverse/scale inverse chi-square, Weibull, Fréchet, Rayleigh, beta-binomial,
  negative-binomial variants, geometric, and Pareto variants.
* Added a checked CSV dataset, an executable Java-builder/script ingestion
  comparison, a control-flow model script, focused execution/gradient tests,
  complete in-memory/cache/CLI compilation instructions, and a migration guide
  for Stan users.
* Expanded the executable catalog to fifty validated model scripts, including
  robust Student-t, logit regression, beta-binomial, negative-binomial log-link,
  scoped control flow, stable log-space mixtures, Weibull, von Mises, and
  exponentially modified normal examples; added a Java comparison of all three
  compilation paths.
* Moved ODE/DAE/algebraic solvers and the remaining full-Stan library surface to
  the unscheduled roadmap.

Version 0.8.1 (August 27, 2026):
* Corrected the tied-sample Mood and Ansari-Bradley asymptotic moments to use
  R's PR#19013 exchangeable-score formulas, including one-sided Ansari tails.
* Corrected negative half-integer Bessel J/Y connection terms instead of
  returning zero, while retaining the existing huge-order allocation guard.
* Added a direct log-domain hypergeometric noncentral-t density path near the
  mode, removing the CDF-subtraction spikes reported in R PR#17519.
* Made Fligner-Killeen ranking stable under affine rescaling by default and
  added a significant-digits overload with an explicit raw-ranking opt-out.
* Added explicit fixed-probability negative-binomial limits for infinite size
  and stable asymptotic density/CDF/quantile behavior near `Double.MAX_VALUE`.
* Added focused R Bugzilla regressions, refreshed the historical Mood vector,
  and retained Java 8-compatible bytecode and the complete 0.8.0 API.

Version 0.8.0 (August 27, 2026):
* Added a named Bayesian model IR and Java builder with observed data, latent
  parameters, dependency-aware factors, real/positive/bounded/ordered/simplex
  transforms, log-Jacobians, analytic factor gradients, forward-mode script
  differentiation, gradient checks, reusable evaluators, and factor caching.
* Added random-walk/component Metropolis, stepping-out slice, composable Gibbs
  and adaptive-rejection conditionals, fixed HMC, and multinomial NUTS with
  dual-averaging step sizes, diagonal/dense metrics, mixed-variable blocks,
  cancellation, state-and-stream checkpoints, and deterministic parallel chains.
* Added rank-normalized split/folded R-hat, bulk/tail ESS, MCSE, divergences,
  tree-depth saturation, E-BFMI, immutable chain/warmup results, and versioned
  JSON/CSV interchange.
* Added chart-neutral trace, rank, ACF, energy and pair datasets; standalone
  SVG, JSON and CSV adapters; Graphviz/JSON model graphs; and self-contained
  HTML diagnostic reports without a desktop UI dependency.
* Added the versioned Stan-inspired 0.8 language with source diagnostics,
  declaration/data validation, transformed blocks, vectorized sampling,
  generated quantities/RNGs, ahead-of-time Java wrappers, SHA-256 compilation
  caching, isolated class loading, and a CLI workflow.
* Added conjugate and sampler reference regressions, constraint/gradient,
  mixed-state, checkpoint, reproducibility, export and code-generation tests,
  compiled examples, a smoke benchmark, and compatibility documentation.
* Added a complete inference tutorial/guide, posterior and diagnostics
  vignettes, a second script walkthrough, and fifteen named executable Bayesian
  examples spanning conjugacy, regression, hierarchy, difficult geometry,
  multimodality, simplex constraints, mixed-state Gibbs, and dense metrics.
* Reduced sampling and reporting allocation pressure with automatic
  chain-local model evaluators, reusable random-walk and covariance buffers,
  allocation-free metric dot/update operations, immutable-data fast paths, and
  scalar chain accessors used by diagnostics, graphing, and export.
* Added forty standalone Stan-inspired `.jdm` models covering the complete
  language 0.8 surface. A packaged-JAR build gate now compiles every script
  with representative data, verifies finite initial density and analytic
  gradients, and executes generated quantities.
* Added feature-focused Java integration examples for continuous and mixed
  copulas, mixtures, affine/nonlinear transforms, truncation/censoring, batch,
  weighted, grouped and online FDR, custom continuous/discrete laws, numerical
  integration, and programmatic/scripted MCMC. Added a browsable website index
  linking the complete verified catalog.

Version 0.7.2 (August 26, 2026):
* Added checked-in 90-digit Decimal reference corpora for boundary-heavy mixed
  Clayton measures and simplified C-/D-vine log densities, with a standalone
  generator derived independently from the Java implementation.
* Added analytic conditional CDFs for the built-in Clayton, Gumbel, and Frank
  pair copulas and analytic inverses for Clayton and Frank, retaining the
  numerical fallback for custom bivariate copulas.
* Added row-level `CopulaLikelihoodDiagnostics`, auditable mixed-measure
  `CopulaLogLikelihoodResult` aggregation, fitted-model diagnostics, and AIC/BIC
  support for vine fits.
* Added deterministic midpoint and explicit-seed mixed fitting/selection
  overloads, defensive marginal-array access, and conventional result getters
  while retaining every 0.7.0 method and field.

Version 0.7.1 (August 26, 2026):
* Added a pure-Java finite-interval CQUAD integration method with nested
  degree-4/8/16/32 Clenshaw-Curtis interpolants, L2 error estimates,
  degree-before-bisection refinement, largest-error-first interval selection,
  hardened callback handling, and CQUAD fallback in `AUTO`.
* Added a general monotone-transform factory and a prominent beginner tutorial
  for mixtures, truncation, censoring, affine changes, and Jacobian-aware
  transformations, backed by a compiled example.
* Added weighted Benjamini–Hochberg adjustment for positive prespecified
  hypothesis weights, including scale-invariant normalization, natural-log
  inputs, and rejection helpers.
* Added the level-dependent Gavrilov–Benjamini–Sarkar adaptive step-down FDR
  procedure with declared-family-size support and an immutable decision result.
* Added weighted Bonferroni and weighted Holm FWER control plus weighted BY for
  arbitrary dependence, with scale-invariant prespecified weights and direct
  natural-log variants.
* Completed declared-family-size and log-domain rejection, count, and threshold
  helpers for the standard batch methods.
* Added two-level Benjamini–Bogomolov grouped testing with explicit Simes group
  selection and selection-adjusted within-family BH results.
* Added separate stateful LORD++ and SAFFRON online-FDR controllers with
  auditable finite gamma spending sequences, immutable per-test decisions, and
  resettable synchronized state.
* Added DBH step-up and step-down procedures for independent heterogeneous
  discrete p-values, backed by explicit finite null support/CDF objects and
  level-dependent critical-value results.

Version 0.7.0 (August 26, 2026):
* Added the stateless `jdistlib.disttest.MultipleTesting` API for Bonferroni,
  Holm, Hochberg, Hommel, Šidák, Holm–Šidák, Benjamini–Hochberg, and
  Benjamini–Yekutieli adjusted p-values, decision thresholds, and Storey
  q-values with smoothing-spline, quantile, or caller-supplied true-null
  proportions. Missing `NaN` positions are preserved and explicit larger test
  families are supported.
* Added level-dependent two-stage BKY FDR testing, direct natural-log p-value
  adjustment, and conservative right-censored-family handling with explicit
  exactness reporting.
* Added a prominent website learning center with beginner tutorials for built-in
  and custom distributions, applied distribution and custom-law vignettes, and
  a copula tutorial plus mixed-marginal vignette clearly labeled as requiring
  JDistlib 0.7.0 or later.
* Expanded `jdistlib.disttest.DistributionTest` with general one-sample
  Cramer-von Mises and Anderson-Darling tests using reproducible parametric
  bootstrap p-values, a two-sample Cramer-von Mises permutation test, and
  Pearson chi-square goodness-of-fit and independence tests.
* Added an immutable `Copula` interface and independence, Gaussian, Student-t,
  Clayton, Gumbel, and Frank implementations. The families provide CDF,
  density/log-density, explicit-engine and seed-convenience sampling, strict
  parameter or correlation validation, and pairwise Kendall's-tau conversion.
* Added `CopulaDistribution` for continuous-marginal joint CDFs, Jacobian-aware
  densities, and sampling through marginal quantiles, plus explicit unit-cube
  boundary diagnostics. Numerical tests cover closed forms, density
  normalization, uniform margins, deterministic streams, and empirical tau.
* Extended composition to discrete and mixed marginals with explicit CDF-jump
  declarations, exact rectangle differences, numerical mixed derivatives,
  evaluation budgets, error comparisons, and typed result statuses.
* Added reusable pair-copula conditional CDF/inversion operations plus
  simplified C-vine and D-vine density, sampling, probability-estimate, and
  sequential fitting APIs.
* Added average-rank and mixed distributional transforms, Kendall and
  likelihood family fitting, correlation regularization, Student-t degrees-of-
  freedom estimation, and automatic AIC/BIC selection across all core families
  and within vine trees.

Version 0.6.1 (August 26, 2026):
* Closed the contributed-distribution screened backlog with complete D/P/Q/R
  APIs for generalized F, beta-negative-binomial, negative hypergeometric,
  discrete Weibull, Skellam, half-Cauchy, half-t, slash, Tukey lambda,
  Feller-Pareto, and phase-type laws. Added discrete-Laplace and logit-normal
  from the 2026-08-21 CRAN Probability Distributions Task View, repaired the
  historical beta-prime transformation and instance truncation bugs, and
  recorded explicit dispositions for ambiguous or infrastructure-heavy
  multivariate families.
* Added asymmetric Laplace, exponentially modified Gaussian, and Huber's
  least-favourable distribution after a full task-view gap sweep. Cleaned all
  legacy Javadoc markup warnings so the API documentation build is warning-free.

Version 0.6.0 (August 25, 2026):
* Closed the historical SourceForge numerical backlog: consistent integer
  validation for hypergeometric parameters, guarded split-deviance Poisson
  densities, compensated noncentral-beta density arithmetic, smaller-tail
  noncentral-chi-square inversion, and conditional quadrature removing the
  PR#16845 noncentral-t cutoff discontinuity. Replaced obsolete open-ticket R
  Bugzilla monitoring with tagged-source regressions, generated standalone R
  source/test patches for PR#16332 and PR#16845, and recorded the one
  out-of-scope `stats::fisher.test` ticket.
* Finalized the multivariate probability result contract with a typed status,
  convergence and usable-estimate queries, a public tolerance calculation, and
  explicit documentation that the replication error is a heuristic rather than
  a rigorous confidence bound.
* Added API-selection and multivariate probability guides, clarified mutable
  instance/random-engine thread safety, made CI run all `check` gates, and added
  an actual packaged-JAR compile/run smoke test on Java 8.
* Added a local, signed Maven-layout Central Portal bundle task with complete
  checksums and a publication runbook; external upload remains an explicit
  maintainer release action.
* Added a custom continuous and discrete distribution guide with mathematical
  quick starts, advanced construction guidance, diagnostics, troubleshooting,
  and a fully typeset distribution formula catalog.
* Completed numerical-distribution hardening with immutable typed integration
  results, callback timing profiles and wall-clock budgets, opt-in daemon-worker
  callback isolation, versioned dependency-free JSON diagnostics, and an
  independent high-precision corpus for difficult integral families.
* Added the complete custom-distribution workflow: fluent builders and named
  diagnostic presets, including native log-kernel analysis; mixture,
  truncation, monotone-transform, affine, and
  censoring composition; Walker-alias and adaptive log-concave rejection
  sampling with strategy explanations; certified tail-bound helpers; numerical
  expectations, moments, entropy, modes, and probability intervals; and
  cache-aware allocation-free batch APIs. Atom-aware composition preserves CDF
  jumps and masses through censoring, mixtures, and decreasing transformations.
* Added call-local, randomized quasi-Monte Carlo rectangle probabilities for
  multivariate normal, Student t, Cauchy, and log-normal laws. Results report
  estimated absolute error, evaluation count, and convergence status; callers
  may supply both accuracy limits and an explicit random stream.
* Added explicitly named equicoordinate quantiles for those vector laws and
  closed-form Mahalanobis/radial quantiles for the normal, t, Cauchy, and
  multivariate power-exponential families, avoiding an ambiguous vector inverse
  CDF contract.
* Corrected the historical Wishart Bartlett sampler to use the documented
  `W(scale, df)` parameterization and added scalar/vector generation overloads,
  scale-matrix convenience methods, and log/non-log matrix densities.
* Added independent quadrature, scalar-reduction, exact-orthant, Wishart-moment,
  invalid-input, reproducibility, and evaluation-budget regression tests.
* Hardened release packaging so the v0.6.0 tag produces non-SNAPSHOT artifacts,
  rejects a mismatched release version, verifies the JAR manifest, and compiles
  the documented custom-distribution example against the packaged JAR.

Version 0.5.0 (August 25, 2026):
* Restructured the historical SourceForge repository as a conventional,
  GitHub-ready Gradle project while retaining the complete commit history.
* Replaced retired build plugins and vendored jars with Gradle 9, Maven Central,
  Maven publication metadata, source/javadoc jars, and Java 8-compatible output.
* Completed the R 4.6.1 `stats::integrate` port with DQAGS finite integration,
  DQAGI semi-infinite and doubly-infinite integration, epsilon extrapolation,
  R-compatible error estimates/statuses, and call-local workspaces; added an
  R-generated reference corpus and concurrency tests.
* Added user-defined numerical continuous and finite-discrete distributions:
  nonnegative formulas are normalized over declared supports and exposed through
  density/mass, CDF, quantile, random, and normalization-diagnostic APIs. The
  hardened integration path adds immutable options, breakpoints, budgets,
  cancellation, callback context, stability repeats, finite tanh-sinh fallback,
  advisory kernel/distribution reports, and log-kernel/log-weight construction.
  Follow-up hardening adds absolute-moment existence checks, reusable adaptive
  monotone CDF tables, mode-searched regional log scaling, interval-union/hole/
  singularity/atom supports, and certified truncation for one- or two-sided
  infinite integer domains. The next hardening pass adds seeded adaptive probes,
  configurable construction policies, arbitrary split-tail absolute moments,
  certified rejection-envelope sampling, and double-exponential quadrature on
  semi-infinite and whole-line domains.
* Began the audited sync from R 3.3.2 to R 4.6.1: modernized normal quantiles,
  discrete quantile search, negative-binomial mean parameterization, Stirling
  error, deviance terms, and selected density edge cases.
* Completed the R 4.6.1 normal-family audit, including subnormal probability
  tails, infinite-scale boundaries, random-parameter validation, and robust
  IEEE-754 binary scaling helpers.
* Completed the remaining R 4.6.1 `src/nmath` audit: beta/TOMS 708, gamma and
  noncentral chi-square, hypergeometric and rank statistics, Bessel functions,
  count and continuous distributions, RNG boundaries, and arithmetic helpers;
  added a reproducible R-generated cross-language regression corpus.
* Ported selected post-4.6.1 R-devel fixes for binomial BTPE sampling,
  large-population hypergeometric sampling, compensated multinomial sampling,
  and Wilcoxon rank/zap controls while retaining an explicit legacy BTPE mode.
* Replaced the low-accuracy AS 226 noncentral-beta CDF with an AS 310-guided,
  Benton--Krishnamoorthy mode-centred mixture that evaluates both tails on the
  log scale; quantiles now invert underflowing log probabilities directly.
* Added a GitHub Pages project site with JavaDoc generated automatically from
  the current `master` branch.
* Completed the previously stubbed Tweedie API: density, CDF, quantile, random
  generation, log-likelihood, dispersion derivative, exact special cases, the
  compound Poisson-gamma model, and the power-above-two series/integration path.
* Completed `PolyGamma.dpsifn` sequences and negative-argument reflection beyond
  the former order-three restriction.
* Added complete triangular, half-normal, shifted Birnbaum-Saunders, and
  unrestricted-shape Gompertz APIs from GPL-compatible CRAN references, with
  R-style logged tails, explicit random engines, provenance, and regression tests.
* Audited the distinct distributions in `distributions3` and added categorical,
  multinomial, exact Poisson-binomial, empirical, sinh-arcsinh, and six hurdle,
  zero-inflated, or zero-truncated Poisson/negative-binomial families.
* Added application-focused VGAM and CRAN task-view families: Stacy generalized
  gamma, GB2, Makeham, Lindley, folded/positive normal, Rice, Maxwell, and
  actuarial Poisson-inverse-Gaussian, with complete scalar distribution APIs.
* Added the ticket-42 `MaxwellBoltzmann` scale-parameter API while retaining
  VGAM-compatible `Maxwell` rate parameters; the two are related by
  `rate = 1 / sigma^2`.
* Added the reusable multivariate task-view core: Dirichlet and
  Dirichlet-multinomial, multivariate hypergeometric, bivariate Poisson and VGAM
  logistic, and multivariate normal, Student t/Cauchy, lognormal, Laplace, and
  power-exponential density/mass and random-vector APIs.
* Preserved JDistlib-only distributions and the explicit per-stream random-state
  design used by cached random generators.
* Added various central tendencies routine per request of Bug #32
* jdistlib.matrix.QMatrixUtils is now being deprecated
* Added Geometric Standard Deviation, per Bug #34
* Bug fix #37 on GEV.cumulative. Thanks Luís!
* Bug fix #36 Underflow of beta (PR #17178)

Version 0.4.5 (October 10, 2016):
* Bug fix for #31 (PR#16972) on Poisson quantile (and related limit bugs)
* Added Median Absolute Deviation (MAD) to jdistlib.math.VectorMath
* Responded to bug #29 by adding an option to sort for Shapiro-Wilk test
* Sync with R-3.3.1, fixes rgamma(1,Inf) or rgamma(1, 0,0) no longer give NaN but the correct limit.
* Added Levy distribution
* Sync with Development build of October 9, 2016
* Fixes: rbeta(4, NA) and similarly rgamma() and rnbinom() now return NaN's with a warning, as other r<dist>(), and as documented. (PR#17155) 

Version 0.4.4 (April 20, 2016):
* Fix for bug #28 about Shapiro-Wilks

Version 0.4.3 (April 19, 2016):
* Fix for bug #27: Regression on bug PR#16489

Version 0.4.2 (April 19, 2016):
* Fix for bug #26: Allow warnings to be cast as an exception.
* Sync with R-devel (Apr 18, 2016-r70508)
* Fix for bug PR#16521: rchisq(*, df=0, ncp=0) now returns 0 instead of NaN, and dchisq(*, df=0, ncp=*) also no longer returns NaN in limit cases (where the limit is unique)
* Fix for pchisq(*, df=0, ncp > 0, log.p=TRUE) no longer underflows (for ncp > ~60).
* Fix for rhyper(nn, m, n, k) no longer returns NA when one of the three parameters exceeds the maximal integer.
* Fix for bug PR#16727: [dpqr]nbinom(..., size = Inf) should behave like [dpqr]pois(...)
* Added WELL 44497b random number generator

Version 0.4.1 (September 15, 2015):
* Sync with R version 3.2.2 (dated August 14, 2015)
* Fix for bug PR#16475: qt(*, df=Inf, ncp=.) now uses the natural qnorm() limit instead of returning NaN.
* Fix for bug PR#16489: rhyper(nn, <large>) now works correctly.

Version 0.4.0 (May 6, 2015):
* Fixed typos on Beta.density that caused failures on regression tests.

Version 0.3.9 (May 5, 2015):
* Sync with the development branch of R 3.2.x (dated April 24, 2015)
* Fix for the second half of PR#15554 regarding Bessel.J and Bessel.Y with huge alpha (>= 2^61)
* Fix for bug #17 regarding the accuracy of Beta.quantile (PR#15755)
* Added MathFunctions.logspace_sum (per 3.2.x API feature)
* Fix for bug #22 for inadvertent use of Java 1.8 API (Double.isFinite). It has been replaced with !Double.isInfinite (Java 1.5-compatible API)

Version 0.3.8 (Dec 15, 2014):
* Fix for bug #20 regarding Kolmogorov-Smirnov (KS) test. Thanks, Gilad Wallach and Eran Avidan!

Version 0.3.7 (Dec 10, 2014):
* Fix for bug #19 regarding Kolmogorov-Smirnov (KS) test
* Added option to allow inexact KS p-value computation method, if needed. Default option is still exact method. See bug #19 entry for details.
* Fixed integer overflow bug when computing KS exact method---only happen with big data sets.

Version 0.3.6 (Aug 18, 2014):
* Fix for bug #18
* Added generalized one-distribution Kolmogorov-Smirnov test
* kolmogorov_smirnov_statistic and kolmogorov_smirnov_pvalue are deprecated in favor of kolmogorov_smirnov_test
* Synced with R version 3.1.1.
* Synced MersenneTwister with Sean Luke's version 20
* Incorporated Bintray / Gradle build system, courtesy Schalk W. Cronjé.

Version 0.3.5 (Apr 14, 2014):
* Synced with R-devel_2014-04-10 (effectively R 3.2.0 alpha or 3.1.1), fixing the following bugs:
   * pchisq(1e-5, 100, 1) == 0 due to underflow in dpois_raw (PR#15635)
   * Calculation error in using function pbinom (PR#15734)

Version 0.3.4 (Apr 7, 2014):
* Synced with R-rc_2014-04-04_r65373, fixing the following bugs:
   * pbeta(x, a,b, log.p=TRUE) sometimes lost all precision for very small and very differently sized a,b. (PR#15641)
   * More precise Normal density when x > 5 (PR#15620)
   * Adding sinpi, cospi, and tanpi for more precise Bessel function and Cauchy distribution computations (PR#15529)
* Fixed bug #16, infinite loop in sort functions when the numbers are all negatives (Thanks Gilad Wallach and Idan Peretz!).
* Imported a lot of comments to sync with the latest R function
* Fixes comment on Bessel functions---Bessel functions can handle negatives already!

Version 0.3.3 (Jan 28, 2014):
* Bessel functions (J, Y, I, K) with fractional orders
* Added Beta Prime and Kumaraswamy distributions.
* Added PolyGamma.lmvpsigammafn (log of multivariate psi-gamma function).

Version 0.3.2 (Jan 24, 2014):
* Fixed bug in MathFunctions.lmvgammafn (see ticket #14)
* Fixed bug in Spearman.quantile (off by 1 issue)
* Fixed bug in binomial test in DistributionTests
* Fixed bug in Logarithmic distributions plus some speed up in Logarithmic.quantile
* Added Bounded Arcsine, Laplace, and Zipf distributions
* Added density functions for Spearman and Tukey distributions (using differentials; not precise!)
* Added MathFunctions.sinc, gharmonic, lgharmonic, and sort for various data types
* Added some incomplete solutions to bug PR#15635
* Added batch calls for PolyGamma functions
* Added Poisson test
* Make MathFunctions.logspace_add and logspace_sub public
* Removed redundant constants from Constants (M_PI_half, M_LN_2, kLog1OverSqrt2Pi)

Version 0.3.1 (Jan 13, 2014):
* Added Spearman quantile (using bisection) and random variates (by inversion)
* Added Order quantile variates (only very minimally tested; caveat emptor!)
* Added Chi, Inverse Gamma, and Nakagami distributions (based on simple transform from the Gamma distribution)
* Added many two-distribution tests: Ansari-Bradley, Mood, Bartlett, Fligner, T-test (one-sample, paired, two-sample), Variance test, Wilcoxon test, Mann-Whitney-U test, Kruskal-Wallis test, Binomial test
* Added lower_tail flag to Ansari distribution
* Utilities.rank is now index 1 based (not index 0) since many routines seem to depend on that fact
* Various bug fixes

Version 0.3.0 (Jan 10, 2014):
* Remove the Q prefix of QRandomEngine, QMersenneTwister, QRandomCMWC, and QRandomSampler
* Added Beta binomial distribution (with parameterization of mu, sigma, and size)
* Added hazard, cumulative hazard, survival, and inverse survival functions for all distributions (instance only)
* Fixed bugs on Kolmogorov-Smirnov two-sample test when the second array (Y) is longer than the first array (X)
* Fixed bugs for Binomial.cumulative when x < 0 or x >= n (improperly returns 0 or 1).
* Updated to R-patched_2014-01-08_r64705 that contains the following bug fixes:
  * dbeta(x, a, b) with a or b within a factor of 2 of the largest representable number could infinite-loop. (Reported by Ioannis Kosmidis.)
  * qcauchy(p, *) is now fully accurate even when p is very close to 1. (PR#15521)
  * In some extreme cases (more than 10^15) integer inputs to dpqrxxx functions might have been rounded up by one (with a warning about being non-integer).  (PR#15624)

Version 0.2.1 (Jan 9, 2014):
* Fixed crash on Poisson.random (and consequently NegBinomial.random) when mu >= 10
* Fixed bugs on NonCentralF.random
* Added codes from p-r-tests.R

Version 0.2.0 (Jan 8, 2014):
* Deprecated GenericDistribution.random(QRandomEngine)
* Added an API to create multiple random variables
* Added an API to query multiple values of density, cumulative, and quantile (instance only)
* Added more codes from d-p-q-r-tests.R for unit testing.
* Fixed bugs on SignRank.quantile. Variable n was set incorrectly.
* Fixed bugs on T.quantile(x, df, true, false) that causes NaN when df is close to 1 and x is very small
* Fixed bugs on many distributions when x is close to the limit of double precision floating point
* Remove false non-convergence warning messages in NonCentralT.cumulative
* Fixed bugs on bd0 when np < 1e-306. This will fix the behavior of many distributions when x is very small
* Fixed bugs on Poisson.random that caused the routine to hang up on certain random states (Ticket #10)
* Fixed bugs on LogNormal when x <= 0
* The precision of Gamma.cumulative is on par with R

Version 0.1.3 (Jan 2, 2014):
* Fixed bugs on SignRank.cumulative. The variable n was set incorrectly.
* Fixed bugs on Gamma.cumulative when the scale is +Inf.
* Added some code from d-p-q-r-tests.R for unit testing.
* Noted some precision loss on Gamma.cumulative
* Noted some precision loss on NonCentralChiSquare
* Fixed bugs on most distributions for boundary cases dealing with infinity
* Converted project to Maven

Version 0.1.2 (Dec 26, 2013):
* Added Rayleigh and Inverse Normal distributions
* Bugfixes on Kendall distribution
* Added Brent's optimization and root finding methods (for brute force quantile search)

Version 0.1.1 (Dec 20, 2013):
* Order (no quantile) and Extreme (both maxima and minima) distributions for order statistics (from EVD package)
* Added Box-Muller method to generate random normals
* Added RandomSampler ripped from Colt. Handy for creating a permutation of list of objects.
* The sources should be 100% compatible with JDK 1.5.

Version 0.1.0 (Dec 19, 2013):
* Distributions are now instantiable

Version 0.0.9 (Dec 17, 2013):
* Proper fix for negative binomial distribution with size=0 (PR#15268)
* Synced with R version 3.0.2

Version 0.0.8 (Dec 17, 2013):
* Fix for bug #6. Thanks, Roland Ewald!

Version 0.0.7 (Mar 29, 2013):
* Proper fix for pt / pf distribution (PR#15162)

Version 0.0.6 (Jan 11, 2013):
Further R synchronization fixes the following bugs / adds the following features:
* qgeom() could return -1 for extremely small q. (PR#14967.)
* lgamma(x) for very small x (in the denormalized range) is no longer Inf with a warning.
* plogis(x, lower = FALSE, log.p = TRUE) no longer underflows early for large x (e.g. 800).
* Imported the simplified logic for T.quantile from R
* Added multivariate gamma function (MathFunctions.lmvgammafn)
* Added Wishart distribution sampling (random only)

Version 0.0.5 (Jan 09, 2013):
* Synchronized with R's patched of the same date. Fixes the following bugs:
-- qt(1e-12, 1.2) no longer gives NaN.
-- dt(1e160, 1.2, log=TRUE) no longer gives -Inf.
-- beta(a, b) could overflow to infinity in its calculations when one of a and b was less than one.  (PR#15075)
-- lbeta(a, b) no longer gives NaN if a or b is very small (in the denormalized range).

Version 0.0.4 (Jan 09, 2013):
* Fix for pt / pf distribution. (PR#15162)
* Added Fretchet, GEV, Generalized Pareto, Gumbel, and Reverse Weibull distributions

Version 0.0.3:
* Added Logarithmic distribution
* Fixed visibility of Binomial.quantile to public (as opposed to package)
* Increased constant precision on Shapiro-Wilk p-value computation
* Modified density, cumulative, and quantile method signature of Ansari
  distribution to allow single values (as opposed to arrays of values)
* Added random number generation for Kendall and Tukey distributions
* Added exact method for computing Spearman distribution
* Finalize the method signature of PolyGamma.dpsifn
* Added MathFunctions.log1px
