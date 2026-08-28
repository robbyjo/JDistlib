# JDistlib to-do

## Planned for 0.9.0: financial distributions and implied inference

Version 0.9.0 is planned as a probability-first finance release. It will add
the reusable distribution, numerical, fitting, and inference primitives in the
[finance and options roadmap](docs/FINANCE_ROADMAP.md), including:

* atom-aware tail risk and partial moments: value at risk, expected shortfall,
  expectiles, lower/upper partial moments, downside deviation, and stop-loss or
  option-payoff expectations, with explicit loss/return and tail conventions;
* log characteristic and cumulant-generating function contracts, numerical
  inversion, cumulants, and exponentially tilted distributions with
  domain/existence diagnostics;
* a shared generalized-hyperbolic family design covering generalized
  hyperbolic, normal-inverse-Gaussian, and variance-gamma laws, followed by a
  documented univariate alpha-stable parameterization and its normal, Cauchy,
  and Levy reductions;
* convolution, weighted independent sums, and compound-sum distributions with
  exact discrete, Panjer, FFT, characteristic-function, and reproducible Monte
  Carlo strategies where applicable, including explicit error and truncation
  reports;
* rotated/survival copulas, additional asymmetric pair families, analytical
  and numerical tail-dependence measures, joint stress-region probabilities,
  and tail-aware fitting and selection diagnostics;
* general distribution fitting plus GEV/GPD and tail-index estimation, return
  levels, threshold diagnostics, uncertainty estimates, censored/interval
  likelihoods, and reusable calibration result contracts;
* checked implied-volatility inversion and arbitrage-constrained option-curve
  processing that can infer a normalized risk-neutral numerical distribution,
  its CDF/density/quantiles/moments/tails, and reproducible samples;
* distribution calibration to option observations, including weighted-tail and
  option-price losses, bid/ask or noisy-observation likelihoods, Bayesian
  inference, and posterior-predictive terminal-price, payoff, tail-risk, and
  strike-interval distributions; and
* transformation, product/ratio, conditional, extrema, and scenario helpers
  needed to express those probability outputs without embedding market-data,
  instrument-lifecycle, trading, backtesting, or portfolio-accounting systems
  in JDistlib.

The implementation order and acceptance contracts are recorded in
`docs/FINANCE_ROADMAP.md`. Full instrument pricing, American exercise,
volatility-surface lifecycle management, Greeks, market-data ingestion, and
execution remain integration-library concerns rather than JDistlib 0.9.0 core
features.

## Unscheduled

* Add the specialized Wiener first-passage distribution to the script
  probability catalog after selecting and validating a stable density and RNG
  implementation across small-time and extreme-parameter regimes.
* Add exact lower-orthant CDFs and rectangle probabilities for the multinomial,
  Dirichlet-multinomial, and multivariate hypergeometric distributions, using a
  shared sequential-conditional dynamic-programming framework.
* Add lower-orthant CDFs and rectangle probabilities for the Dirichlet
  distribution, with numerically stable simplex-aware integration and reported
  error estimates.
* Add lower-orthant CDFs and rectangle probabilities for the multivariate
  Laplace distribution by exploiting its normal scale-mixture representation
  and the existing multivariate-normal probability machinery.
* Investigate and implement reliable lower-orthant CDFs and rectangle
  probabilities for the multivariate power-exponential distribution, including
  robust treatment of tail events and non-Gaussian shape parameters.
* Define useful, unambiguous probability APIs for the Wishart distribution,
  favoring named scalar or matrix-order events over an entrywise matrix CDF;
  implement the events that can be evaluated reliably.

## Recently completed

* Added full multi-path Pathfinder with PSIS, many-short-chain superchains and
  nested R-hat, coordinated ChEES/SNAPER adaptive static HMC, and auditable
  automatic adjusted-MCLMC tuning.
* Added optional CUDA and OpenCL compute providers with automatic detection and
  thresholded deterministic CPU fallback, strict programmatic/CLI compute and
  NUTS-offload policies, selection logging and manifest provenance, plus
  portable checkpoints, precision-driven continuation, scalable draw stores,
  profiling, geometry advice, and health policies. The measured boundary and
  release result are consolidated in `docs/INFERENCE_ACCELERATION_RESULT.md`.
* Added differentiable complex scalars/vectors/matrices, nested tuple values and
  tuple-valued functions, one-based tuple member access/assignment, Java tuple
  adapters, immutable CSR matrices and Stan CSR kernels, Java-bound external
  functions with explicit Jacobians, and `integrate_1d` higher-order callbacks.
* Filled the 0.8.3 standard-library compatibility layer with complex elementary
  overloads, container reductions/distances, SPD inverse/solve/log-determinant,
  lower-triangular products, symmetrization, trace quadratic forms, and row/
  column self-dot kernels. The compatibility contract continues to enumerate
  specialized Stan functions and probability laws outside the 0.8.3 surface.
* Lowered compiled script factors onto thread-local reusable reverse tapes and
  added n-ary atomic nodes plus specialized normal, Student-t, dot-product,
  matrix-normal, external-function, distance, and solver-sensitivity kernels.
* Added algebraic implicit sensitivities, ODE/DAE parameter sensitivities,
  differentiable script bindings for algebraic, RK45, BDF, DAE, and 1-D
  integration callbacks, an adaptive stiff BDF path, and a projected
  velocity-Verlet solver for holonomic index-3 mechanical DAEs. Independent
  stiff, sensitivity, and index-3 corpora guard the three numerical paths.
* Expanded the ordinary Stan fixture catalog from thirty to forty-one programs,
  with executable complex, tuple, sparse, external, standard-library,
  quadrature, algebraic, stiff-ODE, and DAE examples.

* Added Stan array, vector, row-vector, and rectangular matrix literals; forward
  user-function declarations; append/head/tail/segment/block/row/column helpers;
  softmax/log-softmax, cumulative/sort/reverse transforms; diagonal, quadratic,
  cross-product, and row/column dot-product matrix operations.
* Added Java-native damped-Newton algebraic solving, adaptive Dormand-Prince
  ODE integration, and implicit-Euler index-1 DAE integration with tolerances,
  work guards, result diagnostics, examples, and analytic reference tests.
* Expanded the checked ordinary `.stan` compatibility catalog from twelve to
  thirty programs (eighty scripts total) and added v0.8.3 container, function,
  solver, compatibility, and migration tutorials.

* Added arbitrary-rank arrays and typed shaped values, slices/range/all indexes,
  indexed assignment, arrays of vectors/matrices, orientation-aware linear
  algebra, decompositions, SPD solves, multivariate-normal kernels, and binding-
  time shape diagnostics.
* Added container-valued user functions, lexical function scopes, guarded
  recursion, overload resolution and integer promotion, `data` arguments,
  probability suffixes, `_lp` effects, and an explicit tested broadcasting
  compatibility matrix.
* Added exact unit-vector, orthogonal sum-to-zero, covariance/correlation matrix,
  and covariance/correlation Cholesky transforms and Jacobians, including arrays
  of structured constraints.
* Added a primitive-array reverse-mode AD tape with reset/mark/rewind lifecycle,
  an HMC/NUTS-facing `ReverseModeLogDensity`, analytic verification, and a
  forward-versus-reverse throughput benchmark.

* Added a Java-native Stan source-compatibility core with explicit
  `compileStan`/`validateStanSyntax` APIs, shaped arrays/matrices, typed
  functions and algebra, general constraints/Jacobians, conditional/container
  expressions, reductions and broadcasting, ordinary `.stan` fixtures, and a
  documented Java-versus-Stan execution contract.

* Expanded the Stan-inspired language with scoped scalar local variables,
  assignment operators, comparisons and boolean expressions, `if`/`else`,
  integer-range `for`, guarded `while`, and Stan's `|` probability-function
  separator.
* Added a broad differentiable Stan scalar-math surface and expanded scalar
  probability/RNG support to more than thirty families, including logit count models,
  Student-t, lognormal, double-exponential, logistic, Gumbel, skew-normal,
  inverse-gamma/chi-square, Weibull, Fréchet, Rayleigh, beta-binomial, negative
  binomial, geometric, Pareto, and related parameterizations.
* Added executable CSV-to-MCMC examples for both the Java builder and compiled
  script frontends, complete script compilation instructions, and a migration
  tutorial for Stan users with an explicit compatibility boundary.
* Corrected the five 0.8.1 R Bugzilla findings: tied Mood/Ansari moments,
  negative half-integer Bessel connections, noncentral-t density cancellation,
  affine-stable Fligner ranking, and infinite/extreme-size negative binomial
  behavior, with focused references and Java 8 regression coverage.
* Added the 0.8.0 Bayesian model IR and Java builder with observed data,
  constrained latent parameters, dependency-aware prior/likelihood factors,
  log-Jacobians, analytic gradients, forward-mode script differentiation,
  validation, allocation-free evaluators, vectorized likelihoods, and cached
  factor evaluation.
* Added reproducible parallel-chain Metropolis, component Metropolis, slice,
  Gibbs, adaptive-rejection conditional, fixed HMC, and multinomial NUTS
  inference with dual-averaging warmup, diagonal/dense metric adaptation,
  mixed-variable blocks, cancellation, immutable results, and in-memory
  state-and-stream checkpoints/restart.
* Added rank-normalized split/folded R-hat, bulk/tail ESS, MCSE, acceptance,
  divergence, tree-depth, and per-chain E-BFMI diagnostics plus versioned chain
  and diagnostic JSON/CSV interchange.
* Added chart-neutral trace, rank, autocorrelation, energy, pair, and dependency
  graph datasets with versioned JSON, tidy CSV, standalone SVG, Graphviz DOT,
  and self-contained HTML report adapters that retain a headless core.
* Added the versioned Stan-inspired 0.8 language with source-located parsing,
  type/dimension/data/constraint checks, transformed blocks, indexed/vectorized
  expressions, generated quantities/RNGs, forward derivatives, ahead-of-time
  Java wrappers, SHA-256 compilation caching, isolated class loading, and CLI.
* Added inference compatibility and reproducibility contracts, sampler and
  reparameterization guidance, compiled Java/script examples, a smoke benchmark,
  and regressions for conjugate, constrained, hierarchical/funnel, multimodal,
  heavy-tailed, mixed discrete/continuous, gradient, checkpoint, diagnostic,
  graph/export, and generated-code behavior.
* Added the complete 0.8.0 inference learning path: beginner tutorial, reference
  guide, posterior-predictive and difficult-geometry diagnostics vignettes,
  expanded graph/diagnostic interpretation, a second model-script tutorial,
  and fifteen named executable Bayesian examples.
* Accelerated sampling and post-processing with automatic chain-local model
  evaluators, reusable Metropolis proposals and covariance scratch space,
  allocation-free metric operations, zero-copy immutable observation reads for
  built-in factors, and scalar chain access during diagnostics, plotting, and
  export.
* Added forty validated Stan-inspired script examples plus a semantic Gradle
  gate that supplies representative data, compiles and gradient-checks each
  model, verifies finite initial density, and exercises generated quantities.
* Added a browsable executable-example center and compilable integration
  workflows for copulas, mixtures, transformations, FDR, custom distributions,
  numerical integration, and Java/script MCMC; linked it from the main website.
* Expanded mixed/vine copula validation with independent high-precision
  boundary-heavy likelihood references and row-level fitted-model diagnostics.
* Incorporated 0.7.0 API feedback through additive result getters, auditable
  likelihood summaries, vine information criteria, and explicit midpoint/seed
  fitting conveniences while retaining Java 8 and the existing public API.
* Added finite-interval CQUAD integration and incorporated it into the hardened
  automatic fallback sequence.
* Added the general monotone-transformation factory and a prominent beginner
  composition tutorial with compiled examples.
* Added prespecified-weight Benjamini–Hochberg adjustment and the adaptive
  Gavrilov–Benjamini–Sarkar step-down procedure for 0.7.1.
* Added weighted Bonferroni, weighted Holm, and weighted BY; completed batch
  family-size/log helper symmetry; added explicit grouped Benjamini–Bogomolov,
  stateful LORD++/SAFFRON, and discrete DBH step-up/down APIs.
* Consolidated the QGeneric FDR plugins into the stateless 0.7.0
  `MultipleTesting` API, adding standard p-value adjustments, Storey q-values,
  missing-value preservation, family-size overrides, and decision helpers.
* Expanded multiple testing with adaptive two-stage BKY, log-p input, and
  conservative right-censored-family results.
* Added a prominent beginner learning center with distribution and custom-law
  tutorials, applied vignettes, and clearly versioned 0.7.0+ copula learning
  paths; tutorial companion code is compiled by `check`.
* Expanded `jdistlib.disttest.DistributionTest` with reproducible general
  Anderson-Darling, one- and two-sample Cramer-von Mises, and Pearson
  chi-square goodness-of-fit and independence tests.
* Added the complete 0.7.0 copula framework with independence,
  Gaussian, Student-t, Clayton, Gumbel, and Frank families, joint-distribution
  composition, log densities, explicit-stream and seeded sampling, parameter
  validation, boundary diagnostics, Kendall's-tau conversions, atom-aware
  discrete/mixed likelihoods, simplified C-vines and D-vines, dependence
  fitting, and AIC/BIC family and pair-family selection.
* Audited and dispositioned all historical SourceForge tickets, with focused
  regression gates replacing obsolete R Bugzilla monitoring links.
* Finalized multivariate probability statuses, error semantics, reproducibility
  guidance, and difficult-case documentation.
* Added `check`-level CI, a packaged-JAR Java 8 smoke run, API-selection and
  thread-safety guidance, and a signed Central Portal bundle workflow.
* Fluent continuous, discrete, and mixed-support builders plus fast, standard,
  and thorough diagnostic presets.
* Distribution composition for mixtures, truncation, monotone/affine
  transformations, and censoring.
* Walker-alias finite-discrete sampling, adaptive log-concave rejection,
  automatic strategy selection, and strategy explanations.
* Reusable geometric, power-law, symmetric, and finite-prefix discrete tail
  certificates.
* Numerical expectations, raw/central moments, entropy, modes, and probability
  intervals.
* Cache-aware and allocation-free array density, CDF, quantile, and random APIs.
* Immutable typed integration results alongside the mutable legacy compatibility
  type.
* Callback cost profiles, total/per-callback wall-clock limits, and opt-in
  private daemon-worker isolation.
* Independent high-precision regression data for oscillation, endpoint/interior
  singularities, extreme scaling, narrow modes, and heavy tails.
* Versioned, machine-readable JSON serialization for diagnostic reports.
* Seeded, budgeted adaptive randomized diagnostic probes.
* User-selected absolute moments with separate left/right convergence reports.
* Strict, warning, and permissive analyzed-construction policies.
* Optional certified rejection-envelope sampling.
* Double-exponential quadrature for finite, semi-infinite, and whole-line
  intervals.
