# JDistlib to-do

## Unscheduled

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
