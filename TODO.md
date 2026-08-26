# JDistlib to-do

## Future candidates

* Expand independent reference validation for mixed and vine copulas,
  especially boundary-heavy likelihoods and fitted-model diagnostics.
* Incorporate feedback from the 0.7.0 APIs while preserving Java 8 bytecode,
  deterministic seeded behavior, and backward compatibility.

## 0.8.0: Bayesian modeling and inference

* Define a stable model intermediate representation and programmatic Java
  builder for unnormalized joint log densities, prior and likelihood factors,
  observed data, latent parameters, and factor-dependency metadata. Keep this
  inference contract separate from normalized scalar `GenericDistribution`
  objects while reusing JDistlib distributions as model factors.
* Support unconstrained sampling through positive, bounded, ordered, simplex,
  and other common parameter transforms, including correct log-Jacobian terms
  and reusable transformed-state buffers.
* Provide immutable chain and warmup results with explicit `RandomEngine`
  streams, reproducible independent chains, retained log densities, sampler
  statistics, warnings, cancellation, checkpointing, and restart support.
* Implement a composable sampler foundation with random-walk and component-wise
  Metropolis, Gibbs updates, slice sampling, and adaptive-rejection updates for
  eligible log-concave full conditionals.
* Implement production HMC and NUTS, including multinomial trajectory selection,
  dual-averaging step-size warmup, diagonal and dense mass-matrix adaptation,
  maximum-tree-depth controls, and blocked updates for mixed discrete/continuous
  models.
* Define differentiable log-density APIs with caller-supplied and analytic
  gradients, gradient validation, and a practical automatic-differentiation
  path. Permit clearly diagnosed finite differences as a limited fallback, not
  as the default for NUTS.
* Supply rank-normalized split R-hat, bulk and tail effective sample sizes,
  Monte Carlo standard errors, acceptance and energy diagnostics, divergences,
  maximum-tree-depth saturation, and cross-chain convergence summaries.
* Expose chart-neutral trace, rank, autocorrelation, energy, pair-plot, and model
  graph data. Add optional lightweight plotting/export adapters without making
  the headless inference core depend on a UI toolkit.
* Optimize only behind the common model and sampler contracts: cache unaffected
  factors from dependency metadata, vectorize likelihood factors, reuse
  allocation-free evaluation buffers, run chains in parallel while preserving
  deterministic per-chain streams, and benchmark scalar, vectorized, gradient,
  warmup, and sampling throughput.
* Add independent reference models and statistical regression gates covering
  conjugate posteriors, constrained parameters, hierarchical models,
  multimodality, funnels, heavy tails, discrete latent variables, gradients,
  diagnostics, reproducibility, and failure reporting.
* Document the Java modeling API, sampler selection, reparameterization,
  convergence assessment, diagnostics, and common pathologies through compiled
  end-to-end examples.

## 0.8.x: Modeling language and stabilized ecosystem

* Design a Stan-inspired JDistlib modeling language that lowers into the same
  model intermediate representation as the Java builder. Do not claim Stan
  compatibility unless its syntax and semantics are independently verified.
* Implement a lexer, source-located parser and AST, type and dimension checking,
  data and parameter validation, constraint lowering, vectorization and
  broadcasting rules, and actionable compile-time diagnostics.
* Add data, transformed-data, parameter, transformed-parameter, model, and
  generated-quantity blocks incrementally, with explicit versioning for the
  supported language subset.
* Provide ahead-of-time Java source/class generation, compilation caching, safe
  class loading, model metadata, and a documented Gradle/CLI workflow; retain an
  interpreter or reference evaluator for correctness comparisons.
* Expand model-graph inspection, diagnostic visualization integrations, result
  interchange, and notebook/headless workflows on top of the chart-neutral
  data contracts introduced in 0.8.0.
* Harden and then freeze the public modeling, inference, diagnostics, generated
  model, serialization, and reproducibility contracts, with migration tests and
  compatibility guidance for every 0.8.x change.
* Broaden language, sampler, automatic-differentiation, and code-generation
  optimization only when reference evaluation, gradients, posterior summaries,
  diagnostics, and seeded chains remain equivalent within documented bounds.

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
