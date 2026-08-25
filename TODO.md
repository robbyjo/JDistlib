# JDistlib to-do

## Planned for 0.7.0

* Add a continuous-marginal copula framework based on a common `Copula`
  interface and a composed joint distribution. Begin with independence,
  Gaussian, Student-t, Clayton, Gumbel, and Frank copulas, including log-density,
  seeded sampling, parameter/correlation validation, boundary diagnostics, and
  Kendall's-tau conversions. Defer discrete/mixed marginals, vine copulas,
  dependence fitting, and automatic family selection until the core API is
  established.

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
