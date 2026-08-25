# JDistlib to-do

## Planned for 0.7.0

* Add a continuous-marginal copula framework based on a common `Copula`
  interface and a composed joint distribution. Begin with independence,
  Gaussian, Student-t, Clayton, Gumbel, and Frank copulas, including log-density,
  seeded sampling, parameter/correlation validation, boundary diagnostics, and
  Kendall's-tau conversions. Defer discrete/mixed marginals, vine copulas,
  dependence fitting, and automatic family selection until the core API is
  established.

## Recently completed

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
