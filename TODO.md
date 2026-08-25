# JDistlib to-do

## Deferred numerical-distribution work

These items remain after the numerical-distribution hardening and
advanced-support work:
* Introduce an immutable modern integration result alongside the mutable legacy
  `IntegrationResult` compatibility type.
* Add kernel cost profiling and benchmark-oriented limits for exceptionally
  expensive callbacks.
* Build an independent high-precision reference corpus covering oscillation,
  endpoint/interior singularities, extreme scaling, narrow modes, and heavy
  tails.
* Add machine-readable diagnostic report serialization for logs, services, and
  user interfaces.
* Investigate opt-in worker isolation for callbacks that may block indefinitely;
  ordinary cooperative cancellation cannot interrupt a callback that never
  returns.

## Recently completed

* Seeded, budgeted adaptive randomized diagnostic probes.
* User-selected absolute moments with separate left/right convergence reports.
* Strict, warning, and permissive analyzed-construction policies.
* Optional certified rejection-envelope sampling.
* Double-exponential quadrature for finite, semi-infinite, and whole-line
  intervals.
