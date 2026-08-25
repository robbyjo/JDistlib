# JDistlib to-do

## Deferred numerical-distribution work

These items were deliberately deferred after the numerical-distribution
hardening and advanced-support work. Revisit them as a separate batch:

* Add adaptive randomized diagnostic probes for narrow spikes that deterministic
  grids can miss, with reproducible seeds and explicit sampling budgets.
* Generalize absolute-moment diagnostics beyond orders one and two, including
  user-selected orders and separate left/right tail reports.
* Add strict, warning, and permissive construction policies that determine which
  diagnostic findings prevent construction.
* Add faster sampling strategies such as adaptive rejection or rejection-
  envelope sampling for suitable log-concave kernels.
* Extend double-exponential quadrature transformations to semi-infinite and
  doubly-infinite intervals.
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
