# R `src/nmath` synchronization

Target: **R 4.6.2**

Historical baseline: **R 3.3.2**, with several later individual fixes

Canonical source: <https://svn.r-project.org/R/tags/R-4-6-2/src/nmath/>

**Ledger status: complete.** The 2026-08-25 audit classified all 120 paths that
changed between the tagged R 3.3.2 and R 4.6.1 `src/nmath` trees (117 modified,
2 added, and 1 removed). The 2026-11-02 audit then classified the complete
five-file R 4.6.1-to-4.6.2 delta. The exact manifests and source-archive hashes
are recorded in [`NMATH_AUDIT.md`](NMATH_AUDIT.md).

This ledger prevents a partially updated library from being presented as a
complete R 4.6.2 port. Each item should be checked only after the Java port has
been compared with the tagged R source and covered by regression vectors.

## Completed in version 0.5.0

- [x] `qnorm.c`: extreme log-tail refinements
- [x] Normal family (`dnorm.c`, `pnorm.c`, `qnorm.c`, `rnorm.c`, `snorm.c`),
  including subnormal probability tails and infinite-scale boundaries
- [x] `qDiscrete_search.h`: shared binomial, Poisson, and negative-binomial search
- [x] `qnbinom_mu.c`: direct mean parameterization
- [x] `stirlerr.c`: current coefficients and selection rules
- [x] `bd0.c`: signed deviance series and stable ratio calculation
- [x] IEEE-754 `ldexp`/`frexp` boundary behavior used by nmath translations
- [x] Selected `dbinom.c`, `dpois.c`, and `dlnorm.c` boundary updates
- [x] `dpsifn` requested-length sequences and general negative-argument
  reflection (extending the current R fallback beyond derivative order 5)
- [x] R 4.6.1 `stats::integrate` behavior: finite DQAGS and infinite DQAGI
  paths, tolerances, error estimates, subdivision limits, extrapolation, and
  all QUADPACK status codes
- [x] TOMS 708 overflow hardening for finite shapes near `Double.MAX_VALUE`,
  plus bracketed `qbeta` polishing and extreme-tail regression vectors

## Completed audit groups

- [x] Full Beta-family source synchronization (`dbeta`, `pbeta`, `qbeta`,
  `rbeta`, noncentral beta, and the remaining TOMS 708 source-level audit)
- [x] Gamma, chi-square, and noncentral chi-square family
- [x] Hypergeometric and Wilcoxon family changes
- [x] Bessel and polygamma changes
- [x] Remaining Poisson, binomial, and negative-binomial source details
- [x] Remaining continuous distributions and RNG implementations
- [x] Header/macro behavior (`nmath.h`, `dpq.h`, arithmetic helpers)
- [x] Full generated regression corpus against an R 4.6.1 reference build,
  preserved as the historical baseline
- [x] Tagged-delta regression corpus against an R 4.6.2 reference build

The file-by-file disposition and reproducible reference-vector details are in
[`NMATH_AUDIT.md`](NMATH_AUDIT.md).

## R 4.6.2 tagged delta

The [official R 4.6.2 tag](https://svn.r-project.org/R/tags/R-4-6-2/) was created
at revision 90785 from branch revision 90784. Comparing the official R 4.6.1
and R 4.6.2 source archives changes exactly five `src/nmath` files:

- [x] `rhyper.c` revision 90223: use the large-population quantile path when
  the combined population exceeds `Integer.MAX_VALUE`, even if each group is
  individually smaller. `RDevelPost461Test` covers the seeded R 4.6.2 stream;
  the version-gated delta generator records its tagged-release provenance.
- [x] `bessel_i.c`, `bessel_j.c`, `bessel_k.c`, and `bessel_y.c` revisions
  90571, 90618, 90697, and 90756: reject unusable I/K orders before work-array
  allocation; normalize negative K orders before applying the limit; reject
  large negative non-integer Y orders; and evaluate J through its asymptotic
  branch for arguments above `1e5` through `1e24`. `R462NmathDeltaTest` covers
  J immediately above the old cutoff, above `Integer.MAX_VALUE`, and at the
  new cutoff. JDistlib intentionally retains its more conservative `1e7` J/Y
  order limit because the Java implementation allocates one double per order.
- [x] Outside `src/nmath`, `stats::predict.smooth.spline` revision 90606
  (R PR#19190) rejects negative derivative orders explicitly. Both interpolation
  and extrapolation paths are covered by `RDevelPost461Test`.

`src/test/R/generate-r462-nmath-delta.R` refuses to run under any release other
than R 4.6.2 and emits the values asserted by `R462NmathDeltaTest` plus the
seeded `rhyper` stream asserted by `RDevelPost461Test`.

## Additional upstream compatibility

- [x] `stats` bandwidth pair counting revision
  [90955](https://svn.r-project.org/R/!svn/rvr/90955/trunk/src/library/stats/R/bandwidths.R)
  ([PR#19351](https://bugs.r-project.org/show_bug.cgi?id=19351)):
  `Bandwidth` retains the terminal histogram bin and folds pair distance `nb`
  into count `nb - 1`. `MathUtilitiesAuditTest` checks exact pair counts and
  finite UCV/BCV/SJ results for the upstream 20- and 60-observation examples.
- [x] Complex tangent revision
  [90957](https://svn.r-project.org/R/!svn/rvr/90957/trunk/src/main/complex.c)
  ([PR#19327](https://bugs.r-project.org/show_bug.cgi?id=19327)) repairs
  platform `ctan` NaNs for large finite imaginary arguments. Java `Complex.tan`
  and Stan use a shared scaled evaluation; `StanComplexValueTest` covers tiny
  real components, both infinite limits, and large-argument derivatives.
  These are outside `src/nmath`; the tagged R 4.6.2 audit remains unchanged.

The following independently tracked changes remain covered by
`RDevelPost461Test`; they are not part of the five-file tagged Nmath delta:

- [x] `rbinom.c` revisions 90299, 90307, and 90310: corrected BTPE Stirling
  signs and small-mean setup. Corrected BTPE is the default; an explicit
  `BinomialKind.BUGGY_BTPE` random state preserves the R 4.6-and-earlier
  stream when reproducibility requires it.
- [x] `rmultinom.c` revision 89909: sequential conditional binomial sampling
  with Kahan compensated addition and subtraction. Public weight
  normalization continues to match R's `FixupProb` wrapper.
- [x] `stats::wilcox.test` revision 90068: `digits.rank` now defaults to 7 and
  the new `digits.zap` control defaults to the same value. JDistlib exposes
  both through overloads of `wilcoxon_test` and `mann_whitney_u_test`.
- [x] `stats::mood.test` revision 89549 and `stats::ansari.test` revision
  89645 (R PR#19013): tied samples use the corrected cumulative-tie variance
  and exchangeable mid-rank score moments. JDistlib also applies those moments
  consistently to its one-sided Ansari overloads.
- [x] `stats::cor.test` revision
  [90891](https://svn.r-project.org/R/!svn/rvr/90891/trunk/src/library/stats/R/cor.test.R)
  ([R PR#19319](https://bugs.r-project.org/show_bug.cgi?id=19319)): exact
  Kendall upper tails use distribution symmetry instead of subtracting a
  lower tail from one, and cumulative probabilities are clamped against
  floating-point overshoot. `Kendall.cumulative` now sums the shorter
  symmetric tail for ordinary and log probabilities; `RDevelPost461Test`
  covers full support and one-permutation tails through `n = 20`.

## R Bugzilla hardening in 0.8.1

The 2026-08-27 open/accuracy sweep additionally produced focused JDistlib
regressions for PR#15554 (negative half-integer Bessel connections), PR#17519
(noncentral-t density cancellation), PR#18395 (Fligner tie stability), and the
reopened PR#16727 (fixed-probability negative-binomial infinite/extreme size).
These are tracked separately from the tagged `src/nmath` audit because several
reports remain open upstream or concern `stats`-package R code rather than a
tagged Nmath source change.

## Porting rules

1. Preserve the R 3.3.2-to-4.6.1 baseline audit and compare each later tagged
   release as an explicit delta. Track post-tag R-development changes separately
   by exact revision, as above.
2. Preserve JDistlib-only distributions and public APIs unless a compatibility
   shim is supplied.
3. Convert R process-global work caches to call-local data or explicit state
   objects. Never introduce a shared mutable numerical cache.
4. Cover both tails, log probabilities, infinities, NaNs, integer boundaries,
   underflow, and overflow for every changed routine.
5. Record the corresponding R source file in the change and add R-generated
   expected values before checking off an audit group.
