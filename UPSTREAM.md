# R `src/nmath` synchronization

Target: **R 4.6.1**

Historical baseline: **R 3.3.2**, with several later individual fixes

Canonical source: <https://svn.r-project.org/R/tags/R-4-6-1/src/nmath/>

**Ledger status: complete.** Audited on 2026-08-25 against all 120 paths that
changed between the tagged R 3.3.2 and R 4.6.1 `src/nmath` trees (117 modified,
2 added, and 1 removed). The exact changed-file manifest is recorded in
[`NMATH_AUDIT.md`](NMATH_AUDIT.md).

This ledger prevents a partially updated library from being presented as a
complete R 4.6.1 port. Each item should be checked only after the Java port has
been compared with the tagged R source and covered by regression vectors.

## Completed in the 0.5.0 development tree

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
- [x] Full generated regression corpus against an R 4.6.1 reference build

The file-by-file disposition and reproducible reference-vector details are in
[`NMATH_AUDIT.md`](NMATH_AUDIT.md).

## Post-4.6.1 R-devel compatibility

The following changes were audited against the official R trunk on 2026-08-25
and are covered by upstream regression vectors in
`RDevelPost461Test`:

- [x] `rhyper.c` revision 90223: use the large-population quantile path when
  the combined population exceeds `Integer.MAX_VALUE`, even if each group is
  individually smaller.
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

## Porting rules

1. Compare the R 3.3.2 and R 4.6.1 tagged files for the baseline audit. Track
   later R-development changes separately by exact revision, as above.
2. Preserve JDistlib-only distributions and public APIs unless a compatibility
   shim is supplied.
3. Convert R process-global work caches to call-local data or explicit state
   objects. Never introduce a shared mutable numerical cache.
4. Cover both tails, log probabilities, infinities, NaNs, integer boundaries,
   underflow, and overflow for every changed routine.
5. Record the corresponding R source file in the change and add R-generated
   expected values before checking off an audit group.
