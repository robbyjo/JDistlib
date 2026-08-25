# R 4.6.1 `src/nmath` audit

This is the completion record for the checklist in [UPSTREAM.md](UPSTREAM.md).
The audit compared the extracted tagged trees in `.upstream/R-3.3.2/src/nmath`
and `.upstream/R-4.6.1/src/nmath`. R 3.3.2 is the historical JDistlib baseline;
R 4.6.1 is the target, not the moving R development branch.

Every changed upstream file was classified as one of:

1. a numerical or boundary change that needed a Java port;
2. behavior already present in JDistlib or supplied by a shared helper; or
3. R-only diagnostics, declarations, comments, or build scaffolding with no
   numerical Java equivalent.

R process-global warning/error bookkeeping was not introduced into the Java
API. Existing Java `NaN`, infinity, and explicit-engine conventions are retained
where the upstream difference only changes an R diagnostic.

## Complete changed-file manifest

The tagged-tree comparison contains **120 changed paths**: 117 modified files,
two additions, and one removal. Every changed path appears exactly once below.
The result column records whether
the differences required a Java numerical port, were already represented by a
Java/shared helper, or were R-only build, diagnostic, or portability machinery.

| Classification | Changed R files | Disposition |
| --- | --- | --- |
| Build and standalone scaffolding (6) | `Makefile.in`, `Makefile.win`, `standalone/Makefile.in`, `standalone/Makefile.win`, `standalone/sunif.c`, `standalone/test.c` | R build/test harness only; no Java numerical counterpart. |
| Arithmetic, special-function helpers, and headers (22) | `bd0.c`, `beta.c`, `chebyshev.c`, `choose.c`, `cospi.c`, `d1mach.c`, `dpq.h`, `expm1.c`, `fprec.c`, `fround.c`, `gamma.c`, `gamma_cody.c`, `gammalims.c`, `i1mach.c`, `lbeta.c`, `lgamma.c`, `lgammacor.c`, `log1p.c`, `mlutils.c`, `nmath.h`, `nmath2.h`, `stirlerr.c` | Ported numerical and boundary changes into `MathFunctions` and shared distribution helpers; retained Java-native IEEE-754/runtime operations and omitted R-only diagnostics/declarations. |
| Beta and TOMS 708 (8) | `dbeta.c`, `dnbeta.c`, `pbeta.c`, `pnbeta.c`, `qbeta.c`, `qnbeta.c`, `rbeta.c`, `toms708.c` | Ported or intentionally improved as described below; covered by R 4.6.1 and independent high-precision vectors. |
| Gamma and chi-square (8) | `dgamma.c`, `dnchisq.c`, `pgamma.c`, `pnchisq.c`, `qgamma.c`, `qnchisq.c`, `rgamma.c`, `rnchisq.c` | Ported numerical, iteration-limit, quantile-boundary, and RNG-input changes. Central chi-square continues to use the shared gamma implementation. |
| Hypergeometric and rank statistics (6) | `dhyper.c`, `phyper.c`, `qhyper.c`, `rhyper.c`, `signrank.c`, `wilcox.c` | Ported support, recurrence, underflow, and large-parameter sampling changes while keeping caches instance-local. |
| Bessel and polygamma (6) | `bessel.h`, `bessel_i.c`, `bessel_j.c`, `bessel_k.c`, `bessel_y.c`, `polygamma.c` | Ported numerical changes; retained the documented higher-order Java reflection extension. |
| Binomial, Poisson, geometric, and negative binomial (19) | `dbinom.c`, `dgeom.c`, `dnbinom.c`, `dpois.c`, `pbinom.c`, `pgeom.c`, `pnbinom.c`, `ppois.c`, `qDiscrete_search.h`, `qbinom.c`, `qgeom.c`, `qnbinom.c`, `qnbinom_mu.c`, `qpois.c`, `rbinom.c`, `rgeom.c`, `rmultinom.c`, `rnbinom.c`, `rpois.c` | Ported density/search/boundary changes or verified the existing shared implementation and explicit random-engine behavior. |
| Normal family (5) | `dnorm.c`, `pnorm.c`, `qnorm.c`, `rnorm.c`, `snorm.c` | Ported in the earlier normal-family audit, including extreme log tails, subnormal inputs, infinite scales, and Java-safe binary scaling. |
| Remaining continuous distributions and RNGs (40) | `dcauchy.c`, `dexp.c`, `df.c`, `dlnorm.c`, `dlogis.c`, `dnf.c`, `dnt.c`, `dt.c`, `dunif.c`, `dweibull.c`, `pcauchy.c`, `pexp.c`, `pf.c`, `plnorm.c`, `plogis.c`, `pnf.c`, `pnt.c`, `pt.c`, `ptukey.c`, `punif.c`, `pweibull.c`, `qcauchy.c`, `qexp.c`, `qf.c`, `qlogis.c`, `qnf.c`, `qnt.c`, `qt.c`, `qtukey.c`, `qunif.c`, `qweibull.c`, `rcauchy.c`, `rchisq.c`, `rexp.c`, `rf.c`, `rlnorm.c`, `rlogis.c`, `rt.c`, `runif.c`, `rweibull.c` | Ported numerical and boundary differences where applicable; verified already-equivalent formulae and omitted R-only warning/declaration changes. RNG APIs retain explicit engines. |

## Audit disposition

| Group | R source reviewed | Result |
| --- | --- | --- |
| Beta and TOMS 708 | `dbeta.c`, `pbeta.c`, `qbeta.c`, `rbeta.c`, `dnbeta.c`, `pnbeta.c`, `qnbeta.c`, `toms708.c` | Ported zero-shape beta limits, raw-CDF boundaries, noncentral overflow-safe scaling and `lbeta` initialization. Retained the hardened TOMS 708 path and bracketed `qbeta` polishing for finite shapes near `Double.MAX_VALUE`; a log-scale continued-fraction fallback fixes the historical PR#16332 BPSER cancellation case. The noncentral CDF intentionally improves on R's AS 226 path with an AS 310 crossover and a log-scale, mode-centred bidirectional mixture. |
| Gamma and chi-square | `dgamma.c`, `pgamma.c`, `qgamma.c`, `rgamma.c`, central and noncentral chi-square `d/p/q/r` sources | Ported subnormal log-density arithmetic, current noncentral-CDF iteration limits, quantile bracketing boundary, and random-input handling. |
| Hypergeometric and rank statistics | hypergeometric, Wilcoxon, and signed-rank `d/p/q/r` sources | Ported support/underflow boundaries and large-parameter sampling. Replaced the historical recursive Wilcoxon cube with R's one-dimensional Loeffler recurrence; fixed signed-rank integer handling. Cache state remains per distribution instance. |
| Bessel and polygamma | `bessel_i.c`, `bessel_j.c`, `bessel_k.c`, `bessel_y.c`, `bessel.h`, `polygamma.c` | Ported current tiny-order Bessel recurrence, normalization details, and the PR#15554 huge-order allocation guard. Current requested-length and negative-argument polygamma behavior is present; JDistlib intentionally extends reflection beyond R's derivative-order fallback. |
| Binomial, Poisson, and negative binomial | count-family `d/p/q/r` sources, including `_mu` forms and `qDiscrete_search.h` | Shared discrete search, binomial/Poisson density details, and mean-parameterized negative binomial were already current. Ported the remaining negative-binomial small-`x` expansions, stable log probability, and `NaN` checks. |
| Continuous distributions and RNGs | remaining scalar distribution and random-generation sources | Compared all remaining changes. Ported the current overflow-safe Student-t quantile correction and outstanding beta, noncentral chi-square, and hypergeometric RNG boundaries. Other differences were already present or diagnostic/declaration-only. |
| Headers and arithmetic helpers | `nmath.h`, `dpq.h`, `mlutils.c`, `fround.c`, `fprec.c`, `choose.c`, beta/gamma helpers, and trigonometric helpers | Ported current relative integer tolerance, choose symmetry, beta underflow, decimal rounding/significance, and exact signed `sinpi`/`tanpi` cases. Audited probability-tail mappings and IEEE-754 scaling helpers. |

## Reproducible reference corpus

[`src/test/R/generate-r461-nmath.R`](src/test/R/generate-r461-nmath.R) refuses to
run under any version other than R 4.6.1 and emits the scalar references used by
[`R461NmathAuditTest`](src/test/java/jdistlib/R461NmathAuditTest.java). The
checked values were generated with:

```text
R version 4.6.1 (2026-06-24 ucrt), SVN revision 90187
```

The corpus exercises every audit group, including both/logged tails, extreme
probabilities, underflow and overflow, infinities and `NaN`s, support and integer
boundaries, large hypergeometric inputs, and explicit random-engine boundaries.
It supplements the larger historical vectors in `TestDPQR` rather than replacing
them. Three legacy noncentral-F expectations were regenerated because the
current noncentral-beta implementation changes their last digits. Independent
100-decimal mixture references for that intentional improvement are generated by
`src/test/python/generate-ncbeta-high-precision.py`.

The numerical design, validation evidence, and proposed R nmath patch are in
[`docs/NONCENTRAL_BETA_ACCURACY.md`](docs/NONCENTRAL_BETA_ACCURACY.md).

Verification command:

```text
gradlew.bat clean test
```

The 2026-08-25 audit ran 81 Java tests with no failures. It also executed the
R generator above under R 4.6.1 and regenerated the independent noncentral-beta
mixtures at 400 decimal digits with mpmath 1.4.1.

## `stats::integrate` audit

Integration is outside `src/nmath`, so it has a separate source trail. The Java
implementation was compared with R 4.6.1's
`src/library/stats/R/integrate.R`, `src/library/stats/src/integrate.c`, and
`src/appl/integrate.c`. It now follows DQAGS for finite intervals and DQAGI for
semi-infinite and doubly-infinite intervals, with all workspace state local to a
call. [`src/test/R/generate-r461-integrate.R`](src/test/R/generate-r461-integrate.R)
generates the values, error estimates, subdivision counts, and failure messages
used by [`R461IntegrateAuditTest`](src/test/java/jdistlib/math/R461IntegrateAuditTest.java),
including QUADPACK statuses 0 through 6.
