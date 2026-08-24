# R `src/nmath` synchronization

Target: **R 4.6.1**

Historical baseline: **R 3.3.2**, with several later individual fixes

Canonical source: <https://svn.r-project.org/R/tags/R-4-6-1/src/nmath/>

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
- [x] R `stats::integrate` behavior: finite and infinite intervals, tolerances,
  error estimates, subdivision limits, and status reporting

## Remaining audit groups

- [ ] Beta family and TOMS 708 (`pbeta`, `qbeta`, `toms708`)
- [ ] Gamma, chi-square, and noncentral chi-square family
- [ ] Hypergeometric and Wilcoxon family changes
- [ ] Bessel and polygamma changes
- [ ] Remaining Poisson, binomial, and negative-binomial source details
- [ ] Remaining continuous distributions and RNG implementations
- [ ] Header/macro behavior (`nmath.h`, `dpq.h`, arithmetic helpers)
- [ ] Full generated regression corpus against an R 4.6.1 reference build

## Porting rules

1. Compare the R 3.3.2 and R 4.6.1 tagged files, not the moving R development
   branch.
2. Preserve JDistlib-only distributions and public APIs unless a compatibility
   shim is supplied.
3. Convert R process-global work caches to call-local data or explicit state
   objects. Never introduce a shared mutable numerical cache.
4. Cover both tails, log probabilities, infinities, NaNs, integer boundaries,
   underflow, and overflow for every changed routine.
5. Record the corresponding R source file in the change and add R-generated
   expected values before checking off an audit group.
