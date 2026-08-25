# Suggested submission

## Subject

Improve noncentral `pbeta()`/`qbeta()` tail accuracy with a mode-centred mixture

## Body

R-devel's `src/nmath/pnbeta.c` still uses the forward-only AS 226 series with an
absolute `1e-9` stopping target, and obtains the upper tail by subtraction.
This produces about `7.5e-10` error in a moderate example and can lose small
upper tails completely. `qnbeta.c` converts log probabilities to ordinary
probabilities before inversion, so targets below double underflow cannot be
inverted.

For example, the independently computed value of

```r
pbeta(.3, 2.5, 7, ncp = 11, lower.tail = FALSE, log.p = TRUE)
```

is `-0.0766532687446536588568`; R 4.6.1 returns
`-0.07665326799660159`. A regression case in the patch has an upper log tail of
`-255.3388790702448814`; another has a lower log tail of
`-774.0719328515007846`.

The attached patch replaces the CDF core with a log-scale Poisson mixture that
starts at the mode and recurses in both directions (Benton and Krishnamoorthy's
Method 2). It computes the selected incomplete-beta tail directly, uses the AS
310 mean approximation to choose the likely smaller tail, and periodically
re-anchors the recurrences through existing nmath routines. The quantile now
brackets and bisects the smaller tail on the log scale.

The references were generated independently with a 400-decimal `mpmath`
mixture. The proposed C files compile cleanly with strict GCC warnings and were
linked into a standalone Rmath test executable. The patch applies cleanly to R
4.6.1 and to the R-devel files checked on 2026-08-25.

Before submission, replace this paragraph with the exact R-devel revision,
platforms, and `make check-devel` result.

I would appreciate review of the stopping criterion and whether R Core would
prefer the crossover helper shared between the two translation units.

References:

- Chattamvelli and Shanmugam (1997),
  <https://doi.org/10.1111/1467-9876.00055>
- Benton and Krishnamoorthy (2003),
  <https://userweb.ucs.louisiana.edu/~kxk4695/CSDA-03.pdf>
