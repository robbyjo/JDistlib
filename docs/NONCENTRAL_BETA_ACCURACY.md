# Noncentral beta accuracy

## Summary

JDistlib's historical noncentral-beta CDF was a direct translation of R's
Algorithm AS 226 implementation. AS 226 begins below the Poisson mode, recurses
only toward larger indices, sums ordinary probabilities with an absolute
`1e-9` target, and obtains the upper tail by subtracting the lower tail from
one. Those choices explain the roughly `1e-11` accuracy ceiling in ordinary
cases and the complete loss of sufficiently small tails.

The replacement evaluates

```text
P(X <= x) = sum(k >= 0) Pois(k; ncp/2) I_x(a+k, b)
```

on the log scale. It starts at the Poisson mode, recurses in both directions,
and evaluates either incomplete-beta tail directly. The AS 310 approximation
to the distribution mean selects the likely smaller tail. This is the same
broad strategy described as Method 2 by Benton and Krishnamoorthy and used by
Boost.Math, with two additional safeguards:

- recurrence state is recomputed from `dpois` and TOMS 708 every 32 terms to
  bound accumulated drift;
- the incomplete-beta recurrence increment is formed through the beta density,
  avoiding separately expanded powers and `lbeta`.

The implementation does not multiply logarithms of partial sums. Each mixture
term is `log(Poisson weight) + log(beta tail)`, and the total uses log-sum-exp.
Log-space subtraction is used only for one adjacent incomplete-beta recurrence;
if rounding makes it indeterminate, the term is re-anchored through TOMS 708.

## Tail and quantile handling

The CDF computes the smaller tail directly and derives the larger tail with
`log1p` or `expm1`. Thus, an upper tail such as `exp(-255.34)` is not rounded to
zero by `1 - lower`.

The quantile solver preserves the target in logarithmic form and switches to
the complementary log tail when appropriate. It brackets and bisects against
log-CDF values, allowing inversion below `log(Double.MIN_VALUE)`.

## Independent validation

[`generate-ncbeta-high-precision.py`](../src/test/python/generate-ncbeta-high-precision.py)
uses a 400-decimal `mpmath` Poisson mixture independent of JDistlib and R.

| Case | High-precision result | Historical R 4.6.1 | New result |
| --- | ---: | ---: | ---: |
| upper log tail, `x=.3, a=2.5, b=7, ncp=11` | `-0.0766532687446536589` | `-0.07665326799660159` | `-0.07665326874465380` |
| extreme upper log tail | `-255.3388790702448814` | approximately `-21` after cancellation | `-255.33887907024487` |
| underflowing lower log tail | `-774.0719328515007846` | ordinary-probability path underflows | `-774.07193285150083` |

An 80-case sweep spanning both tails and varied shapes/noncentralities had a
maximum relative probability error of approximately `2.86e-13`. Focused Java
regressions cover direct tails, underflow, quantile inversion, noncentral-F
consumers, and the random representation.

## R nmath proposal

[`contrib/r-nmath-noncentral-beta`](../contrib/r-nmath-noncentral-beta/README.md)
contains a proposed R patch. It replaces `pnbeta.c`, makes `qnbeta.c` invert log
tails directly, and adds independent regression cases. On 2026-08-25 R-devel's
two files were byte-for-byte identical to R 4.6.1, so one patch targets both.

The proposed C sources compile with Rtools GCC and were linked into a standalone
Rmath test executable. It reproduced the three references above and inverted
the underflowing log probability to `1e-40` with about `5.4e-13` relative error.

## References

- Chattamvelli and Shanmugam, “Algorithm AS 310: Computing the Non-Central
  Beta Distribution Function” (1997),
  <https://doi.org/10.1111/1467-9876.00055>.
- Benton and Krishnamoorthy, “Computing discrete mixtures of continuous
  distributions” (2003),
  <https://userweb.ucs.louisiana.edu/~kxk4695/CSDA-03.pdf>.
- Gil, Segura, and Temme, “New asymptotic representations of the noncentral
  beta distribution,” <https://arxiv.org/abs/1905.07206>.
- Boost.Math, “Noncentral Beta Distribution,”
  <https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/dist_ref/dists/nc_beta_dist.html>.
