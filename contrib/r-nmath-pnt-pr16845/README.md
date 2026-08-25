# Proposed R nmath patch for PR#16845

This bundle replaces the abrupt noncentral-t normal-approximation cutoff near
`abs(ncp) = 37.62` with a conditional-probability quadrature for large finite
noncentralities.  It was prepared against R 4.6.1 (`src/nmath/pnt.c`) on
2026-08-25; R PR#16845 remains assigned and the same cutoff is present in that
release.

The identity used for positive `t` is

```
P(T <= t) = Phi(-delta)
          + integral phi(z) P(ChiSq(df) >= df*((z+delta)/t)^2) dz,
```

where the integral starts at `-delta`.  The complementary tail is evaluated
directly with the lower chi-square probability.  A small adaptive-Simpson
driver is kept private to `pnt.c`; fixed breakpoints at standard-normal
landmarks and at the chi-square crossover prevent it from skipping a narrow
transition.  The integral is truncated at `z = +/-10`, whose omitted normal
mass is below `8e-24`.

Files:

- `pnt-pr16845.patch`: source and R regression patch;
- `tests/pnt-pr16845.R`: standalone reproduction of the four reported points;
- `SUBMISSION.md`: draft submission notes.

Apply from a clean R 4.6.1 or current R-devel checkout:

```sh
git apply --check pnt-pr16845.patch
git apply pnt-pr16845.patch
make
make check-devel
```

The four expected values were independently evaluated at 80 decimal digits
from the normal/chi-square conditional identity, using the exact binary64
inputs produced by the R expressions.  The patch should be refreshed against
current R-devel and run through the full sanitizer and platform checks before
submission.
