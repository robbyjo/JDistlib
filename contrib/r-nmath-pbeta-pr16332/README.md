# Proposed R nmath patch for PR#16332

This is a submission bundle for the open `pbeta(..., log.p = TRUE)` precision
problem tracked as R PR#16332 and JDistLib SourceForge ticket #23.

The patch was prepared on 2026-08-25 against R 4.6.1, SVN revision 90187. It is
deliberately narrow: when TOMS 708's BPSER result lies below the ordinary
floating-point probability range, the logarithmic lower tail is recomputed with
a modified-Lentz continued fraction. Ordinary-range BPSER results and all
non-logarithmic paths are unchanged.

## Files

- `pbeta-pr16332-log-tail.patch`: R source and regression-test patch;
- `SUBMISSION.md`: proposed R-devel/Bugzilla submission text.

## Apply and test

From a clean, current R-devel checkout:

```sh
git apply --check pbeta-pr16332-log-tail.patch
git apply pbeta-pr16332-log-tail.patch
make
make check-devel
```

For an SVN checkout, apply the patch, update and resolve any conflicts, then
create the attachment expected by R developers with `svn diff`.

The patch includes the two published PR#16332 inputs, the parameter-swapped
upper tail, and an integer-shape case with an independent high-precision value.
`git apply --check` succeeds against the included R 4.6.1 source snapshot, and
the same algorithm and all four reference values are exercised by JDistLib's
`SourceForgeTicketRegressionTest`.
Refresh the patch against current R-devel and run the full R test suite before
submission.
