# Proposed R nmath noncentral-beta patch

This is a submission bundle for replacing R's AS 226 noncentral-beta CDF with
a log-scale, mode-centred, bidirectional Poisson mixture and making the
noncentral-beta quantile invert log tails directly.

The patch was prepared on 2026-08-25 against R 4.6.1, SVN revision 90187. The
official R-devel `pnbeta.c` and `qnbeta.c` were checked that day and were
byte-for-byte identical, so the patch also targets current R-devel. Refresh it
with `svn update` and `svn diff` immediately before submission.

## Files

- `pnbeta-mode-centred.patch`: unified source and regression patch;
- `src/nmath/pnbeta.c` and `src/nmath/qnbeta.c`: complete proposed files;
- `tests/pnbeta-accuracy.R`: independent high-precision regressions;
- `tests/pnbeta-smoke.c`: standalone Rmath numerical smoke test;
- `SUBMISSION.md`: an R-devel/Bugzilla submission draft.

## What changes

- starts at the mode of the `Poisson(ncp/2)` weights and sums both directions;
- accumulates mixture terms with log-sum-exp;
- evaluates lower and upper incomplete-beta mixtures directly;
- uses the AS 310 mean approximation only as a tail-selection crossover;
- re-anchors recurrence values through nmath every 32 terms;
- preserves log probabilities during quantile bracketing and bisection;
- tests cancellation, double underflow, and log-tail quantile inversion.

This is a bounded replacement for the existing Poisson mixture, not a port of
the full AS 310 program. Modern asymptotic methods may be faster for extremely
large parameters, but are not required for the failures covered here.

## Apply and test

From a clean, up-to-date R-devel checkout:

```sh
git apply --check pnbeta-mode-centred.patch
git apply pnbeta-mode-centred.patch
make
make check-devel
```

For an SVN-native submission, apply the changes, run `svn update`, resolve any
conflicts, and create the attachment with:

```sh
svn diff > pnbeta-mode-centred.diff
```

The included patch passes `git apply --check` against the exact R 4.6.1 tree.
The C files compile with `-std=gnu17 -Wall -Wextra -pedantic`. A standalone
Rmath build reproduced:

```text
moderate upper log       -0.076653268744653696
extreme upper log        -255.33887907024487
underflow lower log      -774.07193285150083
underflow log quantile    9.9999999999946159e-41
```

Run the full `make check-devel` on a fresh R-devel build before submission.

The constants in `tests/pnbeta-accuracy.R` were generated independently at 400
decimal digits by `src/test/python/generate-ncbeta-high-precision.py`; it calls
neither JDistlib, R, nor Boost.
