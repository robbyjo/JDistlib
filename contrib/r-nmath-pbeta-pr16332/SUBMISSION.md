# Draft submission for R PR#16332

Subject: `[PATCH] pbeta(log.p=TRUE): avoid BPSER cancellation to -Inf (PR#16332)`

`bpser()` can lose substantial precision below the ordinary floating-point
probability range, and can compute `a * sum <= -1` after cancellation even
though the requested incomplete-beta probability is nonzero. In the latter
case log mode returns `-Inf`. Examples include:

```r
pbeta(0.5555555, 1925.74, 33.7179, log.p = TRUE)
pbeta(0.555555,  1925.74, 33.7179, log.p = TRUE)
```

The attached patch preserves the existing TOMS 708 selection and series. It
only invokes a modified-Lentz incomplete-beta continued fraction when the
BPSER log result is below `log(DBL_MIN)`. The continued-fraction front factor uses
`brcomp(..., log_p=TRUE)` so it retains the existing stable beta/deviance
arithmetic, and the complementary orientation is selected when appropriate.

Expected results for the examples above are approximately
`-994.767594138466967` and `-994.769290541658`. Regression tests also cover the
swapped upper tail and the integer-shape value
`pbeta(5/9, 1925, 34, log.p=TRUE) = -993.424624967607243...`.

The change is isolated to `src/nmath/toms708.c`; non-logarithmic behavior and
ordinary-range BPSER evaluations are unchanged. The patch was initially
prepared against R 4.6.1 (SVN revision 90187).
