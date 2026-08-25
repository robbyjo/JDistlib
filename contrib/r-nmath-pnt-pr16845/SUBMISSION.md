# Draft R PR#16845 submission

PR#16845 is still reproducible because `pnt()` switches abruptly from AS 243
to the A&S 26.7.10 normal approximation when `ncp^2` exceeds the binary64
underflow threshold.  At the original examples this changes the CDF by about
0.015 and 0.047.

The attached patch uses a separate conditional-probability representation at
the existing binary64 underflow cutoff. Conditioning on the normal numerator turns the remaining
probability into a one-dimensional integral of a standard-normal density times
a central chi-square tail.  Both requested tails are integrated directly, so
the method neither starts from `exp(-ncp^2/2)` nor subtracts a small tail from
one.  The existing large-`df` approximation remains unchanged and is retained
as an extreme-boundary fallback.

The regression covers the four values transcribed into PR#16845 comment 2.
Expected constants came from an independent 80-decimal evaluation of the
conditional identity at the exact binary64 inputs, not from the patched code.

Before submitting, please update an R-devel checkout, refresh the diff with
`svn diff`, run `make check-devel`, and include platform/compiler results.  The
private adaptive quadrature is deliberately small, but R Core may prefer to
move it to a shared nmath facility or use one of the asymptotic methods cited
in PR#16845 comment 3.
