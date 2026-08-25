## High-precision references generated independently with a 400-decimal
## Poisson mixture (mpmath 1.4.1).  The cases exercise direct upper-tail
## evaluation, a probability below ordinary double underflow, and log-tail
## quantile inversion.

near <- function(actual, expected, tolerance)
    abs(actual - expected) <= tolerance * max(1, abs(expected))

stopifnot(
    near(pbeta(.3, 2.5, 7, ncp = 11, lower.tail = FALSE, log.p = TRUE),
         -0.0766532687446536588568411685997801, 5e-14),

    near(pbeta(.847507562537541, 3.009369292533997,
               187.95519621829496, ncp = 37.405600013184028,
               lower.tail = FALSE, log.p = TRUE),
         -255.338879070244881404777315694966, 5e-13),

    near(pbeta(1e-40, 1.0149511645559415, 126.03373615036026,
               ncp = 1370.9880095911913, log.p = TRUE),
         -774.071932851500784583717105502188, 2e-12),

    abs(qbeta(-774.071932851500784583717105502188,
              1.0149511645559415, 126.03373615036026,
              ncp = 1370.9880095911913, log.p = TRUE) - 1e-40) <= 1e-52,

    near(qbeta(.91, 2.5, 7, ncp = 11),
         .714379926903104725858801470722063, 5e-14)
)
