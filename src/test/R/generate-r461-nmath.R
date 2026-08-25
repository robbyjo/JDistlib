# Generate the scalar reference values used by R461NmathAuditTest.
# Run with the exact target release, for example:
#   "C:/Program Files/R/R-4.6.1/bin/Rscript.exe" src/test/R/generate-r461-nmath.R

if (getRversion() != package_version("4.6.1"))
    stop("reference vectors must be generated with R 4.6.1")
cat("# ", R.version.string, "\n", sep = "")

emit <- function(name, value) {
    if (is.nan(value)) text <- "NaN"
    else if (is.infinite(value)) text <- if (value > 0) "Infinity" else "-Infinity"
    else text <- sprintf("%.17g", value)
    cat(name, text, sep = "\t", fill = TRUE)
}

# Beta and noncentral beta
emit("dbeta.log", dbeta(.4, .25, 3.5, log = TRUE))
emit("pbeta.lower.log", pbeta(.9833, 43779, .06728, log.p = TRUE))
emit("qbeta.log", qbeta(-248.06320004817743, 25, 6, log.p = TRUE))
emit("pbeta.a0.left", pbeta(-1, 0, 2))
emit("pbeta.a0.inside", pbeta(.4, 0, 2))
emit("pbeta.b0.inside", pbeta(.4, 2, 0))
emit("dnbeta.log", dbeta(.3, 2.5, 7, ncp = 11, log = TRUE))
emit("pnbeta.as226.upper.log", pbeta(.3, 2.5, 7, ncp = 11, lower.tail = FALSE, log.p = TRUE))
emit("qnbeta.as226", qbeta(.91, 2.5, 7, ncp = 11))

# Gamma, chi-square, and noncentral chi-square
emit("dgamma.subnormal.log", dgamma(5e-324, .5, scale = 2, log = TRUE))
emit("pgamma.upper.log", pgamma(1e-8, .125, scale = 3, lower.tail = FALSE, log.p = TRUE))
emit("qgamma.log", qgamma(-700, .125, scale = 3, log.p = TRUE))
emit("dchisq.log", dchisq(41, 7.5, log = TRUE))
emit("pchisq.upper.log", pchisq(41, 7.5, lower.tail = FALSE, log.p = TRUE))
emit("qchisq.log", qchisq(-650, 7.5, log.p = TRUE))
emit("dnchisq.log", dchisq(17, 4.5, ncp = 80, log = TRUE))
emit("pnchisq.lower.log", pchisq(17, 4.5, ncp = 80, log.p = TRUE))
emit("pnchisq.upper.log", pchisq(220, 4.5, ncp = 80, lower.tail = FALSE, log.p = TRUE))
emit("qnchisq", qchisq(.975, 4.5, ncp = 80))

# Hypergeometric and Wilcoxon
emit("dhyper.log", dhyper(23, 150, 230, 70, log = TRUE))
emit("phyper.upper.log", phyper(35, 150, 230, 70, lower.tail = FALSE, log.p = TRUE))
emit("qhyper", qhyper(.999, 150, 230, 70))
emit("dwilcox.log", dwilcox(137, 20, 30, log = TRUE))
emit("pwilcox.upper.log", pwilcox(137, 20, 30, lower.tail = FALSE, log.p = TRUE))
emit("qwilcox", qwilcox(.975, 20, 30))
emit("dsignrank.log", dsignrank(20, 12, log = TRUE))
emit("psignrank.upper.log", psignrank(20, 12, lower.tail = FALSE, log.p = TRUE))
emit("qsignrank", qsignrank(.975, 12))

# Bessel and polygamma
emit("besselJ.tiny.order", besselJ(2, 2e-16))
emit("besselJ.fractional", besselJ(12.25, 7.75))
emit("besselY.fractional", besselY(12.25, 7.75))
emit("besselI.scaled", besselI(1500, 2.25, expon.scaled = TRUE))
emit("besselK.scaled", besselK(700, 2.25, expon.scaled = TRUE))
emit("psigamma.negative.5", psigamma(-.375, deriv = 5))
emit("psigamma.positive.12", psigamma(3.25, deriv = 12))

# Binomial, Poisson, and negative binomial
emit("dbinom.log", dbinom(2, 1e16, 1e-18, log = TRUE))
emit("pbinom.upper.log", pbinom(3, 1e9, 1e-8, lower.tail = FALSE, log.p = TRUE))
emit("qbinom.log", qbinom(-700, 1e9, 1e-8, log.p = TRUE))
emit("dpois.log", dpois(1e200, 1e200 + 1e192, log = TRUE))
emit("ppois.upper.log", ppois(170, 100, lower.tail = FALSE, log.p = TRUE))
emit("qpois.log", qpois(-700, 100, log.p = TRUE))
emit("dnbinom.smallx.log", dnbinom(2, 1e16, .999999999999, log = TRUE))
emit("dnbinom.mu.smallx.log", dnbinom(2, 1e16, mu = 10000, log = TRUE))
emit("pnbinom.mu.upper.log", pnbinom(15000, 1e16, mu = 10000, lower.tail = FALSE, log.p = TRUE))
emit("qnbinom.mu.log", qnbinom(-700, 1e16, mu = 10000, log.p = TRUE))

# Remaining continuous distributions
emit("dcauchy.log", dcauchy(1e200, -3, 2.5, log = TRUE))
emit("pcauchy.upper.log", pcauchy(1e200, -3, 2.5, lower.tail = FALSE, log.p = TRUE))
emit("qexp.upper.log", qexp(-700, rate = 1/3, lower.tail = FALSE, log.p = TRUE))
emit("df.log", df(1e100, 3.5, 17, log = TRUE))
emit("qf.upper.log", qf(-700, 3.5, 17, lower.tail = FALSE, log.p = TRUE))
emit("dlnorm.infinity.zero", dlnorm(Inf, Inf, 0))
emit("plogis.upper.log", plogis(1000, 2, 3, lower.tail = FALSE, log.p = TRUE))
emit("qt.log", qt(-700, 1.25, log.p = TRUE))
emit("qweibull.upper.log", qweibull(-700, .75, 2, lower.tail = FALSE, log.p = TRUE))
emit("pnf.upper.log", pf(8, 4.5, 11, ncp = 7, lower.tail = FALSE, log.p = TRUE))
emit("pnt.upper.log", pt(9, 7.5, ncp = 4, lower.tail = FALSE, log.p = TRUE))
emit("ptukey.upper.log", ptukey(8, nmeans = 7, df = 20, nranges = 3,
                                lower.tail = FALSE, log.p = TRUE))
for (df2 in c(1e6, 1e7, 1e8))
    for (x in c(1 / pi, 1, pi))
        emit(sprintf("pnf.df2.%.0f.x.%.17g", df2, x),
             pf(x, 5, df2, ncp = 1))

# nmath arithmetic helpers
emit("round.tie.even.down", round(2.5, 0))
emit("round.tie.even.up", round(3.5, 0))
emit("round.subnormal", round(4.9406564584124654e-324, 323))
emit("round.small", round(1.234567890123456e-300, 315))
emit("signif.negative", signif(-12345.6789, 6))
emit("signif.large", signif(1.09e308, 2))
