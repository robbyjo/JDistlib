# Reproducible scalar distribution audit. R is a comparator, not an oracle.
# Rscript benchmarks/audit/compare-r.R build/r-audit [sample-count=1024]
args <- commandArgs(TRUE)
out <- if (length(args)) args[1] else "build/r-audit"
n <- if (length(args) > 1) as.integer(args[2]) else 1024L
stopifnot(n >= 16L)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
writeLines(c(R.version.string, capture.output(sessionInfo())), file.path(out, "r-version.txt"))
options(digits = 17)
families <- list(
  normal = list("norm", list(mean = 0.3, sd = 1.7), -8, 8),
  gamma = list("gamma", list(shape = 2.7, scale = 1.3), 0.001, 30),
  beta = list("beta", list(shape1 = 0.7, shape2 = 3.2), 0.00001, 0.99999),
  t = list("t", list(df = 5.5), -15, 15),
  f = list("f", list(df1 = 4.5, df2 = 11), 0.0001, 20),
  exponential = list("exp", list(rate = 1/1.3), 0.001, 25),
  weibull = list("weibull", list(shape = 0.7, scale = 1.3), 0.0001, 30),
  lognormal = list("lnorm", list(meanlog = 0.3, sdlog = 1.7), 0.0001, 30),
  cauchy = list("cauchy", list(location = 0.3, scale = 1.7), -30, 30),
  logistic = list("logis", list(location = 0.3, scale = 1.7), -30, 30),
  poisson = list("pois", list(lambda = 17.3), 0, 55),
  binomial = list("binom", list(size = 73, prob = 0.27), 0, 73),
  negbinomial = list("nbinom", list(size = 4.5, prob = 0.27), 0, 100),
  ncchisquare = list("chisq", list(df = 4.5, ncp = 12), 0.01, 80),
  ncbeta = list("beta", list(shape1 = 2.5, shape2 = 7, ncp = 11), 0.001, 0.999)
)
rows <- list(); timings <- list(); k <- 0L
for (family in names(families)) {
  spec <- families[[family]]
  for (op in c("d", "p", "q")) for (logged in c(FALSE, TRUE)) {
    for (lower in if (op == "d") TRUE else c(TRUE, FALSE)) {
      # Permute an evenly spaced grid: same deterministic inputs in both runtimes.
      u <- (((seq_len(n) * 313L) %% n) + 0.5) / n
      stopifnot(all(u > 0 & u < 1), length(unique(u)) == n)
      x <- if (op == "q") 0.0001 + u * 0.9998 else spec[[3]] + u * (spec[[4]] - spec[[3]])
      if (op != "q" && family %in% c("poisson", "binomial", "negbinomial")) x <- floor(x)
      if (op == "q" && logged) x <- log(x)
      fun <- get(paste0(op, spec[[1]]), envir = asNamespace("stats"))
      callargs <- c(list(x), spec[[2]])
      if (op == "d") callargs$log <- logged else {
        callargs$lower.tail <- lower; callargs$log.p <- logged
      }
      expected <- suppressWarnings(do.call(fun, callargs))
      stopifnot(all(is.finite(expected)))
      key <- paste(family, op, as.integer(lower), as.integer(logged), sep = ":")
      k <- k + 1L
      rows[[k]] <- data.frame(key = key, x = x, expected = expected)
      # Calibrate enough vector calls to exceed coarse system.time resolution.
      reps <- 1L
      repeat {
        elapsed <- system.time(for (i in seq_len(reps)) ans <- do.call(fun, callargs))[["elapsed"]]
        if (elapsed >= 0.2 || reps >= 65536L) break
        reps <- reps * 2L
      }
      samples <- replicate(5, system.time(for (i in seq_len(reps)) ans <- do.call(fun, callargs))[["elapsed"]])
      timings[[k]] <- data.frame(key = key, ns_per_value = median(samples) * 1e9 / (n * reps),
                                 repetitions = reps, checksum = sum(ans))
    }
  }
  cat("completed", family, "\n")
}
references <- do.call(rbind, rows)
references$x <- sprintf("%.17g", references$x)
references$expected <- sprintf("%.17g", references$expected)
write.table(references, file.path(out, "reference.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(do.call(rbind, timings), file.path(out, "r-timing.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
