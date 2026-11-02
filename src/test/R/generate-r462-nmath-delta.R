# Generate the tagged R 4.6.1-to-4.6.2 src/nmath delta references.
# Run with the exact target release, for example:
#   "C:/Program Files/R/R-4.6.2/bin/Rscript.exe" --vanilla src/test/R/generate-r462-nmath-delta.R

if (getRversion() != package_version("4.6.2"))
    stop("delta references must be generated with R 4.6.2")
cat("# ", R.version.string, "\n", sep = "")

emit <- function(name, value) {
    if (is.nan(value)) text <- "NaN"
    else if (is.infinite(value)) text <- if (value > 0) "Infinity" else "-Infinity"
    else text <- sprintf("%.17g", value)
    cat(name, text, sep = "\t", fill = TRUE)
}

# r90571: besselJ() no longer returns zero immediately for x > 1e5.
emit("besselJ.100001.nu0", besselJ(100001, 0))
emit("besselJ.1e6.nu0", besselJ(1e6, 0))
emit("besselJ.1e6.nu0.25", besselJ(1e6, .25))
emit("besselJ.2^31.nu0", besselJ(2^31, 0))
emit("besselJ.2^31.nu3.5", besselJ(2^31, 3.5))
emit("besselJ.1e10.nu0.25", besselJ(1e10, .25))
emit("besselJ.1e24.nu0", besselJ(1e24, 0))

# r90223: m + n, rather than each population separately, selects the safe path.
set.seed(19)
cat("rhyper.combined-boundary", rhyper(3, 2^30, 2^30, 18), sep = "\t", fill = TRUE)
