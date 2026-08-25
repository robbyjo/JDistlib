# Generate the reference values in R461IntegrateAuditTest.
# This script intentionally refuses to run with any other R release.
if (getRversion() != "4.6.1") {
    stop("R 4.6.1 is required; found ", getRversion())
}

options(digits = 17)

emit <- function(name, result) {
    cat(name, "\n", sep = "")
    dput(unclass(result)[c("value", "abs.error", "subdivisions", "message")])
}

control <- list(rel.tol = 1e-10, abs.tol = 1e-10, stop.on.error = FALSE)
emit("polynomial", integrate(function(x) x^2, 0, 1,
    rel.tol = 1e-12, abs.tol = 1e-12, stop.on.error = FALSE))
emit("endpoint singularity", do.call(integrate, c(list(
    f = function(x) 1 / sqrt(x), lower = 0, upper = 1, subdivisions = 200L),
    control)))
emit("interior singularity", integrate(function(x) 1 / sqrt(abs(x - .3)),
    0, 1, subdivisions = 300L, rel.tol = 1e-9, abs.tol = 1e-9,
    stop.on.error = FALSE))
emit("discontinuity", integrate(function(x) ifelse(x < .12345, -2, 3),
    -1, 2, subdivisions = 200L, rel.tol = 1e-11, abs.tol = 1e-11,
    stop.on.error = FALSE))
emit("oscillatory", integrate(function(x) sin(1000 * x), 0, 1,
    subdivisions = 500L, rel.tol = 1e-10, abs.tol = 1e-10,
    stop.on.error = FALSE))
emit("semi-infinite", do.call(integrate, c(list(
    f = function(x) exp(-x), lower = 0, upper = Inf, subdivisions = 200L),
    control)))
emit("doubly-infinite", do.call(integrate, c(list(
    f = function(x) 1 / (1 + x*x), lower = -Inf, upper = Inf,
    subdivisions = 200L), control)))
emit("subdivision limit", integrate(function(x) 1 / sqrt(x), 0, 1,
    subdivisions = 1L, rel.tol = 1e-14, abs.tol = 1e-14,
    stop.on.error = FALSE))
emit("finite roundoff", integrate(exp, 0, 1,
    subdivisions = 1000L, rel.tol = 1e-14, abs.tol = 1e-14,
    stop.on.error = FALSE))
emit("extrapolation roundoff", integrate(sin, 0, Inf,
    subdivisions = 200L, rel.tol = 1e-8, abs.tol = 1e-8,
    stop.on.error = FALSE))
emit("divergent", integrate(function(x) 1 / sqrt(x), 0, Inf,
    subdivisions = 200L, rel.tol = 1e-8, abs.tol = 1e-8,
    stop.on.error = FALSE))
