# Install reference implementations in an isolated audit library, never globally.
lib <- "build/audit2/Rlib"
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(normalizePath(lib), .Library), include.site=FALSE)
packages <- c("extraDistr", "actuar", "VGAM", "statmod", "mvtnorm", "copula",
              "posterior", "loo", "numDeriv", "deSolve", "evd", "VineCopula", "nortest", "moments")
available <- available.packages(repos="https://cloud.r-project.org", type="binary")
required <- unique(c(packages, unlist(tools::package_dependencies(packages, db=available, recursive=TRUE))))
present <- rownames(installed.packages(lib.loc=c(lib, .Library)))
missing <- setdiff(required, present)
if (length(missing)) install.packages(missing, lib=lib, repos="https://cloud.r-project.org",
                                    type="binary", Ncpus=2)
stopifnot(all(vapply(packages, requireNamespace, logical(1), quietly=TRUE)))
writeLines(c(R.version.string, capture.output(installed.packages(lib.loc=lib)[, c("Package", "Version")])),
           "build/audit2/reference-package-versions.txt")
