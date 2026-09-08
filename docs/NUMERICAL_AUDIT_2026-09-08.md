# Accuracy and performance audit — September 8, 2026

Baseline: Git commit `067d9c5`. Comparator: R 4.6.1 (2026-06-24 ucrt),
Windows amd64. Java: Oracle HotSpot 25+37, compiled for Java 8. Host:
Intel Core i9-14900K. The audit fixes errors in nine distribution classes and
shared log-space arithmetic, and removes repeated allocation of immutable
gamma/Stirling coefficient tables.

R was used as a comparator, not as numerical ground truth. Analytic identities
and 100-digit mpmath references identify several cases where both implementations
previously shared a bug. Existing regression expectations that preserved those
R defects were corrected with explicit explanations.

## Coverage and results

* A reproducible grid compares **153,600 values**: 15 families, density/CDF/
  quantile, both tails, and ordinary/log representations. Parameters and grids
  are in [`compare-r.R`](../benchmarks/audit/compare-r.R).
* The central-family grid's largest relative difference from R is
  **2.793e-13**. The screen uses 5e-10 for central and 2e-7 for noncentral
  values, relative to the magnitude of the result, including tiny log values.
* All **584** screened discrepancies are noncentral-beta CDFs, where JDistlib
  already used a more accurate algorithm before this audit. Every discrepancy
  was resolved against a separately summed Poisson/incomplete-beta mixture at
  80 and 120 decimal digits. Java was closer in all 584 cases; its largest
  relative probability error was **1.552e-14**. The omitted-mass bound was below
  9.75e-112. See the [reference summary](../benchmarks/audit/ncbeta-reference-summary.json)
  and [validator](../benchmarks/audit/validate-ncbeta.py).
* New tests exercise subnormal scales/noncentralities, underflowed densities
  whose scale makes the final answer representable, infinite endpoints,
  zero-degree noncentral chi-square, log tails, and negative-binomial limits.
  They supplement the ordinary grid and existing tests.
* Full `gradlew.bat check` passes: **451 tests pass, six CUDA tests skip**
  because that backend is unavailable on this host. This includes distribution/inference tests,
  accelerator modules, Maven metadata checks, documentation example compilation,
  92 model scripts, and the all-in-one JAR smoke check. API documentation also
  builds successfully. No API signatures or dependency versions changed.

This is a targeted numerical audit plus a full regression run, not an exhaustive
proof of every distribution, RNG, inference engine or accelerator implementation.
The grid uses one parameter set per family. Timing does not assess GPUs, random
sampling, inference workloads, cold startup or allocation-heavy applications.

## Correctness fixes

| Area | Problem and corrected behavior |
|---|---|
| Exponential | The upper CDF at/below zero returned zero instead of one. Log CDFs and scaled densities now survive intermediate underflow. |
| Gamma | Dividing positive subnormal inputs by scale erased them before density/CDF evaluation. Log-domain fallbacks preserve the result and handle tiny shape/density factors. |
| Weibull | Intermediate powers could underflow or overflow while the requested log density, log CDF or scaled quantile remained representable. Exceptional paths now use log hazards. |
| Lognormal, Cauchy, logistic | Products, squared standardized distances and unscaled exponentials could overflow/underflow before taking logarithms or applying scale. Stable fallbacks retain finite results. |
| Noncentral chi-square density | Unscaled mixture terms underflowed, causing an inaccurate central approximation; absolute truncation also damaged relative accuracy. A scaled mixture with a relative remainder criterion retains tails and supports zero degrees of freedom. Subnormal Poisson means retain their unrounded logarithm. |
| Noncentral chi-square CDF | For small noncentrality, stopping by omitted Poisson mass could omit components that dominate a tiny upper tail. A log-domain mixture now covers the bounded far-upper-tail regime described below. |
| Negative binomial | Small count relative to size did not justify replacing the zero-mass term by `-mu`. The exact stable term is retained; cases with significant `x*x/size` use a symmetric binomial representation. |
| Noncentral F | Infinite denominator degrees multiplied a log density by the Jacobian instead of adding its logarithm. Infinity endpoints and invalid random noncentrality are also corrected. |
| Log-space helpers | Sums/differences of zero masses and sums of infinite masses could return NaN. Correct limits now preserve NaN propagation for genuinely invalid inputs. |

Representative independent references:

| Calculation | R 4.6.1 | Verified result |
|---|---:|---:|
| `dgamma(2^-1074, .5, scale=2, log=TRUE)` | `-Inf` | `-.5*log(2*pi*x)`, approximately 371.3011 |
| `dcauchy(1e200, log=TRUE)` | `-Inf` | approximately -922.1787671 |
| `dchisq(10000, 1, ncp=10, log=TRUE)` | -2605.522741 | -4694.989489882915 |
| `dchisq(17, 4.5, ncp=80, log=TRUE)` | -16.023767809163 | -16.023767798131 |
| `pchisq(1736.00391113281, 4.5, ncp=12, lower.tail=FALSE, log.p=TRUE)` | -755.886315043866 | -729.882167817043 |
| `dnbinom(1, size=1e12, mu=1e12, log=TRUE)` | -999999999973.062 | -693147180533.0074 |
| `pbeta(.9985126953125, 2.5, 7, ncp=11, lower.tail=FALSE)` | 2.2549703e-10 | 1.3569202528333144e-16 |

The noncentral-beta improvement in the final row predates this audit and was
retained. Density references use the modified-Bessel identity and the square of
a shifted normal; CDF references sum incomplete-gamma mixtures independently.
The negative-binomial references use the log-gamma definition. Reproduce them
with [`generate-ncchisq-density-high-precision.py`](../src/test/python/generate-ncchisq-density-high-precision.py).
The continuous regressions use closed forms and asymptotic terms whose omitted
contribution is below binary64 precision, without using R outputs as expected
values.

## Speed

The [matched benchmark](../benchmarks/audit/README.md) uses vectorized R calls
and warmed scalar Java loops over identical inputs. Timed regions exclude
startup, parsing and I/O. R's batches are at least 200 ms to reduce clock
quantization; Java uses its nanosecond clock. Results are five-sample medians.
All 150 workload measurements, including slower cases and before/after
comparisons, are in [`results-2026-09-08.csv`](../benchmarks/audit/results-2026-09-08.csv).

Representative ordinary-density results (nanoseconds per value):

| Family | R | Java after fixes | R time / Java time |
|---|---:|---:|---:|
| Normal | 38.1 | 6.6 | 5.75x |
| Gamma | 162.1 | 87.2 | 1.86x |
| Beta | 157.4 | 90.1 | 1.75x |
| Poisson | 152.6 | 46.3 | 3.29x |
| Noncentral chi-square | 314.7 | 172.5 | 1.82x |
| Noncentral beta | 314.7 | 1029.3 | 0.31x |

Ratios greater than one favor Java. Java was faster in 131 of the 150 measured
workloads; this count includes near-ties and is only descriptive of this run.

The separate three-fork allocation benchmark isolates the coefficient-table
change from the correctness fixes:

| Workload | Before ns/call | After ns/call | Speedup |
|---|---:|---:|---:|
| Gamma function | 28.29 | 17.14 | 1.65x |
| Log gamma | 35.59 | 29.39 | 1.21x |
| Stirling correction | 30.22 | 5.58 | 5.41x |
| Gamma density | 107.14 | 83.53 | 1.28x |
| Poisson density | 48.83 | 45.04 | 1.08x |
| Beta density | 239.69 | 147.82 | 1.62x |

Gamma, Stirling and beta-density benchmark allocations drop from approximately
352, 433 and 2,304 bytes/call to zero. Before/after fingerprints match across
458,752 values, preserving the arithmetic exactly. Raw measurements and
reproduction details are in [`benchmarks/numerical`](../benchmarks/numerical/README.md).

Java is not uniformly faster than R. Beta/F quantiles and noncentral-beta
operations remain slower in the measured grid. Exceptional correctness paths
can also cost more; the audit does not trade a verified answer for a quicker
approximation. These measurements are workload/host-specific, and small timing
differences should not be treated as significant.

Beta quantiles retain a second safeguarded polishing solve; F quantiles delegate
to beta quantiles. It costs normalization and repeated CDF evaluations, but
protects skewed-shape and extreme-tail accuracy covered by `BetaPrecisionTest`.
No removal or relaxation of that solve was supported by paired accuracy/timing
evidence in this audit.

## Remaining numerical limits

The new noncentral chi-square upper-tail CDF path is deliberately bounded to
`0 < ncp < 80`, `x < 1e8`, and points more than eight standard deviations above
the mean (plus an explicit subnormal-noncentrality case). Other existing AS275
regimes retain their prior limitations; large-noncentrality extreme tails and
extreme log quantiles are not certified by this audit. At mixture indices beyond
`2^52`, noncentral chi-square density still uses a moment approximation, now
including the density Jacobian. Those astronomical-parameter results are not
claimed to be uniformly accurate. No claim is made that R agreement elsewhere
establishes mathematical correctness.
