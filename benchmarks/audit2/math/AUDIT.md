# Numerical utilities and normality tests

Baseline: commit `4bdbb8a`, freshly compiled from `git archive`.
R 4.6.1, nortest 1.0-4, moments 0.14.1; Java 25, release-8 bytecode.

`generate-reference.R` produces 2,076 values: 1,820 kernel-density values,
all six bandwidth methods at three sample sizes and a 100,000-observation
pair-count overflow case, moments, Shapiro-Wilk, five additional normality
statistics and p-values, seven multiple-testing adjustments with missing
values, spline predictions, and scalar optimization references.
`MathUtilitiesAuditTest` checks these families plus independent extreme-scale,
interpolation and weighted/permuted spline invariants. Some ancillary summary
and raw R spline values are retained for inspection rather than asserted.

## Corrections and reference disagreements

- Epanechnikov and biweight kernels previously evaluated to zero because their
  leading constants used integer division. All seven kernels now match R's
  current FFT coordinates, including the R 4.4 correction. One FFT plan is
  reused for the three transforms. Invalid and nonfinite inputs are handled
  before binning; excluded observations retain their original missing mass.
- With explicit weights and infinite observations, R's `density` applies the
  finite mass twice. The independent reference here is the finite subset with
  its original weights (`subdensity=TRUE`), applying that mass once. This is
  an intentional disagreement with the direct infinite-observation R call.
- UCV/BCV/SJ used quadratic observation-pair loops and 32-bit pair counters.
  Histogram pair counting reduces work to O(n + nb^2), prevents count overflow,
  and avoids integer-bin saturation on translated data. SJ bracket expansion
  now follows the alternating upper/lower sequence used by R.
- Mean uses compensated summation with an overflow fallback; SD/variance and
  RMS use centered/scaled calculations. Tests include offsets of 1e12,
  opposing values of 1e300 with a 1e-100 residual, subnormals, and scales of
  1e-200 and 1e200. Nonrepresentable variances can still overflow normally.
  A rare overflow-plus-cancellation fallback sums exact binary64 values with
  BigDecimal so scaling cannot erase a representable tiny residual. Ordinary
  data and same-sign overflow cases continue to use double arithmetic.
- Constant interpolation had reversed endpoint weights. Smoothing splines
  modified caller weights and used unsorted responses/weights when computing
  the final cross-validation score. Duplicate-x permutation tests protect both.
- The spline roughness integral correctly uses 1/3; R `smooth.spline` retains
  a historical 0.3330 coefficient. An independent cubic B-spline design and
  exact piecewise Simpson integration confirms Java's predictions within
  2e-9. The raw R predictions can differ by about 7e-4 on this fixture and
  were not adopted as ground truth.
- Jarque-Bera and D'Agostino-Pearson p-values selected the wrong chi-square
  tail. The statistics used unstable raw moments and were capped at 50.
  Central scaled moments and the signed skew/kurtosis transforms now agree
  with `moments` on ordinary, translated, skewed, and outlier-heavy data.
  Tiny Jarque-Bera p-values use the independent exp(-statistic/2) identity;
  `moments::jarque.test` can lose them through subtraction from one.
- Anderson-Darling now uses normal log tails directly. AD, Cramer-von Mises
  and Lilliefors use stable mean/SD. All five new normality statistics retain
  affine invariance even when a centered difference or sample SD exceeds
  MAX_VALUE; this is tested against the corresponding unit-scale data.
  Undefined constant-data Lilliefors statistics return NaN.
  The AD/CVM p-value fits are restricted to
  their published approximation domains, retaining `nortest`'s respective
  3.7e-24 and 7.37e-10 reporting floors beyond them. These floors are bounds
  imposed by the approximation, not accurate extreme p-values.
- The log harmonic sum used log(1) instead of log(0) as its initial value,
  biasing every result by an extra unit before taking the logarithm. Scalar
  Zipf regressions independently verify the corrected normalization.

The D'Agostino-Pearson combination is the squared sum of skewness and kurtosis
z-scores, as documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html).
R package sources were inspected locally; their versioned functions are used
by the reference generator. Sample-size assumptions of asymptotic normality
tests remain applicable.

## Measured speed

Median of five runs after three warmups; Java time per call. Bandwidth batches
contain five calls, density batches 200; R batches contain 500 calls.
The deterministic input contains 20,000 observations. Density has 512 grid
points, effective bandwidth 0.4, bounds [-4,5]. Java's legacy `width` argument
is bandwidth times the kernel factor, which is accounted for in the harness.

| Operation | Baseline Java ms | Corrected Java ms | R ms |
| --- | ---: | ---: | ---: |
| SJ direct plug-in | 169.81624 | 1.13460 | 0.88 |
| Unbiased cross-validation bandwidth | 172.66684 | 0.36080 | 0.96 |
| Gaussian density | 0.1926585 | 0.2017610 | 0.40 |

Bandwidth selection improved about 150x and 479x over the baseline. Density
was about 5% slower despite eliminating duplicate FFT plans: input validation,
filtering and corrected coordinates are included. Its corrected midpoint
0.37637343542511453 agrees with R; baseline was 0.37674471896538203.
No universal speed advantage over R is claimed.

Run `MathBenchmark.java` with either the pristine classes or current
`build/classes/java/main` first on the classpath, followed by the all-in-one
distribution JAR for FFT dependencies. Run `benchmark.R` separately. Timings
are not CI pass/fail thresholds.
