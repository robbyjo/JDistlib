# Remaining scalar distributions audit, 2026-09-08

Baseline: `4bdbb8af27bd5ae940acb5ca9f3a829d7d64e607`. This is the scalar-distribution part of the second audit; other audit directories cover inference, joint distributions, finance and numerical utilities.

## Accuracy evidence

`reference.R` produces **30,710 d/p/q values for 36 families**, using R 4.6.1, extraDistr 1.10.0.5, actuar 3.3-7, VGAM 1.1-14, evd 2.3-7.1 and statmod 1.5.2. The saved `scalar-summary.txt` identifies the family counts. Not every package implements every operation: absent functions are explicitly skipped. Arcsine, chi and Levy use independent transformed beta/chi-square identities in base R. Both requested tails and both probability representations are exercised where the operation supports them.

The initial comparison tolerance is relative `2e-9`, except inverse Gaussian `5e-8`. **28,921 values agreed at that tolerance; 1,789 discrepancies were investigated.** The independent `adjudicate.py` uses mpmath at 80 decimal digits with the actual binary64 inputs. Quantile checks use either an independent analytic inverse or an independent CDF inversion residual; discrete quantiles check the two adjacent CDF steps.

At the adjudication tolerance of `2e-8`, **1,116 discrepancies favor Java and 673 have both implementations within tolerance**. There are **zero unresolved or remaining Java discrepancies on this grid**. Log-value comparisons use an absolute floor of one, and near-zero location-shifted EVD quantiles allow absolute binary64 cancellation (`2e-14`). These criteria and every classification are executable in the script. This is not a claim of uniform `2e-8` accuracy over all real parameter values.

“R discrepancy” in the saved output includes cancellation, underflow, reference evaluation errors and differing density-at-boundary conventions, not 1,116 distinct R bugs. Examples:

* evd upper-tail quantiles often subtract the requested probability from one. For GEV(0,1,1), an upper probability `1e-12` has a finite quantile near `1e12`; decimal-to-binary subtraction changes the R result by about `2.2e-5` relatively. Java retains the requested tail with `log1p`.
* extraDistr log-density and log-CDF wrappers sometimes log an already underflowed ordinary result. Java retains finite logs.
* statmod inverse-Gaussian quantile evaluation errors occur at some tested small-dispersion/extreme-probability inputs. They are retained as missing R values and checked by independent inversion rather than copied into Java.
* evd defines the generalized Pareto density as zero at the lower boundary. Java returns the finite right-hand density limit `1/scale`, as it does for the exponential limit; this convention has no effect on probabilities.

## Confirmed corrections

* **Arcsine:** reversed/logarithmically incorrect CDF and inaccurate bisection quantiles replaced by the exact arcsine integral and inverse; support and requested tails fixed.
* **Chi / inverse gamma:** squared/reciprocal transformations now include the density Jacobian and correct CDF direction. Inverse gamma preserves the existing reciprocal-Gamma sampling parameterization: its public `scale` is the underlying Gamma scale, so conventional inverse-gamma scale is its reciprocal. This is now documented in the class.
* **Inverse Gaussian (`InvNormal`):** integer division `3/2` had given the wrong density exponent. Fixed support, logged quantile inputs, endpoint termination, normal-tail cancellation, and replaced inverse-CDF sampling with the Michael–Schucany–Haas transformation.
* **Kumaraswamy / log-logistic:** corrected reversed or broken quantiles, negative ordinary survival probabilities, support/log boundary behavior and object APIs that cast continuous observations to integers.
* **Levy:** corrected survival probabilities greater than one, reciprocal-transform tail direction and quantile boundaries.
* **EVD:** fixed Fréchet support/sign errors, generalized Pareto quantile and zero-shape sampling formulas, ordinary densities returning negative infinity outside support, Rayleigh boundary/tail handling, and missing logged-tail support. GEV uses `log1p`/`expm1` through the small-shape limit. Existing static signatures remain available; logged overloads were added.
* **Extreme / Order:** order-statistic CDFs and quantiles now follow the beta law of the transformed base CDF; removed the unbounded squared-objective root search and integer rounding of continuous quantiles. Corrected maximum/minimum tail orientation and log densities.
* **Beta prime / GB2 / Feller–Pareto:** retain the smaller beta argument/quantile tail, avoid exponentiating large positive log odds, and compute finite extreme log tails without rounding the beta argument to one.
* **Beta-binomial:** quantile and random routines had passed parameters in the wrong order. Fixed support, integer-CDF flooring and log inputs; evaluate successive masses by recurrence and sample a beta–binomial mixture.
* **Zipf:** fixed the missing midpoint division that could hang quantiles, incorrect upper-tail boundary/log sum handling and fractional-mass acceptance. The separate math audit also fixed `lgharmonic`'s extra unit of mass.
* **Logarithmic:** quantiles had started summing at zero and therefore returned zero; corrected positive-integer support, log probabilities and directly evaluated small upper tails.
* **Positive/half normal and half-t:** retain very small positive quantiles near zero using local analytic expansions. Positive-normal inversion now calls the normal inverse directly instead of repeatedly evaluating a truncated CDF.
* **Discrete Weibull:** quantile corrections preserve logged or directly requested upper probabilities, including a tested adjacent-step boundary.
* **Poisson–inverse Gaussian:** rationalized the zero-mass exponent; added log-space mass recurrence and directly summed small upper tails for the tested finite-dispersion range.
* **Skew-t, slash, Tukey lambda:** preserve tail/log probability information through inversion; slash keeps finite extreme log densities, and Tukey lambda retains its logistic limit.
* **Truncation / mixtures / censoring:** evaluate requested log tails directly. Truncated normals on `[10,11]` and `[40,41]` now work even when ordinary CDF subtraction or retained probability underflows; `getLogRetainedProbability()` exposes the retained log mass. Mixtures preserve log quantiles and infinite log densities. Generic scalar bisection terminates at adjacent doubles while allowing enough iterations for subnormal-scale roots.

`ScalarRemainingAuditTest` contains **21 regression methods**, including independent densities and transforms, normal/Student-t/logistic limiting identities, order-statistic beta identities, extreme logged tails, bounded quantile termination, and deterministic seeded sampler-moment checks. The parent audit runs these alongside the existing numerical-construction and distribution suites.

## Measured speed

Single-thread best of five timed trials after two warmups, Java 25, 20,000 calls/trial. Baseline was compiled from the complete pristine git archive into `build/audit2/pristine-main`, placed before candidate classes. R uses vectorized calls, 200,000 values repeated 50 times per timed trial to avoid timer quantization; two warmups and five timed trials. Times exclude setup and are workload-specific.

| Operation / parameters | Baseline Java ns/value | Fixed Java ns/value | Java improvement | Vectorized R ns/value |
|---|---:|---:|---:|---:|
| InvNormal.random, mean=2, sigma=.5 | 1679.995 | 36.470 | 46.06x | 100 |
| PositiveNormal.quantile, mean=-1, sd=2 | 9086.690 | 85.740 | 105.98x | 39 |
| BetaBinomial.cumulative, n=100, mu=.3, sigma=.2 | 6638.730 | 1407.695 | 4.72x | 45 |

R remains faster for the last two workloads. This audit does not claim Java is universally faster; the beta-binomial scalar CDF still performs work proportional to the requested count range. Saved timings and `ScalarSpeed.java` / `speed.R` make these measurements reproducible.

## Scope and retained limits

The 36 R-grid families are listed explicitly in `scalar-summary.txt`. EVD Order/Extreme, Zipf, skew-t, Tukey lambda and construction wrappers additionally have independent executable identity/regression coverage. General numerical construction, analysis, table interpolation, rejection envelopes, piecewise/certified discrete distributions and the generic batch API received source review and the existing project tests, not an exhaustive new R differential grid. Other scalar families outside the saved family list should not be described as having received new package-level differential validation in this audit.

Finite grids cannot certify every extreme parameter combination. The explicit logarithmic quantile overload retains its caller-specified finite search maximum, and the default remains 10,000; it reports NaN if that search range is insufficient. Extremely large/ill-conditioned shape or scale combinations can still exceed intermediate representability. Poisson–inverse Gaussian direct survival summation is enabled for `2*dispersion*mean^2 < 1000`; the broader overdispersion range retains the original complementary-CDF path. Random moment checks test the selected parameter settings and do not establish statistical quality for every generator or parameter regime.

## Reproduction

Run from repository root after building Java classes. Install the package versions above into `build/audit2/Rlib`, or adjust the first `.libPaths` line for another local library. mpmath must be on Python's import path.

```text
Rscript benchmarks/audit2/scalar/reference.R
javac -cp build/classes/java/main -d build/audit2/scalar-harness benchmarks/audit2/scalar/ScalarReferenceCheck.java benchmarks/audit2/scalar/ScalarSpeed.java
java -cp "build/audit2/scalar-harness;build/classes/java/main" ScalarReferenceCheck build/audit2/scalar-reference.tsv build/audit2/scalar-failures.tsv
python benchmarks/audit2/scalar/adjudicate.py
java -cp "build/audit2/scalar-harness;build/classes/java/main" ScalarSpeed 20000
Rscript benchmarks/audit2/scalar/speed.R
```

The Java classpath separator shown is for Windows. Substitute `:` on Unix. The adjudicator writes `build/audit2/scalar-adjudication.tsv` with every discrepant row, reference result, Java result, classification and high-precision check. Fixture-generation scripts and compact result summaries are committed; large generated TSV files stay under `build/`.
