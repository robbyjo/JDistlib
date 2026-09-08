# Remaining-components numerical audit — 2026-09-08

This follows the [core numerical audit](NUMERICAL_AUDIT_2026-09-08.md) at
commit `4bdbb8a`. It fixes confirmed errors in contributed distributions,
extreme-value distributions, composition wrappers, copulas, compound counts,
statistical utilities, inference diagnostics, numerical solvers, linear
algebra and random engines. Comparisons use R 4.6.1 and package implementations,
with analytic or high-precision checks when R disagrees.

This is a risk-based review with reproducible numerical coverage. It does not
certify every method or every parameter combination in this large repository.
The component reports distinguish tested paths from remaining limitations.

## Coverage and evidence

| Area | Reference evidence | Details and reproduction |
| --- | --- | --- |
| Contributed and extreme-value scalar distributions | 30,710 density/CDF/quantile comparisons across 36 families; 80-digit mpmath adjudication of discrepancies; support, endpoint, inversion and sampling regressions | [Scalar report](../benchmarks/audit2/scalar/AUDIT.md) |
| Copulas, multivariate laws and financial constructions | 10,905 comparisons using R copula, VineCopula, mvtnorm, extraDistr and independent compound-count sums; 8,861 frozen trusted assertions plus independent extreme-parameter checks | [Joint report](../benchmarks/audit2/joint/AUDIT.md) |
| Density estimation, bandwidths, moments, normality, multiple testing, splines and optimization | 2,076 generated R/reference values, with independent spline Gram integration, tail identities and permutation invariants | [Math report](../benchmarks/audit2/math/AUDIT.md) |
| Inference, solvers and linear algebra | 51 frozen R values using posterior, loo, deSolve, numDeriv, base R and Matrix; analytic derivatives/trajectories, scaled reconstructions and checkpoint tests | [Inference report](../benchmarks/audit2/inference/AUDIT.md) |
| Random engines | WELL against Apache Commons Math across 20,000 words; CMWC against an independent unsigned BigInteger recurrence across 16,000 words; four-engine range, moment, cloning, reseeding and serialization tests | [RNG evidence](../benchmarks/audit2/inference/AUDIT.md#rng-audit) |

Reference packages are installed into `build/audit2/Rlib` by
`benchmarks/audit2/install-reference-packages.R`. The audit records actual
package versions; package installation is separate from the ordinary test
suite, which runs frozen fixtures without needing R or network access.

## Principal corrections

- Corrected transformation Jacobians, support boundaries, lower/upper and log
  tails, endpoint quantiles and random generation across the contributed and
  extreme-value families. Removed broken discrete quantile loops and incorrect
  parameter ordering. Truncated and mixture laws retain representable extreme
  tails instead of subtracting CDFs that have both rounded to one.
- Replaced finite-difference Joe/BB1 copula formulas with analytic evaluations;
  stabilized strong-dependence calculations and sampling. Bivariate Gaussian
  and Student copulas now integrate their conditional distributions adaptively.
  Fixed large-parameter multivariate normalization cancellation and compound
  count CDFs that repeatedly recomputed whole density sums.
- Repaired two zero-valued density kernels, corrected FFT coordinates and
  weighted finite mass, and replaced quadratic bandwidth pair counting with
  histogram counting. Stabilized moments and normality statistics, corrected
  reversed chi-square p-values, and preserved spline caller weights/order.
- Corrected rank-based MCMC diagnostics and mean MCSE against `posterior`.
  PSIS now fits a generalized Pareto tail, including negative shape, instead
  of the previous nonnegative Hill heuristic. FFT autocovariances improve the
  long-chain workload.
- Normalized extreme-scale eigen/SVD inputs before decomposition and rescaled
  their outputs. Fixed zero-coefficient BLAS handling and numerical solver
  termination/scaled-pivot issues.
- Repaired WELL/CMWC initialization, recurrences, bounded draws, Gaussian
  generation and complete engine state handling. **Their corrected seeded
  streams change. Old WELL/CMWC serialized states are explicitly rejected.**
  Mersenne Twister's core recurrence is unchanged.

## R was checked, not assumed correct

The scalar report records every strict discrepancy and the independent
adjudication method, including quantile inversion and discrete inequalities.
Other examples include Joe copula overflow in R at strong dependence, Student
multivariate density normalization at enormous degrees of freedom, explicit
infinite-observation density weights, and R's historical spline roughness
coefficient. The implementation keeps the independently supported result.

Normality p-value approximations have finite validity domains. AD/CVM retain
the documented numerical reporting floors used by `nortest`; those floors
are not precise extreme-tail probabilities. Ordinary and extreme-tail
probability assertions are separated in the regression evidence.

## Performance

Benchmarks use warmed Java processes and separate R processes on the same
host. The Java baseline is freshly compiled from `git archive 4bdbb8a`.
R sometimes uses vectorized package calls while Java uses scalar calls; the
individual reports identify the workload and timing unit. There are no CI
timing thresholds or claims of universal superiority to R.

At 20,000 observations, SJ direct plug-in bandwidth selection improved from
169.816 ms to 1.135 ms (about 150x), and UCV from 172.667 ms to 0.361 ms
(about 479x). R took 0.88 ms and 0.96 ms respectively. Corrected Gaussian
density took 0.202 ms versus 0.193 ms before and 0.40 ms in R; its accuracy
and validation changes produced a small slowdown.

The component reports contain the additional scalar, copula, compound-count,
sampling and inference timing tables, including slower cases and the exact
R/Java workload definitions.
Examples include 33–79x faster compound-count CDFs, 106x faster positive-normal
quantiles, and a 2.26x improvement in diagnostics for four chains of 16,384
draws. The latter took 38.889 ms in Java and 50 ms in R. Positive-normal
quantiles and beta-binomial CDFs remain slower than vectorized R on the tested
workloads despite improving over the old Java implementation.

## Validation and limits

Validation command:

```powershell
.\gradlew.bat check javadoc
```

This runs the core and provider tests, documentation examples, packaging
smoke checks and Javadoc generation. The integrated run passed **514 tests**:
495 core, 7 OpenCL, 8 Vulkan and 4 native CPU. All **8 CUDA tests were skipped**
because the provider was unavailable. The all-in-one JAR smoke check detected
four backends; 92 JDistlib/Stan model scripts, distribution catalogs, modular
POMs and release documentation were validated. Javadoc generation passed.

Higher-dimensional CDFs still use numerical integration with finite budgets.
Fixed-grid Fourier financial families and specialized inference/compiler paths
have not all received broad external parameter grids. First-order stiff/DAE
solvers are not substitutes for every algorithm in deSolve. PSIS assumes
independent importance ratios (`r_eff=1`). Fixed-seed RNG checks do not replace
a comprehensive statistical test battery. CUDA hardware validation depends
on provider availability and is explicitly distinguished from compilation.
