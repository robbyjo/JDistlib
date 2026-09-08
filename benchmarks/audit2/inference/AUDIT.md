# Inference, solvers, linear algebra, and RNG audit (2026-09-08)

Baseline: `4bdbb8a`. Inventory: 227 inference Java files, 3 matrix files,
52 accelerator API/CPU files, and the CUDA, OpenCL, Vulkan, and native CPU
provider modules. This is a risk-based audit of numerical and state-transition
paths, not a claim that every method in that inventory received an independent
reference test.

## Deterministic reference coverage

`generate-reference.R` produces 51 checked values in
`src/test/resources/jdistlib/inference/audit2-reference.csv` using the package
versions in `reference-versions.txt`. The posterior algorithm attribution and
BSD notice are in `THIRD_PARTY_NOTICES.md`. Run it from the repository root after
`benchmarks/audit2/install-reference-packages.R` installs the isolated R library.

- `posterior`: rank-normalized/folded split R-hat, bulk ESS, tail ESS, and
  mean MCSE on five four-chain fixtures: correlated, antithetic, skewed, tied,
  and odd-length draws. Constant chains are separately required to produce
  unavailable diagnostics and an unreliable result.
- `loo`: PSIS shape and normalized log weights for bounded, smooth, and heavy
  importance ratios; PSIS-LOO and WAIC ELPD, effective parameter count, and
  WAIC ELPD standard error on paired pointwise likelihoods. The PSIS reference
  assumes independent draws (`r_eff=1`), matching this Java API.
- `deSolve`: harmonic-oscillator trajectory, independently checked against
  sine/cosine. Tiny representable intervals are checked using constant
  derivatives in both explicit and stiff solvers.
- `numDeriv`: a smooth multivariate gradient, also checked analytically.
- R/base and `Matrix`: thin SVD and dense/sparse solves. Independent spectral
  identities and reconstruction tests cover FP64 global scales 1e-200 and
  1e200 and FP32 scales 1e-25 and 1e25.

`InferenceAudit2Test` contains these checks and scaled algebraic root tests.
Existing inference, sampler/restart, autodiff, sparse-matrix, solver, and CPU
accelerator suites supplement the new references. They are not R RNG bit
comparisons.

## Corrections

1. ESS used a different variogram estimator, paired the lags incorrectly for
   the modern estimator, and capped antithetic ESS at the draw count. It now
   uses multi-chain autocovariances and Geyer's positive/monotone sequences,
   including the final positive even lag. FFT autocovariances avoid quadratic
   work for long correlated chains. Ranking follows splitting, which matters
   for odd-length chains; tied upper-tail indicators now match quantile ESS.
   Mean MCSE uses mean ESS instead of rank ESS. Constant/nonfinite draws cannot
   become reassuring finite diagnostics through rank normalization.
2. The PSIS implementation was a Hill heuristic constrained to nonnegative
   shape, rather than a generalized Pareto tail fit. It now uses the
   empirical-Bayes inverse-scale grid and weak shape prior implemented by
   `posterior::gpdfit`, with the `loo` tail length and truncation rules. Fewer
   than five tail observations or a degenerate tail are reported as an
   unavailable fit (`k=Inf`). Non-tail log weights retain their logarithms
   instead of underflowing through unnecessary exponentiation.
3. Jacobi eigendecomposition stopping thresholds erased small matrices'
   off-diagonal structure; squared SVD products underflowed/overflowed at
   extreme scales. CPU and CUDA/OpenCL/Vulkan FP32/FP64 entry points now
   normalize the input and rescale eigenvalues/singular values. The vectors
   retain their normalization, and caller arrays are copied.
4. CPU and CUDA/OpenCL/Vulkan matrix-product accumulation now respects zero coefficients when
   the disabled operand contains NaN/Infinity (including beta=0 output buffers).
5. Explicit and stiff ODE solvers no longer reject a successfully completed
   tiny interval because its step is below the ULP at time 1. They check
   whether time can actually advance, and reject nonfinite controls/times.
6. Algebraic Newton solves no longer classify a nonsingular Jacobian as
   singular merely because all residuals were expressed in units below 1e-15.

## RNG audit

`RandomEngineAudit2Test` validates the corrected WELL44497b recurrence against
Apache Commons Math's independent implementation for 20,000 words, crossing
multiple complete state wraps. CMWC4096 is compared with a BigInteger unsigned
recurrence for 16,000 words. Four engines receive fixed-seed uniform/Gaussian
moment and range checks (200,000 draws per engine), invalid-bound checks,
clone/reseed/serialization comparisons with mixed draws and a populated
Gaussian cache, and inherited Java Random method checks.

- WELL construction previously called `setSeed` before its state existed.
  The recurrence used signed shifts and omitted the unused-bit state mask;
  Gaussian generation used only the positive quadrant; bounds multiplied the
  draw; clone and reseeding did not preserve/reset the complete state.
- CMWC previously began with an all-zero buffer, failed to retain unsigned
  32-bit words, and supplied only one recurrence word to `nextLong`. It now
  expands all 64 seed bits through SplitMix64 into 4096 words, uses carry
  362436, performs unsigned 32-bit recurrence arithmetic, and builds 64-bit
  draws from two words. Bounds, Gaussian draws, float endpoints, cloning,
  reseeding, and seed metadata are repaired.
- Inherited `Random` methods now obtain bits from the selected engine rather
  than an unrelated superclass LCG. Existing MersenneTwister and
  MersenneTwisterSafe cloning already copied the cursor and Gaussian cache;
  both passed the new invariants without changing their recurrences.

WELL/CMWC corrected seeded streams differ from older releases. Their old
serialized states are explicitly rejected (serialVersionUID 2), rather than
silently treating incompatible state as valid. Current-version checkpoint
round trips reproduce mixed draws exactly. These checks do not establish
cryptographic suitability or replace a comprehensive statistical RNG battery.

## Measured diagnostic performance

Median of five timed in-process runs after two warmups, Java 25 and R 4.6.1 on
the same host. `InferenceBenchmark.java` and `benchmark.R` compute comparable
summaries and modern diagnostics of four deterministic correlated chains.
Java includes sampler-report bookkeeping. R elapsed timing has coarser resolution.
The baseline was freshly compiled from all Java sources in `git archive 4bdbb8a`
into `build/audit2/pristine-main`; the mutable Gradle output was not used as
the baseline. See the parent audit README for the build commands.

| Draws per chain | Baseline Java | Corrected Java | R posterior |
| --- | ---: | ---: | ---: |
| 4,096 | 15.432 ms | 15.397 ms | 20 ms |
| 16,384 | 87.881 ms | 38.889 ms | 50 ms |

The larger case improved 2.26x over the baseline; the smaller case was effectively unchanged. These are workload-specific measurements, not a universal
speedup. The baseline estimator returned bulk ESS 28.426938721253617 on the
larger case; corrected Java returned 30.574094059945967, versus R
30.574094059946194. The estimator corrections are part of the comparison.

## Execution and limits

The parent full Gradle check and Javadoc build passed: 514 tests passed and
8 CUDA tests were skipped. The changed provider suites
executed 4 native CPU, 7 OpenCL, and 8 Vulkan tests without failures, including extreme-scale decomposition and zero-coefficient tests. All 8 CUDA tests were skipped because the
CUDA provider was unavailable; Java compilation does not establish CUDA
hardware correctness. No GPU-vs-R speed claim is made.

The language/compiler, every specialized sampler, nested-R-hat variants,
projection/stacking algorithms, DAE/higher-index equations, and every
accelerator operation were not exhaustively compared against external
implementations. Existing integration tests cover representative sampler and
restart invariants. The DAE solver remains fixed-output implicit Euler and the
stiff ODE solver remains first-order BDF with finite-difference Newton solves;
this audit does not claim they have the accuracy/performance of all deSolve
algorithms. PSIS uses r_eff=1 and does not estimate autocorrelation adjustments
for pointwise likelihood chains. Normality or statistical convergence is not
proved by the fixed-seed sampler checks.
