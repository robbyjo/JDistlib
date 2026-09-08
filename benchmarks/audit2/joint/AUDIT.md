# Joint distributions, copulas, and finance audit

Baseline: `4bdbb8af27bd5ae940acb5ca9f3a829d7d64e607`. R 4.6.1; copula
1.1-7, VineCopula 2.6.1, mvtnorm 1.4-2, extraDistr 1.10.0.5. Base R's
Poisson and negative-binomial functions provide independently assembled
compound-count references. VGAM and actuar were also inspected, but no
matching Delaporte/Polya-Aeppli density implementation was available in those
installed namespaces, so their names must not be credited for those checks.

## Changes and evidence

* **Joe and BB1 copulas:** replaced fixed-width differences of four CDF
  evaluations with analytic log densities. The old method cancels to zero
  and clips to `Double.MIN_NORMAL` in tails. Joe's CDF and BB1's generator
  now use logarithmic arithmetic; their pair conditional CDFs are analytic.
  Joe Kendall tau uses the digamma identity with a series across its removable
  singularity at theta=2, replacing 8,192-panel numerical integration.
* **Clayton and Frank:** strong-dependence CDF/density/conditional calculations
  avoid exponent overflow. At theta=1000, the old Clayton gamma frailty
  underflowed and the old Frank logarithmic frailty overflowed, distorting
  uniform margins. Clayton now samples the logarithm of its frailty using
  gamma shape augmentation; strong bivariate Frank uses exact conditional
  inversion. A seeded 10,000-draw marginal test guards each sampler.
* **Gaussian and Student-t copula CDFs:** bivariate probabilities now integrate
  the analytic conditional CDF with adaptive Simpson quadrature. This removes
  183 discrepancies from the old default randomized multivariate integration
  on the comparison grid, and enforces the exact Frechet bounds. Higher
  dimensions retain their existing randomized probability algorithms.
* **Multivariate Student t and its copula:** the log normalizer uses integer
  gamma recurrences and a stable `lbeta` half-step instead of subtracting huge
  `lgamma` values. This restores the normal limit at df=1e16 in dimensions
  1, 2, 3, 5, and 10. The previous normalization could be wrong by tens of
  log-density units.
* **Multivariate hypergeometric and Dirichlet-multinomial:** stable `lchoose`
  and gamma increments remove cancellation for billion-sized populations and
  concentration parameters of order 1e16. The shared exact conditional
  probability calculation receives the same gamma-increment correction.
* **ConditionalDistribution:** normalizing masses, densities, CDFs, and
  quantiles use log probabilities and the appropriate tail. Normal
  conditioning on (40,41] and [-41,-40] no longer fails because both ordinary
  CDFs round to the same number.
* **OrderStatisticDistribution:** preserves logarithmic tails, computes
  declared atom probabilities by CDF jumps, and generates an extremum with
  one inverse transform instead of `count` base draws. Tiny complementary
  probabilities are retained even when their ordinary value underflows.
* **Delaporte and Polya-Aeppli:** CDFs condition directly on the cluster count,
  reducing quadratic nested density sums to one sum and evaluating upper tails
  directly. Polya-Aeppli density uses a log-term recurrence and its generator
  draws the sum of geometric clusters as a negative-binomial count. NaN,
  infinity, degenerate parameters, and quantile endpoints are handled explicitly.
* **StableDistribution and EmpiricalDistribution:** exact normal/Cauchy/Levy
  stable specializations preserve log/tail flags; one-sided stable support is
  exposed correctly and the Gaussian specialization admits all finite moment
  orders. Empirical NaN and upper-tail handling are corrected.

## Accuracy accounting

The full reproducible grid compares **10,905 scalar values**:

| Reference | Values | Result |
|---|---:|---|
| copula: six bivariate families, CDF/log density/Kendall tau | 7,533 | 89 R arithmetic failures independently adjudicated; all others agree |
| mvtnorm: normal and t log densities, dimensions 1/2/3/5/10 | 300 | 35 R `dmvt` gamma-subtraction discrepancies; remaining values agree |
| VineCopula: BB1 CDF/log density/conditional CDF | 1,296 | all agree within 2e-7 absolute |
| Base R mixtures: Delaporte and Polya-Aeppli log density/lower/upper CDF | 1,728 | all agree within 2e-8 absolute in log space |
| extraDistr: Dirichlet log density | 48 | all agree within 1e-10 |

The copula grid spans coordinates 1e-12 through 1-1e-12, and dependence
parameters through 1000 where the family permits it. The comparison tolerance
is 2e-7 for CDFs, 1e-7 for log densities, and 2e-10 for Kendall tau. The
trusted subset is frozen in `src/test/resources/jdistlib/audit2-joint.tsv`:
**8,861 scalar assertions**, run without R by `JointRReferenceTest`.
`JointAuditRegressionTest` adds 12 tests for independent identities,
normal limits, density integration, sampling, rare tails, and boundaries.
Existing copula, multivariate, Wishart, Finance090, and FinanceFollowup suites
also passed in focused runs; use the parent audit's final full-check count.

R was explicitly not treated as an unquestionable oracle:

* `pCopula(claytonCopula(1000))` returns zero on 48 comparison points with
  representable nonzero probabilities. On the diagonal the independent exact
  expression is `u * (2-u^theta)^(-1/theta)`; Java returns 0.399722837196181
  for u=0.4.
* Strong Gumbel/Joe implementations in R underflow their powered generators.
  Joe(theta=1000,u=v=0.7) returns 1 in R, which exceeds either marginal. The
  correct value is approximately 0.699791983761226. R also produces NaN log
  densities at several such Joe points. There are three Gumbel and 38 Joe
  discrepancies in this grid.
* `mvtnorm::dmvt` loses precision in its gamma subtraction: for df=1e16,
  dimension 1 and x=0 it reports -32 instead of the normal-limit value
  -0.91893853320467. The exact dimension-2 gamma recurrence and the limiting
  normal law independently resolve these disagreements. Ten less severe
  discrepancies occur at df=1e8, and 25 at df=1e16.

## Performance

Milliseconds per indicated batch; Java reports five-run medians after two
warmups. The benchmark's old classes were compiled from a **fresh git archive
of the baseline** into `build/audit2/joint-baseline-main`; early copied build
classes were discovered to contain edits and their first copula timings were
discarded. The parent subsequently also built a complete pristine baseline.
R uses its bulk package APIs where available, and vectorized base-R mixtures
for compound counts. These are workload measurements, not universal speedups.

| Operation | Calls | Baseline Java ms | Fixed Java ms | Speedup | R ms |
|---|---:|---:|---:|---:|---:|
| Joe log density | 20,000 | 3.0190 | 2.4906 | 1.21x | 48 |
| BB1 log density | 20,000 | 9.8622 | 3.3008 | 2.99x | 11 |
| Joe Kendall tau | 1,000 | 190.7376 | 0.4738 | 402.6x | 6 |
| Polya-Aeppli CDF | 100 | 134.0147 | 4.1046 | 32.65x | 9 |
| Delaporte CDF | 100 | 288.3793 | 3.6694 | 78.59x | 9 |
| Polya-Aeppli log density | 1,000 | 11.0599 | 5.4946 | 2.01x | 64 |
| Gaussian copula CDF | 100 | 24.0786 | 1.6833 | 14.30x | 9 |
| Student-t copula CDF | 100 | 8667.3301 | 21.5921 | 401.4x | 9 |
| Maximum of 1,000 normals: sampling | 1,000 | 11.4696 | 0.1122 | 102.2x | below timer resolution |

The Student-t CDF remains slower than R's specialized implementation, despite
its large improvement. The sampler's draw sequence changes, preserving its
distribution rather than the former number of RNG calls.

## Inventory and limits

This pass covers the joint/copula/discrete-multivariate density families and
the exact finance wrappers above. Existing regression coverage was rerun for
Wishart probabilities, multivariate lognormal/Laplace/power-exponential,
Dirichlet rectangles, bivariate Poisson/logistic, C-vines/D-vines, copula
selection/fitting, and mixed marginals. Those reruns and source inspection are
not new independent R certifications of every function or parameter regime.

The finance directory was inventoried: transforms (CGMY, tempered stable,
variance gamma, Meixner, GH/NIG/GIG), risk measures, aggregation, option
calibration/implied distributions, and inference retain their existing
numerical/Monte Carlo contracts. General stable Fourier inversion and GH/GIG
fixed-grid quadrature are **not certified across all tails/scales** by this
pass. Finite-grid risk quadrature is not a proof that a requested moment
exists; reported quadrature differences are not rigorous error bounds.
Vine fitting remains a sequential numerical estimator, not an independently
proved global optimum. Extremely ill-conditioned covariance matrices,
unrepresentable quadratic forms, arbitrarily high copula dimension/dependence,
and densities at singular boundaries remain outside this audit grid.

Compound-count direct sums use integer indexing and now explicitly reject
finite indices at or beyond `Integer.MAX_VALUE`, replacing previous array
overflow or looping behavior. Their cost still grows with the count.
Order-statistic atom support relies on the base implementing
`AtomAwareDistribution`; legacy discrete laws without that contract must be
adapted before relying on its density as a mass function.

## Reproduction

Run from repository root, with the isolated R packages installed under
`build/audit2/Rlib`:

```powershell
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' benchmarks/audit2/joint/reference.R
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' benchmarks/audit2/joint/finance-reference.R
.\gradlew.bat :test --tests jdistlib.JointRReferenceTest --tests jdistlib.JointAuditRegressionTest
javac -cp build/classes/java/main -d build/audit2/joint benchmarks/audit2/joint/JointAudit.java benchmarks/audit2/joint/FinanceAudit.java benchmarks/audit2/joint/JointBenchmark.java
java -cp 'build/classes/java/main;build/audit2/joint' JointAudit
java -cp 'build/classes/java/main;build/audit2/joint' FinanceAudit
java -cp 'build/classes/java/main;build/audit2/joint' JointBenchmark
java -cp 'build/audit2/pristine-main;build/audit2/joint' JointBenchmark
& 'C:\Program Files\R\R-4.6.1\bin\Rscript.exe' benchmarks/audit2/joint/benchmark.R
```

`reference.R` and `finance-reference.R` retain their full audit grids under
`build/audit2/joint`; the latter also regenerates the trusted JUnit fixture.
