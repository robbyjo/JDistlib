# MCMC roadmap beyond Stan's default workflow

Assessment date: 2026-08-27. The comparison baseline is Stan 2.39: Euclidean
static HMC and NUTS with windowed step-size/metric adaptation, standard HMC
diagnostics, and single-/multi-path Pathfinder outside the sampler.

JDistlib should keep NUTS as its conservative default. New methods must earn
their place through independent reference checks, posterior accuracy, and
ESS-per-gradient and ESS-per-second benchmarks—not novelty alone.

## Implementation status (0.8.4)

The four recommended milestone items and the high-value QoL list below now have
public APIs and regression tests: multi-path Pathfinder/PSIS, superchain execution
and nested R-hat, coordinated ChEES/SNAPER static HMC, adjusted-MCLMC pilot tuning,
portable checkpoints, precision continuation, arbitrary-function diagnostics,
geometry advice, factor profiling, fingerprinted warmup reuse, compressed
selected-coordinate streaming, and actionable health findings. Optional JCuda
and JOCL providers implement the accelerator boundary. The experimental methods
at the end of this document remain research candidates rather than implied API.

## Delivered milestone

### 1. Multi-path Pathfinder beyond the legacy mode initializer

The legacy `PathfinderInitializer` remains a lightweight compatible initializer.
`Pathfinder` retains L-BFGS inverse-Hessian approximations along every path,
selects candidates by a Monte Carlo ELBO estimate, scores the multi-path Gaussian
mixture against the exact target, applies Pareto smoothing, and exposes
Pareto-\(\hat{k}\) before returning systematically resampled draws.

This is the clearest QoL win: robust starts reduce wasted warmup, and the same
implementation supports approximate inference when its diagnostics are good.

### 2. Many-short-chain execution and nested R-hat

`SuperchainPlan` and `ManyShortChains` launch groups from shared initial points;
`McmcDiagnostics` implements basic and rank-normalized/folded nested R-hat.
Ordinary R-hat remains the default for a few
long chains; nested R-hat is specifically for many short chains, and
`ManyShortChainsResult` retains the exact `SuperchainPlan` used by the run.

This is the natural execution model for CUDA and high-core-count CPUs. Pair it
with quantile/SD ESS and MCSE, per-chain failure maps, and explicit minimum
warmup safeguards.

### 3. ChEES/SNAPER-style adaptive static HMC

`AdaptiveStaticHamiltonianMonteCarlo` learns trajectory length using either the
centered ChEES objective or SNAPER's difficult principal direction. Its synchronized
leapfrog stages call `BatchedDifferentiableLogDensity` when available. SNAPER's
focus on a difficult principal direction makes regular trajectories competitive
while remaining SIMD-friendly. Once reference validation passes, use this as the
first sampler over `BatchedDifferentiableLogDensity`.

Do not call this GPU NUTS: it is adaptive static HMC with different transition
and adaptation rules. Compare it against CPU NUTS on identical gradients using
ESS/gradient, ESS/second, divergences, and bias-sensitive reference quantities.

### 4. Automated tuning for adjusted microcanonical sampling

`AdjustedMclmcTuner` runs deterministic pilots over decorrelation lengths, adapts
step size, estimates diagonal mass scaling, records every score and acceptance,
then runs the adjusted sampler with the selected geometry. Adjusted and unadjusted
algorithms remain distinct in names, results, and warnings.

## Delivered QoL work

1. **Portable checkpoints.** `CheckpointIO` writes a versioned, SHA-256-checked
   binary envelope containing model/options fingerprints, RNG and adaptation
   state, and platform metadata; mismatched resumes fail explicitly.
2. **Precision-driven continuation.** `PrecisionGoal` accepts a coordinate or
   arbitrary scalar quantity plus absolute/relative MCSE. `PrecisionContinuation`
   extends deterministic chunks after minimum-draw and sampler-health gates.
3. **Posterior diagnostics.** `MonteCarloError` handles arbitrary functions and
   indicators; nested R-hat and Pareto smoothing/\(\hat{k}\) are first-class.
   Classifier-based multivariate \(R^*\) remains optional research work.
4. **Geometry adviser.** `GeometryAdvisor` ranks unconstrained coordinates by
   divergence separation and pairs the evidence with scale/non-centering advice;
   sampler statistics retain metric condition and factor profiling supplies the
   complementary model-side evidence.
5. **Factor profiler.** `FactorProfiler` reports calls, time, positive heap-growth
   estimates, and non-finite results while preserving analytic-gradient identity.
6. **Warmup reuse with fingerprints.** `WarmupBundle` checks model/geometry
   fingerprints and treats metric/step size as an adapting initial guess unless
   the caller explicitly fixes adaptation.
7. **Columnar and streaming results.** `ChunkedDrawSink` provides compressed,
   recoverable selected columns; `MappedDrawStore` provides a fixed-capacity
   memory-mapped path; `GeneratedQuantitySink` composes derived scalar outputs.
8. **Actionable health policy.** `InferenceHealth` returns machine-readable
   severity, evidence, and remediation for sampler failures, divergences, depth,
   E-BFMI, R-hat/ESS, unstable importance weights, and failed gradient checks.

## Experimental algorithms worth isolating

| Method | Potential value | Why it should not be a default yet |
|---|---|---|
| hierarchical/Riemannian HMC | position-dependent geometry for funnels and multiscale targets | Hessians, log determinants, implicit or specialized integrators, and stricter reversibility tests greatly increase cost and maintenance |
| transport-map or normalizing-flow preconditioning | learn a transformation that makes Euclidean HMC easier | training can fail or overfit; exactness requires Jacobians and an exact correction; needs a separate accelerator/autodiff design |
| Bouncy Particle and Zig-Zag PDMP samplers | non-reversible exploration and sparse/tall-data opportunities | event-rate bounds and factor-local structure are model dependent; generic implementations can be slower than HMC |
| annealed/tempered SMC | multimodality, evidence estimates, sequential targets | population resampling and adaptive temperature schedules require a result model distinct from ordinary chains |
| particle MCMC | state-space and latent time-series models | needs particle-filter infrastructure and careful pseudo-marginal diagnostics |
| coupled unbiased MCMC | unbiased finite-time estimators and parallel replication | meeting-time tails can be poor and require diagnostics unfamiliar to most users |
| MEADS/underdamped ensemble adaptation | rapid adaptation from many parallel chains | best suited to large ensembles; reference implementations and tuning practice are less mature |

The first four experimental entries should use separate interfaces rather than
adding switches to `NoUTurnSampler`. This keeps exactness, state, and diagnostics
auditable.

## Acceptance gates

Every new sampler or accelerator backend must pass:

- analytic targets with known moments and independent implementations;
- difficult PosteriorDB-style models, including funnels and multimodality;
- reversibility, volume/Jacobian, and detailed-balance checks where applicable;
- deterministic seeded CPU regression tests and checkpoint round trips;
- bias-sensitive SBC or rank tests, not only means and acceptance rates;
- ESS/gradient and ESS/second comparisons including warmup and compilation;
- explicit behavior for non-finite values, cancellation, streaming, and resume;
- Java 8 core bytecode and no mandatory native dependency.

## Primary references

- [Stan 2.39 reference manual](https://mc-stan.org/docs/reference-manual/)
- [Stan Pathfinder documentation](https://mc-stan.org/docs/reference-manual/pathfinder.html)
- [Adaptive HMC trajectory lengths / ChEES-HMC](https://proceedings.mlr.press/v130/hoffman21a.html)
- [SNAPER-HMC](https://arxiv.org/abs/2110.11576)
- [Adjusted microcanonical HMC](https://arxiv.org/abs/2503.01707)
- [Nested R-hat](https://arxiv.org/abs/2110.13017)
- [Pareto-smoothed importance sampling](https://jmlr.org/papers/v25/19-556.html)
- [R* multivariate convergence diagnostic](https://arxiv.org/abs/2003.07900)
- [Piecewise-deterministic splitting samplers](https://www.jmlr.org/papers/v26/23-0036.html)
- [Modern-hardware MCMC patterns](https://arxiv.org/abs/2411.04260)
