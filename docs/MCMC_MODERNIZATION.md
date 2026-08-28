# Modern MCMC API

JDistlib 0.8.5 keeps the original `Sampler.sample(...)` entry
point and Java 8 bytecode while adding a composable, restartable inference layer.
The consolidated [inference acceleration result](INFERENCE_ACCELERATION_RESULT.md)
records the delivered surface, validation evidence, and measured GPU boundary.

## What is now available

| Area | API | Notes |
|---|---|---|
| warmup | `WarmupSchedule.stanDefault()` | initial fast adaptation, expanding slow metric windows, final fast adaptation |
| transition composition | `TransitionKernel`, `KernelTransition`, `RandomWalkKernel` | reusable one-step kernels for meta-samplers |
| exact NUTS restart | `SamplerCheckpoint`, `ResumableSampler`, `Chains.resume` | restores RNG, metric, dual averaging, covariance accumulator, and warmup position |
| metrics | `MetricConfiguration` | unit, diagonal, dense, block diagonal, supplied, and low-rank-plus-diagonal |
| trajectory control | `integrationTime`, `stepSizeJitter` | static HMC can use integration time; HMC-family steps can be jittered |
| result facade | `Inference.fit`, `Fit`, `RunManifest` | chains, diagnostics, elapsed time, version, seed, compute/device provenance, and SHA-256 option/model identities |
| initialization | `InitialStates`, `PathfinderInitializer`, `LbfgsOptimizer` | named constrained starts, deterministic retry, and quasi-Newton initialization |
| full approximation | `Pathfinder`, `PathfinderOptions`, `PathfinderFit` | L-BFGS path/ELBO selection, multiple paths, mixture scoring, PSIS resampling, and Pareto-k diagnostics |
| many short chains | `ManyShortChains`, `ManyShortChainsResult`, `McmcDiagnostics.nestedRankNormalizedRHat` | common-start superchains and robust nested R-hat |
| adaptive static HMC | `AdaptiveStaticHamiltonianMonteCarlo` | coordinated ChEES or SNAPER trajectory-length adaptation for regular many-chain work |
| additional samplers | `MetropolisAdjustedLangevin`, `BarkerGradientSampler`, `EllipticalSliceSampler`, `ParallelTempering` | exact gradient proposals, tuning-free Gaussian-reference updates, and replica exchange |
| microcanonical | `AdjustedMicrocanonicalLangevin` | MH-adjusted isokinetic MCLMC/MHMCHMC; dimension must exceed one |
| adjusted-MCLMC tuning | `AdjustedMclmcTuner` | pilot selection of step size, decorrelation length, and diagonal mass scaling with auditable scores |
| scale control | `DrawSink`, `storeDraws(false)`, `ProgressListener` | stream draws without retaining the chain and report/cancel progress |
| diagnostics | `MonteCarloError`, `EvaluationCounter`, `Divergences`, `WarmupTrace` | MCSE for SD/quantiles, ESS per work/time, evaluation counts, divergence coordinates, and adaptation traces |
| workflow QoL | `CheckpointIO`, `PrecisionContinuation`, `WarmupBundle`, `FactorProfiler`, `ChunkedDrawSink`, `MappedDrawStore`, `GeneratedQuantitySink`, `InferenceHealth`, `GeometryAdvisor` | portable restart, precision goals, safe warmup reuse, profiling, streaming/mapped output, generated quantities, and actionable health findings |
| acceleration | `Compute`, `ComputeNuts`, `ComputeBackend`, modular CUDA/OpenCL/Vulkan providers, and `jdistlib-all` | thresholded AUTO routing, strict provider selection, forced-NUTS validation, batched likelihoods, and a unified direct download |
| validation | `SimulationBasedCalibration` | deterministic SBC ranks for model/sampler test suites |

The low-rank metric uses deterministic eigendirection extraction so seeded runs
remain reproducible. Supplied metrics are positive-definiteness checked. Existing
`denseMassMatrix(boolean)` calls retain their behavior; `metric(...)` is the more
explicit replacement.

Java callers configure accelerator-aware targets with
`SamplingOptions.builder().backend(Compute.AUTO).nutsBackend(ComputeNuts.AUTO)`.
See the [GPU acceleration webpage](gpu-acceleration.html) for strict modes,
command-line switches, measured thresholds, and reproducibility boundaries.

## Adjusted MCLMC terminology

`AdjustedMicrocanonicalLangevin` is not the unadjusted, potentially biased MCLMC
algorithm. It draws a unit momentum, advances the reversible isokinetic
McLachlan splitting, accounts for the kinetic-energy change, and applies a
Metropolis correction. This adjusted method is often called MHMCHMC. The
correction makes retained draws asymptotically exact, while divergences and
acceptance still need review.

## Initialization and optimization

BOBYQA remains useful for derivative-free objectives. L-BFGS is included because
Pathfinder and gradient models need curvature information accumulated along an
optimization path, not because every MCMC run needs an optimizer. A valid user
start can go directly to warmup. Use `InitialStates.retry(...)` when the support
is awkward and `PathfinderInitializer` when a quick gradient-based route toward
typical high-density geometry is useful.

## Compatibility contract

All additions are additive. The build continues to use `--release 8`, existing
sampler defaults are retained, and randomness comes only from caller-provided
`RandomEngine` instances. The NUTS checkpoint format is explicitly versioned;
unsupported future checkpoint versions fail rather than silently restarting with
different adaptation.

See the [post-Stan MCMC roadmap](MCMC_FUTURE_ROADMAP.md) for the design record,
acceptance gates, and experimental methods that remain deliberately isolated.
