# Inference acceleration and workflow result

**Release status:** CUDA/OpenCL workflow in 0.8.3; unified JAR and Vulkan in 0.8.4
**Compatibility:** additive APIs, Java 8 bytecode, native-free core, deterministic
CPU reference behavior

This release turns JDistlib's original single-machine MCMC layer into
a broader inference workflow. It adds approximate initialization, coordinated
many-chain algorithms, optional accelerators, precision-driven continuation,
portable restart, scalable draw storage, and machine-readable diagnostics while
retaining the 0.8.0 sampler APIs.

## Delivered capabilities

| Area | Current result |
|---|---|
| Pathfinder | Multi-path L-BFGS traces, retained inverse-Hessian approximations, ELBO candidate selection, exact target/mixture scoring, Pareto-smoothed importance sampling, Pareto-\(\hat{k}\), and systematic resampling |
| Many short chains | Common-start superchain execution plus basic, rank-normalized, and folded nested R-hat diagnostics |
| Adaptive static HMC | Coordinated ChEES or SNAPER trajectory-length adaptation, with batched gradient stages when a model supports them |
| Adjusted MCLMC | Pilot selection of step size, decorrelation length, leapfrog count, and diagonal mass scaling, with immutable auditable tuning results |
| Precision and diagnostics | Arbitrary-function and indicator ESS/MCSE, MCSE-driven continuation, factor cost/nonfinite/heap profiling, geometry advice, and policy-based health findings |
| Reuse and restart | Fingerprinted warmup bundles and checksummed binary checkpoints with guarded RNG deserialization |
| Large-output workflows | Compressed selected-column storage, memory-mapped draws, generated-quantity sinks, and operation without retaining every draw in heap |
| Acceleration | A Java 8 `ComputeBackend` SPI, deterministic CPU implementation, JCuda/JNvrtc, JOCL, and LWJGL Vulkan providers, automatic detection, safe fallback, and a standalone all-in-one distribution |

The primary API map and compatibility rules are in
[MCMC_MODERNIZATION.md](MCMC_MODERNIZATION.md). The design record, acceptance
gates, and experimental boundary are in
[MCMC_FUTURE_ROADMAP.md](MCMC_FUTURE_ROADMAP.md).

## Accelerator boundary

The modular core has no GPU dependency. Optional `jdistlib-cuda`,
`jdistlib-opencl`, and `jdistlib-vulkan` artifacts are discovered through Java's service-provider
mechanism. `-Djdistlib.compute.backend=auto|cpu|gpu|cuda|opencl|vulkan` controls
selection and `-Djdistlib.compute.nuts=off|auto|force` controls NUTS offload;
Java callers use `SamplingOptions.backend(Compute...)` and
`nutsBackend(ComputeNuts...)`, while embedding command-line applications can use
`--compute`, `--nuts-offload`, or `--gpu-nuts`. `AUTO` tries CUDA, OpenCL, and Vulkan but
keeps small operations on CPU through conservative profitability thresholds.
Missing drivers, native libraries, compiler runtimes, or FP64 support fall back
only in `AUTO`; explicit GPU/provider requests fail immediately.

All three providers implement double-precision unary vector operations,
AXPY, dot products, dense GEMM, and batched logistic-regression log densities
and gradients. CUDA and OpenCL keep prepared inputs resident; Vulkan 0.8.4
caches compiled shaders and compute pipelines but uses portable host-visible
storage for each evaluation. The same batched target can be consumed by regular many-chain algorithms.
Backend selection is logged, and `RunManifest` retains the requested policy,
concrete provider, device, and NUTS offload mode.

The Vulkan provider uses LWJGL, shaderc, a compute queue with `shaderFloat64`,
host-visible coherent storage, GLSL-to-SPIR-V compilation, descriptors,
pipelines, command buffers, and fences. Deterministic FP64 shader routines
replace transcendental overloads not guaranteed by Vulkan GLSL. The 0.8.4
`jdistlib-all` artifact merges every provider and its x86-64 Java/JNI runtime;
small modules remain available for dependency-managed applications.

## Measured CUDA result

The repeatable smoke benchmark used an NVIDIA GeForce RTX 2080 and an
8,192-row, 32-predictor double-precision logistic regression. Values are
five-trial medians after context and runtime-compilation warmup.

| Batched states | CPU | CUDA, resident data | Resident speedup | CUDA, including copies | End-to-end speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7165 ms | 0.8915 ms | 0.804x | 1.8337 ms | 0.391x |
| 4 | 2.6209 ms | 0.8043 ms | 3.259x | 1.8411 ms | 1.424x |
| 16 | 10.6519 ms | 1.4382 ms | 7.406x | 2.3092 ms | 4.613x |
| 64 | 45.6427 ms | 2.7388 ms | 16.665x | 3.6493 ms | 12.507x |

The largest CPU/CUDA likelihood or gradient difference was below
`3.1e-11`. JNvrtc compiled kernels successfully without `nvcc` or MSVC; a
compatible NVIDIA driver, NVRTC, and `nvrtc-builtins` are still required.

These measurements establish a useful batching boundary, not a universal GPU
speed claim. One modest state is slower on this device, while resident data and
four or more simultaneous states benefit. Re-run
`gradlew :jdistlib-cuda:cudaSmokeBenchmark` on the intended hardware. Detailed
environment and transfer measurements are retained in
[the benchmark record](../benchmarks/cuda/RESULTS-2026-08-27.md).

## NUTS decision remains provisional

JDistlib does not move the NUTS tree algorithm wholesale to the GPU. Per-chain
tree depths and U-turn stopping decisions make ordinary few-chain NUTS an
irregular workload. The implemented accelerator boundary instead targets large
likelihoods, vector math, linear algebra, Pathfinder candidates, simulation-
based calibration, adaptive static HMC, and many-short-chain inference.

Moving more of NUTS requires representative end-to-end measurements of
effective samples per second, including adaptation, synchronization, transfer,
and divergence behavior. The current likelihood benchmark answers the compiler
and primitive-throughput questions but intentionally does not answer that
sampler-level question. See [the GPU acceleration webpage](gpu-acceleration.html) for the
decision record.

## Validation completed

The implementation was checked through:

- the clean full Gradle verification suite, including core, CUDA, OpenCL, Vulkan,
  and the standalone unified-JAR smoke
  tests;
- CPU-reference comparisons for every accelerator primitive and the prepared
  batched likelihood;
- an accelerated batched-HMC integration test;
- focused Pathfinder, nested R-hat, ChEES/SNAPER, adjusted-MCLMC, checkpoint,
  continuation, storage, and health-policy tests;
- the complete ninety-one-model semantic script gate;
- Java 8 release compilation; and
- warning-free generated JavaDoc.

Seeded CPU execution remains the reproducibility reference. Accelerator
reductions are numerically checked within documented tolerances, and backend and
device identity should be recorded with accelerated runs.

## Release status

This result is released through **v0.8.4**. The changes are additive, preserve the
0.8 API, retain Java 8 bytecode, and keep native dependencies outside the core
artifact.
