# Optional GPU acceleration and measured CUDA smoke

> **Provisional architecture decision (2026-08-27):** accelerate batched model
> evaluation and regular multi-chain algorithms first. Keep NUTS tree control on
> the CPU until representative end-to-end ESS/second measurements justify moving
> it. The likelihood numbers below settle the compiler question and establish a
> batching break-even point; they do not by themselves settle whole-sampler NUTS.

## Decision

Do not move the NUTS tree algorithm wholesale to CUDA yet. Tree depths differ by
chain and every chain can stop at a different doubling, making ordinary four-chain
NUTS an irregular GPU workload. JDistlib now provides optional CUDA and OpenCL
backends for large batched log-density, vector-math, and linear-algebra work.
Backend absence never prevents the core artifact from loading; CPU is the
deterministic reference and fallback.

Static HMC, Pathfinder candidate evaluation, SBC, and many-short-chain workloads
are better GPU clients because their work batches predictably. Models with large
dense likelihoods can also benefit while the CPU retains sampler control.

## CUDA likelihood smoke on 2026-08-27

The repeatable `:jdistlib-cuda:cudaSmokeBenchmark` task uses an 8,192-row,
32-predictor synthetic logistic regression. It evaluates log density and gradient
in double precision, validates every result against the CPU reference, and reports
five-trial median timings for batches of 1, 4, 16, and 64 independent states. The
resident column keeps observations on device; end-to-end copies them every call.
Context and NVRTC compilation are warmup costs and excluded from both columns.

| Item | Result |
|---|---|
| device | NVIDIA GeForce RTX 2080, compute capability 7.5, 8 GiB |
| CUDA | runtime 12.9; driver 13.3 |
| device query | PASS |
| pinned host → device | 12,156.3 MB/s |
| pinned device → host | 11,437.1 MB/s |
| device → device | 306,178.9 MB/s |
| kernel compilation | PASS through JCuda/JNvrtc; no `nvcc` or MSVC required |
| CUDA CPU-reference tests | PASS; maximum likelihood/gradient error below 3.1e-11 |
| OpenCL CPU-reference tests | PASS on the same GPU through JOCL |

The initial measured run was:

| batched states | CPU ms | CUDA resident ms | resident speedup | CUDA end-to-end ms | end-to-end speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7165 | 0.8915 | 0.804x | 1.8337 | 0.391x |
| 4 | 2.6209 | 0.8043 | 3.259x | 1.8411 | 1.424x |
| 16 | 10.6519 | 1.4382 | 7.406x | 2.3092 | 4.613x |
| 64 | 45.6427 | 2.7388 | 16.665x | 3.6493 | 12.507x |

These are smoke-test numbers on one machine, not portable performance claims.
They establish that copying a modest model for one state can erase the gain,
while resident data and many simultaneous states are worthwhile. Re-run the task
on target hardware before choosing a backend:

```text
gradlew :jdistlib-cuda:cudaSmokeBenchmark
```

## Implemented backend boundary

- `ComputeBackend` is a Java 8 service-provider interface. `ComputeBackends`
  detects providers and honors `-Djdistlib.compute.backend=auto|cpu|cuda|opencl`.
- `CpuComputeBackend` is always available. Missing drivers, native libraries,
  compiler runtimes, or FP64 support make optional providers unavailable.
- `jdistlib-cuda` uses JCuda Driver plus JNvrtc. `jdistlib-opencl` uses JOCL 2.0.
- Both implement vector math, AXPY, dot reduction, dense GEMM, and batched
  logistic likelihood/gradient primitives. `PreparedLogisticRegression` keeps
  reusable observations in backend storage.
- `AcceleratedLogisticRegression` exposes the prepared likelihood through
  `BatchedDifferentiableLogDensity` without adding native dependencies to core.
- `AdaptiveStaticHamiltonianMonteCarlo` implements coordinated ChEES or SNAPER
  trajectory adaptation. Its CPU semantics are established before a future
  scheduler fuses its gradients into backend batches.

Parallel floating-point reductions are validated within documented tolerances;
the CPU path remains the bit-stable seeded reference. Backend and device identity
should be retained in run provenance whenever an accelerator is selected.

## Why JCuda resolves the compiler blocker

Compiling a Windows `.cu` executable with `nvcc` requires a supported host C++
compiler such as MSVC. JCuda distributes its JNI binding prebuilt, and JNvrtc
invokes NVIDIA's runtime compiler directly. NVRTC compiles device source without
a host compiler; JCuda's Driver API loads the resulting PTX.

JCuda does not eliminate every native requirement. A compatible NVIDIA driver,
NVRTC, and `nvrtc-builtins` are still required. PTX caching and CI-built PTX are
useful deployment follow-ups. CPU remains the safe fallback.

## Why Vulkan is not a third backend yet

Vulkan compute is possible from Java, but it needs a SPIR-V compilation and
reflection pipeline, substantially more descriptor/synchronization boilerplate,
and does not guarantee portable FP64 support. OpenCL already supplies the
comparable non-CUDA path with a smaller statistical-kernel surface. The provider
interface permits a future Vulkan module, but adding one before a supported target
needs it would increase native and numerical validation work without improving
present coverage.
