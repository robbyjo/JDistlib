# GPU acceleration plan and CUDA smoke decision

## Decision

Do **not** move the NUTS tree algorithm wholesale to CUDA. Keep tree construction,
U-turn checks, multinomial candidate selection, adaptation, and checkpoint state
on the CPU. Add an optional CUDA backend for large, batched log-density and
gradient evaluations, and only select it after a measured break-even test.

NUTS gives each chain a different tree depth and may stop at any doubling. Four
ordinary chains therefore provide little regular parallel work and introduce
warp divergence if the control algorithm is placed on-device. Static HMC,
Pathfinder candidate evaluation, SBC, and many-chain workloads are better GPU
clients because their work can be batched predictably. Models with large dense
likelihoods can also benefit while CPU NUTS remains in control.

## CUDA-first smoke on 2026-08-27

The repository contains `benchmarks/cuda/nuts_gradient_smoke.cu`. It keeps a
synthetic logistic-regression data set resident on the GPU, evaluates log density
and gradient in double precision, validates against a CPU reference, and reports
batch sizes 1, 4, 16, and 64. Transfers and context startup are excluded.

The local hardware smoke established:

| Item | Result |
|---|---|
| device | NVIDIA GeForce RTX 2080, compute capability 7.5, 8 GiB |
| CUDA | runtime 12.9; driver 13.3 |
| device query | PASS |
| pinned host → device | 12,156.3 MB/s |
| pinned device → host | 11,437.1 MB/s |
| device → device | 306,178.9 MB/s |
| model-kernel build | blocked: CUDA `nvcc` found no required `cl.exe` host compiler |

Consequently, this run proves that CUDA is usable and quantifies transfer cost,
but does **not** provide a defensible model speedup number. The checked-in source
turns that limitation into a repeatable smoke as soon as MSVC Build Tools are
available. The whole-NUTS decision is still negative because its irregular
control structure is independent of the missing kernel timing; the threshold
for offloading model evaluations remains deliberately undecided until measured.

## Integration stages

1. Keep the core artifact CPU-only and Java 8 compatible. Use
   `BatchedDifferentiableLogDensity` as the portable batching boundary.
2. Build a separate optional `jdistlib-cuda` artifact with a narrow JNI layer.
   JNI is preferable to Panama here because the supported core bytecode remains
   Java 8. Load it explicitly and fall back to the serial default when absent.
3. Initially accelerate model primitives—matrix products, reductions, and batched
   likelihood/gradient calls—with data and work buffers resident on-device.
4. Add a scheduler that batches independent chains or static-HMC trajectories.
   Never reorder RNG draws inside a chain; record backend and device identity in
   `RunManifest`.
5. Gate selection on warmup measurements. Report CPU/GPU evaluations per second,
   ESS per evaluation, ESS per second, transfer bytes, and numerical differences.
6. Require agreement tests for log density, gradient, transition statistics, and
   posterior summaries. Use deterministic CPU execution as the reproducibility
   reference; document that parallel floating-point reductions need tolerances.

Practical first targets are observation-heavy generalized linear models with at
least tens of thousands of rows, batched SBC/Pathfinder work, and static HMC.
Small models and the usual four asynchronous NUTS chains should stay on CPU.
