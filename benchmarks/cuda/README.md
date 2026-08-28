# CUDA NUTS smoke test

This standalone benchmark measures the part of NUTS that is suitable for a GPU:
batched log-density and gradient evaluations with model data resident on-device.
It deliberately does not claim that a fixed HMC trajectory is NUTS; NUTS has
per-chain recursive tree growth, U-turn checks, rejection and early termination.

Compile on an NVIDIA system:

```text
nvcc -O3 -std=c++14 -arch=native -o nuts_gradient_smoke nuts_gradient_smoke.cu
nuts_gradient_smoke 8192 32 100
```

Transfers and CUDA context creation are excluded. The four batch sizes distinguish
the common four-chain case from workloads large enough to occupy a GPU. Results and
the integration decision are recorded in `docs/gpu-acceleration.html`.
