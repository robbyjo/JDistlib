// CUDA-first smoke test for the log-density/gradient hot path used by NUTS.
// This is deliberately standalone: the Java 8 artifact acquires no CUDA dependency.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(call) do { cudaError_t error_ = (call); if (error_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error_)); std::exit(2); } } while (0)

__device__ double log1p_exp(double x) {
  return x > 0.0 ? x + log1p(exp(-x)) : log1p(exp(x));
}

__global__ void residual_kernel(const double* x, const double* y,
    const double* q, double* residual, double* terms, int observations,
    int dimensions, int chains) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = observations * chains;
  if (index >= total) return;
  int chain = index / observations;
  int observation = index - chain * observations;
  double eta = 0.0;
  for (int d = 0; d < dimensions; ++d)
    eta += x[observation * dimensions + d] * q[chain * dimensions + d];
  double probability = eta >= 0.0 ? 1.0 / (1.0 + exp(-eta))
      : exp(eta) / (1.0 + exp(eta));
  residual[index] = y[observation] - probability;
  terms[index] = y[observation] * eta - log1p_exp(eta);
}

__global__ void gradient_kernel(const double* x, const double* q,
    const double* residual, double* gradient, int observations,
    int dimensions, int chains) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int total = dimensions * chains;
  if (index >= total) return;
  int chain = index / dimensions;
  int dimension = index - chain * dimensions;
  double sum = -q[index];
  for (int observation = 0; observation < observations; ++observation)
    sum += residual[chain * observations + observation]
        * x[observation * dimensions + dimension];
  gradient[index] = sum;
}

__global__ void log_density_kernel(const double* terms, const double* q,
    double* log_density, int observations, int dimensions) {
  __shared__ double partial[256];
  int chain = blockIdx.x;
  double sum = 0.0;
  for (int i = threadIdx.x; i < observations; i += blockDim.x)
    sum += terms[chain * observations + i];
  for (int d = threadIdx.x; d < dimensions; d += blockDim.x) {
    double value = q[chain * dimensions + d]; sum -= 0.5 * value * value;
  }
  partial[threadIdx.x] = sum; __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width /= 2) {
    if (threadIdx.x < width) partial[threadIdx.x] += partial[threadIdx.x + width];
    __syncthreads();
  }
  if (threadIdx.x == 0) log_density[chain] = partial[0];
}

static void cpu_evaluate(const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>& q, std::vector<double>& log_density,
    std::vector<double>& gradient, int observations, int dimensions, int chains) {
  std::fill(log_density.begin(), log_density.end(), 0.0);
  std::fill(gradient.begin(), gradient.end(), 0.0);
  for (int chain = 0; chain < chains; ++chain) {
    for (int d = 0; d < dimensions; ++d) {
      double value = q[chain * dimensions + d];
      gradient[chain * dimensions + d] = -value;
      log_density[chain] -= 0.5 * value * value;
    }
    for (int observation = 0; observation < observations; ++observation) {
      double eta = 0.0;
      for (int d = 0; d < dimensions; ++d)
        eta += x[observation * dimensions + d] * q[chain * dimensions + d];
      double probability = eta >= 0.0 ? 1.0 / (1.0 + std::exp(-eta))
          : std::exp(eta) / (1.0 + std::exp(eta));
      double residual = y[observation] - probability;
      log_density[chain] += y[observation] * eta
          - (eta > 0.0 ? eta + std::log1p(std::exp(-eta)) : std::log1p(std::exp(eta)));
      for (int d = 0; d < dimensions; ++d)
        gradient[chain * dimensions + d] += residual * x[observation * dimensions + d];
    }
  }
}

int main(int argc, char** argv) {
  int observations = argc > 1 ? std::atoi(argv[1]) : 8192;
  int dimensions = argc > 2 ? std::atoi(argv[2]) : 32;
  int repetitions = argc > 3 ? std::atoi(argv[3]) : 100;
  const int batches[] = {1, 4, 16, 64};
  cudaDeviceProp property; CUDA_OK(cudaGetDeviceProperties(&property, 0));
  std::printf("device,%s\nobservations,%d\ndimensions,%d\nrepetitions,%d\n",
      property.name, observations, dimensions, repetitions);
  std::printf("chains,cpu_ms_per_eval,gpu_ms_per_eval,speedup,max_abs_error\n");
  unsigned long long seed = 0x123456789abcdefULL;
  auto uniform = [&seed]() { seed = seed * 6364136223846793005ULL + 1ULL;
    return ((seed >> 11) & 0x1fffff) / double(0x200000) - 0.5; };
  std::vector<double> x(observations * dimensions), y(observations);
  for (double& value : x) value = uniform();
  for (double& value : y) value = uniform() > 0.0 ? 1.0 : 0.0;
  double *dx, *dy; CUDA_OK(cudaMalloc(&dx, x.size() * sizeof(double)));
  CUDA_OK(cudaMalloc(&dy, y.size() * sizeof(double)));
  CUDA_OK(cudaMemcpy(dx, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(dy, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));
  for (int chains : batches) {
    std::vector<double> q(chains * dimensions), cpu_log(chains), cpu_gradient(q.size());
    std::vector<double> gpu_log(chains), gpu_gradient(q.size());
    for (double& value : q) value = 0.1 * uniform();
    double *dq, *dresidual, *dterms, *dgradient, *dlog;
    CUDA_OK(cudaMalloc(&dq, q.size() * sizeof(double)));
    CUDA_OK(cudaMalloc(&dresidual, chains * observations * sizeof(double)));
    CUDA_OK(cudaMalloc(&dterms, chains * observations * sizeof(double)));
    CUDA_OK(cudaMalloc(&dgradient, q.size() * sizeof(double)));
    CUDA_OK(cudaMalloc(&dlog, chains * sizeof(double)));
    CUDA_OK(cudaMemcpy(dq, q.data(), q.size() * sizeof(double), cudaMemcpyHostToDevice));
    cpu_evaluate(x, y, q, cpu_log, cpu_gradient, observations, dimensions, chains);
    for (int warm = 0; warm < 10; ++warm) {
      residual_kernel<<<(chains * observations + 255) / 256, 256>>>(dx, dy, dq,
          dresidual, dterms, observations, dimensions, chains);
      gradient_kernel<<<(chains * dimensions + 127) / 128, 128>>>(dx, dq,
          dresidual, dgradient, observations, dimensions, chains);
      log_density_kernel<<<chains, 256>>>(dterms, dq, dlog, observations, dimensions);
    }
    CUDA_OK(cudaDeviceSynchronize());
    auto cpu_start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < repetitions; ++repetition)
      cpu_evaluate(x, y, q, cpu_log, cpu_gradient, observations, dimensions, chains);
    auto cpu_end = std::chrono::steady_clock::now();
    cudaEvent_t start, end; CUDA_OK(cudaEventCreate(&start)); CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaEventRecord(start));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      residual_kernel<<<(chains * observations + 255) / 256, 256>>>(dx, dy, dq,
          dresidual, dterms, observations, dimensions, chains);
      gradient_kernel<<<(chains * dimensions + 127) / 128, 128>>>(dx, dq,
          dresidual, dgradient, observations, dimensions, chains);
      log_density_kernel<<<chains, 256>>>(dterms, dq, dlog, observations, dimensions);
    }
    CUDA_OK(cudaEventRecord(end)); CUDA_OK(cudaEventSynchronize(end));
    float gpu_ms; CUDA_OK(cudaEventElapsedTime(&gpu_ms, start, end));
    CUDA_OK(cudaMemcpy(gpu_gradient.data(), dgradient, q.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(gpu_log.data(), dlog, chains * sizeof(double), cudaMemcpyDeviceToHost));
    double error = 0.0;
    for (size_t i = 0; i < gpu_gradient.size(); ++i) error = std::max(error, std::abs(gpu_gradient[i] - cpu_gradient[i]));
    for (int i = 0; i < chains; ++i) error = std::max(error, std::abs(gpu_log[i] - cpu_log[i]));
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double cpu_each = cpu_ms / repetitions, gpu_each = gpu_ms / repetitions;
    std::printf("%d,%.6f,%.6f,%.3f,%.3e\n", chains, cpu_each, gpu_each,
        cpu_each / gpu_each, error);
    cudaEventDestroy(start); cudaEventDestroy(end); cudaFree(dq); cudaFree(dresidual);
    cudaFree(dterms); cudaFree(dgradient); cudaFree(dlog);
  }
  cudaFree(dx); cudaFree(dy);
  return 0;
}
