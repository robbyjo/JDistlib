# Unified dense and sparse linear algebra

JDistlib exposes parallel FP64 and FP32 linear-algebra contracts across its
deterministic CPU reference and optional CUDA, OpenCL, and Vulkan providers. Select and own a
backend through the existing compute policy:

```java
try (ComputeSelection selection = ComputeBackends.select(Compute.AUTO)) {
    ComputeBackend blas = selection.backend();
    double[] product = new double[rows * columns];
    blas.dgemm(MatrixTranspose.NONE, MatrixTranspose.NONE,
        rows, columns, shared, 1.0, left, right, 0.0, product);
}
```

Use the standard `s*` methods and `float[]` storage for FP32:

```java
float[] product = new float[rows * columns];
blas.sgemm(MatrixTranspose.NONE, MatrixTranspose.NONE,
    rows, columns, shared, 1.0f, leftFloat, rightFloat, 0.0f, product);
```

Dense matrices are contiguous row-major `double[]` or `float[]` values. The
public BLAS surface uses the standard precision prefix and mutates caller-owned
outputs:

| FP64 | FP32 | Operation |
| --- | --- | --- |
| `daxpy` | `saxpy` | strided `y := alpha*x + y` |
| `ddot` | `sdot` | strided dot product |
| `dnrm2` | `snrm2` | scaled Euclidean norm |
| `dgemv` | `sgemv` | `y := alpha*op(A)*x + beta*y` |
| `dgemm` | `sgemm` | `C := alpha*op(A)*op(B) + beta*C` |
| `dcsrmv` | `scsrmv` | CSR matrix times dense vector |
| `dcsrmm` | `scsrmm` | CSR matrix times row-major dense matrix |
| `dpotrf` | `spotrf` | lower Cholesky factorization |
| `dgeqp3` | `sgeqp3` | column-pivoted Householder QR |
| `dsyev` | `ssyev` | symmetric eigenvalues and eigenvectors |
| `dgesvd` | `sgesvd` | thin singular-value decomposition |

`MatrixTranspose` makes every transpose explicit. The older allocating
`axpy`, `dot`, and `matrixMultiply(double[][], double[][])` methods remain
available and compatible.

## Sparse storage

The sparse operations accept immutable `CsrMatrix` (FP64) and `FloatCsrMatrix`
(FP32) values. Their column indices and row starts are one-based for compatibility
with Stan CSR arrays.
Every backend preserves those public semantics internally. CPU, CUDA, OpenCL,
and Vulkan have native CSR kernels; `Compute.AUTO` keeps small heap-backed work
on CPU and routes sufficiently large work to the selected accelerator.

```java
CsrMatrix matrix = new CsrMatrix(rows, columns, values,
    oneBasedColumnIndices, oneBasedRowStarts);
double[] result = new double[rows];
blas.dcsrmv(1.0, matrix, vector, 0.0, result);
```

## Reusable decompositions

`dpotrf`/`spotrf` compute lower Cholesky factors with repeated vector/matrix
solves and a stable log determinant. `dgeqp3`/`sgeqp3` compute column-pivoted
Householder QR with rank reporting and a full-rank least-squares solve. The
results are `CholeskyFactor`/`FloatCholeskyFactor` and
`PivotedQrFactor`/`FloatPivotedQrFactor`. Factor objects own their storage and
never expose a mutable internal array.

```java
CholeskyFactor factor = blas.dpotrf(covariance, dimension);
double[] solution = factor.solve(rightHandSide);
double logDeterminant = factor.logDeterminant();
```

`dsyev`/`ssyev` accept a fully stored real symmetric matrix and return ascending
eigenvalues plus a row-major orthogonal eigenvector matrix whose columns are the
corresponding eigenvectors. `dgesvd`/`sgesvd` work for tall, square, and wide
matrices and return the thin decomposition `A = U*S*Vt`, with singular values
in descending order. Thin `U` has shape `rows` by `min(rows,columns)` and thin
`Vt` has shape `min(rows,columns)` by `columns`.

The deterministic Java CPU algorithms are the reference. CUDA, OpenCL, and
Vulkan execute all four decomposition families natively for FP64 and FP32.
Their initial device implementation uses serial device kernels optimized for a
portable correctness baseline and avoiding host-side factorization; large-scale
tiled or vendor-library implementations can replace those kernels behind the
same API. `Compute.AUTO` uses a cubic work estimate to keep small decompositions
on CPU and route sufficiently large ones to the selected accelerator.

## Provider and numerical contract

`ComputeCapabilities` separately reports dense BLAS, sparse BLAS, and native
factorization support. CUDA, OpenCL, and Vulkan currently accelerate both FP64
and FP32 dense and CSR operations with native precision buffers and kernels.
Reduction order and final rounding can differ across providers, so callers
should compare numerical tolerances rather than bits and record the selected
backend and device with reproducible results. FP32 is intended for workloads
that accept its smaller dynamic range and precision in exchange for lower
storage and potentially higher device throughput; it is never silently used
for an FP64 call.

Heap-backed calls include host/device transfer and synchronization. Small
operations normally favor CPU. Prepared or resident matrix APIs remain the
appropriate future extension for repeated large operations such as mixed-model
solves and fine-mapping residual updates.

The contract permits a future oneMKL or OpenBLAS `ComputeBackend` provider
without changing callers. Legacy convenience operations and the statistical
kernel have portable defaults, so such a provider can focus on the BLAS and
factorization methods. Service-loaded providers can be selected by their exact
identifier through `ComputeBackends.byId` or the
`jdistlib.compute.backend` system property. No native CPU BLAS is bundled yet;
adding one should remain optional, preserve the Java CPU reference, and publish
its threading and library-loading behavior explicitly.
