# Unified dense and sparse linear algebra

**Release status:** the complete API described here is released in JDistlib
0.10.0. The shorter [website guide](linear-algebra.html) provides an overview
for downstream library authors.

JDistlib exposes FP64 and FP32 linear-algebra contracts across its deterministic
Java CPU reference, optional oneMKL/OpenBLAS native CPU providers, and optional
CUDA, OpenCL, and Vulkan providers. Select and own a backend through the
existing compute policy:

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
| `dscal` | `sscal` | strided in-place scaling |
| `dcopy` | `scopy` | alias-safe strided copy |
| `dswap` | `sswap` | strided exchange |
| `dasum` | `sasum` | absolute sum |
| `idamax` | `isamax` | zero-based index of the first absolute maximum |
| `daxpy` | `saxpy` | strided `y := alpha*x + y` |
| `ddot` | `sdot` | strided dot product |
| `dnrm2` | `snrm2` | scaled Euclidean norm |
| `dgemv` | `sgemv` | `y := alpha*op(A)*x + beta*y` |
| `dgemm` | `sgemm` | `C := alpha*op(A)*op(B) + beta*C` |
| `dger` | `sger` | general rank-one update |
| `dsyr`/`dsyr2` | `ssyr`/`ssyr2` | symmetric rank-one/rank-two updates |
| `dsyrk` | `ssyrk` | symmetric rank-k update, returned as a full matrix |
| `dsyr2k` | `ssyr2k` | symmetric rank-2k update, returned as a full matrix |
| `dsymm` | `ssymm` | left/right symmetric matrix multiplication |
| `dtrsv` | `strsv` | in-place triangular vector solve |
| `dtrsm` | `strsm` | in-place left/right triangular multi-RHS solve |
| `dcsrmv` | `scsrmv` | CSR matrix times dense vector |
| `dcsrmm` | `scsrmm` | CSR matrix times row-major dense matrix |
| `dcsrgemm` | `scsrgemm` | canonical CSR times CSR product |
| `dcsrsv` | `scsrsv` | in-place CSR triangular vector solve |
| `dcsrpotrf` | `scsrpotrf` | reusable sparse SPD Cholesky factorization |
| `dpotrf` | `spotrf` | lower Cholesky factorization |
| `dgetrf` | `sgetrf` | partial-pivoted LU factorization |
| `dsytrf` | `ssytrf` | pivoted symmetric-indefinite LDL' factorization |
| `dgeqp3` | `sgeqp3` | column-pivoted Householder QR |
| `dsyev` | `ssyev` | symmetric eigenvalues and eigenvectors |
| `dsygvd` | `ssygvd` | generalized symmetric eigenproblem with an SPD metric |
| `dgesvd` | `sgesvd` | thin singular-value decomposition |

`MatrixTranspose`, `MatrixTriangle`, `MatrixDiagonal`, and `MatrixSide` make
transpose and triangular conventions explicit. The older allocating
`axpy`, `dot`, and `matrixMultiply(double[][], double[][])` methods remain
available and compatible.

GEMV, GEMM, SYRK, and TRSM also have no-copy overloads with array offsets,
row-major leading dimensions, and vector strides. These operate directly on
submatrices in larger work arrays. `dgemmBatched`/`sgemmBatched`,
`dpotrfBatched`/`spotrfBatched`, and `dgetrfBatched`/`sgetrfBatched` accept
independent same-shaped jobs. Batches preserve provider execution but do not
promise a single fused kernel; measure the selected provider for the matrix
sizes used by the application.

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

Sparse Cholesky reads only the triangle named by `MatrixTriangle`, so callers
may supply lower-only, upper-only, or fully stored symmetric input without
double-counting mirrored entries. The default `SparseOrdering.MINIMUM_DEGREE`
reduces fill; `SparseOrdering.NATURAL` preserves input order. Duplicate entries
within the authoritative triangle are summed.

`dcsrgemm`/`scsrgemm` multiply two sparse matrices without dense
materialization, combine duplicate contributions, remove exact zeros, and
return sorted one-based CSR. `dcsrsv`/`scsrsv` solve lower or upper CSR
triangular systems, including transposed and unit-diagonal forms. These two
operations currently use the portable sparse implementation on every provider;
execution planning reports that fallback explicitly.

`prepareDcsr` and `prepareScsr` return owned CSR handles for repeated products.
The portable implementation retains immutable CSR storage and dispatches each
product through the selected provider. `ComputeCapabilities.preparedSparseMatrices()`
distinguishes providers that keep a provider-optimal prepared representation
from this compatible heap-backed behavior. CUDA retains values, column indices,
and row starts in device memory while each dense vector or right side remains a
per-call transfer; OpenCL and Vulkan currently use the compatible non-resident
handle.

`prepareDge` and `prepareSge` retain a general dense left operand for repeated
GEMM-family products. CUDA, OpenCL, and Vulkan keep that operand in a device
buffer; oneMKL and OpenBLAS retain host storage and dispatch repeated products
through CBLAS. The prepared handles also accept batches. AUTO selects a
provider-resident handle above its conservative preparation threshold.

## Reusable decompositions

`dpotrf`/`spotrf` compute lower Cholesky factors with repeated vector/matrix
solves and a stable log determinant. `dgeqp3`/`sgeqp3` compute column-pivoted
Householder QR with rank reporting and a full-rank least-squares solve. The
results are `CholeskyFactor`/`FloatCholeskyFactor` and
`PivotedQrFactor`/`FloatPivotedQrFactor`. Factor objects own their storage and
never expose a mutable internal array.

`dgetrf`/`sgetrf` return `LuFactor`/`FloatLuFactor` with the final row
permutation, determinant sign, log absolute determinant, and vector or
multi-right-side solves. `dsytrf`/`ssytrf` return pivoted LDL' factors whose D
storage supports both 1x1 and 2x2 blocks, so zero-diagonal indefinite matrices
do not require artificial regularization. Their solves permute inputs and
outputs back to original coordinates.

```java
CholeskyFactor factor = blas.dpotrf(covariance, dimension);
double[] solution = factor.solve(rightHandSide);
double logDeterminant = factor.logDeterminant();
```

`dcsrpotrf` and `scsrpotrf` return `SparseCholeskyFactor` and
`FloatSparseCholeskyFactor`. They retain a CSR lower factor in permuted
coordinates, expose the new-to-original permutation and factor nonzero count,
and solve vectors or row-major multi-RHS matrices in the caller's original
coordinates:

```java
SparseCholeskyFactor factor = blas.dcsrpotrf(relationship,
    MatrixTriangle.LOWER);
double[] coefficients = factor.solve(rightHandSide);
double logDeterminant = factor.logDeterminant();
```

For changing coefficients with a fixed sparsity pattern, use
`prepareDcsrpotrf` or `prepareScsrpotrf`. The initial call performs symbolic
analysis and numerical factorization. `refactor` then reuses that analysis,
rejecting any change to the authoritative-triangle structure; solves and log
determinants always refer to the latest successful numeric factor.

```java
try (PreparedSparseCholesky factor = blas.prepareDcsrpotrf(
        relationship, MatrixTriangle.LOWER)) {
    factor.refactor(updatedRelationship);
    factor.solveInPlace(rightHandSides, columns);
}
```

For repeated REML/GLS solves, `prepareDpotrf` and `prepareSpotrf` return owned
`PreparedCholesky` handles with in-place multi-RHS solves and a cached log
determinant. This avoids refactorizing the same covariance or mixed-model
coefficient matrix at every likelihood evaluation.

`dsyev`/`ssyev` accept a fully stored real symmetric matrix and return ascending
eigenvalues plus a row-major orthogonal eigenvector matrix whose columns are the
corresponding eigenvectors. `dgesvd`/`sgesvd` work for tall, square, and wide
matrices and return the thin decomposition `A = U*S*Vt`, with singular values
in descending order. Thin `U` has shape `rows` by `min(rows,columns)` and thin
`Vt` has shape `min(rows,columns)` by `columns`.

`dsygvd`/`ssygvd` solve `A*x = lambda*B*x` for symmetric A and SPD B. Returned
eigenvectors are columns and are B-orthonormal. This directly supports spectral
REML, genomic relationship transformations, and other generalized Rayleigh
problems without exposing temporary Cholesky transformations.

The deterministic Java CPU algorithms are the reference. oneMKL uses LAPACKE
when the installed runtime exports it; OpenBLAS does likewise and otherwise
uses the Java decomposition fallback. CUDA, OpenCL, and
Vulkan execute all four decomposition families natively for FP64 and FP32.
Their initial device implementation uses serial device kernels optimized for a
portable correctness baseline and avoiding host-side factorization; large-scale
tiled or vendor-library implementations can replace those kernels behind the
same API. `Compute.AUTO` uses a cubic work estimate to keep small decompositions
on CPU and route sufficiently large ones to the selected accelerator.

oneMKL/OpenBLAS use LAPACKE GETRF and SYGVD when the installed runtime exports
them. The Java reference provides LU, LDL', and generalized eigen on every
platform. CUDA, OpenCL, and Vulkan currently use that portable implementation
for those three newly added factorization families while retaining native
Cholesky, QR, ordinary symmetric eigen, and SVD kernels; execution plans expose
the boundary rather than describing a host fallback as device work.

Immutable sparse factors continue to use the deterministic Java representation.
Prepared oneMKL factors use PARDISO when the installed runtime exports
`pardiso`, `pardisoinit`, and `pardiso_getdiag`: phase 11 performs analysis,
phase 22 performs the initial and subsequent numeric factorizations, phase 33
solves one or more right sides, and close releases PARDISO state. Natural
ordering requests retain the portable implementation; minimum-degree requests
allow PARDISO to own its fill-reducing permutation. OpenBLAS and GPU providers
currently retain the portable prepared-factor fallback.

## Native CPU providers

Add `jdistlib-nativecpu` (or use `jdistlib-all`) and select an installed runtime
strictly:

```java
try (ComputeSelection selection = ComputeBackends.select(Compute.ONEMKL)) {
    ComputeBackend blas = selection.backend();
    // CBLAS/LAPACKE calls use the system oneMKL runtime.
}
```

Use `Compute.OPENBLAS` for OpenBLAS. The module uses JNA but deliberately does
not redistribute either native library. It searches ordinary loader paths;
oneMKL also searches `ONEAPI_ROOT`. Explicit paths can be supplied with
`-Djdistlib.onemkl.library=...` or `-Djdistlib.openblas.library=...`. Native CPU
choices are strict and fail if the requested runtime is absent. `Compute.CPU`
always remains the portable Java reference.

## Execution identification

`ComputeSelection.deviceInfo()` reports provider version, API/runtime version,
driver version, vendor, device, architecture, device identifier, and visible
memory. `selection.plan(operation, precision, dimensions...)` reports whether
an operation is expected to run as Java reference code, native CPU code, a
parallel GPU kernel, or a serial GPU factorization/triangular kernel. Under
`Compute.AUTO`, the plan includes the threshold decision and concrete backend.

```java
System.out.println(selection.deviceInfo().description());
System.out.println(selection.plan(LinearAlgebraOperation.SYRK,
    NumericPrecision.FP64, markers, markers, samples).description());
```

For linear models and REML, `SYRK`, `GEMM`, `GEMV`, `TRSM`, prepared Cholesky,
and symmetric eigendecomposition cover the principal cross-product, covariance
solve, log-determinant, and spectral steps. Pedigree relationship matrices can
use CSR matrix-vector/matrix products and reusable sparse Cholesky solves without
dense materialization. Record the `CSR_POTRF` execution plan separately from
the plans for accelerated CSR products.

## Provider and numerical contract

`ComputeCapabilities` separately reports dense BLAS, sparse BLAS, native dense
factorizations, provider-prepared sparse matrices, native sparse factorization,
reusable sparse analysis, prepared dense storage, and batched API support.
CUDA, OpenCL, and Vulkan currently accelerate both FP64
and FP32 dense and CSR operations with native precision buffers and kernels.
Reduction order and final rounding can differ across providers, so callers
should compare numerical tolerances rather than bits and record the selected
backend and device with reproducible results. FP32 is intended for workloads
that accept its smaller dynamic range and precision in exchange for lower
storage and potentially higher device throughput; it is never silently used
for an FP64 call.

Heap-backed calls include host/device transfer and synchronization. Small
operations normally favor CPU. Prepared dense and sparse handles amortize the
fixed operand transfer for repeated large operations such as mixed-model solves
and fine-mapping residual updates; per-call right sides and results still cross
the JVM/provider boundary.

Service-loaded providers can also be selected by exact identifier through
`ComputeBackends.byId` or the `jdistlib.compute.backend` system property.
Thread counts and affinity remain settings of the installed oneMKL/OpenBLAS
runtime so applications can coordinate BLAS threads with model-level
parallelism.

Run `gradlew :jdistlib-nativecpu:nativeSparseBenchmark` for a reproducible
prepared-CSR and sparse analysis/refactor/solve comparison. Override its size
with `-Djdistlib.benchmark.dimension=...` and repetitions with
`-Djdistlib.benchmark.repetitions=...`; benchmark numbers are measurements of
the current machine, runtime, and matrix pattern rather than performance
guarantees.
