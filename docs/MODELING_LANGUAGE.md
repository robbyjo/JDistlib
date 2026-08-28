# JDistlib modeling language 0.8

The JDistlib language has a **Stan source-compatible core with a Java-native
runtime**. It lowers into the same `BayesianModel` intermediate representation
as `ModelBuilder`, so the text and Java frontends share constraint, gradient,
sampler, diagnostic, and graph behavior. `ModelScript.LANGUAGE_VERSION`
identifies the JDistlib language and `ModelScript.STAN_SOURCE_COMPATIBILITY`
identifies the portable Stan core. See `STAN_SOURCE_COMPATIBILITY.md` for the
exact boundary and execution differences.

```stan
data {
  int<lower=0> n;
  int<lower=0> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(2, 2);
  y ~ binomial(n, theta);
}
generated quantities {
  int y_rep = binomial_rng(n, theta);
}
```

Compile and sample it with:

```java
CompiledModelScript script = ModelScript.compile(source, data);
ChainResult chain = new NoUTurnSampler().sample(
    script.model(), script.model().initialState(), options, random);
Map<String, double[]> generated = script.generate(chain.sample(0), random);
```

For source intended to remain ordinary Stan syntax, use the explicit alias:

```java
CompiledModelScript script = ModelScript.compileStan(stanSource, data);
```

For a file-backed workflow, read the script and dataset before binding the
data map:

```java
double[] y = McmcDataIngestionExamples.readNumericColumn(
    Paths.get("examples/data/normal-observations.csv"), "y");
String source = new String(Files.readAllBytes(
    Paths.get("examples/models/41-normal-csv-mean.jdm")),
    StandardCharsets.UTF_8);
Map<String, double[]> data = new LinkedHashMap<String, double[]>();
data.put("N", new double[] {y.length});
data.put("y", y);
CompiledModelScript compiled = ModelScript.compile(source, data);
```

The executable `examples/McmcDataIngestionExamples.java` fits that CSV through
both `ModelBuilder` and the script compiler. Its intentionally small CSV reader
does not implement quoted fields; applications with general CSV input should
use a dedicated CSV library and pass the resulting primitive arrays to JDistlib.

Here is a second complete program: a normal location/scale model with a
vectorized likelihood and a posterior-predictive draw.

```stan
data {
  int<lower=1> N;
  vector[N] y;
}
parameters {
  real mu;
  real<lower=0> sigma;
}
model {
  mu ~ normal(0, 10);
  sigma ~ exponential(1);
  y ~ normal(mu, sigma);
}
generated quantities {
  real y_rep = normal_rng(mu, sigma);
}
```

Supply scalars as one-element arrays and vectors as ordinary arrays. The data
map is copied at compilation, so callers may safely reuse or mutate their input
arrays afterward. `examples/ModelScriptExamples.java` compiles both this model
and a gamma-Poisson count-rate model.

## Supported surface

- `data`, `transformed data`, `parameters`, `transformed parameters`, `model`,
  and `generated quantities` blocks;
- scalar `real`, `int`, and `complex`; real/complex `vector`, `row_vector`, and
  `matrix`; procedural nested `tuple` values; `ordered`,
  `positive_ordered`, `sum_to_zero_vector`, `simplex`, and `unit_vector`;
- arbitrary-rank arrays, including arrays of vectors and matrices, with
  one-based scalar/partial/range/all indexing, slices, and indexed assignment;
- rectangular array (`{...}`), vector (`[...]'`), row-vector (`[...]`), and
  matrix (`[[...], [...]]`) literals with shape checking;
- `cov_matrix`, `corr_matrix`, `cholesky_factor_cov`, and
  `cholesky_factor_corr`, with exact Java-native transforms/Jacobians;
- scalar/container lower, upper, finite lower/upper, offset/multiplier,
  positive-ordered, and sum-to-zero parameter constraints;
- type-checked real/complex matrix/vector products, conjugate transpose (`'` or `transpose`),
  `cholesky_decompose`, thin QR (`qr_thin_Q`/`qr_thin_R`), `inverse`,
  `determinant`, `log_determinant`,
  `mdivide_left_spd`/`mdivide_right_spd`, SPD inverse/log-determinant,
  triangular products/symmetrization, trace quadratic forms, and covariance/
  Cholesky `multi_normal` kernels;
- `+`, `-`, `*`, `/`, exponentiation, comparisons, boolean expressions,
  parentheses, scalar-local assignment (`=`, `+=`, `-=`, `*=`, `/=`), sampling
  statements, and `target +=`;
- scalar and container locals (initialized or uninitialized) in procedural blocks, scoped
  statement blocks, `if`/`else`, integer-range `for`, and guarded `while`;
- Stan's `distribution_lpdf(y | ...)`/`distribution_lpmf(y | ...)` separator as
  well as comma-separated calls;
- elementary exponential/logarithmic, power/root, rounding, trigonometric,
  hyperbolic, error/gamma, logit, normal-CDF, stable log-space, beta, and
  binomial-coefficient scalar functions; see the table below;
- more than thirty scalar probability families: `std_normal`, `normal`,
  `lognormal`, `student_t`, `cauchy`, `double_exponential`, `logistic`, `gumbel`,
  `skew_normal`, `exp_mod_normal`, `von_mises`, `exponential`, `gamma`, `inv_gamma`, `chi_square`,
  `inv_chi_square`, `scaled_inv_chi_square`, `weibull`, `frechet`, `rayleigh`,
  `beta`, `beta_proportion`, `uniform`, `pareto`, `pareto_type_2`, `bernoulli`,
  `bernoulli_logit`, `binomial`, `binomial_logit`, `beta_binomial`,
  `hypergeometric`, `neg_binomial`, `neg_binomial_2`, `neg_binomial_2_log`,
  `poisson`, `poisson_log`, `geometric`, and `discrete_range`;
- corresponding `_lpdf`/`_lpmf` calls and generated-quantity `_rng` calls;
- scalar/container/array/tuple user functions and overloads, `data` arguments,
  forward declarations, guarded recursion, Stan probability suffixes and `_lp` target effects,
  `return`, conditional expressions,
  initialized transformed containers, symmetric scalar/container probability
  broadcasting, explicit `.*`/`./`, reductions, `dot_product`, shape queries,
  conversions, repetition constructors, append/head/tail/segment functions,
  block/row/column extraction, softmax/log-softmax, cumulative/sort/reverse
  transforms, diagonal multiplication, quadratic forms, and cross products;
- Stan CSR matrix multiply/conversion/extraction functions, immutable Java
  `CsrMatrix`, Java external-function bindings with supplied Jacobians, and
  higher-order `integrate_1d`, algebraic, RK45, BDF, and DAE callbacks with
  propagated parameter sensitivities.

Sampling a data vector with scalar distribution parameters is vectorized.
Gamma follows Stan's shape/rate convention in scripts; JDistlib's underlying
`Gamma.random` scale argument is converted internally. Compiled script factors
execute on thread-local reusable reverse tapes. Atomic reverse nodes cover
normal, Student-t, dot/distance, matrix-normal, external callback, and numerical
solver results; numerical callbacks use isolated forward/finite-difference
sensitivity work before attaching their result to the main reverse tape.
Java-authored targets may use `ReverseModeLogDensity` directly.

### Complex, tuple, sparse, and external example

```stan
functions { real java_penalty(real x, real scale); }
parameters { complex z; real x; real scale; }
model {
  tuple(real, complex) state = (x, z);
  vector[2] sparse_product = csr_matrix_times_vector(
      2, 2, [1,2]', {1,2}, {1,2,3}, [x,scale]');
  target += get_real(sin(state.2)) + sum(sparse_product)
            - java_penalty(state.1, scale);
}
```

Bind the forward declaration at compilation:

```java
ExternalFunctionRegistry externals = ExternalFunctionRegistry.builder()
    .bind("java_penalty", args -> ExternalFunctionResult.scalar(
        args[1][0] * args[0][0] * args[0][0],
        2 * args[1][0] * args[0][0], args[0][0] * args[0][0]))
    .build();
CompiledModelScript compiled = ModelScript.compileStan(source, data, externals);
```

External Jacobian columns follow flattened argument order. Complex Java data is
interleaved `[real0, imaginary0, ...]`. Tuples are currently local/function
values; top-level tuple data/parameters and arrays of tuples fail explicitly.

### Indexing example

```stan
data {
  array[2, 3] real x;
  array[4] matrix[3, 2] design;
}
model {
  array[2, 3] real work = x;
  work[1, 2:3] = rep_array(0, 2);
  work[:, 1] += 1;
  target += sum(work) + sum(design[2, 1:3, 2]);
}
```

### Matrix and user-function example

```stan
functions {
  vector predict(data matrix X, vector beta) {
    return X * beta;
  }
}
data {
  matrix[3, 2] X;
  vector[3] y;
  matrix[3, 3] Sigma;
}
parameters {
  vector[2] beta;
}
model {
  y ~ multi_normal(predict(X, beta), Sigma);
}
```

### Scalar function groups

| Group | Functions |
| --- | --- |
| Constants | `pi`, `e`, `sqrt2`, zero-argument `log2`/`log10`, `machine_precision`, infinities, and NaN |
| Exponential and log | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`, `log1m`, `log1p_exp`, `log1m_exp` |
| Powers and rounding | `sqrt`, `cbrt`, `square`, `inv`, `inv_square`, `inv_sqrt`, `abs`/`fabs`, `floor`, `ceil`, `round`, `trunc` |
| Trigonometric/hyperbolic | `sin`, `cos`, `tan`, `sinpi`, `cospi`, `tanpi`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Special | `erf`, `erfc`, `tgamma`, `lgamma`, `digamma`, `trigamma`, `Phi`, `Phi_approx`, `inv_Phi` |
| Stable transforms | `inv_logit`, `logit`, `log_inv_logit`, `log1m_inv_logit`, `log_inv_logit_diff`, `log_sum_exp`, `log_diff_exp`, `log_mix`, `multiply_log` |
| Other scalar | `pow`, `fma`, `hypot`, `fmin`/`min`, `fmax`/`max`, `fdim`, `fmod`, `lbeta`, `binomial_coefficient_log`, rising/falling factorials, `step`, `int_step`, `sign`, and finite/NaN predicates |

These scalar kernels broadcast elementwise over vectors, row vectors, arrays,
and matrices. Multi-argument calls accept scalars plus at most one typed shape;
two non-scalars must have identical base type, array rank, and dimensions.
Reductions and linear algebra retain their explicit signatures.

Straight-line sampling and `target +=` statements remain separate dependency-
aware model factors, preserving incremental factor-cache behavior. A model that
uses locals or control flow is lowered as one ordered procedural factor so local
assignments and branch/loop execution have deterministic statement semantics.

The parser reports line and column locations. It validates required data,
integer values, declared lengths and bounds, parameter dimensions, supported
constraints, distribution arity, and unknown names while lowering the model.

## Ahead-of-time workflow

`ModelSourceGenerator.generate` emits a Java class implementing
`GeneratedModelFactory`. `ModelScriptCli` provides the equivalent shell/Gradle
entry point:

```text
ModelScriptCli model.jdm com.example.GeneratedModel build/generated/GeneratedModel.java
```

`ModelCompilationCache` validates syntax, hashes the exact source with SHA-256,
compiles a wrapper with the host JDK, caches it under a caller-selected root,
and loads it through a closeable isolated class loader. Generated paths and
class names are validated before writing. The reference evaluator remains the
semantic oracle for generated wrappers.

The CLI generates Java source; it does not run `javac`, bind a dataset, or
sample the model. A complete Windows command-line workflow is:

```text
gradlew.bat jar
java -cp build\libs\jdistlib-0.8.4.jar jdistlib.inference.lang.ModelScriptCli examples\models\41-normal-csv-mean.jdm com.example.NormalCsvMean build\generated\com\example\NormalCsvMean.java
javac -cp build\libs\jdistlib-0.8.4.jar -d build\generated-classes build\generated\com\example\NormalCsvMean.java
```

Instantiate `com.example.NormalCsvMean` as a `GeneratedModelFactory` and call
`compile(data)`. Alternatively, `ModelCompilationCache.compile(source,
cacheDirectory)` performs generation, `javac`, hashing, caching, and isolated
loading in one JDK-backed call.

Language additions within 0.8.x are additive and versioned. Existing accepted
programs retain their meaning; incompatible syntax or distribution
parameterization changes require a new language version.

## Example catalog and build gate

`examples/models` contains fifty standalone `.jdm` programs spanning every
supported declaration, constraint, transformed block, distribution, RNG,
vectorization/indexing form, and manual-target pattern. Models 42–50 are focused
0.8.2 examples for the expanded distributions, scalar math, locals, and control
flow. `examples/stan` adds forty-one ordinary `.stan` fixtures, including
twenty-nine v0.8.3 examples for literals, forward declarations, complex/tuple
values, sparse and external functions, container algorithms, matrix pipelines,
structured types, quadrature, and numerical solvers. Run:

```text
./gradlew validateModelScripts
```

The validator compiles every script with representative data, checks its
analytic gradient against central finite differences, requires finite initial
density, and evaluates generated quantities. `check` depends on this task, so
the catalog is part of the release gate. Browse annotated links at
`docs/model-script-examples.html`. Stan users should also read
`docs/stan-users.html` for a concept-by-concept migration guide and the exact
compatibility boundary.
