# JDistlib modeling language 0.8

The JDistlib language is **Stan-inspired**, not Stan-compatible. It lowers into
the same `BayesianModel` intermediate representation as `ModelBuilder`, so the
text and Java frontends share constraint, gradient, sampler, diagnostic, and
graph behavior. `ModelScript.LANGUAGE_VERSION` identifies the supported subset.

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
- scalar `real` and `int`, `vector[N]`, `ordered[N]`, and `simplex[N]`;
- scalar lower, upper, and finite lower/upper parameter constraints;
- one-based vector indexing;
- `+`, `-`, `*`, `/`, exponentiation, comparisons, boolean expressions,
  parentheses, scalar-local assignment (`=`, `+=`, `-=`, `*=`, `/=`), sampling
  statements, and `target +=`;
- initialized scalar `real` and `int` locals in the `model` block, scoped
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
- corresponding `_lpdf`/`_lpmf` calls and generated-quantity `_rng` calls.

Sampling a data vector with scalar distribution parameters is vectorized.
Gamma follows Stan's shape/rate convention in scripts; JDistlib's underlying
`Gamma.random` scale argument is converted internally. Expressions use
forward-mode derivatives, including `lgamma` through the digamma function.

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

These are scalar overloads. Array, matrix, reduction, and broadcasting overloads
are intentionally not implied by a shared Stan function name.

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
java -cp build\libs\jdistlib-0.8.2-SNAPSHOT.jar jdistlib.inference.lang.ModelScriptCli examples\models\41-normal-csv-mean.jdm com.example.NormalCsvMean build\generated\com\example\NormalCsvMean.java
javac -cp build\libs\jdistlib-0.8.2-SNAPSHOT.jar -d build\generated-classes build\generated\com\example\NormalCsvMean.java
```

Instantiate `com.example.NormalCsvMean` as a `GeneratedModelFactory` and call
`compile(data)`. Alternatively, `ModelCompilationCache.compile(source,
cacheDirectory)` performs generation, `javac`, hashing, caching, and isolated
loading in one JDK-backed call.

Language additions within 0.8.x are additive and versioned. Existing accepted
programs retain their meaning; incompatible syntax or distribution
parameterization changes require a new language version.

## Example catalog and build gate

`examples/models` contains forty-one standalone `.jdm` programs spanning every
supported declaration, constraint, transformed block, distribution, RNG,
vectorization/indexing form, and manual-target pattern. Run:

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
