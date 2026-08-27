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
- `+`, `-`, `*`, `/`, exponentiation, parentheses, assignments, sampling
  statements, and `target +=`;
- `exp`, `log`, `sqrt`, `abs`, `lgamma`, and `inv_logit`;
- `normal`, `beta`, `gamma`, `exponential`, `bernoulli`, `binomial`, `poisson`,
  `uniform`, and `cauchy` log probabilities;
- corresponding `_lpdf`/`_lpmf` calls and generated-quantity `_rng` calls.

Sampling a data vector with scalar distribution parameters is vectorized.
Gamma follows Stan's shape/rate convention in scripts; JDistlib's underlying
`Gamma.random` scale argument is converted internally. Expressions use
forward-mode derivatives, including `lgamma` through the digamma function.

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

Language additions within 0.8.x are additive and versioned. Existing accepted
programs retain their meaning; incompatible syntax or distribution
parameterization changes require a new language version.

## Example catalog and build gate

`examples/models` contains forty standalone `.jdm` programs spanning every
supported declaration, constraint, transformed block, distribution, RNG,
vectorization/indexing form, and manual-target pattern. Run:

```text
./gradlew validateModelScripts
```

The validator compiles every script with representative data, checks its
analytic gradient against central finite differences, requires finite initial
density, and evaluates generated quantities. `check` depends on this task, so
the catalog is part of the release gate. Browse annotated links at
`docs/model-script-examples.html`.
