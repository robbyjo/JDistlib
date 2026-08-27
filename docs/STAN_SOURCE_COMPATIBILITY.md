# Stan source compatibility and Java execution semantics

JDistlib accepts an expanding core of ordinary Stan source through
`ModelScript.compileStan`. The goal is to preserve the statistical meaning of
supported source while executing it with JDistlib's Java-native model,
autodiff, random-number, sampler, diagnostic, and graph infrastructure. It is
not a wrapper around `stanc3`, Stan Math, CmdStan, or generated C++.

The compatibility identifier is
`ModelScript.STAN_SOURCE_COMPATIBILITY == "core-2026-08-v0.8.3"`. A program outside
the documented core must fail clearly; JDistlib must not silently reinterpret
an unsupported matrix operation, constraint, probability law, or solver.

## Source accepted by the compatibility core

The core accepts:

* the standard `functions`, `data`, `transformed data`, `parameters`,
  `transformed parameters`, `model`, and `generated quantities` blocks;
* arbitrary-rank modern and legacy arrays, including arrays of vectors and
  matrices, flattened only at the Java `double[]` data boundary while retaining
  complete runtime shape and base-type metadata;
* rectangular array, vector, row-vector, and matrix literals, with nested shape
  inference and rejection of ragged literals;
* one-based scalar, partial, chained, range, and all (`:`) indexing, slices,
  and scalar/container indexed assignment;
* container-valued user functions, forward declarations, lexical scopes, overload selection with
  integer-to-real promotion, `data`-qualified arguments, guarded recursion,
  `_lpdf`/`_lpmf` distribution functions, `_lp` target effects, and `return`;
* general scalar/vector/matrix `lower` and `upper` parameter bounds,
  `offset`/`multiplier`, `positive_ordered`, the orthogonal
  `sum_to_zero_vector`, `unit_vector`, covariance/correlation matrices, and
  covariance/correlation Cholesky factors, with exact transforms and
  log-Jacobians;
* type-checked matrix/vector products and transpose, Cholesky and thin-QR
  decompositions, inverse, determinant/log determinant, SPD solves, and differentiable
  `multi_normal` covariance and Cholesky kernels;
* scalar/container broadcasting for arithmetic, scalar functions, and
  probability arguments,
  explicit elementwise `.*` and `./`, conditional expressions, `sum`, `prod`,
  `mean`, `dot_product`, `num_elements`, `size`, `rows`, `cols`, `to_vector`,
  `to_row_vector`, `to_array_1d`, `rep_*` constructors, append/slice/sequence
  helpers, softmax/log-softmax, quadratic forms, and cross products; and
* the scalar math, probability, RNG, target-increment, generated-quantity,
  control-flow, diagnostic, and sampler surface listed in
  `MODELING_LANGUAGE.md`.

The checked `examples/stan` directory contains thirty ordinary `.stan` fixtures for
each compatibility family. `./gradlew validateModelScripts` binds data,
compiles them through `compileStan`, checks analytic gradients against central
differences, requires a finite initial density, and executes generated
quantities.

Stan's current type system includes arbitrary arrays, vectors, row vectors,
matrices, complex and tuple values, plus specialized constrained containers.
JDistlib currently implements only the real-valued core above. See the
[Stan type reference](https://mc-stan.org/docs/reference-manual/types.html) and
[Stan expression/indexing reference](https://mc-stan.org/docs/reference-manual/expressions.html)
for the upstream language definition.

## Execution semantics that intentionally differ

Source compatibility does not mean implementation identity:

| Concern | Stan/CmdStan | JDistlib |
| --- | --- | --- |
| Compiler/runtime | `stanc3`, generated C++, Stan Math | Java parser/lowering into `BayesianModel` |
| Differentiation | Stan Math reverse-mode and specialized kernels | Forward-mode script differentiation; reusable primitive-arena reverse mode is also available to Java model factors |
| Container storage | Typed Eigen/C++ containers | Shape metadata plus flattened Java arrays at the host boundary |
| Floating point | C++ compiler, math library, and Stan kernels | JVM `Math`, JDistlib distribution kernels, and Java evaluation order |
| Random streams | Stan RNG ownership and draw order | Caller-owned JDistlib `RandomEngine` streams |
| Sampling | CmdStan services and Stan warmup | JDistlib Metropolis, slice, Gibbs, HMC, or multinomial NUTS implementations |
| Output | Stan CSV and constrained draws | `ChainResult`, JSON/CSV export, graph data, SVG, and HTML reports |
| Work guards | Stan service limits | Explicit Java loop, recursion, cancellation, and sampler limits |

Consequently, the same seed does not promise the same draws, transition path,
adaptation state, divergence locations, or last bits of a log density. For a
supported model, the compatibility target is the same constrained parameter
space and log-density meaning, equivalent generated-quantity definitions, and
posterior agreement within appropriate numerical and Monte Carlo tolerances.
The [Stan execution reference](https://mc-stan.org/docs/reference-manual/execution.html)
is the upstream definition used when adding a compatible construct.

JDistlib extensions are allowed when they are unambiguous. For example,
elementwise container expressions may be explicit with `.*` and `./`.
Container `*` follows the declared `vector`, `row_vector`, and `matrix` types;
container right division remains explicit through named solve operations.

## Broadcasting compatibility matrix

| Operation family | Scalar + container | Same typed shape | Vector vs row vector | Different shapes | Array vs vector in a probability call |
| --- | --- | --- | --- | --- | --- |
| `+`, `-`, `.*`, `./` | broadcast | elementwise | rejected | rejected | rejected |
| `*` | scalar scaling | typed linear algebra | outer/dot product according to orientation | checked inner dimensions | rejected |
| scalar math functions | broadcast elementwise | elementwise | each preserves its input type | rejected for multi-argument calls | rejected |
| univariate `_lpdf`/`_lpmf` and `~` | broadcast | vectorized | compatible by scalar-element count | rejected when element counts differ | compatible by scalar-element count |
| matrix probability kernels | only where the signature says so | signature-specific | orientation checked | rejected | no implicit array batching |

This table is enforced by `StanAdvancedCompatibilityTest`; diagnostics include
the conflicting types and dimensions.

## Not yet source compatible

The following still require implementation before arbitrary Stan source can be
claimed:

* complex values, tuples, and sparse matrix operations;
* higher-order functions and external functions;
* complete Stan standard-library overloads, probability families,
  truncation/CDF semantics, reduce/map parallel constructs, and external
  functions;
* automatic lowering of every script expression onto the reverse tape and the
  remainder of Stan Math's specialized derivative kernels; and
* direct script bindings and sensitivity differentiation for numerical solvers.

JDistlib 0.8.3 provides Java-native damped-Newton algebraic, adaptive
Dormand-Prince ODE, and implicit-Euler index-1 DAE solvers. They use typed Java
callback interfaces until higher-order function syntax is source-compatible.
See `stan-solvers-tutorial.html` for the migration path and current algorithmic
scope (non-stiff ODEs and index-1 DAEs).

Use `compileStan` when the source is intended to be portable Stan syntax and
`compile` for the JDistlib language, which may include explicitly documented
extensions. Both return `CompiledModelScript` and use the same Java execution
and diagnostics pipeline.
