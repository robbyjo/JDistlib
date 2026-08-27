# JDistlib 0.8.0

JDistlib 0.8.0 introduces a separate Bayesian inference layer without changing
the normalized distribution APIs. Named Java models and a versioned,
Stan-inspired script frontend lower into the same constraint- and
Jacobian-aware representation. The release includes Metropolis, slice, Gibbs,
adaptive-rejection conditional updates, HMC, and adaptive multinomial NUTS with
deterministic multi-chain execution.

Inference results are immutable and include warmup, retained sampler
statistics, cancellation status, warnings, and in-memory state-and-stream restart points.
Diagnostics cover rank-normalized split and folded R-hat, bulk/tail ESS, MCSE,
divergences, tree-depth saturation, acceptance, and E-BFMI. Chart-neutral trace,
rank, ACF, energy, pair, and model graph data export to JSON, CSV, SVG,
self-contained HTML, and Graphviz DOT without requiring a UI toolkit.

The 0.8 modeling language supports validated data and constrained parameter
declarations, transformed blocks, vectorized sampling statements, `target +=`,
forward-mode derivatives, generated quantities/RNGs, ahead-of-time Java source,
SHA-256 compilation caching, isolated class loading, and a CLI workflow. It is
Stan-inspired rather than a claim of Stan compatibility.

Reference and regression coverage includes conjugate beta-binomial and normal
posteriors, gradients/Jacobians, ordered and simplex transforms, heavy-tailed
and mixed discrete/continuous sampling, checkpoints, deterministic chains,
diagnostic/export schemas, model graphs, generated source, and strict Java 8
compilation. See `docs/INFERENCE.md`, `docs/MODELING_LANGUAGE.md`, and
`docs/INFERENCE_COMPATIBILITY.md` for the complete contracts.

The learning center now includes an end-to-end inference tutorial, a sampler
and diagnostics guide, posterior-predictive and difficult-geometry vignettes, a
second complete Stan-inspired script lesson, and a catalog backed by fifteen
named executable models. Sampling and post-processing also reuse chain-local
model buffers, Metropolis proposals, covariance scratch storage, and
allocation-free chain/metric read paths.

The release also ships forty standalone Stan-inspired `.jdm` examples and a
semantic catalog validator run by every `check` build. Feature-focused Java
examples cover copulas, mixtures, transformations, FDR, custom laws, numerical
integration, and both programmatic and scripted MCMC. The website’s **Browse
examples** center links every catalog and source file.

## Previous release: JDistlib 0.7.2

JDistlib 0.7.2 hardens the copula framework introduced in 0.7.0. It expands
independent validation for mixed and simplified-vine likelihoods, adds
auditable fitted-model diagnostics, and incorporates API feedback through
additive conveniences while preserving the existing public API, deterministic
seeded behavior, and Java 8 bytecode.

Validation and numerical improvements include:

- checked-in 90-digit Decimal reference corpora for central, rare-event, and
  boundary-heavy Clayton mixed measures and three-dimensional C-/D-vine log
  densities, with an implementation-independent generator;
- analytic conditional CDFs for Clayton, Gumbel, and Frank pair copulas, direct
  analytic inverses for Clayton and Frank, and the existing numerical fallback
  for custom copulas; and
- a scaled/log-domain Clayton conditional calculation that remains stable near
  unit-cube boundaries.

API and diagnostic improvements include:

- row-level `CopulaLikelihoodDiagnostics`, including individual log-density
  contributions, boundary distances, summary statistics, and the first
  non-finite observation;
- `CopulaLogLikelihoodResult`, which retains mixed-measure contributions,
  numerical warnings, CDF evaluation cost, maximum reported error, and the
  first failing row;
- fitted pair and vine diagnostics, observation counts, and vine AIC/BIC;
- deterministic midpoint and explicit-seed overloads for mixed fitting,
  selection, and vine fitting; and
- defensive marginal-array access and conventional getters on existing result
  objects without removing their original fields or methods.

The beginner copula tutorial, complete copula guide, compiled examples, API
guide, website, changelog, and TODO ledger describe the additions. Regression
coverage includes independent mixed/vine references, boundary conditionals,
diagnostic aggregation, fitted-model information criteria, defensive copies,
and exact equivalence between seed overloads and caller-owned
`MersenneTwister` streams.

Release assets include the binary library, sources, JavaDoc, and SHA-256
checksums. See `CHANGELOG.md` for the detailed change list and `PUBLISHING.md`
for the separate Maven Central maintainer step.
