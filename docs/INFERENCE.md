# Bayesian modeling and MCMC

JDistlib 0.8.0 adds a separate inference layer in `jdistlib.inference`. A
`GenericDistribution` is a normalized scalar law with a CDF and quantile. A
`LogDensity` is an unnormalized, possibly multivariate target on an
unconstrained Euclidean space. Keeping these contracts separate lets inference
reuse the distribution catalogue without inventing CDFs for posterior models.

## Choose the page that matches your goal

| Goal | Start here | What it provides |
| --- | --- | --- |
| Complete first analysis | [MCMC tutorial](inference-tutorial.html#worked) | A fully commented CSV → JDM → compilation → four-chain NUTS → diagnostics → plots → conclusion workflow |
| Bind data or compare frontends | [Data and model-script tutorial](modeling-language-tutorial.html) | The same data through `ModelBuilder` and `ModelScript`, plus compilation choices |
| Select/tune a sampler | [Inference reference](inference-guide.html) | Warmup, metrics, diagnostics, graphing, performance, checkpoints, and failures |
| Interpret an applied model | [Posterior vignette](inference-vignette.html) | Posterior prediction and communication |
| Debug difficult chains | [Diagnostics vignette](inference-diagnostics-vignette.html) | Trace/rank/pairs/energy evidence and geometry fixes |
| Migrate Stan code | [Guide for Stan users](stan-users.html) | Data, compilation, execution differences, output, and compatibility boundaries |
| Find more runnable code | [Example center](examples.html) | JDM, ordinary Stan, and Java integrations |

The complete executable is
[`examples/WorkedMcmcCsvJdmExample.java`](../examples/WorkedMcmcCsvJdmExample.java).
It is intentionally verbose and comments every stage of a reviewable analysis.

## Programmatic models

`ModelBuilder` names observed data, constrained parameters, and additive prior
or likelihood factors. The compiled `BayesianModel` owns the constraint maps and
automatically includes their log-Jacobian terms.

```java
BayesianModel model = new ModelBuilder()
    .data("trials", 10)
    .data("successes", 7)
    .parameter("theta", Constraints.bounded(0, 1), 0.5)
    .factor("theta prior", new String[] {"theta"},
        ModelFactors.betaPrior("theta", 2, 2))
    .factor("observations", new String[] {"successes", "trials", "theta"},
        ModelFactors.binomialObservation("successes", "trials", "theta"))
    .build();
```

Observed values can come from CSV, JSON, a database, or any application object;
the host application converts them to primitive arrays before calling
`ModelBuilder.data` or `ModelScript.compile`. The executable
`examples/McmcDataIngestionExamples.java` reads
`examples/data/normal-observations.csv` and fits the same posterior through
both frontends. `docs/modeling-language-tutorial.html` shows all three script
compilation modes, and `docs/stan-users.html` maps the workflow from Stan.

Built-in constraints cover real and positive scalars/vectors, finite bounded
scalars, ordered vectors, and simplexes. `DifferentiableModelFactor` adds
derivatives in constrained coordinates; the model pulls them through each
transform. `Gradients.check` compares a supplied gradient with central finite
differences. `model.evaluator()` creates a non-thread-safe, allocation-free
transform/gradient evaluator for one chain. `ModelEvaluationCache` reuses
unaffected factor values when a proposal declares its changed coordinates.
All bundled samplers select a chain-local evaluator automatically when given a
`BayesianModel`; callers only need `model.evaluator()` for custom algorithms or
benchmarks.

## Samplers

| Target or model | Recommended starting point |
| --- | --- |
| Differentiable continuous posterior | `NoUTurnSampler` |
| Short, predictable differentiable trajectory | `HamiltonianMonteCarlo` |
| No gradients, moderate dimension | `RandomWalkMetropolis` or `ComponentWiseMetropolis` |
| Difficult one-dimensional conditionals | `SliceSampler` |
| Known full conditionals or mixed variables | `GibbsSampler` with exact, ARS, or `MetropolisBlockKernel` updates |

NUTS uses multinomial candidate selection, dual-averaging step-size warmup,
diagonal or dense covariance metrics, and divergence and tree-depth reporting.
It never silently substitutes whole-target finite differences. A caller must set
`allowFiniteDifferences(true)` when deliberately accepting that slower and less
reliable fallback.

Every sampler takes a caller-owned `RandomEngine`. `Chains.parallel` derives
independent deterministic streams from one base seed and preserves results
regardless of worker scheduling. Each chain returns immutable retained draws,
log densities, iteration statistics, a warmup summary, warnings, and an
in-memory `ChainCheckpoint` containing the exact last state and cloned stream.
Use `SamplingOptions.cancellation` for cooperative cancellation and
`Chains.resume` to restart from a checkpoint. A generic checkpoint does not
serialize NUTS/HMC metric adaptation; provide zero warmup to retain caller
settings or rerun warmup when recalibration is desired.

## Diagnostics

Always run more than one independently initialized chain. `McmcDiagnostics`
reports:

- rank-normalized split R-hat and folded rank R-hat;
- bulk and tail effective sample sizes;
- Monte Carlo standard error for the posterior mean;
- posterior mean, standard deviation, median, and 95% interval;
- mean acceptance, divergences, maximum-tree-depth saturation, numerical
  failures, and energy Bayesian fraction of missing information (E-BFMI).

The default reliability flag requires R-hat below 1.01 and bulk/tail ESS of at
least 100. Those thresholds are a screening rule, not proof of convergence.
Divergences, low E-BFMI, multimodality, or scientifically implausible results
must be investigated even when R-hat is acceptable.

Interpret the diagnostics together:

| Signal | First interpretation | Typical response |
| --- | --- | --- |
| R-hat >= 1.01 | chains do not explore the same stationary distribution | check initialization, modes, and parameterization before merely running longer |
| low bulk ESS | central posterior summaries are noisy | improve geometry or collect more post-warmup draws |
| low tail ESS | intervals and tail probabilities are noisy | reparameterize and collect more draws |
| divergence | numerical trajectory crossed difficult geometry | check gradients, non-center hierarchies, then consider higher target acceptance |
| tree-depth saturation | NUTS used its full trajectory budget | inspect geometry and ESS before raising the limit |
| E-BFMI < 0.3 | momentum resampling explores energy poorly | rescale/reparameterize and inspect the energy plot |

`McmcDiagnostics` compares the common retained suffix when chain lengths differ.
E-BFMI is computed per chain and the report retains the minimum, so one unhealthy
chain cannot be hidden by averaging. A single chain can yield ESS and MCSE but
cannot yield a meaningful between-chain R-hat.

`McmcDiagnosticReport.toJson()` uses schema `jdistlib.mcmc-diagnostics/1`.
`ChainExport` supplies `jdistlib.chains/1` JSON and tidy CSV. Non-finite JSON
numbers are represented as `null`.

## Graphing and reports

`DiagnosticGraphs` creates immutable, chart-neutral `ChartSpec` datasets for:

- traces;
- rank histograms;
- autocorrelation;
- Hamiltonian energy; and
- pair plots.

Each chart exports versioned JSON, tidy CSV, or standalone SVG without JavaFX,
AWT, or a web runtime. `InferenceHtmlReport` combines summaries and embedded
SVGs into a self-contained report. `BayesianModel.graph()` exposes the
parameter/data/factor dependency graph, and `ModelGraphExport` writes versioned
JSON or Graphviz DOT. UI libraries and notebooks can consume these same neutral
contracts rather than scraping renderer-specific objects.

Start every review with trace and rank plots, add autocorrelation when ESS is
low, use energy plots for HMC/NUTS, and use pairs plots to locate funnels or
strong posterior correlations. Chart coordinates are unconstrained sampler
coordinates; use `model.state(draw)` or `constrainedSamples(model)` when a
scientific plot must show bounded, positive, ordered, or simplex values.

```java
McmcDiagnosticReport report = McmcDiagnostics.analyze(names, chains);
ChartSpec trace = DiagnosticGraphs.trace("theta", 0, chains);
ChartSpec energy = DiagnosticGraphs.energy(30, chains);
String html = InferenceHtmlReport.render("Posterior review", report,
    model.graph(), trace, energy);
```

## Performance notes

The 0.8.0 samplers reuse chain-local model buffers; random-walk Metropolis also
reuses proposal storage. Metric operations avoid temporary velocity arrays,
warmup covariance reuses its scratch vector, and observation factors read the
immutable data store without cloning it. Diagnostics, chart construction, and
CSV/JSON export use scalar chain accessors rather than deep-copying draws.

For best throughput, retain only scientifically useful draws, prefer thinning
only when storage or downstream cost requires it, use analytic gradients, and
run independent chains in parallel. Dense mass matrices can pay off for a
moderate strongly correlated parameter block, but have quadratic storage and
linear-algebra costs. `examples/InferenceBenchmark.java` is a reproducible smoke
benchmark; use JMH for publishable measurements.

## Reparameterization and failures

Start with finite initial log density and gradients. For hierarchical scales,
prefer a non-centered parameterization when a funnel geometry causes
divergences. Increase `targetAcceptance` only after checking the model and its
gradients. A saturated tree-depth limit is not fixed merely by raising the
limit; examine posterior geometry and effective sample size first.

Discrete variables cannot be evolved by HMC. Use a Gibbs or Metropolis block
for discrete coordinates and NUTS/HMC blocks for differentiable continuous
coordinates. Slice and adaptive-rejection updates remain useful for scalar full
conditionals. See `InferenceExamples.java` for a compiled end-to-end workflow.
The website adds a beginner tutorial, reference guide, posterior vignette,
diagnostics vignette, language tutorial, and a catalog backed by fifteen named
tests in `BayesianShowcaseTest`.

## Reproducibility contract

Within one JDistlib release, the same sampler, options, initial state, explicit
engine implementation, and seed produce the same chain on supported Java
runtimes. Parallel execution never shares an engine. Algorithmic changes may
change exact draws across minor releases; convergence summaries and reference
posterior tests are the compatibility contract unless a method explicitly
promises bitwise persistence.
