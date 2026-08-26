# Contributed distribution provenance

JDistlib contains distributions beyond R's `src/nmath`. This ledger records
where those APIs came from and keeps that work separate from the R 4.6.1 audit
in [UPSTREAM.md](UPSTREAM.md).

## Porting policy

A contributed distribution is eligible when:

* it is not already represented by a JDistlib distribution or a trivial fixed
  parameterization of one;
* its source license is compatible with JDistlib's GPL-2.0-or-later license;
* scalar laws can supply density/mass, cumulative, quantile, and random
  generation together; vector laws supply every mathematically canonical
  operation in the reference API (there is no general scalar quantile for a
  random vector); and
* the parameterization, source version, attribution, edge cases, and reference
  tests can be recorded.

New implementations use explicit `RandomEngine` instances and do not introduce
shared mutable caches.

## Audited ports

| JDistlib API | Reference implementation | Version | License | Verification |
| --- | --- | ---: | --- | --- |
| `Triangular` | [`extraDistr`](https://cran.r-project.org/package=extraDistr), `d/p/q/rtriang` | 1.10.0.5 | GPL-2 | R 4.6.1 reference vectors, endpoint-mode cases, inversion, simulation |
| `HalfNormal` | [`extraDistr`](https://cran.r-project.org/package=extraDistr), `d/p/q/rhnorm` | 1.10.0.5 | GPL-2 | R 4.6.1 reference vectors, logged tails, inversion, simulation |
| `BirnbaumSaunders` | [`extraDistr`](https://cran.r-project.org/package=extraDistr), `d/p/q/rfatigue` | 1.10.0.5 | GPL-2 | R 4.6.1 reference vectors, shifted support, inversion, simulation |
| `Gompertz` | [`flexsurv`](https://cran.r-project.org/package=flexsurv), `d/p/q/rgompertz` | 2.3.2 | GPL-2-or-later | Upstream regression vectors, exponential limit, hazards, defective negative-shape mass |
| `Categorical`, `Multinomial`, `PoissonBinomial`, `Empirical` | [`distributions3`](https://cran.r-project.org/package=distributions3) | 0.2.4 plus 0.3.0 development additions | MIT | Exact finite masses/CDFs, inverse empirical CDF, support and simulation checks |
| `ZeroInflatedPoisson`, `ZeroTruncatedPoisson`, `HurdlePoisson` | `distributions3`, `d/p/q/rzipois`, `ztpois`, `hpois` | 0.2.4 | MIT | Definition identities, logged tails, degenerate-rate limits, quantile inversion, simulation |
| `ZeroInflatedNegativeBinomial`, `ZeroTruncatedNegativeBinomial`, `HurdleNegativeBinomial` | `distributions3`, `d/p/q/rzinbinom`, `ztnbinom`, `hnbinom` | 0.2.4 | MIT | Mean/size parameterization, geometric special cases, logged tails, quantile inversion |
| `SinhArcsinh` | `distributions3`, `d/p/q/rsinharcsinh` | 0.3.0 development | MIT | Normal special case, asymmetric inversion, logged tails, simulation |
| `GeneralizedGamma` | [`VGAM`](https://cran.r-project.org/package=VGAM), `d/p/q/rgengamma.stacy` | 1.1-14 | GPL-3 | Gamma transformation identities, endpoints, logged tails, simulation |
| `GeneralizedBetaSecondKind` | `VGAM`, `dgenbetaII` and beta-transform definition | 1.1-14 | GPL-3 | Log-logistic and beta-transform special cases, logged tails, simulation |
| `Makeham`, `Lindley` | `VGAM`, `d/p/q/rmakeham`, `d/p/rlind` | 1.1-14 | GPL-3 | Gompertz special case, survival identities, numerical quantile inversion, simulation |
| `FoldedNormal`, `PositiveNormal`, `Rice`, `Maxwell`, `MaxwellBoltzmann` | `VGAM`, corresponding `d/p/q/r` APIs; JDistlib ticket 42 scale convention | 1.1-14 | GPL-3 | Half-normal, Rayleigh, noncentral-chi-square, gamma, and rate/scale identities |
| `PoissonInverseGaussian` | [`actuar`](https://cran.r-project.org/package=actuar), `d/p/q/rpoisinvgauss` | 3.3-7 | GPL-2-or-later | Published mass recurrence, quantile inversion, mixture mean and simulation |
| `GeneralizedF` | [`flexsurv`](https://cran.r-project.org/package=flexsurv), `d/p/q/rgenf` | 2.3.2 | GPL-2-or-later | Upstream reference vectors, generalized-gamma/lognormal limits, direct beta tails and inversion |
| `BetaNegativeBinomial`, `NegativeHypergeometric`, `DiscreteWeibull`, `Skellam` | [`extraDistr`](https://cran.r-project.org/package=extraDistr), corresponding `d/p/q/r` definitions completed where the package omits an operation | 1.10.0.5 | GPL-2 | Exact masses, finite-urn identities, geometric/Poisson reductions, inversion and simulation |
| `HalfCauchy`, `HalfT`, `Slash`, `TukeyLambda` | `extraDistr`, corresponding probability APIs and quantile definitions | 1.10.0.5 | GPL-2 | Symmetric-parent reductions, direct logged tails, completed numerical inverses and seeded simulation |
| `FellerPareto`, `PhaseType` | [`actuar`](https://cran.r-project.org/package=actuar), `d/p/q/rfpareto` and `d/p/rphtype` definitions | 3.3-7 | GPL-2-or-later | Beta-transform identities, Erlang reduction, matrix-exponential probabilities, atoms and Markov-chain simulation |
| `DiscreteLaplace`, `LogitNormal` | [`extraDistr`](https://cran.r-project.org/package=extraDistr), [`logitnorm`](https://cran.r-project.org/package=logitnorm), and CRAN task-view definitions | 1.10.0.5 / 0.8.39 / task view 2026-08-21 | GPL-2 | Two-sided geometric and normal-logit transforms, lattice/boundary behavior, inversion and simulation |
| `AsymmetricLaplace`, `ExponentiallyModifiedGaussian`, `Huber` | [`ald`](https://cran.r-project.org/package=ald), [`emg`](https://cran.r-project.org/package=emg), and [`extraDistr`](https://cran.r-project.org/package=extraDistr) | 1.3.1 / 1.0.9 / 1.10.0.5 | GPL-2-compatible | Published piecewise/normal-convolution formulas, symmetric limits, direct logged tails, inversion and seeded simulation |
| `Dirichlet`, `DirichletMultinomial` | [`MCMCpack`](https://cran.r-project.org/package=MCMCpack) and CRAN task-view definitions | 1.7-1 / task view 2026-05-07 | GPL-3 | Closed-form densities/masses, simplex support, Pólya mixture simulation and means |
| `MultivariateHypergeometric` | [`extraDistr`](https://cran.r-project.org/package=extraDistr) and task-view definition | 1.10.0.5 / task view 2026-05-07 | GPL-2 | Exact combinatorial mass, sequential conditional sampling, support checks |
| `BivariatePoisson` | [`bivpois`](https://cran.r-project.org/package=bivpois) | 1.2 | GPL-2-or-later | Independent-plus-shared Poisson identity, exact joint CDF, simulated covariance |
| `BivariateLogistic` | `VGAM`, `d/p/rbilogis` | 1.1-14 | GPL-3 | Closed-form density/CDF, endpoint behavior and conditional generator |
| `MultivariateNormal`, `MultivariateStudentT`, `MultivariateCauchy`, `MultivariateLogNormal` | [`mvtnorm`](https://cran.r-project.org/package=mvtnorm), Genz conditional transformation, task-view definitions | 1.4-2 / task view 2026-05-07 | GPL-2 | Cholesky density/generation, randomized rectangle probabilities with error estimates, independent quadrature, exact orthants, equicoordinate/radial quantiles |
| `MultivariateLaplace`, `MultivariatePowerExponential` | [`LaplacesDemon`](https://cran.r-project.org/package=LaplacesDemon), task-view definitions | 16.1.6 / task view 2026-05-07 | MIT | Normal-mixture and radial-gamma constructions, Bessel density, Gaussian/Laplace reductions and power-exponential radial quantiles |
| `Wishart` | R `stats::rWishart`, Bartlett decomposition and standard matrix density | R 4.6.1 / AS 53 | GPL-2-or-later | Scaled-chi-square reduction, density/log-density agreement, exact seeded Bartlett reduction, matrix means and support validation |

The numerical reference checks are in `AdditionalDistributionsTest`,
`ContributedPackageDistributionsTest`, `BacklogDistributionsTest`,
`TaskViewAdditionalDistributionsTest`,
`MultivariateDistributionsTest`, and `MultivariateProbabilityTest`.
The source packages are used as references; the public Java APIs follow
JDistlib naming conventions. Vector-valued laws are intentionally static vector
APIs rather than `GenericDistribution` instances, matching the absence of a
canonical one-dimensional CDF or quantile. Their shared covariance validation
and Cholesky workspaces are call-local and thread-safe.

## Package coverage decisions

The `distributions3` distribution constructors were audited individually.
Bernoulli and Erlang are fixed-parameter uses of `Binomial` and `Gamma`;
Fisher F, Student t, and the classical families already have direct JDistlib
counterparts; and its GEV, GP, Gumbel, Frechet, and reversed-Weibull families
are represented in `jdistlib.evd`. The table above therefore contains every
distinct missing family rather than adding aliases.

VGAM contains more than one hundred regression families, many of which are
model-fitting machinery rather than complete probability APIs. This batch
prioritizes distinct exported probability laws with complete or safely
completable density/CDF/quantile/random behavior and direct biomedical,
survival, actuarial, reliability, or financial use. Zero-altered aliases and
special cases of the new GB2/generalized-gamma families remain represented by
parameter conversion rather than duplicate classes.

The discrete and continuous multivariate sections of the CRAN task view were
also screened. This batch establishes the broadly reusable count, simplex, and
elliptical foundations. It includes density or mass and random generation, exact
joint CDFs for the bivariate Poisson and VGAM bivariate logistic laws, and
explicit lower/upper rectangular probabilities for the normal, Student t,
Cauchy, and log-normal families. Their named equicoordinate and radial quantiles
avoid implying a unique vector inverse CDF. Simplex/count CDFs and matrix-valued
quantiles remain intentionally absent because their region or ordering semantics
and computational cost require a separate contract.

## Historical contributed APIs

These predate this ledger and remain part of JDistlib:

| JDistlib API | Recorded source |
| --- | --- |
| `BetaBinomial`, `InvNormal` | `gamlss.dist` |
| `LogLogistic` | `actuar` 2.3-0 |
| `Tweedie` | `tweedie` 2.2.1, subsequently completed and regression-tested against 3.1.0 |
| Other JDistlib-only APIs | See their class documentation and repository history |

Historical entries will receive the same version/license/reference-vector audit
as they are revisited.

The beta-prime implementation was revisited during the 2026-08-21 task-view
screen. Its beta transformation, Jacobian, unbounded support, instance methods,
and generator are now covered by regression tests; the earlier implementation
incorrectly used a bounded transformation and truncated instance inputs.

## Completed screen and dispositions

The [CRAN Probability Distributions Task View](https://cran.r-project.org/view=Distributions)
was reviewed through version 2026-08-21. The former candidate queue has been
closed as follows. An implementation row means that JDistlib now provides the
complete meaningful probability API. A disposition row records why a family is
not silently represented by a partial or arbitrarily parameterized class.

| Screened group | Disposition | Notes |
| --- | --- | --- |
| Generalized F | Implemented as `GeneralizedF` | Direct smaller-tail beta evaluation avoids the underflow cases recorded by `flexsurv`. |
| Beta-negative-binomial, negative hypergeometric, discrete Weibull, Skellam | Implemented | Complete D/P/Q/R APIs include operations missing from some reference packages. |
| Half-Cauchy, half-t, slash, Tukey lambda | Implemented | Slash and Tukey lambda quantiles/CDFs complete reference APIs that expose only a subset. |
| Feller-Pareto and phase-type | Implemented | `PhaseType` preserves the optional atom at zero and uses call-local matrix workspaces. |
| Huber, asymmetric Laplace, exponentially modified Gaussian | Implemented | These distinct scalar laws have complete logged-tail D/P/Q/R APIs; the EMG quantile replaces the reference package's unconstrained optimizer with monotone inversion. |
| Truncated and zero-inflated families | Covered by composition | `TruncatedContinuousDistribution`, `ModifiedCount`, and the named count laws provide reusable semantics without a class per base family. |
| Mixture distributions | Covered by composition | `MixtureDistribution` is the immutable normalized scalar mixture API; specialist aliases would be duplicates. |
| Variance-gamma, normal-inverse-Gaussian, generalized hyperbolic | Deferred as one design unit | These nested parameterizations need a shared financial-family contract, exponentially scaled Bessel-density audit, and numerical-CDF error contract. Adding any one package convention alone would make later compatibility worse. |
| Multivariate inverted beta, Burr, F, Lomax/Pareto and Cook-Johnson uniform | Not selected | `NonNorMvtDist` assigns distribution-specific meanings to multivariate CDF and quantile operations; those do not establish a reusable JDistlib vector contract. |
| Bivariate geometric/logarithmic/negative-binomial and Poisson-lognormal | Not selected | The task view lists inequivalent dependence constructions. No unqualified class name can choose one without silently fixing a model. |
| Multivariate gamma, inverse-Gaussian, generalized-hyperbolic, stable and extreme-value laws | Deferred | Each requires a separately specified dependence/spectral model and family-specific probability algorithm, not a thin distribution port. |
| Truncated/skew multivariate normal and t laws | Deferred to composition design | Existing rectangle probabilities supply the normalization primitive, but a vector truncation/skew contract must define regions, conditioning and sampler diagnostics together. |
| Hyper-Dirichlet and multiplicative multinomial | Not selected | Graph- or dimension-dependent normalizing constants make these model frameworks rather than fixed probability-law ports. |
| Delaporte and Pólya-Aeppli | Deferred to compound-count design | Both are useful, but robust extreme-tail CDF/quantile recurrences should share one compound-count contract rather than two isolated summation loops. |
| Circular and directional laws | Deferred to a periodic-support API | Von Mises and wrapped laws need modulo-equivalent observations, interval conventions, and circular quantiles defined consistently. |
| Discrete gamma and discrete normal | Not selected under those names | The listed `extraDistr` laws are floor-discretizations with density-only reference implementations; unqualified names would hide that construction. |
| Inverse chi-squared and shifted/truncated aliases | Covered by transformation/composition | `InvGamma`, location-scale transformations, and truncation wrappers already supply these laws without duplicate numerical code. |

The 2026-08-21 task-view review also selected `DiscreteLaplace`,
`LogitNormal`, `AsymmetricLaplace`, `ExponentiallyModifiedGaussian`, and
`Huber`, distinct complete scalar laws with broad modeling use. The
discrete-Laplace API supports a translated integer lattice rather than silently
restricting the location to zero. Fixed parameterizations and aliases remain
excluded under the policy above; package lists were not treated as a requirement
to duplicate every spelling of an existing law.

Aliases and exact special cases are intentionally not separate classes. For
example, Bernoulli is `Binomial` with size one, Wald is `InvNormal`, and several
Pareto/Lomax forms are covered by `evd.GeneralizedPareto` after parameter
conversion.
