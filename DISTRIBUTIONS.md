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
| `Dirichlet`, `DirichletMultinomial` | [`MCMCpack`](https://cran.r-project.org/package=MCMCpack) and CRAN task-view definitions | 1.7-1 / task view 2026-05-07 | GPL-3 | Closed-form densities/masses, simplex support, Pólya mixture simulation and means |
| `MultivariateHypergeometric` | [`extraDistr`](https://cran.r-project.org/package=extraDistr) and task-view definition | 1.10.0.5 / task view 2026-05-07 | GPL-2 | Exact combinatorial mass, sequential conditional sampling, support checks |
| `BivariatePoisson` | [`bivpois`](https://cran.r-project.org/package=bivpois) | 1.2 | GPL-2-or-later | Independent-plus-shared Poisson identity, exact joint CDF, simulated covariance |
| `BivariateLogistic` | `VGAM`, `d/p/rbilogis` | 1.1-14 | GPL-3 | Closed-form density/CDF, endpoint behavior and conditional generator |
| `MultivariateNormal`, `MultivariateStudentT`, `MultivariateCauchy`, `MultivariateLogNormal` | [`mvtnorm`](https://cran.r-project.org/package=mvtnorm), Genz conditional transformation, task-view definitions | 1.4-2 / task view 2026-05-07 | GPL-2 | Cholesky density/generation, randomized rectangle probabilities with error estimates, independent quadrature, exact orthants, equicoordinate/radial quantiles |
| `MultivariateLaplace`, `MultivariatePowerExponential` | [`LaplacesDemon`](https://cran.r-project.org/package=LaplacesDemon), task-view definitions | 16.1.6 / task view 2026-05-07 | MIT | Normal-mixture and radial-gamma constructions, Bessel density, Gaussian/Laplace reductions and power-exponential radial quantiles |
| `Wishart` | R `stats::rWishart`, Bartlett decomposition and standard matrix density | R 4.6.1 / AS 53 | GPL-2-or-later | Scaled-chi-square reduction, density/log-density agreement, exact seeded Bartlett reduction, matrix means and support validation |

The numerical reference checks are in `AdditionalDistributionsTest`,
`ContributedPackageDistributionsTest`, `MultivariateDistributionsTest`, and
`MultivariateProbabilityTest`.
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

## Screened backlog

The [CRAN Probability Distributions Task View](https://cran.r-project.org/view=Distributions)
and the packages below were reviewed for missing, useful families. This is a
candidate queue, not a completeness claim or release promise.

| Candidate group | Likely reference | Notes |
| --- | --- | --- |
| Generalized F | `flexsurv` | High-value survival family; numerical tail audit required |
| Beta-negative-binomial, negative hypergeometric, discrete Weibull, Skellam | `extraDistr` | Complete discrete APIs; require careful integer and extreme-tail tests |
| Half-Cauchy, half-t, slash, Tukey lambda | `extraDistr` | Continuous families with reusable JDistlib base distributions |
| Feller-Pareto and phase-type | [`actuar`](https://cran.r-project.org/package=actuar) | High-value actuarial families not fully subsumed by GB2/GPD |
| Variance-gamma, normal-inverse-Gaussian, generalized hyperbolic | `VarianceGamma` / `ghyp` | Financial return and risk models; require a shared, carefully audited Bessel/integration core |
| Truncated and zero-inflated families | `extraDistr` / `actuar` | Best implemented as reusable wrappers after scalar-family work |
| Mixture distributions | `extraDistr` and specialist packages | Needs a general immutable mixture API before individual aliases |
| Multivariate inverted beta, Burr, F, Lomax/Pareto and Cook-Johnson uniform | `NonNorMvtDist` | Application families; parameterization and joint-quantile semantics need a dedicated audit |
| Bivariate geometric/logarithmic/negative-binomial and Poisson-lognormal | `bivgeom`, `trawl`, `MNB`, `poilog` | Several inequivalent dependence constructions; avoid choosing one silently |
| Multivariate gamma, inverse-Gaussian, generalized-hyperbolic, stable and extreme-value laws | `joker`, `mig`, `ghyp`, `mvpd`, `evd` | Require family-specific special functions or numerical probability algorithms |
| Truncated/skew multivariate normal and t laws | `mvtnorm`, `TruncatedNormal`, `tmvtnorm`, `sn` | Rectangle probabilities now exist; reusable truncated/skew densities and samplers remain to be designed |
| Hyper-Dirichlet and multiplicative multinomial | `hyper2`, `MM` | Graph- or dimension-dependent normalizing constants make these more than thin distribution ports |

Aliases and exact special cases are intentionally not separate classes. For
example, Bernoulli is `Binomial` with size one, Wald is `InvNormal`, and several
Pareto/Lomax forms are covered by `evd.GeneralizedPareto` after parameter
conversion.
