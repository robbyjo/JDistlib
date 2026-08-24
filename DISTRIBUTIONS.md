# Contributed distribution provenance

JDistlib contains distributions beyond R's `src/nmath`. This ledger records
where those APIs came from and keeps that work separate from the R 4.6.1 audit
in [UPSTREAM.md](UPSTREAM.md).

## Porting policy

A contributed distribution is eligible when:

* it is not already represented by a JDistlib distribution or a trivial fixed
  parameterization of one;
* its source license is compatible with JDistlib's GPL-2.0-or-later license;
* density/mass, cumulative, quantile, and random-generation behavior can be
  supplied together, including R-style lower/upper and logarithmic tails; and
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

The numerical reference checks for this batch are in
`AdditionalDistributionsTest`. The source packages are used as references; the
public Java APIs follow JDistlib naming and scalar-call conventions.

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
| Generalized gamma and generalized F | `flexsurv` | High-value survival families; numerical tail audit required |
| Beta-negative-binomial, negative hypergeometric, discrete Weibull, Skellam | `extraDistr` | Complete discrete APIs; require careful integer and extreme-tail tests |
| Half-Cauchy, half-t, slash, Tukey lambda | `extraDistr` | Continuous families with reusable JDistlib base distributions |
| Burr, generalized/transformed beta and gamma, Poisson-inverse-Gaussian | [`actuar`](https://cran.r-project.org/package=actuar) | GPL-2-or-later; prioritize distinct parameterizations and full APIs |
| Truncated and zero-inflated families | `extraDistr` / `actuar` | Best implemented as reusable wrappers after scalar-family work |
| Mixture distributions | `extraDistr` and specialist packages | Needs a general immutable mixture API before individual aliases |

Aliases and exact special cases are intentionally not separate classes. For
example, Bernoulli is `Binomial` with size one, Wald is `InvNormal`, and several
Pareto/Lomax forms are covered by `evd.GeneralizedPareto` after parameter
conversion.
