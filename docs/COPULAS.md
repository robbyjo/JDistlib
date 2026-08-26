# Copulas, mixed marginals, vines, and fitting

> **Version requirement:** copula features are available only in JDistlib 0.7.0
> and later.

`Copula` models dependence separately from univariate distributions. A copula
accepts a vector in `[0, 1]^d`; its random output has uniform marginals.
`CopulaDistribution` turns those uniforms into a continuous joint law;
`MixedCopulaDistribution` also supports declared discrete coordinates.

## Families and parameter domains

| Class | Parameters | Kendall's tau |
| --- | --- | --- |
| `IndependenceCopula` | dimension at least 1 | 0 |
| `GaussianCopula` | positive-definite correlation matrix | `2 asin(rho) / pi` |
| `StudentTCopula` | positive-definite correlation matrix, finite `df > 0` | `2 asin(rho) / pi` |
| `ClaytonCopula` | dimension at least 2, finite `theta >= 0` | `theta / (theta + 2)` |
| `GumbelCopula` | dimension at least 2, finite `theta >= 1` | `1 - 1 / theta` |
| `FrankCopula` | finite theta; negative values only in dimension 2 | Debye-function relationship |

Zero Clayton or Frank theta and unit Gumbel theta are represented exactly as
independence. Clayton is intentionally limited to its completely monotone,
nonnegative-dependence range. Frank permits negative dependence for the
bivariate family, but rejects it in higher dimensions.

Elliptical families expose `fromKendallsTau(double[][])`. Clayton, Gumbel, and
Frank expose both `fromKendallsTau` and `parameterFromKendallsTau`. Invalid or
non-positive-definite matrices and out-of-domain scalar parameters fail in the
constructor rather than creating a partly valid object.

## Evaluation and boundaries

```java
Copula copula = new ClaytonCopula(2, 1.5);
double cdf = copula.cumulative(new double[] {0.3, 0.8});
double logDensity = copula.logDensity(new double[] {0.3, 0.8});
```

CDFs accept the closed unit cube and implement its exact zero/one boundary
identities. Densities are defined on the open cube. `logDensity` returns `NaN`
at an exact boundary for dependent families because a multivariate boundary
limit can be infinite or path-dependent. `copula.diagnose(u)` reports whether a
point is interior, boundary, or invalid and counts its lower and upper boundary
coordinates. Independence has unit density everywhere, including boundaries.

Gaussian and Student-t CDFs use JDistlib's deterministic default multivariate
probability calculation. Their density and sampling paths are direct Cholesky
calculations.

## Continuous-marginal composition

```java
CopulaDistribution joint = new CopulaDistribution(
    new GumbelCopula(2, 1.8),
    new LogNormal(0.0, 0.4),
    new Gamma(3.0, 2.0));

double probability = joint.cumulative(new double[] {1.2, 7.0});
double density = joint.density(new double[] {1.2, 7.0});
double[][] sample = joint.random(1000, 12345L);
```

The joint density is the copula density at the marginal CDF values multiplied
by all marginal densities. Sampling draws dependent uniforms once and applies
the corresponding marginal quantiles. The marginal objects must describe
continuous distributions. Use the measure-aware API below when any coordinate
has atoms.

## Discrete and mixed marginals

Wrap each scalar law with `CopulaMarginal.continuous` or
`CopulaMarginal.discrete`. The latter derives the left CDF limit as
`F(x) - mass(x)`; an overload accepts an explicit left-limit function for a
distribution with different legacy conventions.

```java
MixedCopulaDistribution joint = new MixedCopulaDistribution(
    new ClaytonCopula(2, 1.2),
    CopulaMarginal.continuous(new Normal()),
    CopulaMarginal.discrete(new Poisson(2.5)));

CopulaMeasureResult contribution = joint.measure(new double[] {0.3, 2.0});
if (!contribution.isSuccess()) {
    System.err.println(contribution.message());
}
```

For all-continuous inputs, `measure` uses the analytic copula density and
marginal Jacobians. For all-discrete inputs it takes the alternating difference
of the copula CDF over the marginal jump rectangle. Mixed inputs differentiate
that rectangle numerically over continuous coordinates, repeat the calculation
with a smaller step, and report the difference as `absoluteError`.
`CopulaMeasureOptions` bounds the CDF-corner count and controls derivative steps
and negative-cancellation tolerance. The calculation has `2^d` corner growth,
so high-dimensional mixed likelihoods should use an explicit evaluation budget.

For fitting data with atoms, `CopulaFitter.marginalTransforms` accepts the same
marginal declarations. It uses CDF-jump midpoints deterministically or a
caller-owned random engine for the distributional transform.

## Pair copulas and vines

`PairCopula` adapts any bivariate `Copula` with conditional CDF and inverse
conditional operations. Gaussian and Student-t conditionals are analytic;
other families use bounded finite differences and bisection.

`CVineCopula` and `DVineCopula` implement simplified pair-copula constructions.
Their triangular `PairCopula` arrays follow the documented C-vine root order or
D-vine tree/interval order. Densities and samples use the conditional recursion
directly. General lower-orthant CDFs are Monte Carlo estimates, so
`cumulativeResult` should be preferred over the scalar convenience method when
the standard error matters.

```java
PairCopula independent = new PairCopula(new IndependenceCopula(2));
VineCopula vine = new CVineCopula(
    new PairCopula[] {independent, independent},
    new PairCopula[] {independent});
```

These are simplified vines: each conditional pair copula has fixed parameters
rather than parameters that vary with conditioning values. Arbitrary R-vine
matrix structures are not represented by the C-vine/D-vine constructors.

## Dependence fitting and automatic selection

`CopulaFitter.pseudoObservations` applies average ranks to raw continuous data.
`fitUniforms` accepts data already in `(0,1)^d`. Gaussian correlation can be
estimated from normal scores, Student-t degrees of freedom are likelihood-
optimized, and Archimedean fits use Kendall's tau initialization with optional
likelihood refinement. Fitted correlation matrices are shrunk toward identity
only when needed to restore positive definiteness.

```java
CopulaSelectionResult selected = CopulaSelector.select(data,
    new CopulaFitOptions(), CopulaSelectionCriterion.BIC,
    CopulaFamily.INDEPENDENCE, CopulaFamily.GAUSSIAN,
    CopulaFamily.STUDENT_T, CopulaFamily.CLAYTON,
    CopulaFamily.GUMBEL, CopulaFamily.FRANK);
Copula fitted = selected.getSelected().getCopula();
```

Every candidate remains in the ranked result. An incompatible family, such as
a multivariate negative Clayton fit, reports its failure rather than silently
changing the data or parameter domain. AIC and BIC use the family parameter
counts exposed in `CopulaFitResult`.

`VineFitter` performs sequential pair fitting for C-vines and D-vines and runs
the same family selector at every tree edge. `VineFitResult` retains each pair's
complete ranking so the selected structure is auditable. Structure ordering is
caller-supplied; automatic variable reordering and arbitrary R-vine structure
search are outside this core API.

## Random streams and concurrency

All implementations are immutable and retain no random engine. Prefer
`random(count, RandomEngine)` when a caller owns the stream. Seed overloads
create a fresh `MersenneTwister`, so repeated calls with the same seed reproduce
the same sample. As elsewhere in JDistlib, do not share one mutable
`RandomEngine` between threads without synchronization.
