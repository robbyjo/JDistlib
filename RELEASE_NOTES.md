# JDistlib 0.7.0

JDistlib 0.7.0 adds a composable copula framework while preserving
Java 8-compatible bytecode, explicit per-stream random state, and the
GPL-2.0-or-later license.

The project website now prominently features a beginner learning center with
tutorials for built-in and custom distributions and applied vignettes for both.
A separate copula tutorial and mixed-marginal vignette are visibly labeled as
requiring JDistlib 0.7.0 or later.

`jdistlib.disttest.DistributionTest` adds general Anderson-Darling and
Cramer-von Mises goodness-of-fit procedures with reproducible resampling, plus
Pearson categorical goodness-of-fit and contingency-table independence tests.

The adjacent `jdistlib.disttest.MultipleTesting` facade complements generated
p-values with standard FWER and FDR adjustments, decision helpers, and Storey
q-values. It replaces the older QGeneric plugin hierarchy with one stateless
API, uses `NaN` for missing values, and corrects the legacy Šidák exponent
direction.
It also provides two-stage Benjamini–Krieger–Yekutieli testing, fully
log-domain adjustment for underflow-scale p-values, and explicit handling of
right-censored families when the censoring limit and total test count are known.

Highlights:

- introduces the immutable `Copula` interface with independence, Gaussian,
  Student-t, Clayton, Gumbel, and Frank families;
- provides CDF, density and log-density evaluation, explicit-engine and seeded
  sampling, strict parameter/correlation validation, and pairwise Kendall's-tau
  reporting and conversions;
- adds `CopulaDistribution`, which composes a copula with continuous
  `GenericDistribution` marginals for joint CDFs, Jacobian-aware densities, and
  sampling through marginal quantiles;
- adds atom-aware discrete and mixed composition with explicit CDF left limits,
  rectangle-difference masses, numerical mixed likelihoods, evaluation budgets,
  and typed accuracy/status results;
- adds simplified C-vine and D-vine construction and fitting, reusable pair
  conditional CDFs and inverses, and Monte Carlo CDF results with uncertainty;
- adds pseudo-observations and mixed distributional transforms, Kendall or
  likelihood fitting, and automatic AIC/BIC family selection for ordinary and
  vine copulas;
- classifies interior, exact-boundary, and invalid unit-cube inputs with
  `CopulaDiagnostics`, avoiding misleading density values where limits can be
  singular or path-dependent; and
- documents family domains, marginal-measure declarations, approximation
  boundaries, random-stream ownership, and selection semantics.

Regression coverage checks closed forms, uniform-margin identities, numerical
density normalization, correlation and scalar-parameter rejection, deterministic
streams, empirical Kendall's tau, discrete mass normalization, mixed
likelihoods, generated-data parameter recovery, and sequential vine fitting.

Release assets include the binary library, sources, JavaDoc, and SHA-256
checksums. See `CHANGELOG.md` for the detailed change list, `docs/COPULAS.md` for
the usage contract, and `PUBLISHING.md` for the separate Maven Central
maintainer step.
