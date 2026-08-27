# JDistlib 0.7.2

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
