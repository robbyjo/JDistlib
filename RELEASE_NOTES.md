# JDistlib 0.6.1

JDistlib 0.6.1 completes the screened distribution backlog and incorporates the
practical scalar additions identified in the CRAN Probability Distributions
Task View audit. It preserves Java 8-compatible bytecode, explicit per-stream
random state, and the GPL-2.0-or-later license.

Highlights:

- adds complete density/mass, cumulative, quantile, and random-generation APIs
  for generalized F, beta-negative-binomial, negative hypergeometric, discrete
  Weibull, Skellam, half-Cauchy, half-t, slash, Tukey lambda, Feller-Pareto, and
  phase-type distributions;
- adds asymmetric Laplace, exponentially modified Gaussian, Huber,
  discrete-Laplace, and logit-normal distributions from the broader CRAN
  task-view audit;
- corrects the historical beta-prime transformation, density Jacobian, support
  handling, tail stability, and instance truncation behavior;
- expands the distribution catalog and provenance ledger, including explicit
  dispositions for compound-count, periodic, discretized, compositional, and
  infrastructure-heavy families that were not selected for this release; and
- cleans legacy JavaDoc markup throughout the numerical and statistical APIs so
  the documentation build completes without warnings.

The release adds focused regression coverage for the backlog and task-view
families, including boundary behavior, tail conventions, quantile inversion,
and deterministic random-stream checks. The full Gradle check and JavaDoc gates
pass cleanly.

Release assets include the binary library, sources, JavaDoc, and SHA-256
checksums. See `CHANGELOG.md` for the detailed change list,
`DISTRIBUTIONS.md` for sources and audit dispositions, and `PUBLISHING.md` for
the separate Maven Central maintainer step.
