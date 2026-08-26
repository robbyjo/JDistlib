# JDistlib 0.7.1

JDistlib 0.7.1 expands multiple-testing and numerical-integration support while
preserving Java 8-compatible bytecode, deterministic seeded behavior, and the
GPL-2.0-or-later license.

Multiple-testing additions include:

- prespecified-weight Benjamini–Hochberg, Benjamini–Yekutieli, Bonferroni, and
  Holm procedures with scale-invariant mean-one weight normalization;
- direct natural-log variants and completed family-size, rejection-count, and
  threshold helpers for batch methods;
- the adaptive Gavrilov–Benjamini–Sarkar step-down FDR procedure for independent
  tests;
- two-level Benjamini–Bogomolov testing for explicitly grouped families;
- separate stateful LORD++ and SAFFRON controllers for hypotheses arriving over
  time; and
- DBH step-up and step-down procedures for independent heterogeneous discrete
  p-values with explicit null support and CDF declarations.

Numerical and modeling additions include:

- a pure-Java finite-interval CQUAD integration strategy with nested
  Clenshaw–Curtis interpolants, degree and interval refinement, largest-error
  prioritization, hardened callback handling, and automatic fallback after
  QUADPACK; and
- `Distributions.transform`, a concise factory for differentiable monotone
  transformations, accompanied by a prominent beginner tutorial and compiled
  examples covering mixtures, truncation, censoring, affine changes, nonlinear
  transformations, and Jacobians.

The documentation site includes updated multiple-testing guidance and the new
composition tutorial. Regression coverage includes weighted and adaptive
decisions, grouped and online procedures, discrete null distributions, CQUAD
smooth and difficult integrands, callback budgets, breakpoints, and general
transformation identities.

Release assets include the binary library, sources, JavaDoc, and SHA-256
checksums. See `CHANGELOG.md` for the detailed change list and `PUBLISHING.md`
for the separate Maven Central maintainer step.
