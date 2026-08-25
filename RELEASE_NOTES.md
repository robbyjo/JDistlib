# JDistlib 0.6.0

JDistlib 0.6.0 makes user-defined probability distributions a first-class part
of the library and adds hardened numerical tools for assessing, composing, and
sampling them. It preserves Java 8-compatible bytecode, explicit per-stream
random state, and the GPL-2.0-or-later license.

Highlights:

- adds fluent builders for continuous, finite-discrete, and mixed-support
  distributions, with fast, standard, and thorough diagnostic presets;
- analyzes ordinary and log-space kernels under explicit strict, warning, or
  permissive construction policies, including deterministic and seeded adaptive
  probes, callback budgets, repeatability checks, and normalization stability;
- composes scalar distributions through normalized mixtures, truncation,
  censoring, and general monotone or affine transformations;
- adds Walker-alias sampling for finite discrete laws, adaptive rejection for
  caller-certified finite-support log-concave densities, and explicit sampling
  strategy reports;
- adds reusable geometric and power-law discrete-tail certificates, numerical
  expectations, moments, entropy, modes, probability intervals, and cache-aware
  allocation-free batch APIs;
- hardens integration with immutable results, callback timing profiles and
  wall-clock budgets, optional daemon-worker isolation, double-exponential
  quadrature, versioned JSON diagnostics, and independently generated
  high-precision regression data;
- adds randomized quasi-Monte Carlo rectangle probabilities for multivariate
  normal, Student t, Cauchy, and log-normal laws, with reproducible streams,
  error estimates, evaluation budgets, and explicit equicoordinate or radial
  quantiles; and
- corrects the Wishart Bartlett sampler parameterization and expands its
  density and generation overloads.

The custom-distribution guide includes mathematical quick starts for continuous
and discrete laws, advanced construction, diagnostics, troubleshooting, and a
typeset distribution catalog. Adaptive rejection remains conditional on the
caller's global log-concavity assertion, and infinite discrete truncation
certificates remain mathematical promises supplied by the caller.

Release assets include the binary library, sources, JavaDoc, and SHA-256
checksums. See `CHANGELOG.md` for the detailed change list and
`docs/custom-distributions.html` for the complete workflow.
