# JDistlib 0.5.0

JDistlib 0.5.0 is the first release since 0.4.5 and the first published from
GitHub. It modernizes the build while preserving Java 8-compatible bytecode,
the GPL-2.0-or-later license, JDistlib-specific distributions, and explicit
per-stream random state.

Highlights:

- completes the file-by-file synchronization of R `src/nmath` from the
  historical R 3.3.2 baseline through R 4.6.1, with selected later R-devel
  sampling and Wilcoxon fixes tracked separately;
- completes finite and infinite numerical integration using R 4.6.1's
  DQAGS/DQAGI behavior, including extrapolation, error estimates, and all
  QUADPACK status codes;
- improves noncentral-beta tails and quantiles with a mode-centred, log-scale
  bidirectional mixture and includes an independently validated proposal for
  the corresponding R nmath change;
- completes previously unfinished APIs, including Tweedie and higher-order
  polygamma behavior;
- adds a broad set of scalar, count, multivariate, and contributed-package
  distributions while retaining all historical JDistlib-only distributions;
- provides a Gradle 9 build, source and JavaDoc JARs, automated CI across JDK
  17, 21, and 25, and a GitHub Pages distribution reference and JavaDoc site.

Corrected BTPE sampling is now the binomial default. Applications that require
the historical R 4.6-and-earlier stream can request
`Binomial.BinomialKind.BUGGY_BTPE` explicitly.

Release assets include the binary library, sources, JavaDoc, and SHA-256
checksums. See `CHANGELOG.md` for the detailed change list and `UPSTREAM.md`
for the audited R source ledger.
