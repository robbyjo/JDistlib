# JDistlib

JDistlib is a Java library for probability distributions and related numerical
methods. Its core is a manual Java translation of R's `src/nmath`, designed to
retain R-compatible results without inheriting the process-global caches that
made older native implementations awkward to call concurrently.

The library also contains distributions and utilities that are not part of R,
including arcsine, beta-binomial, beta-prime, inverse gamma, inverse normal,
Kumaraswamy, Laplace, Levy, log-logistic, logarithmic, Nakagami, skewed t,
Tweedie, Wishart, Zipf, and the `jdistlib.evd` package. These are first-class
JDistlib features and are not removed during upstream synchronization.

## Project status

The `0.5.0-SNAPSHOT` branch is the modernization line. The repository layout,
build, publication metadata, integration implementation, and the first audited
R 4.6.1 numerical updates are present. The complete file-by-file R 4.6.1 sync is
still in progress; [UPSTREAM.md](UPSTREAM.md) is the source-of-truth checklist.
Until that checklist is complete, do not describe 0.5.0 as fully equivalent to
R 4.6.1.

## Building

JDK 17 or newer is recommended for building. Produced class files remain Java 8
compatible.

```text
./gradlew test
./gradlew build
```

On Windows, use `gradlew.bat` instead.

## Using the distribution APIs

Distribution classes expose static density, cumulative, quantile, and random
methods. For example:

```java
double p = Normal.cumulative(1.96, 0.0, 1.0, true, false);
double x = Normal.quantile(0.975, 0.0, 1.0, true, false);
```

Boolean arguments follow R's `lower.tail` and `log.p` conventions.

## Numerical integration

`Integrate.integrate` supports finite, semi-infinite, and doubly-infinite
intervals. The default tolerances match R's `integrate()` default of
`.Machine$double.eps^0.25`.

```java
IntegrationResult result = Integrate.integrate(
    x -> Math.exp(-x * x),
    Double.NEGATIVE_INFINITY,
    Double.POSITIVE_INFINITY
);

if (!result.isSuccess()) {
    throw new ArithmeticException(result.message());
}
```

The result includes the estimated integral, absolute error, number of
subdivisions, and a QUADPACK-compatible status code.

## Thread safety

Pure density, cumulative, and quantile calls use call-local state. Cached random
algorithms for binomial, hypergeometric, and Poisson sampling accept an explicit
`RandomState`; use one state and one `RandomEngine` per random stream. `SignRank`
and `Wilcoxon` intentionally keep their work tables in instances, so do not share
one mutable instance across concurrent callers.

## Upstream and license

The nmath-derived code is synchronized against the official R sources and is
distributed under the GNU General Public License, version 2 or later. See
[LICENSE](LICENSE), [UPSTREAM.md](UPSTREAM.md), and the historical
[JDistlib website](https://jdistlib.sourceforge.net/).
