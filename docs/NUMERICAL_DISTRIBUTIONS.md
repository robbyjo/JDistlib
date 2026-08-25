# Numerical distributions and function diagnostics

JDistlib can turn an unnormalized nonnegative formula into a continuous or
finite-discrete probability distribution. This feature evaluates user code, so
its diagnostic APIs deliberately distinguish observed evidence from proof.

No finite computation can prove that an arbitrary callback has no unsampled
negative spike, singularity, state change, or divergent tail. A report that has
no errors means “no error was observed under these settings.”

## Hardened integration

The original `Integrate.integrate(f, lower, upper, ...)` overloads preserve
R/QUADPACK-compatible behavior. The `IntegrationOptions` overload is additive
and supplies:

* immutable tolerances and method selection;
* a maximum function-evaluation budget;
* cooperative cancellation checked before each evaluation;
* declared finite breakpoints, with absolute tolerance divided across pieces;
* callback exceptions, non-finite values, and their coordinates in the result;
* QUADPACK, finite-interval tanh-sinh, or automatic fallback selection.

Cancellation is cooperative. JDistlib cannot safely interrupt a callback that
enters an infinite loop or blocks forever inside a single evaluation.

`Integrate.assessStability` repeats the calculation with tighter tolerances and
an additional interval split. Agreement strengthens confidence but does not
prove convergence: related algorithms can agree on the same wrong answer.

Known discontinuities and interior integrable singularities should be declared
as breakpoints. Tanh-sinh is useful for difficult finite endpoints. Because a
double callback cannot resolve points beyond the nearest representable number,
the tanh-sinh result uses a conservative square-root-machine-epsilon error floor
when transformed points round onto an endpoint.

## Kernel analysis

`ProbabilityFunctionAnalyzer` samples a transformed grid that supports finite,
semi-infinite, and doubly-infinite domains. It reports observed:

* callback exceptions, NaNs, infinities, and negative values;
* changes when the same coordinate is evaluated repeatedly;
* sharp adjacent changes and candidate split points;
* frequent slope reversals suggesting oscillation;
* large dynamic range suggesting a log-kernel;
* weak sampled tail decay;
* normalization failures or sensitivity to tighter/split integration.

Suggested breakpoints are advisory. `NumericalContinuousDistribution.analyze`
uses them when it attempts construction and returns both the report and either
a distribution or a retained construction failure.

## Log-scale construction

`NumericalContinuousDistribution.fromLogKernel` subtracts a finite reference
before exponentiation. The default factory chooses a reference from deterministic
interior probes; an overload accepts an expert-provided reference. If an
unsampled peak still overflows after scaling, construction fails rather than
silently clipping it.

`NumericalDiscreteDistribution.fromLogWeights` performs the analogous operation
over a finite declared support. It can retain a finite log normalizer even when
the ordinary normalization constant overflows.

Automatic continuous log construction probes for local modes, refines their
locations, divides the domain into as many as eight scaling regions, and combines
regional integrals using log-sum-exp. Expert-supplied single-reference overloads
remain available. A peak narrower than every deterministic probe can still be
missed; declared support splits or an explicit reference remain important.

## Reusable CDF tables

`NumericalCdfTable` evaluates directly integrated CDF values on a transformed
domain, refines intervals whose midpoint validation exceeds tolerance, and uses
shape-preserving monotone cubic interpolation. `getCdfTable()` builds it lazily;
`rebuildCdfTable` accepts `CdfTableOptions`.

The standard `cumulative` method remains directly integrated for compatibility
and maximum accuracy. `cumulativeCached` opts into the table. Central quantiles
use the table as an initial inverse and then perform direct Newton corrections;
logged and extreme tails continue through direct quadrature.

## Piecewise and mixed support

`NumericalSupport` represents a union of intervals after optional hole
subtraction. It also carries declared singularities and finite atom locations.
`NumericalPiecewiseDistribution` normalizes one component per interval and adds
explicit atom weights. At an atom, its `density` method returns probability mass;
elsewhere it returns the ordinary continuous density.

Log-piecewise construction independently scales every interval and combines
continuous and atomic mass by log-sum-exp. This is useful when separated regions
have radically different heights or widths.

## Certified infinite discrete support

`CertifiedInfiniteDiscreteDistribution` supports right-infinite, left-infinite,
and two-sided integer domains. A `DiscreteTailBound` must bound all unnormalized
mass beginning with the first omitted integer. Truncation stops only when that
certificate implies the requested `CertifiedDiscreteOptions` omitted-probability
tolerance.

The returned object is the normalized finite truncation and exposes both the raw
tail-weight bound and the resulting omitted-probability bound. Its guarantee is
conditional on the user-provided certificate being mathematically correct;
JDistlib can check that the bound includes the first omitted term but cannot
prove the remainder formula.

## Constructed-distribution checks

`analyzeDistribution()` checks numerical self-consistency:

* normalization status and relative error;
* exact CDF endpoint values;
* CDF range and quantile monotonicity;
* direct lower-tail plus upper-tail agreement;
* CDF/quantile round trips over central and tail probabilities;
* stability of the first and second absolute continuous moments;
* exact finite-support moments for discrete distributions.

Absolute-moment stability prevents odd or symmetric cancellation from being
mistaken for existence. It still means only that the tested calculations agreed
under the selected budgets and tolerances, not that existence was proven.

## Choosing a better kernel

The most effective remedies are usually structural:

1. Supply the smallest correct support.
2. Declare known discontinuities and singularities.
3. Prefer a deterministic, side-effect-free callback.
4. Use a log-kernel when values span many orders of magnitude.
5. Algebraically remove cancellation or removable singularities.
6. Supply a log reference near the mode when automatic probes can miss a narrow
   peak.
7. Compare results under tighter tolerances, more subdivisions, and meaningful
   problem-specific splits.

Infinite discrete support is accepted only through a user-supplied tail bound;
observing several small terms is never treated as sufficient evidence that a
series converges.
