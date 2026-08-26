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
* immutable `ImmutableIntegrationResult` snapshots with typed statuses;
* callback attempt/completion counts and total, average, maximum, and overall
  wall-clock timing;
* benchmark-oriented total and per-callback wall-clock limits;
* opt-in private daemon-worker execution for callbacks that may not return;
* QUADPACK, finite CQUAD, finite tanh-sinh, infinite exp-sinh/sinh-sinh, or
  automatic fallback selection.

Cancellation and caller-thread time limits are cooperative: a per-callback limit
is observed only after that direct callback returns. `ISOLATED_DAEMON` execution
waits through a `Future` and releases the integrating thread at the configured
deadline. Java cannot safely kill arbitrary code, so a callback that ignores
interruption can leave at most that integration's private daemon worker alive.
The daemon cannot keep the JVM alive, but untrusted callbacks or strict CPU and
memory containment still require isolation in another process.

`Integrate.assessStability` repeats the calculation with tighter tolerances and
an additional interval split. Agreement strengthens confidence but does not
prove convergence: related algorithms can agree on the same wrong answer.

Known discontinuities and interior integrable singularities should be declared
as breakpoints. `DOUBLE_EXPONENTIAL` chooses tanh-sinh for finite intervals,
exp-sinh for a semi-infinite interval, and sinh-sinh for the whole line. Because a
double callback cannot resolve points beyond the nearest representable number,
the tanh-sinh result uses a conservative square-root-machine-epsilon error floor
when transformed points round onto an endpoint.

`CQUAD` is a finite-interval, doubly adaptive Clenshaw-Curtis implementation.
Each active interval uses nested polynomial interpolants of degrees 4, 8, 16,
and 32. It raises the polynomial degree before bisecting, and always processes
the interval with the largest estimated error. The estimate uses the ordinary
L2 difference between successive interpolants plus a floating-point roundoff
floor. CQUAD is a robust alternative for difficult finite integrands;
QUADPACK usually spends fewer evaluations on smooth problems, while tanh-sinh
remains the better first choice for endpoint singularities. CQUAD deliberately
rejects infinite bounds. `AUTO` tries QUADPACK first, then CQUAD for a finite
interval, and finally the applicable double-exponential rule.

## Kernel analysis

`ProbabilityFunctionAnalyzer` samples a transformed grid that supports finite,
semi-infinite, and doubly-infinite domains. It also performs seeded, stratified
random exploration, then spends the remaining explicit probe budget around
observed high-variation intervals. It reports observed:

* callback exceptions, NaNs, infinities, and negative values;
* changes when the same coordinate is evaluated repeatedly;
* sharp adjacent changes and candidate split points;
* frequent slope reversals suggesting oscillation;
* large dynamic range suggesting a log-kernel;
* weak sampled tail decay;
* normalization failures or sensitivity to tighter/split integration.

Suggested breakpoints are advisory. `NumericalContinuousDistribution.analyze`
uses them when it attempts construction and returns both the report and either
a distribution or a retained construction failure. `ConstructionPolicy.STRICT`
rejects warnings and errors, `WARNING` rejects errors, and `PERMISSIVE` attempts
construction regardless of advisory findings. Hard integration and kernel
validity failures are never bypassed.

`ProbabilityFunctionAnalyzer.analyzeLogKernel` and
`NumericalContinuousDistribution.analyzeLogKernel` perform the corresponding
probes without first exponentiating the formula. Negative infinity is valid
zero mass, while NaN and positive infinity are errors. Dynamic-range and sharp-
change checks compare log values directly. The fluent continuous builder
selects this path automatically whenever `logKernel` is used.

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

Array operations inherited from `GenericDistribution` now have allocation-free
`densityInto`, `cumulativeInto`, `quantileInto`, and `randomInto` forms.
Continuous ordinary-probability CDF batches build the monotone table once and
reuse it across the batch; logged and extreme tails still use direct integration.

## Fluent construction and diagnostic presets

`NumericalContinuousDistribution.builder()` collects a kernel or log-kernel,
support, singularity breakpoints, integration options, analysis policy, CDF
table settings, and an optional sampling configuration.
`NumericalDiscreteDistribution.builder()` accepts ordinary or log weights and
explicit or consecutive support. `NumericalPiecewiseDistribution.builder()`
handles interval unions, holes, singularities, and atoms.

`DiagnosticPreset.FAST`, `STANDARD`, and `THOROUGH` scale deterministic and
random probes, refinement rounds, integration tolerances, subdivisions, and
evaluation budgets. A preset produces an ordinary `FunctionAnalysisOptions`
object, so every value remains independently editable. Presets apply equally
to `kernel` and `logKernel` construction; `withoutAnalysis()` is the explicit
opt-out for either representation.

## Distribution composition

`MixtureDistribution`, `TruncatedContinuousDistribution`,
`MonotoneTransformDistribution`, and `CensoredDistribution` compose scalar
distribution objects. The `Distributions` class supplies short factories.
Mixtures combine densities and masses in log space when requested. Monotone
transforms require the inverse map and its log absolute derivative; the affine
factory supplies these automatically. Censoring follows the same convention as
piecewise distributions: `density` at a censoring bound returns its atom mass.
Discrete, mixed, censored, mixture, and transformed laws implement
`AtomAwareDistribution`; decreasing transforms use that exact mass information
to preserve CDF jumps and do not apply a continuous Jacobian to an atom.

Truncation and the Jacobian transform formulas are

\[
f_{[a,b]}(x)=\frac{f(x)}{F(b)-F(a)},\qquad
f_Y(y)=f_X(h^{-1}(y))\left|\frac{d h^{-1}(y)}{dy}\right|.
\]

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

`DiscreteTailBounds` supplies conditional certificates for geometric ratio
bounds, right/left/symmetric power-law integral bounds, constants, and delayed
remainders after a verified finite prefix. These helpers make the formula less
error-prone; their mathematical assumptions remain promises from the caller.

## Constructed-distribution checks

`analyzeDistribution()` checks numerical self-consistency:

* normalization status and relative error;
* exact CDF endpoint values;
* CDF range and quantile monotonicity;
* direct lower-tail plus upper-tail agreement;
* CDF/quantile round trips over central and tail probabilities;
* stability of user-selected absolute continuous moments, independently on
  either side of a selected split point;
* exact finite-support moments for discrete distributions.

Absolute-moment stability prevents odd or symmetric cancellation from being
mistaken for existence. It still means only that the tested calculations agreed
under the selected budgets and tolerances, not that existence was proven.

`MomentAnalysisOptions` selects orders and the reporting split. The default
retains orders one and two split at zero, preserving the original mean/variance
convenience accessors.

## Machine-readable diagnostics

`FunctionAnalysis`, `DistributionAnalysis`, `NumericalDistributionBuildResult`,
`DiagnosticFinding`, `IntegrationResult`, `ImmutableIntegrationResult`, and
`IntegrationStabilityResult` expose `toJson()` methods. The dependency-free
records carry `schemaVersion: 1` and a `type` discriminator. JSON has no NaN or
infinity numeric literals, so non-finite diagnostic measurements are emitted as
`null`; status and finding fields retain the reason those measurements are
unavailable.

## Independent accuracy corpus

`src/test/resources/jdistlib/math/integration-reference.csv` retains 70-digit
decimal targets obtained independently from closed-form arbitrary-precision
identities. It covers oscillation, endpoint and interior singularities, extreme
scaling, a narrow mode, and a heavy tail. The regression test loads decimal
strings rather than values produced by JDistlib, so implementation changes are
checked against an independent oracle.

## Faster optional sampling

`RejectionEnvelope` describes a normalized proposal and a certified constant
`M` satisfying `target <= M * proposal`. Installing it with
`configureRejectionSampling` makes `random()` use rejection sampling.
`UniformRejectionEnvelope` is supplied for finite supports when the caller knows
a global normalized log-density upper bound. JDistlib checks every encountered
ratio and fails on a violation or exhausted attempt budget, but cannot prove a
user-supplied bound over unsampled coordinates.

Finite `NumericalDiscreteDistribution` objects automatically build a Walker
alias table, making each draw constant-time. Continuous builders select inverse
CDF sampling by default, a supplied certified envelope when configured, or an
adaptive tangent-envelope sampler when supplied a log-density derivative and
initial knots. `getSamplingStrategy()` and
`getSamplingStrategyExplanation()` make that selection observable.

Adaptive rejection currently requires finite support and a differentiable,
strictly log-concave density at the retained knots. JDistlib checks decreasing
sampled derivatives and every encountered tangent-envelope inequality, but the
global log-concavity claim remains the caller's responsibility.

## Numerical summaries

Continuous numerical distributions expose `expectation`, `rawMoment`,
`centralMoment`, `entropy`, `mode`, and `probabilityInterval`. Integrals return
`ImmutableIntegrationResult`, retaining error and callback diagnostics.
Finite-discrete counterparts are exact sums in double arithmetic. Fractional
raw moments require nonnegative support, while central moments require integer
orders. Mode search for arbitrary continuous callbacks is evidence from a
transformed grid plus local refinement, not proof that a narrower mode does not
exist.

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
