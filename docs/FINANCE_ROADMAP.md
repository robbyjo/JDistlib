# JDistlib 0.9.0 finance and options roadmap

## Release objective

JDistlib 0.9.0 will add reusable probability machinery for financial loss and
return distributions and for distributions inferred from option observations.
The release remains probability-first: every public feature must be useful
outside a particular exchange, asset class, pricing convention, or trading
workflow.

The release covers the complete chain from tail-risk calculations through
option-implied and posterior-predictive distributions. The phases below are an
implementation order, not separate release commitments; all are part of the
0.9.0 scope.

## Shared contracts

Numerical operations that can approximate, truncate, fail to converge, or use
sampling must return immutable typed results. As applicable, results report the
estimate, absolute or relative error, convergence status, work count, chosen
strategy, warnings, and deterministic seed/stream information. Convenience
methods may return a scalar only when the operation is exact or its documented
default contract has succeeded.

All tail APIs distinguish losses from returns and upper from lower tails.
Expected shortfall and partial moments specify their behavior at atoms and
report a nonexistent required moment instead of returning a plausible finite
number. Probability inputs retain JDistlib's lower-tail/upper-tail and
log-probability conventions where those conventions are meaningful.

Parameter-rich families use one documented canonical internal representation
plus named conversion factories for established external conventions. No class
silently adopts an ambiguous package or publication parameterization.

## Phase 1: tail and payoff functionals

Add a generic scalar-distribution analysis API covering:

* value at risk and quantiles under explicit loss/return conventions;
* atom-aware upper- and lower-tail expected shortfall;
* upper and lower partial moments of configurable order;
* stop-loss expectations and the equivalent call/put payoff expectations;
* downside deviation, shortfall probability and expected shortfall magnitude;
* expectiles with a numerical convergence report; and
* batch evaluation into caller-owned arrays where it materially reduces
  allocation.

The implementation will reuse analytic formulas when supplied by a family and
otherwise use the distribution's CDF, quantile, atoms, support, and hardened
integration machinery. Tests cover continuous, discrete, empirical, mixture,
censored, truncated, transformed, and numerically constructed distributions,
including infinite-moment and quantile-atom cases.

## Phase 2: transform-domain foundations

Introduce log characteristic-function and cumulant-generating-function
contracts with explicit domains of existence. Add stable complex arithmetic,
phase handling, numerical Fourier inversion with error reporting, derivative or
cumulant helpers, and reference reductions for existing normal, gamma,
Poisson, compound-Poisson/Tweedie, and Student-t-compatible cases.

Add exponential/Esscher tilting as distribution composition:

\[
f_\theta(x)=\frac{\exp(\theta x)f(x)}{M_X(\theta)}.
\]

Construction validates that the moment-generating function exists at the tilt,
retains normalization diagnostics, and supports importance sampling and
rare-event workflows in addition to risk-neutral transformations.

## Phase 3: financial heavy-tailed families

Design and implement generalized hyperbolic, normal-inverse-Gaussian, and
variance-gamma distributions as one nested family. Coverage includes density
and log density, both CDF tails, quantiles, explicit-stream random generation,
characteristic functions, moments/cumulants with existence checks, and fitting
hooks. Numerical work includes exponentially scaled modified-Bessel kernels,
extreme-parameter tests, special-case reductions, and a typed numerical CDF
error contract.

Implement a univariate alpha-stable family after the transform-domain contract
is established. It will document a canonical parameterization, provide named
S0/S1-style conversions as needed, handle the alpha-equals-one neighborhood,
and supply density, both CDF tails, quantiles, characteristic functions, and
Chambers-Mallows-Stuck-style explicit-stream sampling. Tests include exact
normal, Cauchy, and Levy reductions, tail asymptotics, and moment-existence
boundaries.

## Phase 4: aggregation and transformations

Add immutable distribution composition for:

* convolution and weighted sums of independent scalar variables;
* compound sums with caller-supplied count and severity distributions;
* products and ratios when their support and atom semantics can be stated
  unambiguously;
* reusable conditional distributions for supported scalar events; and
* named maximum/minimum and scenario transformations where existing order and
  monotone-transform APIs do not already express the operation.

Strategies may include exact finite-discrete convolution, certified recurrence,
Panjer recursion, controlled-grid FFT, characteristic-function inversion, and
reproducible Monte Carlo. The result reports discretization, omitted-tail or
truncation error and the strategy actually used. The compound-count work also
provides robust Delaporte and Polya-Aeppli implementations rather than isolated
uncertified summation loops.

## Phase 5: tail-sensitive dependence

Extend pair copulas with rotated and survival forms and finance-relevant
asymmetric families, beginning with Joe and BB1 and adding BB6/BB7/BB8 only
where density, conditional CDF/inversion, parameter-domain, and limiting-case
contracts can be validated.

Add lower and upper tail-dependence coefficients, finite-level tail
concentration diagnostics, joint upper/lower and mixed stress-region
probabilities, and tail-weighted fitting/selection criteria. These features
integrate with continuous, discrete/mixed, C-vine, and D-vine likelihood result
contracts and preserve explicit-stream reproducibility.

## Phase 6: fitting and extreme-value inference

Add a reusable distribution-fitting framework supporting bounded parameters,
maximum likelihood, optional MAP objectives, analytic or numerical gradients,
observed or sandwich covariance estimates, likelihood diagnostics,
censored/interval observations, and immutable convergence results. Calibration
objectives may also use quantile, moment, option-price, or weighted-tail losses.

Build EVT inference on that framework:

* GEV maximum-likelihood and probability-weighted-moment fitting;
* GPD peaks-over-threshold fitting;
* Hill, Pickands, and related tail-index estimators;
* return levels with profile, asymptotic, or reproducible-bootstrap
  uncertainty;
* mean residual life and threshold-stability diagnostics; and
* explicit policies for threshold equality, finite endpoints, and non-finite
  estimated moments.

Time-series declustering is exposed only if it can be defined as a generic,
documented preprocessing policy. Data calendars and market-session conventions
remain outside JDistlib.

## Phase 7: option-implied distributions

Add a narrow reference option-observation layer whose output is a JDistlib
distribution rather than a managed instrument or volatility surface.

Checked implied-volatility inversion validates no-arbitrage price bounds and
returns the implied parameter, pricing residual, iterations, bracket, and
convergence status. Any Black-Scholes/Bachelier formulas included for inversion
are reference transformations, not the beginning of a general pricer hierarchy.

An arbitrage-constrained curve builder accepts forward, discount, maturity,
strike, and call or put observations, including bid/ask intervals or observation
weights. It enforces the relevant monotonicity, convexity, parity, and boundary
conditions before differentiating or otherwise recovering probabilities.
Diagnostics identify repaired or incompatible observations and quantify fit
residuals.

The curve can produce a normalized risk-neutral
`NumericalContinuousDistribution` or an atom-aware numerical distribution with:

* density/mass, CDF, and both tails;
* quantiles, partial moments, and tail risk;
* reproducible random generation;
* strike-interval and terminal-threshold probabilities; and
* normalization, monotonicity, convexity, and differentiation-error reports.

Calibration can fit generalized-hyperbolic, NIG, variance-gamma, stable,
mixture, empirical/numerical, or caller-supplied compatible families to option
prices, implied probabilities, quantiles, moments, or tail-weighted objectives.
It retains parameter conventions and reports identifiability, optimizer, and
observation-residual diagnostics.

## Phase 8: Bayesian and predictive option inference

Integrate option observations with the existing model and MCMC layer through
reusable likelihood factors for quoted prices, bid/ask intervals, censoring,
and heteroskedastic observation noise. Priors and latent dynamics remain
ordinary JDistlib model components rather than a hard-coded market model.

Posterior output adapters construct distributions or ensembles for terminal
prices, log returns, payoffs, strike intervals, tail probabilities, value at
risk, and expected shortfall. Results distinguish risk-neutral distributions
inferred from prices from physical/predictive distributions inferred with an
explicit observation and transition model. Posterior summaries include Monte
Carlo error and retain chain/seed provenance.

## Validation and release gates

The 0.9.0 work is complete only when:

* every new scalar law has density/CDF/quantile/random round-trip, tail,
  normalization, seeded-reproducibility, invalid-parameter, and limiting-case
  coverage against independent references;
* every approximate result exposes failure and error information rather than
  silently returning an unchecked scalar;
* tail-risk tests cover atoms, infinite moments, extreme log probabilities, and
  loss/return sign conventions;
* characteristic-function inversion and aggregation tests include analytic
  reductions and mass/normalization conservation;
* option-implied tests use synthetic arbitrage-free surfaces with known
  distributions plus deliberately inconsistent/noisy observations;
* Bayesian examples distinguish risk-neutral from physical measures and report
  sampler and posterior Monte Carlo diagnostics; and
* all public APIs remain thread-safe under the documented explicit-stream
  rules, Java 8 bytecode compatible, and represented in the API guide, website,
  JavaDoc, and compilable examples.

## Boundary of the release

Version 0.9.0 does not include market-data download, symbol or corporate-action
handling, exchange calendars, technical indicators, portfolio accounting or
optimization, backtesting, execution, order books, yield-curve management, or
instrument lifecycle systems. It also does not promise general American,
barrier or exotic pricing, production volatility-surface management, or a
Greeks framework. Separate finance libraries can build those features on the
probability, calibration, and inference contracts supplied here.
