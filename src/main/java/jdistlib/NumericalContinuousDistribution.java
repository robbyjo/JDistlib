/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.ImmutableIntegrationResult;
import jdistlib.math.Integrate;
import jdistlib.math.IntegrationOptions;
import jdistlib.math.IntegrationResult;
import jdistlib.math.UnivariateFunction;

/**
 * A continuous distribution obtained by numerically normalizing a nonnegative
 * kernel over a real interval.
 *
 * <p>If {@code kernel} is {@code g}, this class uses the density
 * {@code g(x) / Z}, where {@code Z = integral(g(x), lower, upper)}. Bounds may
 * be finite or infinite. The normalization integral is evaluated once during
 * construction; CDF values and quantiles are evaluated numerically on demand.
 * The supplied function must remain deterministic and thread-safe if the
 * distribution is shared between threads.</p>
 *
 * <p>Successful numerical integration cannot prove that an arbitrary function
 * is nonnegative or integrable everywhere. Kernel values encountered by the
 * integrator are checked, and the normalization result is available through
 * {@link #getNormalizationResult()} for inspection.</p>
 */
public class NumericalContinuousDistribution extends GenericDistribution
		implements SupportedDistribution {
	private static final int QUANTILE_ITERATIONS = 128;
	private static final IntegrationOptions DEFAULT_OPTIONS = IntegrationOptions.builder()
			.tolerances(0.0, 1e-10)
			.subdivisions(300)
			.maxEvaluations(500000)
			.method(IntegrationOptions.Method.AUTO)
			.build();

	private final UnivariateFunction kernel;
	private final UnivariateFunction logKernel;
	private final LogKernelIntegrator logIntegrator;
	private final double lower;
	private final double upper;
	private final IntegrationOptions options;
	private final double scaledNormalization;
	private final double normalization;
	private final double logNormalization;
	private final IntegrationResult normalizationResult;
	private volatile NumericalCdfTable cdfTable;
	private volatile CdfTableOptions cdfTableOptions = CdfTableOptions.defaults();
	private volatile RejectionSamplingConfig rejectionSampling;
	private volatile AdaptiveRejectionSampler adaptiveRejectionSampler;

	/** Returns a fluent builder for a custom continuous distribution. */
	public static Builder builder() { return new Builder(); }

	/** Fluent construction with optional analysis and sampling configuration. */
	public static final class Builder {
		private UnivariateFunction kernel;
		private UnivariateFunction logKernel;
		private double lower = Double.NaN;
		private double upper = Double.NaN;
		private IntegrationOptions integrationOptions = DEFAULT_OPTIONS;
		private FunctionAnalysisOptions analysisOptions = DiagnosticPreset.STANDARD.options();
		private double[] singularities = new double[0];
		private CdfTableOptions cdfTableOptions;
		private RejectionEnvelope rejectionEnvelope;
		private int rejectionAttempts = 10000;
		private UnivariateFunction logDerivative;
		private double[] adaptivePoints;
		private int adaptiveMaximumKnots = 128;
		private int adaptiveMaximumAttempts = 10000;

		private Builder() {}
		public Builder kernel(UnivariateFunction value) {
			kernel = value;
			logKernel = null;
			return this;
		}
		public Builder logKernel(UnivariateFunction value) {
			logKernel = value;
			kernel = null;
			return this;
		}
		public Builder support(double lowerBound, double upperBound) {
			lower = lowerBound;
			upper = upperBound;
			return this;
		}
		public Builder singularities(double... values) {
			singularities = values == null ? new double[0] : values.clone();
			return this;
		}
		public Builder integrationOptions(IntegrationOptions value) {
			integrationOptions = value;
			return this;
		}
		public Builder diagnosticPreset(DiagnosticPreset value) {
			if (value == null) throw new IllegalArgumentException("preset must not be null");
			analysisOptions = value.options();
			return this;
		}
		public Builder analysisOptions(FunctionAnalysisOptions value) {
			analysisOptions = value;
			return this;
		}
		public Builder constructionPolicy(ConstructionPolicy value) {
			if (analysisOptions == null) {
				analysisOptions = DiagnosticPreset.STANDARD.options();
			}
			analysisOptions = analysisOptions.toBuilder()
					.constructionPolicy(value).build();
			return this;
		}
		public Builder withoutAnalysis() { analysisOptions = null; return this; }
		public Builder cdfTable(CdfTableOptions value) {
			cdfTableOptions = value;
			return this;
		}
		public Builder rejectionSampling(RejectionEnvelope envelope,
				int maxAttempts) {
			rejectionEnvelope = envelope;
			rejectionAttempts = maxAttempts;
			return this;
		}
		public Builder adaptiveRejectionSampling(UnivariateFunction derivative,
				double... initialPoints) {
			logDerivative = derivative;
			adaptivePoints = initialPoints == null ? null : initialPoints.clone();
			return this;
		}
		public Builder adaptiveRejectionLimits(int maximumKnots,
				int maximumAttempts) {
			adaptiveMaximumKnots = maximumKnots;
			adaptiveMaximumAttempts = maximumAttempts;
			return this;
		}

		public NumericalContinuousDistribution build() {
			if ((kernel == null) == (logKernel == null)) {
				throw new IllegalStateException("exactly one kernel or logKernel is required");
			}
			if (integrationOptions == null) {
				throw new IllegalStateException("integrationOptions must not be null");
			}
			double[] declared = integrationOptions.getBreakpoints();
			double[] combined = new double[declared.length + singularities.length];
			System.arraycopy(declared, 0, combined, 0, declared.length);
			System.arraycopy(singularities, 0, combined, declared.length,
					singularities.length);
			IntegrationOptions effective = integrationOptions.toBuilder()
					.breakpoints(combined).build();
			NumericalContinuousDistribution result;
			if (logKernel != null && analysisOptions != null) {
				FunctionAnalysisOptions checks = analysisOptions.toBuilder()
						.integrationOptions(effective).build();
				result = analyzeLogKernel(logKernel, lower, upper, checks).build();
			} else if (logKernel != null) {
				result = fromLogKernel(logKernel, lower, upper, effective);
			} else if (analysisOptions != null) {
				FunctionAnalysisOptions checks = analysisOptions.toBuilder()
						.integrationOptions(effective).build();
				result = analyze(kernel, lower, upper, checks).build();
			} else {
				result = new NumericalContinuousDistribution(kernel, lower, upper,
						effective);
			}
			if (cdfTableOptions != null) result.rebuildCdfTable(cdfTableOptions);
			if (rejectionEnvelope != null) {
				result.configureRejectionSampling(rejectionEnvelope, rejectionAttempts);
			}
			if (logDerivative != null) {
				result.configureAdaptiveRejectionSampling(logDerivative,
						adaptiveMaximumKnots, adaptiveMaximumAttempts, adaptivePoints);
			}
			return result;
		}
	}

	/**
	 * Constructs a distribution using distribution-oriented integration defaults:
	 * relative tolerance {@code 1e-10}, 300 subdivisions, a finite evaluation
	 * budget, and automatic finite-interval tanh-sinh fallback.
	 *
	 * @param kernel nonnegative unnormalized density
	 * @param lower lower support bound, possibly negative infinity
	 * @param upper upper support bound, possibly positive infinity
	 * @throws IllegalArgumentException if the support or normalization is invalid
	 */
	public NumericalContinuousDistribution(UnivariateFunction kernel, double lower,
			double upper) {
		this(kernel, null, lower, upper, DEFAULT_OPTIONS, 0.0);
	}

	/**
	 * Constructs a distribution with explicit QUADPACK tolerances.
	 *
	 * @param kernel nonnegative unnormalized density
	 * @param lower lower support bound, possibly negative infinity
	 * @param upper upper support bound, possibly positive infinity
	 * @param epsabs absolute integration tolerance
	 * @param epsrel relative integration tolerance
	 * @param subdivisions maximum number of integration subdivisions
	 * @throws IllegalArgumentException if the inputs or normalization are invalid
	 */
	public NumericalContinuousDistribution(UnivariateFunction kernel, double lower,
			double upper, double epsabs, double epsrel, int subdivisions) {
		this(kernel, null, lower, upper, IntegrationOptions.builder()
				.tolerances(epsabs, epsrel)
				.subdivisions(subdivisions)
				.method(IntegrationOptions.Method.QUADPACK)
				.build(), 0.0);
	}

	/** Constructs a distribution with hardened integration options. */
	public NumericalContinuousDistribution(UnivariateFunction kernel, double lower,
			double upper, IntegrationOptions options) {
		this(kernel, null, lower, upper, options, 0.0);
	}

	private NumericalContinuousDistribution(UnivariateFunction kernel,
			UnivariateFunction logKernel, double lower, double upper,
			IntegrationOptions options, double logScale) {
		if (kernel == null) throw new IllegalArgumentException("kernel must not be null");
		if (options == null) throw new IllegalArgumentException("options must not be null");
		if (Double.isNaN(lower) || Double.isNaN(upper) || !(lower < upper)) {
			throw new IllegalArgumentException("lower bound must be less than upper bound");
		}
		if (lower == Double.POSITIVE_INFINITY
				|| upper == Double.NEGATIVE_INFINITY) {
			throw new IllegalArgumentException("support bounds point away from the interval");
		}

		this.kernel = kernel;
		this.logKernel = logKernel;
		this.logIntegrator = null;
		this.lower = lower;
		this.upper = upper;
		this.options = options;

		final double[] invalidAt = {Double.NaN};
		final int[] invalidKind = {0};
		UnivariateFunction checked = x -> {
			double value = kernel.eval(x);
			if (Double.isNaN(value) || value < 0.0) {
				invalidAt[0] = x;
				invalidKind[0] = 1;
				return Double.NaN;
			}
			if (Double.isInfinite(value)) {
				invalidAt[0] = x;
				invalidKind[0] = 2;
				return Double.NaN;
			}
			return value;
		};
		normalizationResult = Integrate.integrate(checked, lower, upper, options);
		if (invalidKind[0] != 0) {
			String problem = invalidKind[0] == 1 ? "negative or NaN" : "infinite";
			throw new IllegalArgumentException("kernel returned a " + problem
					+ " value at x=" + invalidAt[0]);
		}
		if (!normalizationResult.isSuccess()) {
			throw new IllegalArgumentException("normalization failed: "
					+ normalizationResult.detailedMessage(), normalizationResult.cause);
		}
		if (!(normalizationResult.result > 0.0)
				|| !Double.isFinite(normalizationResult.result)) {
			throw new IllegalArgumentException(
					"normalization constant must be finite and positive");
		}
		scaledNormalization = normalizationResult.result;
		logNormalization = Math.log(scaledNormalization) + logScale;
		normalization = Math.exp(logNormalization);
	}

	private NumericalContinuousDistribution(UnivariateFunction logKernel,
			double lower, double upper, IntegrationOptions options,
			LogKernelIntegrator integrator) {
		this.kernel = integrator.scaledKernel();
		this.logKernel = logKernel;
		this.logIntegrator = integrator;
		this.lower = lower;
		this.upper = upper;
		this.options = options;
		this.normalizationResult = integrator.getAggregate();
		this.scaledNormalization = normalizationResult.result;
		this.logNormalization = integrator.getLogNormalization();
		this.normalization = Math.exp(logNormalization);
	}

	/**
	 * Constructs from a log-kernel, automatically selecting a finite reference
	 * value from deterministic interior probes.
	 */
	public static NumericalContinuousDistribution fromLogKernel(
			UnivariateFunction logKernel, double lower, double upper) {
		return fromLogKernel(logKernel, lower, upper, DEFAULT_OPTIONS);
	}

	/** Constructs from a log-kernel with explicit integration options. */
	public static NumericalContinuousDistribution fromLogKernel(
			UnivariateFunction logKernel, double lower, double upper,
			IntegrationOptions options) {
		LogKernelIntegrator integrator = LogKernelIntegrator.build(logKernel, lower,
				upper, options);
		return new NumericalContinuousDistribution(logKernel, lower, upper, options,
				integrator);
	}

	/**
	 * Constructs from a log-kernel with a user-supplied finite scaling reference.
	 */
	public static NumericalContinuousDistribution fromLogKernel(
			UnivariateFunction logKernel, double lower, double upper,
			double referenceLogValue, IntegrationOptions options) {
		if (logKernel == null) {
			throw new IllegalArgumentException("logKernel must not be null");
		}
		if (!Double.isFinite(referenceLogValue)) {
			throw new IllegalArgumentException("referenceLogValue must be finite");
		}
		UnivariateFunction scaled = x -> {
			double value = logKernel.eval(x);
			if (value == Double.NEGATIVE_INFINITY) return 0.0;
			if (!Double.isFinite(value)) return Double.NaN;
			return Math.exp(value - referenceLogValue);
		};
		return new NumericalContinuousDistribution(scaled, logKernel, lower, upper,
				options, referenceLogValue);
	}

	/**
	 * Analyzes a kernel and attempts construction using the analyzer's integration
	 * settings plus any suggested breakpoints. The retained report remains
	 * advisory rather than a proof of validity.
	 */
	public static NumericalDistributionBuildResult analyze(UnivariateFunction kernel,
			double lower, double upper) {
		return analyze(kernel, lower, upper, FunctionAnalysisOptions.defaults());
	}

	/** Analyzes using defaults and an explicit construction policy. */
	public static NumericalDistributionBuildResult analyze(UnivariateFunction kernel,
			double lower, double upper, ConstructionPolicy policy) {
		return analyze(kernel, lower, upper, FunctionAnalysisOptions.builder()
				.constructionPolicy(policy).build());
	}

	/** Analyzes and attempts construction with explicit analysis settings. */
	public static NumericalDistributionBuildResult analyze(UnivariateFunction kernel,
			double lower, double upper, FunctionAnalysisOptions analysisOptions) {
		FunctionAnalysis analysis = ProbabilityFunctionAnalyzer.analyze(kernel, lower,
				upper, analysisOptions);
		NumericalContinuousDistribution distribution = null;
		IllegalArgumentException failure = null;
		ConstructionPolicy policy = analysisOptions.getConstructionPolicy();
		if (analysis.isSuitableForConstruction(policy)) {
			try {
				IntegrationOptions base = analysisOptions.getIntegrationOptions();
				double[] declared = base.getBreakpoints();
				double[] suggested = analysis.getSuggestedBreakpoints();
				double[] combined = new double[declared.length + suggested.length];
				System.arraycopy(declared, 0, combined, 0, declared.length);
				System.arraycopy(suggested, 0, combined, declared.length,
						suggested.length);
				distribution = new NumericalContinuousDistribution(kernel, lower, upper,
						base.toBuilder().breakpoints(combined).build());
			} catch (IllegalArgumentException exception) {
				failure = exception;
			}
		} else {
			failure = new IllegalArgumentException(
					"kernel analysis is rejected by " + policy
							+ " construction policy");
		}
		return new NumericalDistributionBuildResult(analysis, distribution, failure);
	}

	/**
	 * Analyzes a log-kernel in log space and attempts construction using the
	 * default settings. Negative infinity is accepted as zero mass.
	 */
	public static NumericalDistributionBuildResult analyzeLogKernel(
			UnivariateFunction logKernel, double lower, double upper) {
		return analyzeLogKernel(logKernel, lower, upper,
				FunctionAnalysisOptions.defaults());
	}

	/** Analyzes a log-kernel with defaults and an explicit policy. */
	public static NumericalDistributionBuildResult analyzeLogKernel(
			UnivariateFunction logKernel, double lower, double upper,
			ConstructionPolicy policy) {
		return analyzeLogKernel(logKernel, lower, upper,
				FunctionAnalysisOptions.builder().constructionPolicy(policy).build());
	}

	/** Analyzes and attempts log-kernel construction with explicit settings. */
	public static NumericalDistributionBuildResult analyzeLogKernel(
			UnivariateFunction logKernel, double lower, double upper,
			FunctionAnalysisOptions analysisOptions) {
		if (analysisOptions == null) {
			throw new IllegalArgumentException("analysisOptions must not be null");
		}
		FunctionAnalysis analysis = ProbabilityFunctionAnalyzer.analyzeLogKernel(
				logKernel, lower, upper, analysisOptions);
		NumericalContinuousDistribution distribution = null;
		IllegalArgumentException failure = null;
		ConstructionPolicy policy = analysisOptions.getConstructionPolicy();
		if (analysis.isSuitableForConstruction(policy)) {
			try {
				IntegrationOptions base = analysisOptions.getIntegrationOptions();
				double[] declared = base.getBreakpoints();
				double[] suggested = analysis.getSuggestedBreakpoints();
				double[] combined = new double[declared.length + suggested.length];
				System.arraycopy(declared, 0, combined, 0, declared.length);
				System.arraycopy(suggested, 0, combined, declared.length,
						suggested.length);
				distribution = fromLogKernel(logKernel, lower, upper,
						base.toBuilder().breakpoints(combined).build());
			} catch (IllegalArgumentException exception) {
				failure = exception;
			}
		} else {
			failure = new IllegalArgumentException(
					"log-kernel analysis is rejected by " + policy
							+ " construction policy");
		}
		return new NumericalDistributionBuildResult(analysis, distribution, failure);
	}

	/** Runs CDF, quantile, tail, normalization, and moment diagnostics. */
	public DistributionAnalysis analyzeDistribution() {
		return NumericalDistributionAnalyzer.analyze(this);
	}

	/** Runs diagnostics with user-selected absolute-moment orders and tail split. */
	public DistributionAnalysis analyzeDistribution(MomentAnalysisOptions settings) {
		return NumericalDistributionAnalyzer.analyze(this, settings);
	}

	/** Returns the lower support bound. */
	public double getLowerBound() { return lower; }

	/** Returns the upper support bound. */
	public double getUpperBound() { return upper; }

	/** Returns the immutable integration settings used by this distribution. */
	public IntegrationOptions getIntegrationOptions() { return options; }

	/** Number of independently scaled regions used for an automatic log-kernel. */
	public int getLogScalingRegionCount() {
		return logIntegrator == null ? (logKernel == null ? 0 : 1)
				: logIntegrator.getRegionCount();
	}

	/** Returns the cached normalization constant. */
	public double getNormalizationConstant() { return normalization; }

	/** Returns the logarithm of the cached normalization constant. */
	public double getLogNormalizationConstant() { return logNormalization; }

	/**
	 * Returns a defensive copy of the normalization diagnostics. For a
	 * log-kernel, its result and error describe the scaled integral actually sent
	 * to the quadrature routine; {@link #getLogNormalizationConstant()} describes
	 * the original formula's normalizer.
	 */
	public IntegrationResult getNormalizationResult() {
		IntegrationResult copy = new IntegrationResult();
		copy.f = kernel;
		copy.result = normalizationResult.result;
		copy.abserr = normalizationResult.abserr;
		copy.neval = normalizationResult.neval;
		copy.ier = normalizationResult.ier;
		copy.last = normalizationResult.last;
		copy.failureX = normalizationResult.failureX;
		copy.cause = normalizationResult.cause;
		copy.detail = normalizationResult.detail;
		copy.callbackProfile = normalizationResult.getCallbackProfile();
		return copy;
	}

	/** Returns immutable normalization diagnostics without retaining the kernel. */
	public ImmutableIntegrationResult getImmutableNormalizationResult() {
		return normalizationResult.toImmutable();
	}

	/** Returns the lazily built reusable monotone CDF table. */
	public NumericalCdfTable getCdfTable() {
		NumericalCdfTable result = cdfTable;
		if (result == null) {
			synchronized (this) {
				result = cdfTable;
				if (result == null) {
					result = NumericalCdfTable.build(this, cdfTableOptions);
					cdfTable = result;
				}
			}
		}
		return result;
	}

	/** Rebuilds and installs the reusable CDF table with explicit settings. */
	public synchronized NumericalCdfTable rebuildCdfTable(CdfTableOptions settings) {
		if (settings == null) throw new IllegalArgumentException("settings must not be null");
		cdfTableOptions = settings;
		cdfTable = NumericalCdfTable.build(this, settings);
		return cdfTable;
	}

	/** Drops the table so the next CDF or central quantile call rebuilds it. */
	public synchronized void clearCdfTable() { cdfTable = null; }

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		if (!Double.isFinite(x) || x < lower || x > upper) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (logKernel != null) {
			double value = logKernel.eval(x);
			if (Double.isNaN(value) || value == Double.POSITIVE_INFINITY) {
				return Double.NaN;
			}
			double logDensity = value - logNormalization;
			return log ? logDensity : Math.exp(logDensity);
		}
		double value = kernel.eval(x);
		if (Double.isNaN(value) || value < 0.0) return Double.NaN;
		if (log) return Math.log(value) - Math.log(scaledNormalization);
		return value / scaledNormalization;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulativeDirect(x, lowerTail, logP);
	}

	/** Batch CDF evaluation reuses the monotone table for ordinary probabilities. */
	@Override public void cumulativeInto(double[] input, int inputOffset,
			double[] output, int outputOffset, int length, boolean lowerTail,
			boolean logP) {
		if (input == null || output == null || inputOffset < 0 || outputOffset < 0
				|| length < 0 || inputOffset > input.length - length
				|| outputOffset > output.length - length) {
			throw new IllegalArgumentException("invalid batch input or output range");
		}
		if (!logP && length > 0) getCdfTable();
		if (copyBackward(input, inputOffset, output, outputOffset, length)) {
			for (int i = length - 1; i >= 0; i--) {
				output[outputOffset + i] = logP
						? cumulativeDirect(input[inputOffset + i], lowerTail, true)
						: cumulativeCached(input[inputOffset + i], lowerTail, false);
			}
			return;
		}
		for (int i = 0; i < length; i++) {
			output[outputOffset + i] = logP
					? cumulativeDirect(input[inputOffset + i], lowerTail, true)
					: cumulativeCached(input[inputOffset + i], lowerTail, false);
		}
	}

	/**
	 * Evaluates through the reusable CDF table. Extreme or logged tails fall back
	 * to direct integration to avoid subtractive loss.
	 */
	public double cumulativeCached(double x, boolean lowerTail, boolean logP) {
		if (logP) return cumulativeDirect(x, lowerTail, true);
		if (Double.isNaN(x)) return Double.NaN;
		if (x <= lower) return DistributionUtil.boundary(false, lowerTail, false);
		if (x >= upper) return DistributionUtil.boundary(true, lowerTail, false);
		double lowerProbability = getCdfTable().cumulative(x);
		double requested = lowerTail ? lowerProbability : 1.0 - lowerProbability;
		if (requested < 1e-8) return cumulativeDirect(x, lowerTail, false);
		return requested;
	}

	/** Direct integration path used for validation and extreme tails. */
	double cumulativeDirect(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		if (x <= lower) return DistributionUtil.boundary(false, lowerTail, logP);
		if (x >= upper) return DistributionUtil.boundary(true, lowerTail, logP);
		if (logIntegrator != null) {
			return logIntegrator.cumulative(x, lowerTail, logP);
		}

		double from = lowerTail ? lower : x;
		double to = lowerTail ? x : upper;
		IntegrationResult partial = Integrate.integrate(this::checkedKernel, from, to,
				options);
		if (!partial.isSuccess() || Double.isNaN(partial.result)
				|| partial.result < 0.0) return Double.NaN;
		if (partial.result == 0.0) {
			return logP ? Double.NEGATIVE_INFINITY : 0.0;
		}
		if (partial.result >= scaledNormalization) return logP ? 0.0 : 1.0;
		return logP ? Math.log(partial.result) - Math.log(scaledNormalization)
				: partial.result / scaledNormalization;
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		if (logP) return quantileDirect(p, lowerTail, true);
		if (p == 0.0 || p == 1.0) return quantileDirect(p, lowerTail, false);
		double target = lowerTail ? p : 1.0 - p;
		if (target < 1e-8 || target > 1.0 - 1e-8) {
			return quantileDirect(p, lowerTail, false);
		}
		double estimate = getCdfTable().quantile(target);
		for (int i = 0; i < 6; i++) {
			double actual = cumulativeDirect(estimate, true, false);
			double error = actual - target;
			double derivative = density(estimate, false);
			if (!(derivative > 0.0) || !Double.isFinite(derivative)) {
				return quantileDirect(p, lowerTail, false);
			}
			double step = error / derivative;
			if (Math.abs(step) <= 1e-13 * Math.max(1.0, Math.abs(estimate))) {
				return estimate;
			}
			double next = estimate - step;
			if (!(next > lower && next < upper) || !Double.isFinite(next)) {
				return quantileDirect(p, lowerTail, false);
			}
			estimate = next;
		}
		return estimate;
	}

	private double quantileDirect(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		boolean zero = p == (logP ? Double.NEGATIVE_INFINITY : 0.0);
		boolean one = p == (logP ? 0.0 : 1.0);
		if ((lowerTail && zero) || (!lowerTail && one)) return lower;
		if ((lowerTail && one) || (!lowerTail && zero)) return upper;

		double low = Double.isFinite(lower) ? lower
				: (Double.isFinite(upper) ? Math.min(-1.0, upper - 1.0) : -1.0);
		double high = Double.isFinite(upper) ? upper
				: (Double.isFinite(lower) ? Math.max(1.0, lower + 1.0) : 1.0);

		boolean lowCondition = quantileCondition(low, p, lowerTail, logP);
		int expansions = 0;
		while (lowCondition && !Double.isFinite(lower)
				&& expansions++ < QUANTILE_ITERATIONS) {
			high = low;
			double expanded = expandLeft(low);
			if (expanded == low) break;
			low = expanded;
			lowCondition = quantileCondition(low, p, lowerTail, logP);
		}
		boolean highCondition = quantileCondition(high, p, lowerTail, logP);
		expansions = 0;
		while (!highCondition && !Double.isFinite(upper)
				&& expansions++ < QUANTILE_ITERATIONS) {
			low = high;
			double expanded = expandRight(high);
			if (expanded == high) break;
			high = expanded;
			highCondition = quantileCondition(high, p, lowerTail, logP);
		}
		if (lowCondition || !highCondition) return Double.NaN;

		for (int i = 0; i < QUANTILE_ITERATIONS; i++) {
			double middle = midpoint(low, high);
			if (middle == low || middle == high) break;
			boolean condition = quantileCondition(middle, p, lowerTail, logP);
			if (condition) high = middle;
			else low = middle;
		}
		return high;
	}

	@Override public double random() {
		AdaptiveRejectionSampler adaptive = adaptiveRejectionSampler;
		if (adaptive != null) return adaptive.sample(random);
		RejectionSamplingConfig configured = rejectionSampling;
		if (configured != null) {
			return random(configured.envelope, configured.maxAttempts);
		}
		return quantile(random.nextDouble(), true, false);
	}

	/**
	 * Configures rejection-envelope sampling for subsequent {@link #random()}
	 * calls. The caller is responsible for the envelope's global majorization
	 * promise; sampled violations are detected and rejected with an exception.
	 */
	public void configureRejectionSampling(RejectionEnvelope envelope,
			int maxAttempts) {
		if (envelope == null) throw new IllegalArgumentException("envelope must not be null");
		if (maxAttempts < 1) throw new IllegalArgumentException("maxAttempts must be positive");
		if (!Double.isFinite(envelope.getLogMajorizationConstant())) {
			throw new IllegalArgumentException("envelope majorization constant must be finite");
		}
		rejectionSampling = new RejectionSamplingConfig(envelope, maxAttempts);
	}

	/** Configures a uniform rejection envelope over this finite support. */
	public void configureUniformRejectionSampling(double logDensityUpperBound,
			int maxAttempts) {
		if (!Double.isFinite(lower) || !Double.isFinite(upper)) {
			throw new IllegalStateException(
					"uniform rejection sampling requires finite support");
		}
		configureRejectionSampling(new UniformRejectionEnvelope(lower, upper,
				logDensityUpperBound), maxAttempts);
	}

	/** Restores inverse-CDF sampling. */
	public void clearRejectionSampling() { rejectionSampling = null; }

	public boolean isRejectionSamplingConfigured() {
		return rejectionSampling != null;
	}

	/** Configures adaptive rejection under a caller-certified log-concavity promise. */
	public void configureAdaptiveRejectionSampling(UnivariateFunction logDerivative,
			int maximumKnots, int maximumAttempts, double... initialPoints) {
		adaptiveRejectionSampler = new AdaptiveRejectionSampler(this, logDerivative,
				maximumKnots, maximumAttempts, initialPoints);
	}

	public void clearAdaptiveRejectionSampling() { adaptiveRejectionSampler = null; }
	public boolean isAdaptiveRejectionSamplingConfigured() {
		return adaptiveRejectionSampler != null;
	}
	public SamplingStrategy getSamplingStrategy() {
		if (adaptiveRejectionSampler != null) {
			return SamplingStrategy.ADAPTIVE_LOG_CONCAVE_REJECTION;
		}
		return rejectionSampling == null ? SamplingStrategy.INVERSE_CDF
				: SamplingStrategy.CERTIFIED_REJECTION;
	}
	public String getSamplingStrategyExplanation() {
		switch (getSamplingStrategy()) {
		case ADAPTIVE_LOG_CONCAVE_REJECTION:
			return "adaptive tangent envelope selected from a supplied log derivative";
		case CERTIFIED_REJECTION:
			return "caller-supplied certified rejection envelope is configured";
		default:
			return "general inverse-CDF sampling requires no shape assumptions";
		}
	}

	/** Numerically evaluates E[g(X)] with immutable integration diagnostics. */
	public ImmutableIntegrationResult expectation(UnivariateFunction function) {
		if (function == null) throw new IllegalArgumentException("function must not be null");
		return Integrate.integrateImmutable(x -> {
			double probabilityDensity = density(x, false);
			if (probabilityDensity == 0.0) return 0.0;
			return function.eval(x) * probabilityDensity;
		}, lower, upper, options);
	}

	/** Numerically evaluates E[X^order]. Fractional orders require nonnegative support. */
	public ImmutableIntegrationResult rawMoment(double order) {
		validateMomentOrder(order, false);
		return expectation(x -> Math.pow(x, order));
	}

	/** Numerically evaluates E[(X-E[X])^order]. Orders must be integers. */
	public ImmutableIntegrationResult centralMoment(double order) {
		validateMomentOrder(order, true);
		ImmutableIntegrationResult mean = rawMoment(1.0);
		if (!mean.isSuccess()) return mean;
		double center = mean.getValue();
		return expectation(x -> Math.pow(x - center, order));
	}

	/** Numerically evaluates differential entropy, -E[log f(X)]. */
	public ImmutableIntegrationResult entropy() {
		return expectation(x -> -density(x, true));
	}

	/** Returns the best mode observed by transformed-grid search and refinement. */
	public double mode() {
		final int probes = 1025;
		int best = 0;
		double bestValue = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < probes; i++) {
			double unit = (i + 0.5) / probes;
			double value = density(ProbabilityFunctionAnalyzer.mapUnit(unit,
					lower, upper), true);
			if (value > bestValue) { bestValue = value; best = i; }
		}
		double left = Math.max(0.0, (best - 1.0) / probes);
		double right = Math.min(1.0, (best + 2.0) / probes);
		final double ratio = (Math.sqrt(5.0) - 1.0) * 0.5;
		double c = right - ratio * (right - left);
		double d = left + ratio * (right - left);
		for (int i = 0; i < 64; i++) {
			double fc = density(ProbabilityFunctionAnalyzer.mapUnit(c, lower, upper), true);
			double fd = density(ProbabilityFunctionAnalyzer.mapUnit(d, lower, upper), true);
			if (fc >= fd) { right = d; d = c; c = right - ratio * (right - left); }
			else { left = c; c = d; d = left + ratio * (right - left); }
		}
		return ProbabilityFunctionAnalyzer.mapUnit((left + right) * 0.5,
				lower, upper);
	}

	/** Returns the equal-tail interval containing the requested probability. */
	public ProbabilityInterval probabilityInterval(double probability) {
		if (!(probability > 0.0 && probability < 1.0)) {
			throw new IllegalArgumentException("probability must lie between zero and one");
		}
		double tail = (1.0 - probability) * 0.5;
		return new ProbabilityInterval(quantile(tail), quantile(1.0 - tail),
				probability, "equal-tail");
	}

	private void validateMomentOrder(double order, boolean requireInteger) {
		if (!(order >= 0.0) || !Double.isFinite(order)
				|| (requireInteger && order != Math.rint(order))
				|| (lower < 0.0 && order != Math.rint(order))) {
			throw new IllegalArgumentException(
					"moment order must be finite and nonnegative; negative support and central moments require integer orders");
		}
	}

	/** Draws one value using an explicit rejection envelope. */
	public double random(RejectionEnvelope envelope, int maxAttempts) {
		if (envelope == null) throw new IllegalArgumentException("envelope must not be null");
		if (maxAttempts < 1) throw new IllegalArgumentException("maxAttempts must be positive");
		double logMajorization = envelope.getLogMajorizationConstant();
		if (!Double.isFinite(logMajorization)) {
			throw new IllegalArgumentException("envelope majorization constant must be finite");
		}
		for (int attempt = 0; attempt < maxAttempts; attempt++) {
			double candidate = envelope.sample(random);
			double logProposal = envelope.logProposalDensity(candidate);
			if (!Double.isFinite(logProposal)) {
				throw new IllegalStateException(
						"envelope sampled outside its positive proposal density");
			}
			double logTarget = density(candidate, true);
			if (Double.isNaN(logTarget)) {
				throw new IllegalStateException("target density is invalid at sampled x="
						+ candidate);
			}
			if (logTarget == Double.NEGATIVE_INFINITY) continue;
			double logAcceptance = logTarget - logProposal - logMajorization;
			if (logAcceptance > 1e-12) {
				throw new IllegalStateException("rejection envelope violated at x="
						+ candidate + " by log ratio " + logAcceptance);
			}
			if (Math.log(random.nextDouble()) <= Math.min(0.0, logAcceptance)) {
				return candidate;
			}
		}
		throw new IllegalStateException("rejection sampler exceeded " + maxAttempts
				+ " attempts; use a tighter envelope or a larger attempt budget");
	}

	private double checkedKernel(double x) {
		double value = kernel.eval(x);
		return value >= 0.0 && Double.isFinite(value) ? value : Double.NaN;
	}

	private boolean quantileCondition(double x, double p, boolean lowerTail,
			boolean logP) {
		double probability = cumulativeDirect(x, lowerTail, logP);
		if (Double.isNaN(probability)) return false;
		return lowerTail ? probability >= p : probability <= p;
	}

	private static double expandLeft(double x) {
		if (x < 0.0) return x <= -Double.MAX_VALUE / 2.0
				? -Double.MAX_VALUE : x * 2.0;
		return x - Math.max(1.0, Math.abs(x));
	}

	private static double expandRight(double x) {
		if (x > 0.0) return x >= Double.MAX_VALUE / 2.0
				? Double.MAX_VALUE : x * 2.0;
		return x + Math.max(1.0, Math.abs(x));
	}

	private static double midpoint(double low, double high) {
		double middle = low + (high - low) * 0.5;
		if (!Double.isFinite(middle)) middle = low * 0.5 + high * 0.5;
		return middle;
	}

	private static final class RejectionSamplingConfig {
		final RejectionEnvelope envelope;
		final int maxAttempts;

		RejectionSamplingConfig(RejectionEnvelope envelope, int maxAttempts) {
			this.envelope = envelope;
			this.maxAttempts = maxAttempts;
		}
	}

	private static double chooseLogReference(UnivariateFunction logKernel,
			double lower, double upper) {
		if (logKernel == null) throw new IllegalArgumentException("logKernel must not be null");
		if (Double.isNaN(lower) || Double.isNaN(upper) || !(lower < upper)) {
			throw new IllegalArgumentException("lower bound must be less than upper bound");
		}
		double maximum = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < 257; i++) {
			double x = ProbabilityFunctionAnalyzer.mapUnit((i + 0.5) / 257.0,
					lower, upper);
			double value = logKernel.eval(x);
			if (Double.isNaN(value) || value == Double.POSITIVE_INFINITY) {
				throw new IllegalArgumentException(
						"log-kernel must not return NaN or positive infinity at x=" + x);
			}
			maximum = Math.max(maximum, value);
		}
		if (!Double.isFinite(maximum)) {
			throw new IllegalArgumentException(
					"no finite log-kernel value was observed; supply a reference explicitly");
		}
		return maximum;
	}
}
