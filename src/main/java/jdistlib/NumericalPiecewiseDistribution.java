/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Arrays;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.IntegrationOptions;
import jdistlib.math.UnivariateFunction;

/**
 * Numerical distribution over a union of continuous intervals and optional
 * point atoms. At an atom, {@code density} returns its probability mass.
 */
public final class NumericalPiecewiseDistribution extends GenericDistribution
		implements SupportedDistribution {
	/** Returns a fluent builder for interval unions with optional atoms. */
	public static Builder builder() { return new Builder(); }

	public static final class Builder {
		private UnivariateFunction kernel;
		private UnivariateFunction logKernel;
		private UnivariateFunction atomWeight = x -> 0.0;
		private UnivariateFunction logAtomWeight = x -> Double.NEGATIVE_INFINITY;
		private NumericalSupport support;
		private IntegrationOptions options = IntegrationOptions.defaults();
		private Builder() {}
		public Builder kernel(UnivariateFunction value) {
			kernel = value; logKernel = null; return this;
		}
		public Builder logKernel(UnivariateFunction value) {
			logKernel = value; kernel = null; return this;
		}
		public Builder atomWeights(UnivariateFunction value) {
			atomWeight = value; return this;
		}
		public Builder logAtomWeights(UnivariateFunction value) {
			logAtomWeight = value; return this;
		}
		public Builder support(NumericalSupport value) { support = value; return this; }
		public Builder integrationOptions(IntegrationOptions value) {
			options = value; return this;
		}
		public NumericalPiecewiseDistribution build() {
			if ((kernel == null) == (logKernel == null)) {
				throw new IllegalStateException("exactly one kernel or logKernel is required");
			}
			if (support == null || options == null) {
				throw new IllegalStateException("support and integration options are required");
			}
			return logKernel == null
					? new NumericalPiecewiseDistribution(kernel, support, atomWeight,
							options)
					: NumericalPiecewiseDistribution.fromLogKernel(logKernel, support,
							logAtomWeight, options);
		}
	}
	private static final int QUANTILE_ITERATIONS = 128;
	private final NumericalSupport support;
	private final NumericalContinuousDistribution[] components;
	private final double[] componentProbability;
	private final double[] atoms;
	private final double[] atomProbability;
	private final double normalization;
	private final double logNormalization;

	public NumericalPiecewiseDistribution(UnivariateFunction kernel,
			NumericalSupport support, UnivariateFunction atomWeight,
			IntegrationOptions options) {
		this(buildOrdinary(kernel, support, atomWeight, options));
	}

	public NumericalPiecewiseDistribution(UnivariateFunction kernel,
			NumericalSupport support, IntegrationOptions options) {
		this(kernel, support, null, options);
	}

	private NumericalPiecewiseDistribution(Prepared prepared) {
		support = prepared.support;
		components = prepared.components;
		componentProbability = prepared.componentProbability;
		atoms = prepared.atoms;
		atomProbability = prepared.atomProbability;
		logNormalization = prepared.logNormalization;
		normalization = Math.exp(logNormalization);
	}

	/** Constructs a regionally scaled piecewise distribution from log formulas. */
	public static NumericalPiecewiseDistribution fromLogKernel(
			UnivariateFunction logKernel, NumericalSupport support,
			UnivariateFunction logAtomWeight, IntegrationOptions options) {
		return new NumericalPiecewiseDistribution(
				buildLog(logKernel, support, logAtomWeight, options));
	}

	public NumericalSupport getSupport() { return support; }
	@Override public double getLowerBound() { return support.getLowerBound(); }
	@Override public double getUpperBound() { return support.getUpperBound(); }
	public double getNormalizationConstant() { return normalization; }
	public double getLogNormalizationConstant() { return logNormalization; }

	@Override public double density(double x, boolean log) {
		if (Double.isNaN(x)) return Double.NaN;
		int atom = Arrays.binarySearch(atoms, x);
		if (atom >= 0) return log ? Math.log(atomProbability[atom])
				: atomProbability[atom];
		NumericalSupport.Interval[] intervals = support.getIntervals();
		for (int i = 0; i < intervals.length; i++) {
			if (intervals[i].contains(x)) {
				double logDensity = components[i].density(x, true)
						+ Math.log(componentProbability[i]);
				return log ? logDensity : Math.exp(logDensity);
			}
		}
		return log ? Double.NEGATIVE_INFINITY : 0.0;
	}

	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x)) return Double.NaN;
		double probability = 0.0;
		if (lowerTail) {
			for (int i = 0; i < atoms.length; i++) {
				if (atoms[i] <= x) probability += atomProbability[i];
			}
			for (int i = 0; i < components.length; i++) {
				probability += componentProbability[i]
						* components[i].cumulative(x, true, false);
			}
		} else {
			for (int i = 0; i < atoms.length; i++) {
				if (atoms[i] > x) probability += atomProbability[i];
			}
			for (int i = 0; i < components.length; i++) {
				probability += componentProbability[i]
						* components[i].cumulative(x, false, false);
			}
		}
		probability = Math.max(0.0, Math.min(1.0, probability));
		return logP ? Math.log(probability) : probability;
	}

	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		if (Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) {
			return Double.NaN;
		}
		boolean zero = p == (logP ? Double.NEGATIVE_INFINITY : 0.0);
		boolean one = p == (logP ? 0.0 : 1.0);
		double lower = support.getLowerBound();
		double upper = support.getUpperBound();
		if ((lowerTail && zero) || (!lowerTail && one)) return lower;
		if ((lowerTail && one) || (!lowerTail && zero)) return upper;
		double target = logP ? Math.exp(p) : p;
		if (!lowerTail) target = logP ? -Math.expm1(p) : 1.0 - p;

		double low = Double.isFinite(lower) ? lower : -1.0;
		double high = Double.isFinite(upper) ? upper : 1.0;
		while (!Double.isFinite(lower) && cumulative(low, true, false) >= target) {
			high = low;
			double next = low < 0.0 ? low * 2.0 : low - 1.0;
			if (next == low || !Double.isFinite(next)) break;
			low = next;
		}
		while (!Double.isFinite(upper) && cumulative(high, true, false) < target) {
			low = high;
			double next = high > 0.0 ? high * 2.0 : high + 1.0;
			if (next == high || !Double.isFinite(next)) break;
			high = next;
		}
		for (int i = 0; i < QUANTILE_ITERATIONS; i++) {
			double middle = low + (high - low) * 0.5;
			if (!Double.isFinite(middle)) middle = low * 0.5 + high * 0.5;
			if (middle == low || middle == high) break;
			if (cumulative(middle, true, false) >= target) high = middle;
			else low = middle;
		}
		/* Snap a neighboring jump to its exact atom instead of returning an
		 * adjacent representable double. */
		int atomIndex = Arrays.binarySearch(atoms, high);
		if (atomIndex < 0) atomIndex = -atomIndex - 1;
		if (atomIndex < atoms.length && jumpContains(atoms[atomIndex], target)) {
			return atoms[atomIndex];
		}
		if (atomIndex > 0 && jumpContains(atoms[atomIndex - 1], target)) {
			return atoms[atomIndex - 1];
		}
		return high;
	}

	@Override public double random() {
		return quantile(random.nextDouble(), true, false);
	}

	private boolean jumpContains(double atom, double target) {
		return cumulative(Math.nextDown(atom), true, false) < target
				&& cumulative(atom, true, false) >= target;
	}

	private static Prepared buildOrdinary(UnivariateFunction kernel,
			NumericalSupport support, UnivariateFunction atomWeight,
			IntegrationOptions options) {
		validate(kernel, support, options);
		NumericalSupport.Interval[] intervals = support.getIntervals();
		NumericalContinuousDistribution[] components =
				new NumericalContinuousDistribution[intervals.length];
		double[] logWeight = new double[intervals.length + support.getAtoms().length];
		IntegrationOptions regional = withSingularities(options,
				support.getSingularities());
		for (int i = 0; i < intervals.length; i++) {
			components[i] = new NumericalContinuousDistribution(kernel,
					intervals[i].getLower(), intervals[i].getUpper(), regional);
			logWeight[i] = components[i].getLogNormalizationConstant();
		}
		double[] atoms = support.getAtoms();
		if (atoms.length > 0 && atomWeight == null) {
			throw new IllegalArgumentException("atomWeight is required when support has atoms");
		}
		for (int i = 0; i < atoms.length; i++) {
			double weight = atomWeight.eval(atoms[i]);
			if (!(weight >= 0.0) || !Double.isFinite(weight)) {
				throw new IllegalArgumentException("invalid atom weight at x=" + atoms[i]);
			}
			logWeight[intervals.length + i] = Math.log(weight);
		}
		return prepare(support, components, atoms, logWeight);
	}

	private static Prepared buildLog(UnivariateFunction logKernel,
			NumericalSupport support, UnivariateFunction logAtomWeight,
			IntegrationOptions options) {
		validate(logKernel, support, options);
		NumericalSupport.Interval[] intervals = support.getIntervals();
		NumericalContinuousDistribution[] components =
				new NumericalContinuousDistribution[intervals.length];
		double[] atoms = support.getAtoms();
		double[] logWeight = new double[intervals.length + atoms.length];
		IntegrationOptions regional = withSingularities(options,
				support.getSingularities());
		for (int i = 0; i < intervals.length; i++) {
			components[i] = NumericalContinuousDistribution.fromLogKernel(logKernel,
					intervals[i].getLower(), intervals[i].getUpper(), regional);
			logWeight[i] = components[i].getLogNormalizationConstant();
		}
		if (atoms.length > 0 && logAtomWeight == null) {
			throw new IllegalArgumentException("logAtomWeight is required when support has atoms");
		}
		for (int i = 0; i < atoms.length; i++) {
			logWeight[intervals.length + i] = logAtomWeight.eval(atoms[i]);
			if (Double.isNaN(logWeight[intervals.length + i])
					|| logWeight[intervals.length + i] == Double.POSITIVE_INFINITY) {
				throw new IllegalArgumentException("invalid log atom weight at x=" + atoms[i]);
			}
		}
		return prepare(support, components, atoms, logWeight);
	}

	private static Prepared prepare(NumericalSupport support,
			NumericalContinuousDistribution[] components, double[] atoms,
			double[] logWeight) {
		double logTotal = Double.NEGATIVE_INFINITY;
		for (double value : logWeight) logTotal = logAdd(logTotal, value);
		if (!Double.isFinite(logTotal)) {
			throw new IllegalArgumentException("piecewise normalization is not finite and positive");
		}
		double[] componentProbability = new double[components.length];
		for (int i = 0; i < components.length; i++) {
			componentProbability[i] = Math.exp(logWeight[i] - logTotal);
		}
		double[] atomProbability = new double[atoms.length];
		for (int i = 0; i < atoms.length; i++) {
			atomProbability[i] = Math.exp(logWeight[components.length + i] - logTotal);
		}
		return new Prepared(support, components, componentProbability, atoms,
				atomProbability, logTotal);
	}

	private static void validate(UnivariateFunction kernel, NumericalSupport support,
			IntegrationOptions options) {
		if (kernel == null || support == null || options == null) {
			throw new IllegalArgumentException("kernel, support, and options are required");
		}
	}

	private static IntegrationOptions withSingularities(IntegrationOptions options,
			double[] singularities) {
		double[] declared = options.getBreakpoints();
		double[] combined = new double[declared.length + singularities.length];
		System.arraycopy(declared, 0, combined, 0, declared.length);
		System.arraycopy(singularities, 0, combined, declared.length,
				singularities.length);
		return options.toBuilder().breakpoints(combined).build();
	}

	private static double logAdd(double x, double y) {
		if (x == Double.NEGATIVE_INFINITY) return y;
		if (y == Double.NEGATIVE_INFINITY) return x;
		double high = Math.max(x, y);
		return high + Math.log1p(Math.exp(Math.min(x, y) - high));
	}

	private static final class Prepared {
		final NumericalSupport support;
		final NumericalContinuousDistribution[] components;
		final double[] componentProbability;
		final double[] atoms;
		final double[] atomProbability;
		final double logNormalization;
		Prepared(NumericalSupport support,
				NumericalContinuousDistribution[] components,
				double[] componentProbability, double[] atoms,
				double[] atomProbability, double logNormalization) {
			this.support = support;
			this.components = components;
			this.componentProbability = componentProbability;
			this.atoms = atoms;
			this.atomProbability = atomProbability;
			this.logNormalization = logNormalization;
		}
	}
}
