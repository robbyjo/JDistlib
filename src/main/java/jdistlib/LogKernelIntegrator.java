/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import jdistlib.math.Integrate;
import jdistlib.math.IntegrationOptions;
import jdistlib.math.IntegrationResult;
import jdistlib.math.UnivariateFunction;

/** Mode-searched, regionally scaled log-kernel quadrature. */
final class LogKernelIntegrator {
	private static final int PROBES = 513;
	private static final int MAX_REGIONS = 8;

	private final UnivariateFunction logKernel;
	private final IntegrationOptions options;
	private final Region[] regions;
	private final double logNormalization;
	private final double globalReference;
	private final IntegrationResult aggregate;

	private LogKernelIntegrator(UnivariateFunction logKernel,
			IntegrationOptions options, Region[] regions, double logNormalization,
			double globalReference, IntegrationResult aggregate) {
		this.logKernel = logKernel;
		this.options = options;
		this.regions = regions;
		this.logNormalization = logNormalization;
		this.globalReference = globalReference;
		this.aggregate = aggregate;
	}

	static LogKernelIntegrator build(UnivariateFunction logKernel, double lower,
			double upper, IntegrationOptions options) {
		if (logKernel == null || options == null || !(lower < upper)) {
			throw new IllegalArgumentException("log-kernel, support, or options are invalid");
		}
		double[] unit = new double[PROBES];
		double[] value = new double[PROBES];
		List<Candidate> candidates = new ArrayList<Candidate>();
		for (int i = 0; i < PROBES; i++) {
			unit[i] = (i + 0.5) / PROBES;
			double x = ProbabilityFunctionAnalyzer.mapUnit(unit[i], lower, upper);
			value[i] = checkedLogValue(logKernel, x);
		}
		for (int i = 1; i + 1 < PROBES; i++) {
			if (Double.isFinite(value[i]) && value[i] >= value[i - 1]
					&& value[i] >= value[i + 1]) {
				candidates.add(refine(logKernel, lower, upper, unit[i - 1],
						unit[i + 1]));
			}
		}
		int best = 0;
		for (int i = 1; i < PROBES; i++) if (value[i] > value[best]) best = i;
		if (!Double.isFinite(value[best])) {
			throw new IllegalArgumentException("no finite log-kernel value was observed");
		}
		boolean globalRepresented = false;
		for (Candidate candidate : candidates) {
			if (Math.abs(candidate.unit - unit[best]) <= 2.0 / PROBES) {
				globalRepresented = true;
				break;
			}
		}
		if (!globalRepresented) {
			candidates.add(refine(logKernel, lower, upper,
					Math.max(0.0, unit[best] - 1.0 / PROBES),
					Math.min(1.0, unit[best] + 1.0 / PROBES)));
		}
		Collections.sort(candidates,
				Comparator.comparingDouble((Candidate candidate) -> candidate.value)
						.reversed());
		if (candidates.size() > MAX_REGIONS) {
			candidates = new ArrayList<Candidate>(candidates.subList(0, MAX_REGIONS));
		}
		Collections.sort(candidates,
				Comparator.comparingDouble(candidate -> candidate.unit));

		double[] boundaries = new double[candidates.size() + 1];
		boundaries[0] = lower;
		boundaries[boundaries.length - 1] = upper;
		for (int i = 1; i < candidates.size(); i++) {
			double splitUnit = (candidates.get(i - 1).unit
					+ candidates.get(i).unit) * 0.5;
			boundaries[i] = ProbabilityFunctionAnalyzer.mapUnit(splitUnit,
					lower, upper);
		}

		Region[] regions = new Region[candidates.size()];
		double logTotal = Double.NEGATIVE_INFINITY;
		double globalReference = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < regions.length; i++) {
			double reference = candidates.get(i).value;
			UnivariateFunction scaled = scaled(logKernel, reference);
			IntegrationResult integral = Integrate.integrate(scaled, boundaries[i],
					boundaries[i + 1], options);
			if (!integral.isSuccess() || !(integral.result > 0.0)
					|| !Double.isFinite(integral.result)) {
				throw new IllegalArgumentException("regional log-kernel normalization failed: "
						+ integral.detailedMessage(), integral.cause);
			}
			double logArea = reference + Math.log(integral.result);
			regions[i] = new Region(boundaries[i], boundaries[i + 1], reference,
					logArea, integral);
			logTotal = logAdd(logTotal, logArea);
			globalReference = Math.max(globalReference, reference);
		}

		IntegrationResult aggregate = new IntegrationResult();
		aggregate.f = scaled(logKernel, globalReference);
		double result = 0.0;
		double error = 0.0;
		for (Region region : regions) {
			double factor = Math.exp(region.reference - globalReference);
			result += region.integral.result * factor;
			error += region.integral.abserr * factor;
			aggregate.neval += region.integral.neval;
			aggregate.last += region.integral.last;
		}
		aggregate.result = result;
		aggregate.abserr = error;
		aggregate.detail = "mode-searched regional log scaling with "
				+ regions.length + " region(s)";
		return new LogKernelIntegrator(logKernel, options, regions, logTotal,
				globalReference, aggregate);
	}

	double cumulative(double x, boolean lowerTail, boolean logP) {
		double logMass = logIntegral(lowerTail ? Double.NEGATIVE_INFINITY : x,
				lowerTail ? x : Double.POSITIVE_INFINITY);
		double logProbability = logMass - logNormalization;
		if (logP) return logProbability > 0.0 ? 0.0 : logProbability;
		return logProbability >= 0.0 ? 1.0 : Math.exp(logProbability);
	}

	double logIntegral(double lower, double upper) {
		double total = Double.NEGATIVE_INFINITY;
		for (Region region : regions) {
			double from = Math.max(lower, region.lower);
			double to = Math.min(upper, region.upper);
			if (!(from < to)) continue;
			if (from == region.lower && to == region.upper) {
				total = logAdd(total, region.logArea);
				continue;
			}
			IntegrationResult partial = Integrate.integrate(
					scaled(logKernel, region.reference), from, to, options);
			if (!partial.isSuccess() || partial.result < 0.0) return Double.NaN;
			if (partial.result > 0.0) {
				total = logAdd(total, region.reference + Math.log(partial.result));
			}
		}
		return total;
	}

	UnivariateFunction scaledKernel() { return scaled(logKernel, globalReference); }
	double getLogNormalization() { return logNormalization; }
	double getGlobalReference() { return globalReference; }
	int getRegionCount() { return regions.length; }
	IntegrationResult getAggregate() { return aggregate; }

	private static Candidate refine(UnivariateFunction logKernel, double lower,
			double upper, double left, double right) {
		final double ratio = (Math.sqrt(5.0) - 1.0) * 0.5;
		double c = right - ratio * (right - left);
		double d = left + ratio * (right - left);
		double fc = checkedLogValue(logKernel,
				ProbabilityFunctionAnalyzer.mapUnit(c, lower, upper));
		double fd = checkedLogValue(logKernel,
				ProbabilityFunctionAnalyzer.mapUnit(d, lower, upper));
		for (int i = 0; i < 48; i++) {
			if (fc >= fd) {
				right = d;
				d = c;
				fd = fc;
				c = right - ratio * (right - left);
				fc = checkedLogValue(logKernel,
						ProbabilityFunctionAnalyzer.mapUnit(c, lower, upper));
			} else {
				left = c;
				c = d;
				fc = fd;
				d = left + ratio * (right - left);
				fd = checkedLogValue(logKernel,
						ProbabilityFunctionAnalyzer.mapUnit(d, lower, upper));
			}
		}
		return fc >= fd ? new Candidate(c, fc) : new Candidate(d, fd);
	}

	private static double checkedLogValue(UnivariateFunction function, double x) {
		double value = function.eval(x);
		if (Double.isNaN(value) || value == Double.POSITIVE_INFINITY) {
			throw new IllegalArgumentException(
					"log-kernel returned " + value + " at x=" + x);
		}
		return value;
	}

	private static UnivariateFunction scaled(UnivariateFunction logKernel,
			double reference) {
		return x -> {
			double value = logKernel.eval(x);
			if (value == Double.NEGATIVE_INFINITY) return 0.0;
			if (!Double.isFinite(value)) return Double.NaN;
			return Math.exp(value - reference);
		};
	}

	private static double logAdd(double x, double y) {
		if (x == Double.NEGATIVE_INFINITY) return y;
		if (y == Double.NEGATIVE_INFINITY) return x;
		double high = Math.max(x, y);
		return high + Math.log1p(Math.exp(Math.min(x, y) - high));
	}

	private static final class Candidate {
		final double unit;
		final double value;
		Candidate(double unit, double value) { this.unit = unit; this.value = value; }
	}

	private static final class Region {
		final double lower;
		final double upper;
		final double reference;
		final double logArea;
		final IntegrationResult integral;
		Region(double lower, double upper, double reference, double logArea,
				IntegrationResult integral) {
			this.lower = lower;
			this.upper = upper;
			this.reference = reference;
			this.logArea = logArea;
			this.integral = integral;
		}
	}
}
