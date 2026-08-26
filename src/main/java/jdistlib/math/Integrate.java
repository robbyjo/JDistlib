/*
 *  Java translation of R's QUADPACK integration routines.
 *  Copyright (C) 2001-2025 The R Core Team
 *  Java port and thread-safety adaptation Copyright (C) 2026 Roby Joehanes
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  <http://www.gnu.org/licenses/>.
 */
package jdistlib.math;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static jdistlib.math.Constants.DBL_EPSILON;
import static jdistlib.math.Constants.DBL_MAX;
import static jdistlib.math.Constants.DBL_MIN;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * Adaptive numerical integration corresponding to R 4.6.1
 * {@code stats::integrate}. Finite intervals use QUADPACK {@code dqags};
 * infinite intervals use {@code dqagi}. Invocation state is call-local.
 */
public class Integrate {
	private static final double DEFAULT_TOLERANCE = pow(DBL_EPSILON, 0.25);
	private static final int DEFAULT_SUBDIVISIONS = 100;

	private Integrate() {}

	/**
	 * Integrates {@code f} over a finite, semi-infinite, or infinite interval.
	 * The defaults match R's {@code integrate}: 100 subdivisions and
	 * {@code .Machine$double.eps^.25} absolute and relative tolerances.
	 */
	public static IntegrationResult integrate(UnivariateFunction f, double lower, double upper) {
		return integrate(f, lower, upper, DEFAULT_TOLERANCE, DEFAULT_TOLERANCE,
				DEFAULT_SUBDIVISIONS);
	}

	/** Integrates with hardened defaults and returns an immutable modern result. */
	public static ImmutableIntegrationResult integrateImmutable(UnivariateFunction f,
			double lower, double upper) {
		return integrate(f, lower, upper, IntegrationOptions.defaults()).toImmutable();
	}

	/** Integrates with hardened options and returns an immutable modern result. */
	public static ImmutableIntegrationResult integrateImmutable(UnivariateFunction f,
			double lower, double upper, IntegrationOptions options) {
		return integrate(f, lower, upper, options).toImmutable();
	}

	/**
	 * Integrates {@code f} with QUADPACK-compatible status codes.
	 * All work arrays are local to the invocation, so concurrent calls are safe.
	 */
	public static IntegrationResult integrate(UnivariateFunction f, double lower, double upper,
			double epsabs, double epsrel, int limit) {
		IntegrationResult invalid = new IntegrationResult();
		invalid.f = f;
		if (f == null || Double.isNaN(lower) || Double.isNaN(upper)
				|| Double.isNaN(epsabs) || Double.isNaN(epsrel) || limit < 1 ||
				(epsabs <= 0 && epsrel < max(50 * DBL_EPSILON, 5e-29))) {
			invalid.ier = 6;
			return invalid;
		}

		if (Double.isFinite(lower) && Double.isFinite(upper)) {
			return integrateFinite(f, lower, upper, epsabs, epsrel, limit);
		}

		/* Match stats::integrate: once either bound is infinite, the finite bound
		 * selects the semi-infinite direction; two infinite bounds select the
		 * whole real line. */
		if (Double.isFinite(lower))
			return dqagie(f, lower, 1, epsabs, epsrel, limit);
		if (Double.isFinite(upper))
			return dqagie(f, upper, -1, epsabs, epsrel, limit);
		return dqagie(f, 0.0, 2, epsabs, epsrel, limit);
	}

	/**
	 * Integrates with hardened callback handling, evaluation budgets, optional
	 * breakpoints, cancellation, and selectable quadrature methods. The legacy
	 * overloads remain unchanged for R/QUADPACK compatibility.
	 */
	public static IntegrationResult integrate(UnivariateFunction f, double lower,
			double upper, IntegrationOptions options) {
		IntegrationResult invalid = new IntegrationResult();
		invalid.f = f;
		if (f == null || options == null || Double.isNaN(lower)
				|| Double.isNaN(upper)) {
			invalid.ier = 6;
			return invalid;
		}

		EvaluationGuard guard = new EvaluationGuard(f, options);
		try {
			return integrateGuarded(f, lower, upper, options, guard);
		} finally {
			guard.close();
		}
	}

	private static IntegrationResult integrateGuarded(UnivariateFunction f,
			double lower, double upper, IntegrationOptions options,
			EvaluationGuard guard) {
		boolean reverse = lower > upper;
		double low = reverse ? upper : lower;
		double high = reverse ? lower : upper;
		double[] points = internalBreakpoints(options.getBreakpoints(), low, high);
		int pieces = points.length + 1;
		double absolute = options.getAbsoluteTolerance() / pieces;
		double sum = 0.0;
		double correction = 0.0;
		double error = 0.0;
		int last = 0;
		IntegrationResult combined = new IntegrationResult();
		combined.f = f;
		double from = low;
		for (int i = 0; i < pieces; i++) {
			double to = i == points.length ? high : points[i];
			IntegrationResult part = integrateSelected(guard, from, to, absolute,
					options);
			double adjusted = part.result - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
			error += part.abserr;
			last += part.last;
			if (part.detail != null && combined.detail == null) {
				combined.detail = part.detail;
			}
			if (!part.isSuccess()) {
				combined.ier = part.ier;
				combined.failureX = part.failureX;
				combined.cause = part.cause;
				combined.detail = part.detail;
				break;
			}
			from = to;
		}
		combined.result = reverse ? -sum : sum;
		combined.abserr = error;
		combined.last = last;
		combined.neval = guard.evaluations;
		guard.decorate(combined);
		return combined;
	}

	/**
	 * Repeats an integral with tighter tolerances and an additional midpoint
	 * split. Agreement is a useful stability check, but is not a proof of
	 * correctness or convergence.
	 */
	public static IntegrationStabilityResult assessStability(UnivariateFunction f,
			double lower, double upper, IntegrationOptions options) {
		if (options == null) throw new IllegalArgumentException("options must not be null");
		IntegrationResult baseline = integrate(f, lower, upper, options);
		double tighterAbsolute = options.getAbsoluteTolerance() * 0.1;
		double tighterRelative = options.getRelativeTolerance() * 0.1;
		if (tighterAbsolute == 0.0 && tighterRelative < 100.0 * DBL_EPSILON) {
			tighterRelative = 100.0 * DBL_EPSILON;
		}
		int tighterLimit = options.getSubdivisions() > Integer.MAX_VALUE / 2
				? Integer.MAX_VALUE : options.getSubdivisions() * 2;
		IntegrationOptions tightenedOptions = options.toBuilder()
				.tolerances(tighterAbsolute, tighterRelative)
				.subdivisions(tighterLimit)
				.build();
		IntegrationResult tightened = integrate(f, lower, upper, tightenedOptions);

		double[] declared = options.getBreakpoints();
		double[] splitPoints;
		if (Double.isFinite(lower) && Double.isFinite(upper) && lower != upper) {
			splitPoints = new double[declared.length + 1];
			System.arraycopy(declared, 0, splitPoints, 0, declared.length);
			splitPoints[declared.length] = lower * 0.5 + upper * 0.5;
		} else {
			splitPoints = new double[declared.length + 1];
			System.arraycopy(declared, 0, splitPoints, 0, declared.length);
			splitPoints[declared.length] = Double.isFinite(lower) ? lower + 1.0
					: (Double.isFinite(upper) ? upper - 1.0 : 0.0);
		}
		IntegrationOptions splitOptions = tightenedOptions.toBuilder()
				.breakpoints(splitPoints).build();
		IntegrationResult split = integrate(f, lower, upper, splitOptions);

		double discrepancy = Math.max(abs(baseline.result - tightened.result),
				abs(tightened.result - split.result));
		double scale = Math.max(1.0, abs(tightened.result));
		double toleranceAllowance = Math.max(tighterAbsolute,
				tighterRelative * scale) * 8.0;
		double errorAllowance = 4.0 * (baseline.abserr + tightened.abserr
				+ split.abserr);
		double allowed = Math.max(toleranceAllowance, errorAllowance);
		boolean stable = baseline.isSuccess() && tightened.isSuccess()
				&& split.isSuccess() && Double.isFinite(discrepancy)
				&& discrepancy <= allowed;
		return new IntegrationStabilityResult(baseline, tightened, split,
				discrepancy, allowed, stable);
	}

	private static IntegrationResult integrateSelected(EvaluationGuard guard,
			double lower, double upper, double epsabs, IntegrationOptions options) {
		IntegrationOptions.Method method = options.getMethod();
		if (method == IntegrationOptions.Method.TANH_SINH) {
			if (!Double.isFinite(lower) || !Double.isFinite(upper)) {
				IntegrationResult invalid = new IntegrationResult();
				invalid.ier = 6;
				invalid.detail = "tanh-sinh currently requires finite bounds";
				return invalid;
			}
			return tanhSinh(guard, lower, upper, epsabs,
					options.getRelativeTolerance(), options.getTanhSinhMaxLevels());
		}
		if (method == IntegrationOptions.Method.DOUBLE_EXPONENTIAL) {
			return doubleExponential(guard, lower, upper, epsabs,
					options.getRelativeTolerance(), options.getTanhSinhMaxLevels());
		}
		if (method == IntegrationOptions.Method.CQUAD) {
			if (!Double.isFinite(lower) || !Double.isFinite(upper)) {
				IntegrationResult invalid = new IntegrationResult();
				invalid.ier = 6;
				invalid.detail = "CQUAD requires finite bounds";
				return invalid;
			}
			return cquad(guard, lower, upper, epsabs,
					options.getRelativeTolerance(), options.getSubdivisions());
		}

		IntegrationResult quadpack = integrate(guard, lower, upper, epsabs,
				options.getRelativeTolerance(), options.getSubdivisions());
		if (method != IntegrationOptions.Method.AUTO || quadpack.isSuccess()
				|| guard.hasFailure()) return quadpack;

		IntegrationResult cquad = null;
		if (Double.isFinite(lower) && Double.isFinite(upper)) {
			cquad = cquad(guard, lower, upper, epsabs,
					options.getRelativeTolerance(), options.getSubdivisions());
			if (cquad.isSuccess()) {
				cquad.detail = "CQUAD fallback used after QUADPACK status "
						+ quadpack.ier;
				return cquad;
			}
			if (guard.hasFailure()) return cquad;
		}

		IntegrationResult alternative = doubleExponential(guard, lower, upper, epsabs,
				options.getRelativeTolerance(), options.getTanhSinhMaxLevels());
		if (alternative.isSuccess()) {
			alternative.detail = "double-exponential fallback used after QUADPACK"
					+ (cquad == null ? "" : " and CQUAD") + " failures";
			return alternative;
		}
		IntegrationResult best = betterEstimate(alternative, quadpack);
		return cquad == null ? best : betterEstimate(cquad, best);
	}

	private static IntegrationResult betterEstimate(IntegrationResult first,
			IntegrationResult second) {
		if (!Double.isFinite(first.abserr)) return second;
		if (!Double.isFinite(second.abserr)) return first;
		return first.abserr < second.abserr ? first : second;
	}

	/**
	 * Doubly-adaptive finite-interval Clenshaw-Curtis integration inspired by
	 * Gonnet's CQUAD algorithm. Each interval is p-refined through nested rules
	 * of degree 4, 8, 16, and 32 before h-refinement by bisection.
	 */
	private static IntegrationResult cquad(EvaluationGuard f, double lower,
			double upper, double epsabs, double epsrel, int maxIntervals) {
		IntegrationResult result = new IntegrationResult();
		result.f = f;
		result.abserr = DBL_MAX;
		if (!(lower < upper) || !Double.isFinite(lower)
				|| !Double.isFinite(upper)) {
			if (lower == upper) {
				result.abserr = 0.0;
				return result;
			}
			result.ier = 6;
			return result;
		}

		PriorityQueue<CquadInterval> intervals =
				new PriorityQueue<CquadInterval>(11,
						new Comparator<CquadInterval>() {
			@Override public int compare(CquadInterval first,
					CquadInterval second) {
				return Double.compare(second.error, first.error);
			}
		});
		CquadInterval root = CquadInterval.initial(f, lower, upper);
		if (root == null) {
			result.ier = 3;
			result.detail = "CQUAD could not construct its initial interpolant";
			return result;
		}
		intervals.add(root);

		while (true) {
			double integral = cquadIntegral(intervals);
			double error = cquadError(intervals);
			result.result = integral;
			result.abserr = error;
			result.last = intervals.size();
			double target = max(epsabs, epsrel * abs(integral));
			if (error <= target) return result;
			CquadInterval worst = intervals.poll();
			if (worst == null) {
				result.ier = 3;
				result.detail = "CQUAD lost its active interval";
				return result;
			}
			if (worst.level < CquadInterval.MAX_LEVEL) {
				if (!worst.refine(f)) {
					result.ier = 3;
					result.detail = "CQUAD refinement produced a non-finite interpolant";
					return result;
				}
				intervals.add(worst);
				continue;
			}

			if (intervals.size() + 2 > maxIntervals) {
				intervals.add(worst);
				result.result = cquadIntegral(intervals);
				result.abserr = cquadError(intervals);
				result.last = intervals.size();
				result.ier = 1;
				result.detail = "CQUAD interval workspace exhausted";
				return result;
			}
			double midpoint = worst.lower * 0.5 + worst.upper * 0.5;
			if (!(midpoint > worst.lower && midpoint < worst.upper)) {
				intervals.add(worst);
				result.ier = 2;
				result.detail = "CQUAD interval cannot be bisected in double precision";
				return result;
			}
			CquadInterval left = CquadInterval.initial(f, worst.lower, midpoint);
			CquadInterval right = CquadInterval.initial(f, midpoint, worst.upper);
			if (left == null || right == null) {
				result.ier = 3;
				result.detail = "CQUAD bisection produced a non-finite interpolant";
				return result;
			}
			intervals.add(left);
			intervals.add(right);
		}
	}

	private static double cquadIntegral(PriorityQueue<CquadInterval> intervals) {
		double sum = 0.0;
		double correction = 0.0;
		for (CquadInterval interval : intervals) {
			double adjusted = interval.integral - correction;
			double next = sum + adjusted;
			correction = (next - sum) - adjusted;
			sum = next;
		}
		return sum;
	}

	private static double cquadError(PriorityQueue<CquadInterval> intervals) {
		double sum = 0.0;
		for (CquadInterval interval : intervals) sum += interval.error;
		return sum;
	}

	private static final class CquadInterval {
		private static final int[] DEGREES = {4, 8, 16, 32};
		private static final int MAX_LEVEL = DEGREES.length - 1;
		private final double lower;
		private final double upper;
		private int level;
		private double[] samples;
		private double[] coefficients;
		private double integral;
		private double error;

		private CquadInterval(double lower, double upper) {
			this.lower = lower;
			this.upper = upper;
		}

		static CquadInterval initial(EvaluationGuard f, double lower,
				double upper) {
			CquadInterval interval = new CquadInterval(lower, upper);
			int degree = DEGREES[0];
			interval.samples = new double[degree + 1];
			for (int index = 0; index <= degree; index++) {
				double x = interval.node(index, degree);
				interval.samples[index] = f.eval(x);
				if (!Double.isFinite(interval.samples[index])) return null;
			}
			interval.coefficients = coefficients(interval.samples);
			interval.integral = interval.integral(interval.coefficients);
			interval.error = Double.POSITIVE_INFINITY;
			return interval.refine(f) ? interval : null;
		}

		boolean refine(EvaluationGuard f) {
			if (level >= MAX_LEVEL) return true;
			int oldDegree = DEGREES[level];
			int newDegree = DEGREES[level + 1];
			double[] refined = new double[newDegree + 1];
			for (int index = 0; index <= oldDegree; index++)
				refined[2 * index] = samples[index];
			for (int index = 1; index < newDegree; index += 2) {
				refined[index] = f.eval(node(index, newDegree));
				if (!Double.isFinite(refined[index])) return false;
			}
			double[] nextCoefficients = coefficients(refined);
			double nextIntegral = integral(nextCoefficients);
			if (!Double.isFinite(nextIntegral)) return false;
			double difference = interpolantDifference(coefficients,
					nextCoefficients);
			double maximum = 0.0;
			for (double sample : refined) maximum = max(maximum, abs(sample));
			double scale = abs(nextIntegral) + (upper - lower) * maximum;
			double floor = 50.0 * DBL_EPSILON * scale;
			error = max(difference, floor);
			level++;
			samples = refined;
			coefficients = nextCoefficients;
			integral = nextIntegral;
			return Double.isFinite(error);
		}

		private double node(int index, int degree) {
			double midpoint = lower * 0.5 + upper * 0.5;
			double halfWidth = upper * 0.5 - lower * 0.5;
			return midpoint + halfWidth * Math.cos(Math.PI * index / degree);
		}

		private double integral(double[] values) {
			double normalized = 0.0;
			for (int order = 0; order < values.length; order += 2)
				normalized += values[order] * chebyshevIntegral(order);
			return (upper * 0.5 - lower * 0.5) * normalized;
		}

		private double interpolantDifference(double[] coarse,
				double[] fine) {
			double[] difference = fine.clone();
			for (int i = 0; i < coarse.length; i++) difference[i] -= coarse[i];
			double squaredNorm = 0.0;
			for (int i = 0; i < difference.length; i++) {
				for (int j = 0; j < difference.length; j++) {
					squaredNorm += difference[i] * difference[j]
							* chebyshevProductIntegral(i, j);
				}
			}
			squaredNorm = max(0.0, squaredNorm);
			double halfWidth = upper * 0.5 - lower * 0.5;
			return halfWidth * Math.sqrt(2.0 * squaredNorm);
		}

		private static double[] coefficients(double[] samples) {
			int degree = samples.length - 1;
			double[] result = new double[degree + 1];
			for (int order = 0; order <= degree; order++) {
				double sum = 0.5 * samples[0]
						+ 0.5 * samples[degree] * (order % 2 == 0 ? 1.0 : -1.0);
				for (int index = 1; index < degree; index++)
					sum += samples[index] * Math.cos(
							Math.PI * order * index / degree);
				result[order] = 2.0 * sum / degree;
			}
			result[0] *= 0.5;
			result[degree] *= 0.5;
			return result;
		}

		private static double chebyshevIntegral(int order) {
			return order % 2 == 0 ? 2.0 / (1.0 - order * (double) order)
					: 0.0;
		}

		private static double chebyshevProductIntegral(int first, int second) {
			return 0.5 * (chebyshevIntegral(abs(first - second))
					+ chebyshevIntegral(first + second));
		}
	}

	private static IntegrationResult doubleExponential(UnivariateFunction f,
			double lower, double upper, double epsabs, double epsrel,
			int maxLevels) {
		if (Double.isFinite(lower) && Double.isFinite(upper)) {
			return tanhSinh(f, lower, upper, epsabs, epsrel, maxLevels);
		}
		return infiniteDoubleExponential(f, lower, upper, epsabs, epsrel,
				maxLevels);
	}

	private static double[] internalBreakpoints(double[] declared, double lower,
			double upper) {
		double[] selected = new double[declared.length];
		int count = 0;
		for (double point : declared) {
			if (point > lower && point < upper
					&& (count == 0 || point != selected[count - 1])) {
				selected[count++] = point;
			}
		}
		double[] result = new double[count];
		System.arraycopy(selected, 0, result, 0, count);
		return result;
	}

	private static IntegrationResult tanhSinh(UnivariateFunction f, double lower,
			double upper, double epsabs, double epsrel, int maxLevels) {
		IntegrationResult result = new IntegrationResult();
		result.f = f;
		result.abserr = DBL_MAX;
		if (!(lower < upper)) {
			if (lower == upper) return result;
			result.ier = 6;
			return result;
		}
		double midpoint = lower * 0.5 + upper * 0.5;
		double halfWidth = upper * 0.5 - lower * 0.5;
		double previous = Double.NaN;
		final double halfPi = Math.PI * 0.5;
		for (int level = 0; level < maxLevels; level++) {
			double step = Math.scalb(1.0, -level);
			int extent = (int) Math.ceil(3.5 / step);
			double sum = 0.0;
			double correction = 0.0;
			boolean endpointRounded = false;
			for (int k = -extent; k <= extent; k++) {
				double t = k * step;
				double sinh = Math.sinh(t);
				double u = halfPi * sinh;
				double y = Math.tanh(u);
				if (!(y > -1.0 && y < 1.0)) continue;
				double coshU = Math.cosh(u);
				double jacobian = halfWidth * halfPi * Math.cosh(t)
						/ (coshU * coshU);
				double x = midpoint + halfWidth * y;
				/* The transform can round to an exact endpoint before tanh rounds to
				 * +/-1. Endpoint singularities are approached, never evaluated. */
				if (!(x > lower && x < upper)) {
					endpointRounded = true;
					continue;
				}
				double term = f.eval(x) * jacobian;
				if (!Double.isFinite(term)) {
					result.ier = 3;
					result.failureX = x;
					result.detail = "non-finite tanh-sinh contribution";
					return result;
				}
				double adjusted = term - correction;
				double next = sum + adjusted;
				correction = (next - sum) - adjusted;
				sum = next;
			}
			double current = sum * step;
			result.result = current;
			result.last = level + 1;
			if (level > 0) {
				result.abserr = abs(current - previous);
				if (endpointRounded) {
					/* A callback using doubles cannot resolve the final transformed
					 * sliver next to a singular endpoint. Do not claim accuracy below
					 * the conservative square-root-epsilon scale. */
					result.abserr = max(result.abserr,
							Math.sqrt(DBL_EPSILON) * abs(current));
				}
				double target = max(epsabs, epsrel * abs(current));
				if (result.abserr <= target) return result;
			}
			previous = current;
		}
		result.ier = 1;
		return result;
	}

	private static IntegrationResult infiniteDoubleExponential(
			UnivariateFunction f, double lower, double upper, double epsabs,
			double epsrel, int maxLevels) {
		IntegrationResult result = new IntegrationResult();
		result.f = f;
		result.abserr = DBL_MAX;
		if (!(lower < upper) || (Double.isFinite(lower) && Double.isFinite(upper))) {
			result.ier = 6;
			return result;
		}
		double previous = Double.NaN;
		final double halfPi = Math.PI * 0.5;
		for (int level = 0; level < maxLevels; level++) {
			double step = Math.scalb(1.0, -level);
			int extent = (int) Math.ceil(3.5 / step);
			double sum = 0.0;
			double correction = 0.0;
			boolean endpointRounded = false;
			for (int k = -extent; k <= extent; k++) {
				double t = k * step;
				double u = halfPi * Math.sinh(t);
				double x;
				double jacobian;
				if (Double.isFinite(lower)) {
					double radial = Math.exp(u);
					x = lower + radial;
					jacobian = halfPi * Math.cosh(t) * radial;
					if (!(x > lower)) {
						endpointRounded = true;
						continue;
					}
				} else if (Double.isFinite(upper)) {
					double radial = Math.exp(u);
					x = upper - radial;
					jacobian = halfPi * Math.cosh(t) * radial;
					if (!(x < upper)) {
						endpointRounded = true;
						continue;
					}
				} else {
					x = Math.sinh(u);
					jacobian = halfPi * Math.cosh(t) * Math.cosh(u);
				}
				double term = f.eval(x) * jacobian;
				if (!Double.isFinite(term)) {
					result.ier = 3;
					result.failureX = x;
					result.detail = "non-finite infinite double-exponential contribution";
					return result;
				}
				double adjusted = term - correction;
				double next = sum + adjusted;
				correction = (next - sum) - adjusted;
				sum = next;
			}
			double current = sum * step;
			result.result = current;
			result.last = level + 1;
			if (level > 0) {
				result.abserr = abs(current - previous);
				if (endpointRounded) {
					result.abserr = max(result.abserr,
							Math.sqrt(DBL_EPSILON) * abs(current));
				}
				double target = max(epsabs, epsrel * abs(current));
				if (result.abserr <= target) return result;
			}
			previous = current;
		}
		result.ier = 1;
		return result;
	}

	private static final class EvaluationGuard implements UnivariateFunction {
		private final UnivariateFunction delegate;
		private final IntegrationOptions options;
		private final long startedNanos = System.nanoTime();
		private final ExecutorService worker;
		private int evaluations;
		private int completedEvaluations;
		private long totalCallbackNanos;
		private long maximumCallbackNanos;
		private int failureCode;
		private double failureX = Double.NaN;
		private RuntimeException cause;
		private String detail;

		EvaluationGuard(UnivariateFunction delegate, IntegrationOptions options) {
			this.delegate = delegate;
			this.options = options;
			worker = options.getCallbackExecution()
					== IntegrationOptions.CallbackExecution.ISOLATED_DAEMON
					? Executors.newSingleThreadExecutor(new ThreadFactory() {
						@Override public Thread newThread(Runnable task) {
							Thread thread = new Thread(task,
									"jdistlib-isolated-integrand");
							thread.setDaemon(true);
							return thread;
						}
					}) : null;
		}

		@Override public double eval(double x) {
			if (failureCode != 0) return Double.NaN;
			long remaining = remainingTotalNanos();
			if (remaining <= 0L) {
				timeLimitFailure(x, "total integration time limit exceeded");
				return Double.NaN;
			}
			if (options.getCancellation() != null) {
				try {
					if (options.getCancellation().getAsBoolean()) {
						failureCode = 8;
						failureX = x;
						detail = "cancelled before evaluating x=" + x;
						return Double.NaN;
					}
				} catch (RuntimeException exception) {
					failureCode = 7;
					failureX = x;
					cause = exception;
					detail = "cancellation callback threw "
							+ exception.getClass().getSimpleName();
					return Double.NaN;
				}
			}
			if (evaluations >= options.getMaxEvaluations()) {
				failureCode = 9;
				failureX = x;
				detail = "limit=" + options.getMaxEvaluations();
				return Double.NaN;
			}
			evaluations++;
			long callbackStarted = System.nanoTime();
			try {
				double value = worker == null ? delegate.eval(x)
						: evaluateIsolated(x, remaining);
				long elapsed = elapsedSince(callbackStarted);
				recordCompleted(elapsed);
				if (elapsed > options.getMaxCallbackNanos()) {
					timeLimitFailure(x, "callback took " + elapsed
							+ " ns; limit=" + options.getMaxCallbackNanos() + " ns");
					return Double.NaN;
				}
				if (remainingTotalNanos() <= 0L) {
					timeLimitFailure(x, "total integration time limit exceeded");
					return Double.NaN;
				}
				if (!Double.isFinite(value)) {
					failureCode = 10;
					failureX = x;
					detail = "value=" + value + " at x=" + x;
					return Double.NaN;
				}
				return value;
			} catch (TimeoutException exception) {
				long elapsed = elapsedSince(callbackStarted);
				recordElapsed(elapsed);
				timeLimitFailure(x, "isolated callback did not return within "
						+ elapsed + " ns");
				return Double.NaN;
			} catch (InterruptedException exception) {
				Thread.currentThread().interrupt();
				recordElapsed(elapsedSince(callbackStarted));
				failureCode = 8;
				failureX = x;
				detail = "interrupted while waiting for isolated callback";
				return Double.NaN;
			} catch (ExecutionException exception) {
				recordCompleted(elapsedSince(callbackStarted));
				Throwable original = exception.getCause();
				if (original instanceof RuntimeException) {
					return callbackFailure(x, (RuntimeException) original);
				}
				if (original instanceof Error) throw (Error) original;
				return callbackFailure(x, new RuntimeException(original));
			} catch (RuntimeException exception) {
				recordCompleted(elapsedSince(callbackStarted));
				return callbackFailure(x, exception);
			}
		}

		private double evaluateIsolated(final double x, long remaining)
				throws InterruptedException, ExecutionException, TimeoutException {
			Future<Double> future = worker.submit(new Callable<Double>() {
				@Override public Double call() { return delegate.eval(x); }
			});
			long limit = Math.min(options.getMaxCallbackNanos(), remaining);
			try {
				return future.get(limit, TimeUnit.NANOSECONDS);
			} catch (TimeoutException exception) {
				future.cancel(true);
				throw exception;
			}
		}

		private double callbackFailure(double x, RuntimeException exception) {
			failureCode = 7;
			failureX = x;
			cause = exception;
			detail = exception.getClass().getSimpleName() + " at x=" + x
					+ (exception.getMessage() == null ? ""
							: ": " + exception.getMessage());
			return Double.NaN;
		}

		private long remainingTotalNanos() {
			long limit = options.getMaxTotalNanos();
			if (limit == Long.MAX_VALUE) return Long.MAX_VALUE;
			return limit - elapsedSince(startedNanos);
		}

		private void timeLimitFailure(double x, String message) {
			failureCode = 11;
			failureX = x;
			detail = message;
		}

		private void recordCompleted(long elapsed) {
			completedEvaluations++;
			recordElapsed(elapsed);
		}

		private void recordElapsed(long elapsed) {
			if (Long.MAX_VALUE - totalCallbackNanos < elapsed) {
				totalCallbackNanos = Long.MAX_VALUE;
			} else {
				totalCallbackNanos += elapsed;
			}
			maximumCallbackNanos = Math.max(maximumCallbackNanos, elapsed);
		}

		private static long elapsedSince(long start) {
			long elapsed = System.nanoTime() - start;
			return elapsed < 0L ? Long.MAX_VALUE : elapsed;
		}

		boolean hasFailure() { return failureCode != 0; }

		void decorate(IntegrationResult result) {
			if (failureCode != 0) result.ier = failureCode;
			result.failureX = failureX;
			result.cause = cause;
			if (detail != null) result.detail = detail;
			result.callbackProfile = new CallbackProfile(evaluations,
					completedEvaluations, totalCallbackNanos, maximumCallbackNanos,
					elapsedSince(startedNanos));
		}

		void close() {
			if (worker != null) worker.shutdownNow();
		}
	}

	private static IntegrationResult integrateFinite(UnivariateFunction f, double lower, double upper,
			double epsabs, double epsrel, int limit) {
		double[] alist = new double[limit + 1];
		double[] blist = new double[limit + 1];
		double[] rlist = new double[limit + 1];
		double[] elist = new double[limit + 1];
		int[] iord = new int[limit + 1];
		IntegrationResult result = new IntegrationResult();
		result.f = f;
		alist[1] = lower;
		blist[1] = upper;

		double[] error = {0.0};
		double[] resabsHolder = {0.0};
		double[] resascHolder = {0.0};
		result.result = dqk21(f, lower, upper, error, resabsHolder, resascHolder);
		result.abserr = error[0];
		double defabs = resabsHolder[0];
		double resabs = resascHolder[0];
		result.neval = 21;
		result.last = 1;
		rlist[1] = result.result;
		elist[1] = result.abserr;
		iord[1] = 1;
		if (!Double.isFinite(result.result) || !Double.isFinite(result.abserr)) {
			result.ier = 3;
			return result;
		}

		double dres = abs(result.result);
		double errbnd = max(epsabs, epsrel * dres);
		if (result.abserr <= 100.0 * DBL_EPSILON * defabs
				&& result.abserr > errbnd) result.ier = 2;
		if (limit == 1) result.ier = 1;
		if (result.ier != 0 || (result.abserr <= errbnd && result.abserr != resabs)
				|| result.abserr == 0.0) return result;

		double[] rlist2 = new double[53];
		double[] res3la = new double[4];
		rlist2[0] = result.result;
		double errmax = result.abserr;
		int maxerr = 1;
		double area = result.result;
		double errsum = result.abserr;
		result.abserr = DBL_MAX;
		int nrmax = 1;
		int nres = 0;
		int numrl2 = 2;
		int ktmin = 0;
		boolean extrap = false;
		boolean noext = false;
		int ierro = 0;
		int iroff1 = 0;
		int iroff2 = 0;
		int iroff3 = 0;
		int ksgn = dres >= (1.0 - 50.0 * DBL_EPSILON) * defabs ? 1 : -1;
		double small = 0.0;
		double erlarg = 0.0;
		double ertest = 0.0;
		double correc = 0.0;

		for (result.last = 2; result.last <= limit; result.last++) {
			double a1 = alist[maxerr];
			double intervalSum = alist[maxerr] + blist[maxerr];
			double b1 = Double.isFinite(intervalSum) ? intervalSum * 0.5
					: alist[maxerr] * 0.5 + blist[maxerr] * 0.5;
			double a2 = b1;
			double b2 = blist[maxerr];
			double erlast = errmax;

			error[0] = 0.0;
			resabsHolder[0] = 0.0;
			resascHolder[0] = 0.0;
			double area1 = dqk21(f, a1, b1, error, resabsHolder, resascHolder);
			double error1 = error[0];
			double defab1 = resascHolder[0];
			error[0] = 0.0;
			resabsHolder[0] = 0.0;
			resascHolder[0] = 0.0;
			double area2 = dqk21(f, a2, b2, error, resabsHolder, resascHolder);
			double error2 = error[0];
			double defab2 = resascHolder[0];

			if (!Double.isFinite(area1) || !Double.isFinite(area2)
					|| !Double.isFinite(error1) || !Double.isFinite(error2)) {
				result.ier = 3;
				result.neval = result.last * 42 - 21;
				return result;
			}

			double area12 = area1 + area2;
			double erro12 = error1 + error2;
			errsum += erro12 - errmax;
			area += area12 - rlist[maxerr];
			if (!(defab1 == error1 || defab2 == error2)) {
				if (abs(rlist[maxerr] - area12) <= 1e-5 * abs(area12)
						&& erro12 >= 0.99 * errmax) {
					if (extrap) iroff2++; else iroff1++;
				}
				if (result.last > 10 && erro12 > errmax) iroff3++;
			}
			rlist[maxerr] = area1;
			rlist[result.last] = area2;
			errbnd = max(epsabs, epsrel * abs(area));
			if (iroff1 + iroff2 >= 10 || iroff3 >= 20) result.ier = 2;
			if (iroff2 >= 5) ierro = 3;
			if (result.last == limit) result.ier = 1;
			if (max(abs(a1), abs(b2)) <= (100.0 * DBL_EPSILON + 1.0)
					* (abs(a2) + 1000.0 * DBL_MIN)) result.ier = 4;

			if (error2 > error1) {
				alist[maxerr] = a2;
				alist[result.last] = a1;
				blist[result.last] = b1;
				rlist[maxerr] = area2;
				rlist[result.last] = area1;
				elist[maxerr] = error2;
				elist[result.last] = error1;
			} else {
				alist[result.last] = a2;
				blist[maxerr] = b1;
				blist[result.last] = b2;
				elist[maxerr] = error1;
				elist[result.last] = error2;
			}

			double[] errmaxHolder = {errmax};
			int[] maxerrHolder = {maxerr};
			int[] nrmaxHolder = {nrmax};
			dqpsrt(limit, result.last, maxerrHolder, errmaxHolder, elist, iord,
					nrmaxHolder);
			errmax = errmaxHolder[0];
			maxerr = maxerrHolder[0];
			nrmax = nrmaxHolder[0];

			if (errsum <= errbnd) {
				result.result = sum(rlist, result.last);
				result.abserr = errsum;
				result.neval = result.last * 42 - 21;
				return result;
			}
			if (result.ier != 0) break;
			if (result.last == 2) {
				small = abs(upper - lower) * 0.375;
				erlarg = errsum;
				ertest = errbnd;
				rlist2[1] = area;
				continue;
			}
			if (noext) continue;

			erlarg -= erlast;
			if (abs(b1 - a1) > small) erlarg += erro12;
			if (!extrap) {
				if (abs(blist[maxerr] - alist[maxerr]) > small) continue;
				extrap = true;
				nrmax = 2;
			}

			boolean largerIntervalFound = false;
			if (ierro != 3 && erlarg > ertest) {
				int jupbnd = result.last;
				if (result.last > limit / 2 + 2) jupbnd = limit + 3 - result.last;
				for (int k = nrmax; k <= jupbnd; k++) {
					maxerr = iord[nrmax];
					errmax = elist[maxerr];
					if (abs(blist[maxerr] - alist[maxerr]) > small) {
						largerIntervalFound = true;
						break;
					}
					nrmax++;
				}
			}
			if (largerIntervalFound) continue;

			numrl2++;
			rlist2[numrl2 - 1] = area;
			int[] numrl2Holder = {numrl2};
			double[] absepsHolder = {0.0};
			int[] nresHolder = {nres};
			double reseps = dqelg(numrl2Holder, rlist2, absepsHolder, res3la,
					nresHolder);
			numrl2 = numrl2Holder[0];
			double abseps = absepsHolder[0];
			nres = nresHolder[0];
			ktmin++;
			if (ktmin > 5 && result.abserr < 0.001 * errsum) result.ier = 5;
			if (abseps < result.abserr) {
				ktmin = 0;
				result.abserr = abseps;
				result.result = reseps;
				correc = erlarg;
				ertest = max(epsabs, epsrel * abs(reseps));
				if (result.abserr <= ertest) break;
			}
			if (numrl2 == 1) noext = true;
			if (result.ier == 5) break;
			maxerr = iord[1];
			errmax = elist[maxerr];
			nrmax = 1;
			extrap = false;
			small *= 0.5;
			erlarg = errsum;
		}

		if (result.abserr == DBL_MAX) {
			result.result = sum(rlist, result.last);
			result.abserr = errsum;
		} else {
			if (result.ier + ierro != 0) {
				if (ierro == 3) result.abserr += correc;
				if (result.ier == 0) result.ier = 3;
				if (result.result == 0.0 || area == 0.0) {
					if (result.abserr > errsum) {
						result.result = sum(rlist, result.last);
						result.abserr = errsum;
					} else if (area == 0.0) {
						result.neval = result.last * 42 - 21;
						return result;
					}
				} else if (result.abserr / abs(result.result) > errsum / abs(area)) {
					result.result = sum(rlist, result.last);
					result.abserr = errsum;
				}
			}
			if (!(ksgn == -1 && max(abs(result.result), abs(area)) <= 0.01 * defabs)
					&& (result.result / area < 0.01 || result.result / area > 100.0
							|| errsum > abs(area))) result.ier = 5;
		}
		result.neval = result.last * 42 - 21;
		return result;
	}

	private static double sum(double[] values, int last) {
		double result = 0.0;
		for (int i = 1; i <= last; i++) result += values[i];
		return result;
	}
	//static double c_b6 = 0.;
	//static double c_b7 = 1.;

	/*
	***begin prologue  dqagi
	***date written   800101   (yymmdd)
	***revision date  830518   (yymmdd)
	***category no.  h2a3a1,h2a4a1
	***keywords  automatic integrator, infinite intervals,
	            general-purpose, transformation, extrapolation,
	            globally adaptive
	***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
	          de doncker,elise,appl. math. & progr. div. -k.u.leuven
	***purpose  the routine calculates an approximation result to a given
	           integral   i = integral of f over (bound,+infinity)
	           or i = integral of f over (-infinity,bound)
	           or i = integral of f over (-infinity,+infinity)
	           hopefully satisfying following claim for accuracy
	           abs(i-result) <= max(epsabs,epsrel*abs(i)).
	***description

	       integration over infinite intervals
	       standard fortran subroutine

	       parameters
	        on entry
	           f      - double precision
	                    function subprogram defining the integrand
	                    function f(x). the actual name for f needs to be
	                    declared e x t e r n a l in the driver program.

	           bound  - double precision
	                    finite bound of integration range
	                    (has no meaning if interval is doubly-infinite)

	           inf    - int
	                    indicating the kind of integration range involved
	                    inf = 1 corresponds to  (bound,+infinity),
	                    inf = -1            to  (-infinity,bound),
	                    inf = 2             to (-infinity,+infinity).

	           epsabs - double precision
	                    absolute accuracy requested
	           epsrel - double precision
	                    relative accuracy requested
	                    if  epsabs <= 0
	                    and epsrel < max(50*rel.mach.acc.,0.5d-28),
	                    the routine will end with ier = 6.


	        on return
	           result - double precision
	                    approximation to the integral

	           abserr - double precision
	                    estimate of the modulus of the absolute error,
	                    which should equal or exceed abs(i-result)

	           neval  - int
	                    number of integrand evaluations

	           ier    - int
	                    ier = 0 normal and reliable termination of the
	                            routine. it is assumed that the requested
	                            accuracy has been achieved.
	                  - ier > 0 abnormal termination of the routine. the
	                            estimates for result and error are less
	                            reliable. it is assumed that the requested
	                            accuracy has not been achieved.
	           error messages
	                    ier = 1 maximum number of subdivisions allowed
	                            has been achieved. one can allow more
	                            subdivisions by increasing the value of
	                            limit (and taking the according dimension
	                            adjustments into account). however, if
	                            this yields no improvement it is advised
	                            to analyze the integrand in order to
	                            determine the integration difficulties. if
	                            the position of a local difficulty can be
	                            determined (e.g. singularity,
	                            discontinuity within the interval) one
	                            will probably gain from splitting up the
	                            interval at this point and calling the
	                            integrator on the subranges. if possible,
	                            an appropriate special-purpose integrator
	                            should be used, which is designed for
	                            handling the type of difficulty involved.
	                        = 2 the occurrence of roundoff error is
	                            detected, which prevents the requested
	                            tolerance from being achieved.
	                            the error may be under-estimated.
	                        = 3 extremely bad integrand behaviour occurs
	                            at some points of the integration
	                            interval.
	                        = 4 the algorithm does not converge.
	                            roundoff error is detected in the
	                            extrapolation table.
	                            it is assumed that the requested tolerance
	                            cannot be achieved, and that the returned
	                            result is the best which can be obtained.
	                        = 5 the integral is probably divergent, or
	                            slowly convergent. it must be noted that
	                            divergence can occur with any other value
	                            of ier.
	                        = 6 the input is invalid, because
	                            (epsabs <= 0 and
	                             epsrel < max(50*rel.mach.acc.,0.5d-28))
	                             or limit < 1 or leniw < limit*4.
	                            result, abserr, neval, last are set to
	                            zero. exept when limit or leniw is
	                            invalid, iwork(1), work(limit*2+1) and
	                            work(limit*3+1) are set to zero, work(1)
	                            is set to a and work(limit+1) to b.

	        dimensioning parameters
	           limit - int
	                   dimensioning parameter for iwork
	                   limit determines the maximum number of subintervals
	                   in the partition of the given integration interval
	                   (a,b), limit >= 1.
	                   if limit < 1, the routine will end with ier = 6.

	           lenw  - int
	                   dimensioning parameter for work
	                   lenw must be at least limit*4.
	                   if lenw < limit*4, the routine will end
	                   with ier = 6.

	           last  - int
	                   on return, last equals the number of subintervals
	                   produced in the subdivision process, which
	                   determines the number of significant elements
	                   actually in the work arrays.

	        work arrays
	           iwork - int
	                   vector of dimension at least limit, the first
	                   k elements of which contain pointers
	                   to the error estimates over the subintervals,
	                   such that work(limit*3+iwork(1)),... ,
	                   work(limit*3+iwork(k)) form a decreasing
	                   sequence, with k = last if last <= (limit/2+2), and
	                   k = limit+1-last otherwise

	           work  - double precision
	                   vector of dimension at least lenw
	                   on return
	                   work(1), ..., work(last) contain the left
	                    end points of the subintervals in the
	                    partition of (a,b),
	                   work(limit+1), ..., work(limit+last) contain
	                    the right end points,
	                   work(limit*2+1), ...,work(limit*2+last) contain the
	                    integral approximations over the subintervals,
	                   work(limit*3+1), ..., work(limit*3)
	                    contain the error estimates.

	***routines called  dqagie
	***end prologue  dqagi
	*/
	/**begin prologue  dqagie
	***date written   800101   (yymmdd)
	***revision date  830518   (yymmdd)
	***category no.  h2a3a1,h2a4a1
	***keywords  automatic integrator, infinite intervals,
	            general-purpose, transformation, extrapolation,
	            globally adaptive
	***author  piessens,robert,appl. math & progr. div - k.u.leuven
	          de doncker,elise,appl. math & progr. div - k.u.leuven
	***purpose  the routine calculates an approximation result to a given
	           integral   i = integral of f over (bound,+infinity)
	           or i = integral of f over (-infinity,bound)
	           or i = integral of f over (-infinity,+infinity),
	           hopefully satisfying following claim for accuracy
	           abs(i-result) <= max(epsabs,epsrel*abs(i))
	***description

	integration over infinite intervals
	standard fortran subroutine

	           f      - double precision
	                    function subprogram defining the integrand
	                    function f(x). the actual name for f needs to be
	                    declared e x t e r n a l in the driver program.

	           bound  - double precision
	                    finite bound of integration range
	                    (has no meaning if interval is doubly-infinite)

	           inf    - double precision
	                    indicating the kind of integration range involved
	                    inf = 1 corresponds to  (bound,+infinity),
	                    inf = -1            to  (-infinity,bound),
	                    inf = 2             to (-infinity,+infinity).

	           epsabs - double precision
	                    absolute accuracy requested
	           epsrel - double precision
	                    relative accuracy requested
	                    if  epsabs <= 0
	                    and epsrel < max(50*rel.mach.acc.,0.5d-28),
	                    the routine will end with ier = 6.

	           limit  - int
	                    gives an upper bound on the number of subintervals
	                    in the partition of (a,b), limit >= 1

	        on return
	           result - double precision
	                    approximation to the integral

	           abserr - double precision
	                    estimate of the modulus of the absolute error,
	                    which should equal or exceed abs(i-result)

	           neval  - int
	                    number of integrand evaluations

	           ier    - int
	                    ier = 0 normal and reliable termination of the
	                            routine. it is assumed that the requested
	                            accuracy has been achieved.
	                  - ier > 0 abnormal termination of the routine. the
	                            estimates for result and error are less
	                            reliable. it is assumed that the requested
	                            accuracy has not been achieved.
	           error messages
	                    ier = 1 maximum number of subdivisions allowed
	                            has been achieved. one can allow more
	                            subdivisions by increasing the value of
	                            limit (and taking the according dimension
	                            adjustments into account). however,if
	                            this yields no improvement it is advised
	                            to analyze the integrand in order to
	                            determine the integration difficulties.
	                            if the position of a local difficulty can
	                            be determined (e.g. singularity,
	                            discontinuity within the interval) one
	                            will probably gain from splitting up the
	                            interval at this point and calling the
	                            integrator on the subranges. if possible,
	                            an appropriate special-purpose integrator
	                            should be used, which is designed for
	                            handling the type of difficulty involved.
	                        = 2 the occurrence of roundoff error is
	                            detected, which prevents the requested
	                            tolerance from being achieved.
	                            the error may be under-estimated.
	                        = 3 extremely bad integrand behaviour occurs
	                            at some points of the integration
	                            interval.
	                        = 4 the algorithm does not converge.
	                            roundoff error is detected in the
	                            extrapolation table.
	                            it is assumed that the requested tolerance
	                            cannot be achieved, and that the returned
	                            result is the best which can be obtained.
	                        = 5 the integral is probably divergent, or
	                            slowly convergent. it must be noted that
	                            divergence can occur with any other value
	                            of ier.
	                        = 6 the input is invalid, because
	                            (epsabs <= 0 and
	                             epsrel < max(50*rel.mach.acc.,0.5d-28),
	                            result, abserr, neval, last, rlist(1),
	                            elist(1) and iord(1) are set to zero.
	                            alist(1) and blist(1) are set to 0
	                            and 1 respectively.

	           alist  - double precision
	                    vector of dimension at least limit, the first
	                     last  elements of which are the left
	                    end points of the subintervals in the partition
	                    of the transformed integration range (0,1).

	           blist  - double precision
	                    vector of dimension at least limit, the first
	                     last  elements of which are the right
	                    end points of the subintervals in the partition
	                    of the transformed integration range (0,1).

	           rlist  - double precision
	                    vector of dimension at least limit, the first
	                     last  elements of which are the integral
	                    approximations on the subintervals

	           elist  - double precision
	                    vector of dimension at least limit,  the first
	                    last elements of which are the moduli of the
	                    absolute error estimates on the subintervals

	           iord   - int
	                    vector of dimension limit, the first k
	                    elements of which are pointers to the
	                    error estimates over the subintervals,
	                    such that elist(iord(1)), ..., elist(iord(k))
	                    form a decreasing sequence, with k = last
	                    if last <= (limit/2+2), and k = limit+1-last
	                    otherwise

	           last   - int
	                    number of subintervals actually produced
	                    in the subdivision process

	***routines called  dqelg,dqk15i,dqpsrt
	***end prologue  dqagie


	           the dimension of rlist2 is determined by the value of
	           limexp in subroutine dqelg.

	           list of major variables
	           -----------------------

	          alist     - list of left end points of all subintervals
	                      considered up to now
	          blist     - list of right end points of all subintervals
	                      considered up to now
	          rlist(i)  - approximation to the integral over
	                      (alist(i),blist(i))
	          rlist2    - array of dimension at least (limexp+2),
	                      containing the part of the epsilon table
	                      wich is still needed for further computations
	          elist(i)  - error estimate applying to rlist(i)
	          maxerr    - pointer to the interval with largest error
	                      estimate
	          errmax    - elist(maxerr)
	          erlast    - error on the interval currently subdivided
	                      (before that subdivision has taken place)
	          area      - sum of the integrals over the subintervals
	          errsum    - sum of the errors over the subintervals
	          errbnd    - requested accuracy max(epsabs,epsrel*
	                      abs(result))
	          *****1    - variable for the left subinterval
	          *****2    - variable for the right subinterval
	          last      - index for subdivision
	          nres      - number of calls to the extrapolation routine
	          numrl2    - number of elements currently in rlist2. if an
	                      appropriate approximation to the compounded
	                      integral has been obtained, it is put in
	                      rlist2(numrl2) after numrl2 has been increased
	                      by one.
	          small     - length of the smallest interval considered up
	                      to now, multiplied by 1.5
	          erlarg    - sum of the errors over the intervals larger
	                      than the smallest interval considered up to now
	          extrap    - logical variable denoting that the routine
	                      is attempting to perform extrapolation. i.e.
	                      before subdividing the smallest interval we
	                      try to decrease the value of erlarg.
	          noext     - logical variable denoting that extrapolation
	                      is no longer allowed (true-value)

	           machine dependent constants
	           ---------------------------

	          epmach is the largest relative spacing.
	          uflow is the smallest positive magnitude.
	          oflow is the largest positive magnitude.
	*/
	static IntegrationResult dqagie(UnivariateFunction f, double bound, int inf, double epsabs, double epsrel, int limit) {
		double[] alist = new double[limit+1];
		double[] blist = new double[limit+1];
		double[] rlist = new double[limit+1];
		double[] elist = new double[limit+1];
		int[] iord = new int[limit+1];
		IntegrationResult result = new IntegrationResult();
		result.f = f;

		/* Local variables */
		double area, dres;
		int ksgn;
		double boun;
		int nres;
		double area1, area2, area12;
		int k;
		double small = 0.0, erro12;
		int ierro;
		double a1, a2, b1, b2, defab1 = 0, defab2 = 0, oflow;
		int ktmin, nrmax;
		double uflow;
		boolean noext, extrap;
		int iroff1, iroff2, iroff3;
		double res3la[] = new double[4], error1 = 0, error2 = 0;
		int id;
		double rlist2[] = new double [53];
		int numrl2;
		double deabs = 0, epmach, erlarg = 0.0, abseps = 0, correc = 0.0, errbnd, resabs = 0;
		int jupbnd;
		double erlast, errmax;
		int maxerr;
		double reseps = 0;
		double ertest = 0.0, errsum;
		double[] temp1 = new double[1], temp2 = new double[1], temp3 = new double[1];
		int[] temp4 = new int[1], temp5 = new int[1];


		/* ***first executable statement  dqagie */

		/* Function Body */
		epmach = DBL_EPSILON;

		/*           test on validity of parameters */
		/*           ----------------------------- */

		result.ier = 0;
		result.neval = 0;
		result.last = 0;
		result.result = 0.;
		result.abserr = 0.;
		alist[1] = 0.;
		blist[1] = 1.;
		rlist[1] = 0.;
		elist[1] = 0.;
		iord[1] = 0;
		if (epsabs <= 0. && (epsrel < max(epmach * 50., 5e-29))) result.ier = 6;
		if (result.ier == 6) return result;

		/*           first approximation to the integral */
		/*           ----------------------------------- */

		/*         determine the interval to be mapped onto (0,1).
	           if inf = 2 the integral is computed as i = i1+i2, where
	           i1 = integral of f over (-infinity,0),
	           i2 = integral of f over (0,+infinity). */

		boun = bound;
		if (inf == 2) {
			boun = 0.;
		}
		temp1[0] = result.abserr; temp2[0] = deabs; temp3[0] = resabs;
		result.result = dqk15i(f, boun, inf, 0., 1., temp1, temp2, temp3);
		result.abserr = temp1[0]; deabs = temp2[0]; resabs = temp3[0];

		/*           test on accuracy */

		result.last = 1;
		rlist[1] = result.result;
		elist[1] = result.abserr;
		iord[1] = 1;
		dres = abs(result.result);
		errbnd = max(epsabs, epsrel * dres);
		if (result.abserr <= epmach * 100. * deabs && result.abserr > errbnd) result.ier = 2;
		if (limit == 1) result.ier = 1;
		if (result.ier != 0 || (result.abserr <= errbnd && result.abserr != resabs) || result.abserr == 0.) {
			//goto L130;
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}

		/*           initialization */
		/*           -------------- */

		uflow = DBL_MIN;
		oflow = DBL_MAX;
		rlist2[0] = result.result;
		errmax = result.abserr;
		maxerr = 1;
		area = result.result;
		errsum = result.abserr;
		result.abserr = oflow;
		nrmax = 1;
		nres = 0;
		ktmin = 0;
		numrl2 = 2;
		extrap = false;
		noext = false;
		ierro = 0;
		iroff1 = 0;
		iroff2 = 0;
		iroff3 = 0;
		ksgn = -1;
		if (dres >= (1. - epmach * 50.) * deabs) {
			ksgn = 1;
		}

		/*           main do-loop */
		/*           ------------ */

		for (result.last = 2; result.last <= limit; ++(result.last)) {

			/*           bisect the subinterval with nrmax-th largest error estimate. */

			a1 = alist[maxerr];
			b1 = (alist[maxerr] + blist[maxerr]) * .5;
			a2 = b1;
			b2 = blist[maxerr];
			erlast = errmax;

			temp1[0] = error1; temp2[0] = resabs; temp3[0] = defab1;
			area1 = dqk15i(f, boun, inf, a1, b1, temp1, temp2, temp3);
			error1 = temp1[0]; resabs = temp2[0]; defab1 = temp3[0];

			temp1[0] = error2; temp2[0] = resabs; temp3[0] = defab2;
			area2 = dqk15i(f, boun, inf, a2, b2, temp1, temp2, temp3);
			error2 = temp1[0]; resabs = temp2[0]; defab2 = temp3[0];

			/*           improve previous approximations to integral
		     and error and test for accuracy. */

			area12 = area1 + area2;
			erro12 = error1 + error2;
			errsum = errsum + erro12 - errmax;
			area = area + area12 - rlist[maxerr];
			if (!(defab1 == error1 || defab2 == error2)) {
				if (abs(rlist[maxerr] - area12) <= abs(area12) * 1e-5 &&
						erro12 >= errmax * .99) {
					if (extrap)
						++iroff2;
					else /* if (! extrap) */
						++iroff1;
				}
				if (result.last > 10 && erro12 > errmax)
					++iroff3;
			}

			rlist[maxerr] = area1;
			rlist[result.last] = area2;
			errbnd = max(epsabs, epsrel * abs(area));

			/*           test for roundoff error and eventually set error flag. */

			if (iroff1 + iroff2 >= 10 || iroff3 >= 20)
				result.ier = 2;
			if (iroff2 >= 5)
				ierro = 3;

			/*           set error flag in the case that the number of
		     subintervals equals limit. */

			if (result.last == limit)
				result.ier = 1;

			/*           set error flag in the case of bad integrand behaviour
		     at some points of the integration range. */

			if (max(abs(a1), abs(b2)) <= (epmach * 100. + 1.) * (abs(a2) + uflow * 1e3)) {
				result.ier = 4;
			}

			/*           append the newly-created intervals to the list. */

			if (error2 <= error1) {
				alist[result.last] = a2;
				blist[maxerr] = b1;
				blist[result.last] = b2;
				elist[maxerr] = error1;
				elist[result.last] = error2;
			}
			else {
				alist[maxerr] = a2;
				alist[result.last] = a1;
				blist[result.last] = b1;
				rlist[maxerr] = area2;
				rlist[result.last] = area1;
				elist[maxerr] = error2;
				elist[result.last] = error1;
			}

			/*           call subroutine dqpsrt to maintain the descending ordering
		     in the list of error estimates and select the subinterval
		     with nrmax-th largest error estimate (to be bisected next). */

			temp1[0] = errmax; temp4[0] = maxerr; temp5[0] = nrmax;
			dqpsrt(limit, result.last, temp4, temp1, elist, iord, temp5);
			errmax = temp1[0]; maxerr = temp4[0]; nrmax = temp5[0];
			if (errsum <= errbnd) {
				// goto L115;
				result.result = 0.;
				for (k = 1; k <= result.last; ++k)
					result.result += rlist[k];
				result.abserr = errsum;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
			if (result.ier != 0)	    break;
			if (result.last == 2) { /* L80: */
				small = .375;
				erlarg = errsum;
				ertest = errbnd;
				rlist2[1] = area; continue;
			}
			if (noext) 	    continue;

			erlarg -= erlast;
			if (abs(b1 - a1) > small) {
				erlarg += erro12;
			}
			if (!extrap) {

				/*           test whether the interval to be bisected next is the
		     smallest interval. */

				if (abs(blist[maxerr] - alist[maxerr]) > small) {
					continue;
				}
				extrap = true;
				nrmax = 2;
			}

			if (ierro != 3 && erlarg > ertest) {

				/*	    the smallest interval has the largest error.
		    before bisecting decrease the sum of the errors over the
		    larger intervals (erlarg) and perform extrapolation. */

				id = nrmax;
				jupbnd = result.last;
				if (result.last > limit / 2 + 2) {
					jupbnd = limit + 3 - result.last;
				}
				boolean cont = false;
				for (k = id; k <= jupbnd; ++k) {
					maxerr = iord[nrmax];
					errmax = elist[maxerr];
					if (abs(blist[maxerr] - alist[maxerr]) > small) {
						cont = true;
						break;
						//goto L90;
					}
					++nrmax;
					/* L50: */
				}
				if (cont) continue;
			}
			/*           perform extrapolation.  L60: */
			++numrl2;
			rlist2[numrl2 - 1] = area;
			temp4[0] = numrl2; temp5[0] = nres; temp1[0] = abseps;
			reseps = dqelg(temp4, rlist2, temp1, res3la, temp5);
			numrl2 = temp4[0]; nres = temp5[0]; abseps = temp1[0];

			++ktmin;
			if (ktmin > 5 && result.abserr < errsum * .001) result.ier = 5;
//			if (abseps >= abserr) {
//				goto L70;
//			}
			if (abseps < result.abserr) {
				ktmin = 0;
				result.abserr = abseps;
				result.result = reseps;
				correc = erlarg;
				ertest = max(epsabs, epsrel * abs(reseps));
				if (result.abserr <= ertest)
					break;
			}

			/*            prepare bisection of the smallest interval. */

			//L70:
			if (numrl2 == 1) noext = true;
			if (result.ier == 5) break;
			maxerr = iord[1];
			errmax = elist[maxerr];
			nrmax = 1;
			extrap = false;
			small *= .5;
			erlarg = errsum;
			//L90: ;
		} // end for (last = 2; last <= limit; ++(last))

		/* L100:     set final result and error estimate. */
		/*	     ------------------------------------ */

		if (result.abserr == oflow) {
			// goto L115;
			result.result = 0.;
			for (k = 1; k <= result.last; ++k)
				result.result += rlist[k];
			result.abserr = errsum;
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}
		if (result.ier + ierro == 0) {
			//goto L110;
			if (ksgn == -1 && max(abs(result.result), abs(area)) <= deabs * .01) {
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
			if (.01 > result.result / area || result.result / area > 100. || errsum > abs(area)) {
				result.ier = 6;
			}
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}
		if (ierro == 3) {
			result.abserr += correc;
		}
		if (result.ier == 0) {
			result.ier = 3;
		}
		if (result.result == 0. || area == 0.) {
			if (result.abserr > errsum) {
				//goto L115;
				result.result = 0.;
				for (k = 1; k <= result.last; ++k)
					result.result += rlist[k];
				result.abserr = errsum;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}

			if (area == 0.) {
				//	goto L130;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
		}
		else { /* L105: */
			if (result.abserr / abs(result.result) > errsum / abs(area)) {
				//goto L115;
				result.result = 0.;
				for (k = 1; k <= result.last; ++k)
					result.result += rlist[k];
				result.abserr = errsum;
				result.neval = result.last * 30 - 15;
				if (inf == 2) result.neval <<= 1;
				if (result.ier > 2) --result.ier;
				return result;
			}
		}

		/*           test on divergence */
		//L110:
		if (ksgn == -1 && max(abs(result.result), abs(area)) <= deabs * .01) {
			result.neval = result.last * 30 - 15;
			if (inf == 2) result.neval <<= 1;
			if (result.ier > 2) --result.ier;
			return result;
		}
		if (.01 > result.result / area || result.result / area > 100. || errsum > abs(area)) {
			result.ier = 6;
		}
		result.neval = result.last * 30 - 15;
		if (inf == 2) result.neval <<= 1;
		if (result.ier > 2) --result.ier;
		return result;
	} /* rdqagie_ */


	/*
	 ***begin prologue  dqk15i
	 ***date written   800101   (yymmdd)
	 ***revision date  830518   (yymmdd)
	 ***category no.  h2a3a2,h2a4a2
	 ***keywords  15-point transformed gauss-kronrod rules
	 ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
      de doncker,elise,appl. math. & progr. div. - k.u.leuven
	 ***purpose  the original (infinite integration range is mapped
       onto the interval (0,1) and (a,b) is a part of (0,1).
       it is the purpose to compute
       i = integral of transformed integrand over (a,b),
       j = integral of abs(transformed integrand) over (a,b).
	 ***description

      integration rule
      standard fortran subroutine
      double precision version

      parameters
       on entry
         f      - double precision
                  fuction subprogram defining the integrand
                  function f(x). the actual name for f needs to be
                  declared e x t e r n a l in the calling program.

         boun   - double precision
                  finite bound of original integration
                  range (set to zero if inf = +2)

         inf    - int
                  if inf = -1, the original interval is
                              (-infinity,bound),
                  if inf = +1, the original interval is
                              (bound,+infinity),
                  if inf = +2, the original interval is
                              (-infinity,+infinity) and
                  the integral is computed as the sum of two
                  integrals, one over (-infinity,0) and one over
                  (0,+infinity).

         a      - double precision
                  lower limit for integration over subrange
                  of (0,1)

         b      - double precision
                  upper limit for integration over subrange
                  of (0,1)

       on return
         result - double precision
                  approximation to the integral i
                  result is computed by applying the 15-point
                  kronrod rule(resk) obtained by optimal addition
                  of abscissae to the 7-point gauss rule(resg).

         abserr - double precision
                  estimate of the modulus of the absolute error,
                  which should equal or exceed abs(i-result)

         resabs - double precision
                  approximation to the integral j

         resasc - double precision
                  approximation to the integral of
                  abs((transformed integrand)-i/(b-a)) over (a,b)

	 ***references  (none)
	 ***end prologue  dqk15i


      the abscissae and weights are supplied for the interval
      (-1,1).  because of symmetry only the positive abscissae and
      their corresponding weights are given.

      xgk    - abscissae of the 15-point kronrod rule
               xgk(2), xgk(4), ... abscissae of the 7-point
               gauss rule
               xgk(1), xgk(3), ...  abscissae which are optimally
               added to the 7-point gauss rule

      wgk    - weights of the 15-point kronrod rule

      wg     - weights of the 7-point gauss rule, corresponding
               to the abscissae xgk(2), xgk(4), ...
               wg(1), wg(3), ... are set to zero.





      list of major variables
      -----------------------

      centr  - mid point of the interval
      hlgth  - half-length of the interval
      absc*  - abscissa
      tabsc* - transformed abscissa
      fval*  - function value
      resg   - result of the 7-point gauss formula
      resk   - result of the 15-point kronrod formula
      reskh  - approximation to the mean value of the transformed
               integrand over (a,b), i.e. to i/(b-a)

      machine dependent constants
      ---------------------------

      epmach is the largest relative spacing.
      uflow is the smallest positive magnitude.
	 */
	/** 21-point Gauss-Kronrod rule used by R's finite-interval dqags path. */
	private static double dqk21(UnivariateFunction f, double a, double b,
			double[] abserr, double[] resabs, double[] resasc) {
		final double[] wg = {
			.066671344308688137593568809893332,
			.149451349150580593145776339657697,
			.219086362515982043995534934228163,
			.269266719309996355091226921569469,
			.295524224714752870173892994651338
		};
		final double[] xgk = {
			.995657163025808080735527280689003,
			.973906528517171720077964012084452,
			.930157491355708226001207180059508,
			.865063366688984510732096688423493,
			.780817726586416897063717578345042,
			.679409568299024406234327365114874,
			.562757134668604683339000099272694,
			.433395394129247190799265943165784,
			.294392862701460198131126603103866,
			.14887433898163121088482600112972,
			0.0
		};
		final double[] wgk = {
			.011694638867371874278064396062192,
			.03255816230796472747881897245939,
			.05475589657435199603138130024458,
			.07503967481091995276704314091619,
			.093125454583697605535065465083366,
			.109387158802297641899210590325805,
			.123491976262065851077958109831074,
			.134709217311473325928054001771707,
			.142775938577060080797094273138717,
			.147739104901338491374841515972068,
			.149445554002916905664936468389821
		};

		double[] fv1 = new double[10];
		double[] fv2 = new double[10];
		double intervalSum = a + b;
		double intervalDifference = b - a;
		double center = Double.isFinite(intervalSum) ? intervalSum * 0.5
				: a * 0.5 + b * 0.5;
		double halfLength = Double.isFinite(intervalDifference)
				? intervalDifference * 0.5 : b * 0.5 - a * 0.5;
		double absHalfLength = abs(halfLength);
		double centerValue = f.eval(center);
		double resg = 0.0;
		double resk = wgk[10] * centerValue;
		resabs[0] = abs(resk);

		for (int j = 1; j <= 5; j++) {
			int even = j << 1;
			double abscissa = halfLength * xgk[even - 1];
			double fval1 = f.eval(center - abscissa);
			double fval2 = f.eval(center + abscissa);
			fv1[even - 1] = fval1;
			fv2[even - 1] = fval2;
			double fsum = fval1 + fval2;
			resg += wg[j - 1] * fsum;
			resk += wgk[even - 1] * fsum;
			resabs[0] += wgk[even - 1] * (abs(fval1) + abs(fval2));
		}
		for (int j = 1; j <= 5; j++) {
			int odd = (j << 1) - 1;
			double abscissa = halfLength * xgk[odd - 1];
			double fval1 = f.eval(center - abscissa);
			double fval2 = f.eval(center + abscissa);
			fv1[odd - 1] = fval1;
			fv2[odd - 1] = fval2;
			double fsum = fval1 + fval2;
			resk += wgk[odd - 1] * fsum;
			resabs[0] += wgk[odd - 1] * (abs(fval1) + abs(fval2));
		}

		double reskh = resk * 0.5;
		resasc[0] = wgk[10] * abs(centerValue - reskh);
		for (int j = 0; j < 10; j++) {
			resasc[0] += wgk[j] * (abs(fv1[j] - reskh) + abs(fv2[j] - reskh));
		}
		double result = resk * halfLength;
		resabs[0] *= absHalfLength;
		resasc[0] *= absHalfLength;
		abserr[0] = abs((resk - resg) * halfLength);
		if (resasc[0] != 0.0 && abserr[0] != 0.0) {
			abserr[0] = resasc[0] * min(1.0,
					pow(200.0 * abserr[0] / resasc[0], 1.5));
		}
		if (resabs[0] > DBL_MIN / (50.0 * DBL_EPSILON)) {
			abserr[0] = max(50.0 * DBL_EPSILON * resabs[0], abserr[0]);
		}
		return result;
	}

	static double dqk15i(UnivariateFunction f, double boun, int inf, double a, double b,
			double[] abserr, double[] resabs, double[] resasc)
	{
		/* Initialized data */

		final double wg[] = new double[] {
				0., .129484966168869693270611432679082,
				0., .27970539148927666790146777142378,
				0., .381830050505118944950369775488975,
				0., .417959183673469387755102040816327 };
		final double xgk[] = new double[] {
				.991455371120812639206854697526329,
				.949107912342758524526189684047851,
				.864864423359769072789712788640926,
				.741531185599394439863864773280788,
				.58608723546769113029414483825873,
				.405845151377397166906606412076961,
				.207784955007898467600689403773245, 0. };
		final double wgk[] = new double[] {
				.02293532201052922496373200805897,
				.063092092629978553290700663189204,
				.104790010322250183839876322541518,
				.140653259715525918745189590510238,
				.16900472663926790282658342659855,
				.190350578064785409913256402421014,
				.204432940075298892414161999234649,
				.209482141084727828012999174891714 };

		/* Local variables */
		double absc, dinf, resg, resk, fsum, absc1, absc2, fval1, fval2;
		int j;
		double hlgth, centr, reskh, uflow, result = 0;
		double tabsc1, tabsc2, fc, epmach;
		double[] fv1 = new double[7], fv2 = new double[7], vec = new double[15], vec2 = new double[15];


		/* ***first executable statement  dqk15i */
		epmach = DBL_EPSILON;
		uflow = DBL_MIN;
		dinf = (double) min(1, inf);

		centr = (a + b) * .5;
		hlgth = (b - a) * .5;
		tabsc1 = boun + dinf * (1. - centr) / centr;
		vec[0] = tabsc1;
		if (inf == 2) {
			vec2[0] = -tabsc1;
		}
		for (j = 1; j <= 7; ++j) {
			absc = hlgth * xgk[j - 1];
			absc1 = centr - absc;
			absc2 = centr + absc;
			tabsc1 = boun + dinf * (1. - absc1) / absc1;
			tabsc2 = boun + dinf * (1. - absc2) / absc2;
			vec[(j << 1) - 1] = tabsc1;
			vec[j * 2] = tabsc2;
			if (inf == 2) {
				vec2[(j << 1) - 1] = -tabsc1;
				vec2[j * 2] = -tabsc2;
			}
			/* L5: */
		}
		for (int i = 0; i < 15; i++)
			vec[i] = f.eval(vec[i]); /* -> new vec[] overwriting old vec[] */
		if (inf == 2) {
			for (int i = 0; i < 15; i++)
				vec2[i] = f.eval(vec2[i]); /* -> new vec[] overwriting old vec[] */
		}
		fval1 = vec[0];
		if (inf == 2) fval1 += vec2[0];
		fc = fval1 / centr / centr;

		/*           compute the 15-point kronrod approximation to
	     the integral, and estimate the error. */

		resg = wg[7] * fc;
		resk = wgk[7] * fc;
		resabs[0] = abs(resk);
		for (j = 1; j <= 7; ++j) {
			absc = hlgth * xgk[j - 1];
			absc1 = centr - absc;
			absc2 = centr + absc;
			tabsc1 = boun + dinf * (1. - absc1) / absc1;
			tabsc2 = boun + dinf * (1. - absc2) / absc2;
			fval1 = vec[(j << 1) - 1];
			fval2 = vec[j * 2];
			if (inf == 2) {
				fval1 += vec2[(j << 1) - 1];
			}
			if (inf == 2) {
				fval2 += vec2[j * 2];
			}
			fval1 = fval1 / absc1 / absc1;
			fval2 = fval2 / absc2 / absc2;
			fv1[j - 1] = fval1;
			fv2[j - 1] = fval2;
			fsum = fval1 + fval2;
			resg += wg[j - 1] * fsum;
			resk += wgk[j - 1] * fsum;
			resabs[0] += wgk[j - 1] * (abs(fval1) + abs(fval2));
			/* L10: */
		}
		reskh = resk * .5;
		resasc[0] = wgk[7] * abs(fc - reskh);
		for (j = 1; j <= 7; ++j) {
			resasc[0] += wgk[j - 1] * (abs(fv1[j - 1] - reskh) +
					abs(fv2[j - 1] - reskh));
			/* L20: */
		}
		result = resk * hlgth;
		resasc[0] *= hlgth;
		resabs[0] *= hlgth;
		abserr[0] = abs((resk - resg) * hlgth);
		if (resasc[0] != 0. && abserr[0] != 0.) {
			abserr[0] = resasc[0] * min(1., pow(abserr[0] * 200. / resasc[0], 1.5));
		}
		if (resabs[0] > uflow / (epmach * 50.)) {
			abserr[0] = max(epmach * 50. * resabs[0], abserr[0]);
		}
		return result;
	} /* dqk15i */

	/* ***begin prologue  dqpsrt
	 ***refer to  dqage,dqagie,dqagpe,dqawse
	 ***routines called  (none)
	 ***revision date  810101   (yymmdd)
	 ***keywords  sequential sorting
	 ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
       de doncker,elise,appl. math. & progr. div. - k.u.leuven
	 ***purpose  this routine maintains the descending ordering in the
        list of the local error estimated resulting from the
        interval subdivision process. at each call two error
        estimates are inserted using the sequential search
        method, top-down for the largest error estimate and
        bottom-up for the smallest error estimate.
	 ***description

       ordering routine
       standard fortran subroutine
       double precision version

       parameters (meaning at output)
          limit  - int
                   maximum number of error estimates the list
                   can contain

          last   - int
                   number of error estimates currently in the list

          maxerr - int
                   maxerr points to the nrmax-th largest error
                   estimate currently in the list

          ermax  - double precision
                   nrmax-th largest error estimate
                   ermax = elist(maxerr)

          elist  - double precision
                   vector of dimension last containing
                   the error estimates

          iord   - int
                   vector of dimension last, the first k elements
                   of which contain pointers to the error
                   estimates, such that
                   elist(iord(1)),...,  elist(iord(k))
                   form a decreasing sequence, with
                   k = last if last <= (limit/2+2), and
                   k = limit+1-last otherwise

          nrmax  - int
                   maxerr = iord(nrmax)

	 ***end prologue  dqpsrt
	 */
	static void dqpsrt(int limit, int last, int[] maxerr,
			double[] ermax, double[] elist, int[] iord, int[] nrmax)
	{
		/* Local variables */
		int i, j, k, ido, jbnd, isucc, jupbn;
		double errmin, errmax;

		/* Function Body */

		/*           check whether the list contains more than
	     two error estimates. */
		if (last <= 2) {
			iord[1] = 1;
			iord[2] = 2;
			//goto Last;
			maxerr[0] = iord[nrmax[0]];
			ermax[0] = elist[maxerr[0]];
			return;
		}
		/*           this part of the routine is only executed if, due to a
	     difficult integrand, subdivision increased the error
	     estimate. in the normal case the insert procedure should
	     start after the nrmax-th largest error estimate. */

		errmax = elist[maxerr[0]];
		if (nrmax[0] > 1) {
			ido = nrmax[0] - 1;
			for (i = 1; i <= ido; ++i) {
				isucc = iord[nrmax[0] - 1];
				if (errmax <= elist[isucc])
					break; /* out of for-loop */
				iord[nrmax[0]] = isucc;
				--(nrmax[0]);
				/* L20: */
			}
		}

		/*L30:       compute the number of elements in the list to be maintained
	     in descending order. this number depends on the number of
	     subdivisions still allowed. */
		if (last > limit / 2 + 2)
			jupbn = limit + 3 - last;
		else
			jupbn = last;

		errmin = elist[last];

		/*           insert errmax by traversing the list top-down,
	     starting comparison from the element elist(iord(nrmax+1)). */

		jbnd = jupbn - 1;
		for (i = nrmax[0] + 1; i <= jbnd; ++i) {
			isucc = iord[i];
			if (errmax >= elist[isucc]) {/* ***jump out of do-loop */
				/* L60: insert errmin by traversing the list bottom-up. */
				iord[i - 1] = maxerr[0];
				for (j = i, k = jbnd; j <= jbnd; j++, k--) {
					isucc = iord[k];
					if (errmin < elist[isucc]) {
						/* goto L80; ***jump out of do-loop */
						iord[k + 1] = last;
						// goto Last;
						maxerr[0] = iord[nrmax[0]];
						ermax[0] = elist[maxerr[0]];
						return;
					}
					iord[k + 1] = isucc;
				}
				iord[i] = last;
				//goto Last;
				maxerr[0] = iord[nrmax[0]];
				ermax[0] = elist[maxerr[0]];
				return;
			}
			iord[i - 1] = isucc;
		}

		iord[jbnd] = maxerr[0];
		iord[jupbn] = last;
		maxerr[0] = iord[nrmax[0]];
		ermax[0] = elist[maxerr[0]];
		return;
	} /* dqpsrt_ */

	/* ***begin prologue  dqelg
	***refer to  dqagie,dqagoe,dqagpe,dqagse
	***revision date  830518   (yymmdd)
	***keywords  epsilon algorithm, convergence acceleration,
	            extrapolation
	***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
	          de doncker,elise,appl. math & progr. div. - k.u.leuven
	***purpose  the routine determines the limit of a given sequence of
	           approximations, by means of the epsilon algorithm of
	           p.wynn. an estimate of the absolute error is also given.
	           the condensed epsilon table is computed. only those
	           elements needed for the computation of the next diagonal
	           are preserved.
	***description

	          epsilon algorithm
	          standard fortran subroutine
	          double precision version

	          parameters
	             n      - int
	                      epstab(n) contains the new element in the
	                      first column of the epsilon table.

	             epstab - double precision
	                      vector of dimension 52 containing the elements
	                      of the two lower diagonals of the triangular
	                      epsilon table. the elements are numbered
	                      starting at the right-hand corner of the
	                      triangle.

	             result - double precision
	                      resulting approximation to the integral

	             abserr - double precision
	                      estimate of the absolute error computed from
	                      result and the 3 previous results

	             res3la - double precision
	                      vector of dimension 3 containing the last 3
	                      results

	             nres   - int
	                      number of calls to the routine
	                      (should be zero at first call)

	***end prologue  dqelg


	          list of major variables
	          -----------------------

	          e0     - the 4 elements on which the computation of a new
	          e1       element in the epsilon table is based
	          e2
	          e3                 e0
	                       e3    e1    new
	                             e2

	          newelm - number of elements to be computed in the new diagonal
	          errA   - errA = abs(e1-e0)+abs(e2-e1)+abs(new-e2)
	          result - the element in the new diagonal with least value of errA

	          machine dependent constants
	          ---------------------------

	          epmach is the largest relative spacing.
	          oflow is the largest positive magnitude.
	          limexp is the maximum number of elements the epsilon
	          table can contain. if this number is reached, the upper
	          diagonal of the epsilon table is deleted. */
	static final double dqelg(int[] n, double[] epstab, double[] abserr, double[] res3la, int[] nres)
	{
		final double epmach = DBL_EPSILON;
		final double oflow = DBL_MAX;
		nres[0]++;
		abserr[0] = oflow;
		double result = epstab[n[0] - 1];
		if (n[0] < 3) {
			abserr[0] = max(abserr[0], 5.0 * epmach * abs(result));
			return result;
		}

		final int limexp = 50;
		epstab[n[0] + 1] = epstab[n[0] - 1];
		int newelm = (n[0] - 1) / 2;
		epstab[n[0] - 1] = oflow;
		int num = n[0];
		int k1 = n[0];
		for (int i = 1; i <= newelm; i++) {
			int k2 = k1 - 1;
			int k3 = k1 - 2;
			double res = epstab[k1 + 1];
			double e0 = epstab[k3 - 1];
			double e1 = epstab[k2 - 1];
			double e2 = res;
			double e1abs = abs(e1);
			double delta2 = e2 - e1;
			double err2 = abs(delta2);
			double tol2 = max(abs(e2), e1abs) * epmach;
			double delta3 = e1 - e0;
			double err3 = abs(delta3);
			double tol3 = max(e1abs, abs(e0)) * epmach;
			if (err2 <= tol2 && err3 <= tol3) {
				result = res;
				abserr[0] = max(err2 + err3, 5.0 * epmach * abs(result));
				return result;
			}

			double e3 = epstab[k1 - 1];
			epstab[k1 - 1] = e1;
			double delta1 = e1 - e3;
			double err1 = abs(delta1);
			double tol1 = max(e1abs, abs(e3)) * epmach;
			if (err1 > tol1 && err2 > tol2 && err3 > tol3) {
				double ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
				if (abs(ss * e1) > 1e-4) {
					res = e1 + 1.0 / ss;
					epstab[k1 - 1] = res;
					k1 -= 2;
					double error = err2 + abs(res - e2) + err3;
					if (error <= abserr[0]) {
						abserr[0] = error;
						result = res;
					}
					continue;
				}
			}
			n[0] = i + i - 1;
			break;
		}

		if (n[0] == limexp) n[0] = (limexp / 2 << 1) - 1;
		int ib = (num / 2 << 1) == num ? 2 : 1;
		int ie = newelm + 1;
		for (int i = 1; i <= ie; i++) {
			int ib2 = ib + 2;
			epstab[ib - 1] = epstab[ib2 - 1];
			ib = ib2;
		}
		if (num != n[0]) {
			int index = num - n[0] + 1;
			for (int i = 1; i <= n[0]; i++, index++) {
				epstab[i - 1] = epstab[index - 1];
			}
		}

		if (nres[0] >= 4) {
			abserr[0] = abs(result - res3la[2]) + abs(result - res3la[1])
					+ abs(result - res3la[0]);
			res3la[0] = res3la[1];
			res3la[1] = res3la[2];
			res3la[2] = result;
		} else {
			res3la[nres[0] - 1] = result;
			abserr[0] = oflow;
		}
		abserr[0] = max(abserr[0], 5.0 * epmach * abs(result));
		return result;
	} /* dqelg */

}
