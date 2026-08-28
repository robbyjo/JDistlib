/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;

/** Small deterministic L-BFGS maximizer intended for initialization and MAP fits. */
public final class LbfgsOptimizer {
	private LbfgsOptimizer() {}
	public static OptimizationResult maximize(DifferentiableLogDensity target,
			double[] initial, int maximumIterations, int historySize, double tolerance) {
		return trace(target, initial, maximumIterations, historySize, tolerance).result();
	}
	public static OptimizationTrace trace(DifferentiableLogDensity target,
			double[] initial, int maximumIterations, int historySize, double tolerance) {
		if (target == null || initial == null || maximumIterations < 1
				|| historySize < 1 || !(tolerance > 0.0))
			throw new IllegalArgumentException("invalid L-BFGS arguments");
		double[] x = initial.clone(), gradient = new double[x.length];
		double value = target.logDensityAndGradient(x, gradient);
		int evaluations = 1;
		List<double[]> sHistory = new ArrayList<double[]>();
		List<double[]> yHistory = new ArrayList<double[]>();
		List<Double> rhoHistory = new ArrayList<Double>();
		List<double[]> points = new ArrayList<double[]>(); List<Double> objectives = new ArrayList<Double>();
		List<double[][]> inverseHessians = new ArrayList<double[][]>();
		points.add(x.clone()); objectives.add(value);
		inverseHessians.add(identity(x.length));
		for (int iteration = 0; iteration < maximumIterations; iteration++) {
			if (normInfinity(gradient) <= tolerance)
				return traceResult(points, objectives, inverseHessians, x, value, iteration, evaluations, true);
			double[] direction = inverseHessianProduct(gradient, sHistory,
					yHistory, rhoHistory);
			double directional = dot(gradient, direction);
			if (!(directional > 0.0)) { direction = gradient.clone(); directional = dot(gradient, gradient); }
			double scale = 1.0;
			double[] next = new double[x.length], nextGradient = new double[x.length];
			double nextValue;
			do {
				for (int i = 0; i < x.length; i++) next[i] = x[i] + scale * direction[i];
				nextValue = target.logDensityAndGradient(next, nextGradient); evaluations++;
				if (Double.isFinite(nextValue) && nextValue >= value + 1e-4 * scale * directional) break;
				scale *= 0.5;
			} while (scale > 1e-12);
			if (scale <= 1e-12)
				return traceResult(points, objectives, inverseHessians, x, value, iteration, evaluations, false);
			double[] s = subtract(next, x), y = subtract(gradient, nextGradient);
			double curvature = dot(s, y);
			if (curvature > 1e-12) {
				if (sHistory.size() == historySize) {
					sHistory.remove(0); yHistory.remove(0); rhoHistory.remove(0);
				}
				sHistory.add(s); yHistory.add(y); rhoHistory.add(1.0 / curvature);
			}
			x = next.clone(); gradient = nextGradient.clone(); value = nextValue;
			points.add(x.clone()); objectives.add(value);
			inverseHessians.add(denseInverse(x.length, sHistory, yHistory, rhoHistory));
		}
		return traceResult(points, objectives, inverseHessians, x, value, maximumIterations, evaluations,
				normInfinity(gradient) <= tolerance);
	}
	private static OptimizationTrace traceResult(List<double[]> points, List<Double> objectives, List<double[][]> inverseHessians,
			double[] point, double value, int iterations, int evaluations, boolean converged) {
		double[] values = new double[objectives.size()]; for (int i = 0; i < values.length; i++) values[i] = objectives.get(i);
		return new OptimizationTrace(points.toArray(new double[points.size()][]), values,
				inverseHessians.toArray(new double[inverseHessians.size()][][]),
				new OptimizationResult(point, value, iterations, evaluations, converged));
	}
	private static double[][] denseInverse(int dimension, List<double[]> s, List<double[]> y, List<Double> rho) {
		double[][] result = new double[dimension][dimension]; for (int column = 0; column < dimension; column++) {
			double[] basis = new double[dimension]; basis[column] = 1.0; double[] value = inverseHessianProduct(basis, s, y, rho);
			for (int row = 0; row < dimension; row++) result[row][column] = value[row]; }
		for (int row = 0; row < dimension; row++) for (int column = row + 1; column < dimension; column++) { double value = 0.5 * (result[row][column] + result[column][row]); result[row][column] = value; result[column][row] = value; }
		return result;
	}
	private static double[][] identity(int dimension) { double[][] result = new double[dimension][dimension]; for (int i = 0; i < dimension; i++) result[i][i] = 1.0; return result; }
	private static double[] inverseHessianProduct(double[] gradient,
			List<double[]> s, List<double[]> y, List<Double> rho) {
		double[] result = gradient.clone();
		double[] alpha = new double[s.size()];
		for (int i = s.size() - 1; i >= 0; i--) {
			alpha[i] = rho.get(i) * dot(s.get(i), result);
			addScaled(result, y.get(i), -alpha[i]);
		}
		if (!s.isEmpty()) {
			int last = s.size() - 1;
			double gamma = dot(s.get(last), y.get(last)) / dot(y.get(last), y.get(last));
			for (int i = 0; i < result.length; i++) result[i] *= gamma;
		}
		for (int i = 0; i < s.size(); i++) {
			double beta = rho.get(i) * dot(y.get(i), result);
			addScaled(result, s.get(i), alpha[i] - beta);
		}
		return result;
	}
	private static void addScaled(double[] target, double[] value, double scale) {
		for (int i = 0; i < target.length; i++) target[i] += scale * value[i];
	}
	private static double[] subtract(double[] first, double[] second) {
		double[] result = new double[first.length];
		for (int i = 0; i < result.length; i++) result[i] = first[i] - second[i];
		return result;
	}
	private static double dot(double[] first, double[] second) {
		double result = 0.0;
		for (int i = 0; i < first.length; i++) result += first[i] * second[i];
		return result;
	}
	private static double normInfinity(double[] value) {
		double result = 0.0;
		for (double element : value) result = Math.max(result, Math.abs(element));
		return result;
	}
}
