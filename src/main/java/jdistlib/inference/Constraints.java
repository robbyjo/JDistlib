/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Standard parameter constraints and their Jacobian-aware transforms. */
public final class Constraints {
	private Constraints() {}

	public static ParameterConstraint real() { return new Identity(1); }
	public static ParameterConstraint realVector(int dimension) {
		return new Identity(dimension);
	}
	public static ParameterConstraint positive() { return new Positive(1); }
	public static ParameterConstraint positiveVector(int dimension) {
		return new Positive(dimension);
	}
	public static ParameterConstraint bounded(double lower, double upper) {
		return new Bounded(lower, upper);
	}
	public static ParameterConstraint ordered(int dimension) {
		return new Ordered(dimension);
	}
	public static ParameterConstraint simplex(int dimension) {
		return new Simplex(dimension);
	}

	private abstract static class Base implements ParameterConstraint {
		final int unconstrained;
		final int constrained;
		Base(int unconstrained, int constrained) {
			if (unconstrained < 1 || constrained < 1)
				throw new IllegalArgumentException("constraint dimensions must be positive");
			this.unconstrained = unconstrained;
			this.constrained = constrained;
		}
		@Override public int unconstrainedDimension() { return unconstrained; }
		@Override public int constrainedDimension() { return constrained; }
		void check(double[] source, int offset, int length) {
			if (source == null || offset < 0 || offset > source.length - length)
				throw new IllegalArgumentException("invalid transform slice");
		}
	}

	private static final class Identity extends Base {
		Identity(int dimension) { super(dimension, dimension); }
		@Override public String description() { return constrained == 1 ? "real" : "real[" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			System.arraycopy(source, sourceOffset, target, targetOffset, constrained);
			return 0.0;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			System.arraycopy(source, sourceOffset, target, targetOffset, constrained);
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			for (int i = 0; i < unconstrained; i++) ug[uo + i] += cg[co + i];
		}
	}

	private static final class Positive extends Base {
		Positive(int dimension) { super(dimension, dimension); }
		@Override public String description() { return constrained == 1 ? "positive" : "positive[" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double jacobian = 0.0;
			for (int i = 0; i < constrained; i++) {
				double value = source[sourceOffset + i];
				target[targetOffset + i] = Math.exp(value);
				jacobian += value;
			}
			return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < constrained; i++) {
				if (!(source[sourceOffset + i] > 0.0))
					throw new IllegalArgumentException("positive value required");
				target[targetOffset + i] = Math.log(source[sourceOffset + i]);
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			for (int i = 0; i < unconstrained; i++)
				ug[uo + i] += cg[co + i] * c[co + i] + 1.0;
		}
	}

	private static final class Bounded extends Base {
		private final double lower;
		private final double upper;
		private final double width;
		Bounded(double lower, double upper) {
			super(1, 1);
			if (!Double.isFinite(lower) || !Double.isFinite(upper) || !(lower < upper))
				throw new IllegalArgumentException("finite ordered bounds are required");
			this.lower = lower; this.upper = upper; width = upper - lower;
		}
		@Override public String description() { return "bounded[" + lower + "," + upper + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, 1); check(target, targetOffset, 1);
			double z = source[sourceOffset];
			double p = logistic(z);
			target[targetOffset] = lower + width * p;
			return Math.log(width) + logLogistic(z) + logLogistic(-z);
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, 1); check(target, targetOffset, 1);
			double p = (source[sourceOffset] - lower) / width;
			if (!(p > 0.0 && p < 1.0))
				throw new IllegalArgumentException("value must be strictly inside bounds");
			target[targetOffset] = Math.log(p) - Math.log1p(-p);
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			double p = logistic(u[uo]);
			ug[uo] += cg[co] * width * p * (1.0 - p) + 1.0 - 2.0 * p;
		}
	}

	private static final class Ordered extends Base {
		Ordered(int dimension) { super(dimension, dimension); }
		@Override public String description() { return "ordered[" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			target[targetOffset] = source[sourceOffset];
			double jacobian = 0.0;
			for (int i = 1; i < constrained; i++) {
				jacobian += source[sourceOffset + i];
				target[targetOffset + i] = target[targetOffset + i - 1]
						+ Math.exp(source[sourceOffset + i]);
			}
			return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			target[targetOffset] = source[sourceOffset];
			for (int i = 1; i < constrained; i++) {
				double difference = source[sourceOffset + i] - source[sourceOffset + i - 1];
				if (!(difference > 0.0)) throw new IllegalArgumentException("values must be strictly ordered");
				target[targetOffset + i] = Math.log(difference);
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			double suffix = 0.0;
			for (int i = constrained - 1; i >= 0; i--) {
				suffix += cg[co + i];
				ug[uo + i] += i == 0 ? suffix : suffix * Math.exp(u[uo + i]) + 1.0;
			}
		}
	}

	private static final class Simplex extends Base {
		Simplex(int dimension) { super(dimension - 1, dimension); }
		@Override public String description() { return "simplex[" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double remaining = 1.0;
			double logJacobian = 0.0;
			for (int i = 0; i < unconstrained; i++) {
				double shift = Math.log(constrained - i - 1.0);
				double stick = logistic(source[sourceOffset + i] - shift);
				target[targetOffset + i] = remaining * stick;
				logJacobian += Math.log(remaining) + Math.log(stick) + Math.log1p(-stick);
				remaining *= 1.0 - stick;
			}
			target[targetOffset + constrained - 1] = remaining;
			return logJacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			double remaining = 1.0;
			double sum = 0.0;
			for (int i = 0; i < constrained; i++) sum += source[sourceOffset + i];
			if (Math.abs(sum - 1.0) > 1e-10) throw new IllegalArgumentException("simplex values must sum to one");
			for (int i = 0; i < unconstrained; i++) {
				double stick = source[sourceOffset + i] / remaining;
				if (!(stick > 0.0 && stick < 1.0)) throw new IllegalArgumentException("simplex values must be interior");
				target[targetOffset + i] = Math.log(stick) - Math.log1p(-stick)
						+ Math.log(constrained - i - 1.0);
				remaining -= source[sourceOffset + i];
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			double tailWeightedGradient = cg[co + constrained - 1]
					* c[co + constrained - 1];
			for (int i = unconstrained - 1; i >= 0; i--) {
				double shift = Math.log(constrained - i - 1.0);
				double stick = logistic(u[uo + i] - shift);
				double factorDerivative = cg[co + i] * c[co + i]
						* (1.0 - stick) - stick * tailWeightedGradient;
				double jacobianDerivative = 1.0 - (constrained - i) * stick;
				ug[uo + i] += factorDerivative + jacobianDerivative;
				tailWeightedGradient += cg[co + i] * c[co + i];
			}
		}
	}

	private static double logistic(double x) {
		if (x >= 0.0) return 1.0 / (1.0 + Math.exp(-x));
		double exponential = Math.exp(x);
		return exponential / (1.0 + exponential);
	}
	private static double logLogistic(double x) {
		return x >= 0.0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x));
	}
}
