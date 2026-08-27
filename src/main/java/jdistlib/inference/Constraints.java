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
	public static ParameterConstraint lowerBound(double lower, int dimension) {
		return lower == 0.0 ? new Positive(dimension) : new LowerBound(lower, dimension);
	}
	public static ParameterConstraint upperBound(double upper, int dimension) {
		return new UpperBound(upper, dimension);
	}
	public static ParameterConstraint bounded(double lower, double upper) {
		return new Bounded(lower, upper, 1);
	}
	public static ParameterConstraint boundedVector(double lower, double upper, int dimension) {
		return new Bounded(lower, upper, dimension);
	}
	public static ParameterConstraint offsetMultiplier(double offset, double multiplier,
			int dimension) {
		return new OffsetMultiplier(offset, multiplier, dimension);
	}
	public static ParameterConstraint ordered(int dimension) {
		return new Ordered(dimension);
	}
	public static ParameterConstraint positiveOrdered(int dimension) {
		return new PositiveOrdered(dimension);
	}
	public static ParameterConstraint sumToZero(int dimension) {
		return new SumToZero(dimension);
	}
	public static ParameterConstraint simplex(int dimension) {
		return new Simplex(dimension);
	}
	/** Stan-compatible unit-vector normalization transform. */
	public static ParameterConstraint unitVector(int dimension) {
		return new UnitVector(dimension);
	}
	/** Stan-compatible covariance-matrix transform, stored row-major. */
	public static ParameterConstraint covarianceMatrix(int dimension) {
		return new CovarianceMatrix(dimension);
	}
	/** Stan-compatible correlation-matrix (LKJ/CPC) transform, stored row-major. */
	public static ParameterConstraint correlationMatrix(int dimension) {
		return new CorrelationMatrix(dimension);
	}
	/** Stan-compatible lower Cholesky factor of an {@code rows x columns} covariance matrix. */
	public static ParameterConstraint choleskyFactorCovariance(int rows, int columns) {
		return new CholeskyFactorCovariance(rows, columns);
	}
	/** Stan-compatible Cholesky factor of a correlation matrix, stored row-major. */
	public static ParameterConstraint choleskyFactorCorrelation(int dimension) {
		return new CholeskyFactorCorrelation(dimension);
	}
	/** Repeats an independent constraint transform, as required for arrays of constrained values. */
	public static ParameterConstraint repeated(ParameterConstraint element, int repetitions) {
		return new Repeated(element, repetitions);
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
	private static final class Repeated extends Base {
		private final ParameterConstraint element; private final int repetitions;
		Repeated(ParameterConstraint element, int repetitions) {
			super(repeatedDimension(element, repetitions, true), repeatedDimension(element, repetitions, false));
			this.element = element; this.repetitions = repetitions;
		}
		@Override public String description() { return "array[" + repetitions + "] " + element.description(); }
		@Override public double constrain(double[] source, int sourceOffset, double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double result = 0.0;
			for (int i = 0; i < repetitions; i++) result += element.constrain(source,
					sourceOffset + i * element.unconstrainedDimension(), target,
					targetOffset + i * element.constrainedDimension());
			return result;
		}
		@Override public void unconstrain(double[] source, int sourceOffset, double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < repetitions; i++) element.unconstrain(source,
					sourceOffset + i * element.constrainedDimension(), target,
					targetOffset + i * element.unconstrainedDimension());
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co, double[] cg, double[] ug) {
			for (int i = 0; i < repetitions; i++) element.pullback(u,
					uo + i * element.unconstrainedDimension(), c,
					co + i * element.constrainedDimension(), cg, ug);
		}
		private static int repeatedDimension(ParameterConstraint element, int repetitions, boolean unconstrained) {
			if (element == null || repetitions < 1) throw new IllegalArgumentException("constraint and positive repetition count required");
			long result = (long) repetitions * (unconstrained
					? element.unconstrainedDimension() : element.constrainedDimension());
			if (result > Integer.MAX_VALUE) throw new IllegalArgumentException("repeated constraint is too large");
			return (int) result;
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

	private static final class LowerBound extends Base {
		private final double lower;
		LowerBound(double lower, int dimension) {
			super(dimension, dimension);
			if (!Double.isFinite(lower)) throw new IllegalArgumentException("finite lower bound required");
			this.lower = lower;
		}
		@Override public String description() { return "lower[" + lower + "][" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double jacobian = 0.0;
			for (int i = 0; i < constrained; i++) {
				double value = source[sourceOffset + i];
				target[targetOffset + i] = lower + Math.exp(value); jacobian += value;
			}
			return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < constrained; i++) {
				double shifted = source[sourceOffset + i] - lower;
				if (!(shifted > 0.0)) throw new IllegalArgumentException("value must exceed lower bound");
				target[targetOffset + i] = Math.log(shifted);
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			for (int i = 0; i < unconstrained; i++)
				ug[uo + i] += cg[co + i] * (c[co + i] - lower) + 1.0;
		}
	}

	private static final class UpperBound extends Base {
		private final double upper;
		UpperBound(double upper, int dimension) {
			super(dimension, dimension);
			if (!Double.isFinite(upper)) throw new IllegalArgumentException("finite upper bound required");
			this.upper = upper;
		}
		@Override public String description() { return "upper[" + upper + "][" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double jacobian = 0.0;
			for (int i = 0; i < constrained; i++) {
				double value = source[sourceOffset + i];
				target[targetOffset + i] = upper - Math.exp(value); jacobian += value;
			}
			return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < constrained; i++) {
				double shifted = upper - source[sourceOffset + i];
				if (!(shifted > 0.0)) throw new IllegalArgumentException("value must be below upper bound");
				target[targetOffset + i] = Math.log(shifted);
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			for (int i = 0; i < unconstrained; i++)
				ug[uo + i] += cg[co + i] * (c[co + i] - upper) + 1.0;
		}
	}

	private static final class Bounded extends Base {
		private final double lower;
		private final double upper;
		private final double width;
		Bounded(double lower, double upper, int dimension) {
			super(dimension, dimension);
			if (!Double.isFinite(lower) || !Double.isFinite(upper) || !(lower < upper))
				throw new IllegalArgumentException("finite ordered bounds are required");
			this.lower = lower; this.upper = upper; width = upper - lower;
		}
		@Override public String description() { return constrained == 1
				? "bounded[" + lower + "," + upper + "]"
				: "bounded[" + lower + "," + upper + "][" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double jacobian = 0.0;
			for (int i = 0; i < constrained; i++) {
				double z = source[sourceOffset + i]; double p = logistic(z);
				target[targetOffset + i] = lower + width * p;
				jacobian += Math.log(width) + logLogistic(z) + logLogistic(-z);
			}
			return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < constrained; i++) {
				double p = (source[sourceOffset + i] - lower) / width;
				if (!(p > 0.0 && p < 1.0))
					throw new IllegalArgumentException("value must be strictly inside bounds");
				target[targetOffset + i] = Math.log(p) - Math.log1p(-p);
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			for (int i = 0; i < unconstrained; i++) {
				double p = logistic(u[uo + i]);
				ug[uo + i] += cg[co + i] * width * p * (1.0 - p) + 1.0 - 2.0 * p;
			}
		}
	}

	private static final class OffsetMultiplier extends Base {
		private final double offset;
		private final double multiplier;
		OffsetMultiplier(double offset, double multiplier, int dimension) {
			super(dimension, dimension);
			if (!Double.isFinite(offset) || !Double.isFinite(multiplier) || multiplier == 0.0)
				throw new IllegalArgumentException("finite offset and nonzero multiplier required");
			this.offset = offset; this.multiplier = multiplier;
		}
		@Override public String description() { return "offset_multiplier[" + offset + "," + multiplier + "][" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			for (int i = 0; i < constrained; i++)
				target[targetOffset + i] = offset + multiplier * source[sourceOffset + i];
			return constrained * Math.log(Math.abs(multiplier));
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < constrained; i++)
				target[targetOffset + i] = (source[sourceOffset + i] - offset) / multiplier;
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			for (int i = 0; i < unconstrained; i++) ug[uo + i] += cg[co + i] * multiplier;
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

	private static final class PositiveOrdered extends Base {
		PositiveOrdered(int dimension) { super(dimension, dimension); }
		@Override public String description() { return "positive_ordered[" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double jacobian = 0.0;
			for (int i = 0; i < constrained; i++) {
				jacobian += source[sourceOffset + i];
				double increment = Math.exp(source[sourceOffset + i]);
				target[targetOffset + i] = i == 0 ? increment
						: target[targetOffset + i - 1] + increment;
			}
			return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < constrained; i++) {
				double increment = i == 0 ? source[sourceOffset]
						: source[sourceOffset + i] - source[sourceOffset + i - 1];
				if (!(increment > 0.0))
					throw new IllegalArgumentException("values must be positive and strictly ordered");
				target[targetOffset + i] = Math.log(increment);
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			double suffix = 0.0;
			for (int i = constrained - 1; i >= 0; i--) {
				suffix += cg[co + i];
				ug[uo + i] += suffix * Math.exp(u[uo + i]) + 1.0;
			}
		}
	}

	private static final class SumToZero extends Base {
		SumToZero(int dimension) { super(dimension - 1, dimension); }
		@Override public String description() { return "sum_to_zero_vector[" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			for (int i = 0; i < constrained; i++) target[targetOffset + i] = 0.0;
			double sumWeights = 0.0;
			for (int step = 0; step < unconstrained; step++) {
				int i = unconstrained - step;
				double weight = source[sourceOffset + i - 1] / Math.sqrt(i * (i + 1.0));
				sumWeights += weight;
				target[targetOffset + i - 1] += sumWeights;
				target[targetOffset + i] -= weight * i;
			}
			return 0.0;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			double sum = 0.0;
			for (int i = 0; i < constrained; i++) sum += source[sourceOffset + i];
			if (Math.abs(sum) > 1e-10 * Math.max(1.0, constrained))
				throw new IllegalArgumentException("values must sum to zero");
			target[targetOffset + unconstrained - 1] = -source[sourceOffset + constrained - 1]
					* Math.sqrt(1.0 + 1.0 / unconstrained);
			double sumWeights = 0.0;
			for (int step = 1; step < unconstrained; step++) {
				int i = unconstrained - step;
				double weight = target[targetOffset + i] / Math.sqrt((i + 1.0) * (i + 2.0));
				sumWeights += weight;
				target[targetOffset + i - 1] = (sumWeights - source[sourceOffset + i])
						* Math.sqrt((i + 1.0) * i) / i;
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			double prefix = 0.0;
			for (int i = 1; i <= unconstrained; i++) {
				prefix += cg[co + i - 1];
				ug[uo + i - 1] += (prefix - i * cg[co + i]) / Math.sqrt(i * (i + 1.0));
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

	private static final class UnitVector extends Base {
		UnitVector(int dimension) { super(dimension, dimension); }
		@Override public String description() { return "unit_vector[" + constrained + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double squaredNorm = 0.0;
			for (int i = 0; i < unconstrained; i++) {
				double value = source[sourceOffset + i]; squaredNorm += value * value;
			}
			if (!(squaredNorm > 0.0) || !Double.isFinite(squaredNorm))
				throw new IllegalArgumentException("unit-vector transform is undefined at zero or nonfinite input");
			double inverseNorm = 1.0 / Math.sqrt(squaredNorm);
			for (int i = 0; i < constrained; i++)
				target[targetOffset + i] = source[sourceOffset + i] * inverseNorm;
			return -0.5 * squaredNorm;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			double squaredNorm = 0.0;
			for (int i = 0; i < constrained; i++) squaredNorm += source[sourceOffset + i] * source[sourceOffset + i];
			if (Math.abs(squaredNorm - 1.0) > 1e-8)
				throw new IllegalArgumentException("unit-vector values must have Euclidean norm one");
			System.arraycopy(source, sourceOffset, target, targetOffset, unconstrained);
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			double squaredNorm = 0.0, projection = 0.0;
			for (int i = 0; i < unconstrained; i++) {
				squaredNorm += u[uo + i] * u[uo + i];
				projection += c[co + i] * cg[co + i];
			}
			double inverseNorm = 1.0 / Math.sqrt(squaredNorm);
			for (int i = 0; i < unconstrained; i++)
				ug[uo + i] += (cg[co + i] - c[co + i] * projection) * inverseNorm - u[uo + i];
		}
	}

	private static final class CholeskyFactorCovariance extends Base {
		private final int rows, columns;
		CholeskyFactorCovariance(int rows, int columns) {
			super(choleskyFreeCount(rows, columns), matrixCount(rows, columns));
			if (rows < columns) throw new IllegalArgumentException("Cholesky factor requires rows >= columns");
			this.rows = rows; this.columns = columns;
		}
		@Override public String description() { return "cholesky_factor_cov[" + rows + "," + columns + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			for (int i = 0; i < constrained; i++) target[targetOffset + i] = 0.0;
			int input = sourceOffset; double logJacobian = 0.0;
			for (int row = 0; row < rows; row++) {
				for (int column = 0; column < columns && column <= row; column++) {
					double value = source[input++];
					if (row == column) { target[targetOffset + row * columns + column] = Math.exp(value); logJacobian += value; }
					else target[targetOffset + row * columns + column] = value;
				}
			}
			return logJacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			int output = targetOffset;
			for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++) {
				double value = source[sourceOffset + row * columns + column];
				if (column > row) {
					if (Math.abs(value) > 1e-10) throw new IllegalArgumentException("Cholesky factor must be lower triangular");
				} else if (row == column) {
					if (!(value > 0.0)) throw new IllegalArgumentException("positive Cholesky diagonal required");
					target[output++] = Math.log(value);
				} else target[output++] = value;
			}
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			int input = uo;
			for (int row = 0; row < rows; row++) for (int column = 0; column < columns && column <= row; column++) {
				double gradient = cg[co + row * columns + column];
				ug[input] += row == column ? gradient * c[co + row * columns + column] + 1.0 : gradient;
				input++;
			}
		}
	}

	private static final class CovarianceMatrix extends Base {
		private final int dimension;
		CovarianceMatrix(int dimension) {
			super(triangularCount(dimension), matrixCount(dimension, dimension)); this.dimension = dimension;
		}
		@Override public String description() { return "cov_matrix[" + dimension + "]"; }
		@Override public double constrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double[] lower = new double[constrained]; int input = sourceOffset;
			double logJacobian = dimension * Math.log(2.0);
			for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++) {
				double value = source[input++];
				if (row == column) {
					lower[row * dimension + column] = Math.exp(value);
					logJacobian += (dimension - row + 1.0) * value;
				} else lower[row * dimension + column] = value;
			}
			lowerProduct(lower, dimension, target, targetOffset);
			return logJacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset,
				double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			double[] lower = cholesky(source, sourceOffset, dimension, true);
			int output = targetOffset;
			for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++)
				target[output++] = row == column ? Math.log(lower[row * dimension + column])
						: lower[row * dimension + column];
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co,
				double[] cg, double[] ug) {
			double[] lower = new double[constrained]; int input = uo;
			for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++)
				lower[row * dimension + column] = row == column ? Math.exp(u[input++]) : u[input++];
			int output = uo;
			for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++) {
				double derivative = 0.0;
				for (int k = 0; k < dimension; k++)
					derivative += (cg[co + row * dimension + k] + cg[co + k * dimension + row])
							* lower[k * dimension + column];
				if (row == column) derivative = derivative * lower[row * dimension + column]
						+ dimension - row + 1.0;
				ug[output++] += derivative;
			}
		}
	}

	private abstract static class CorrelationBase extends Base {
		final int dimension;
		CorrelationBase(int dimension) { super(correlationFreeCount(dimension), matrixCount(dimension, dimension)); this.dimension = dimension; }
		double fillLower(double[] source, int sourceOffset, double[] lower, boolean matrixJacobian) {
			for (int i = 0; i < lower.length; i++) lower[i] = 0.0;
			lower[0] = 1.0; int input = sourceOffset; double jacobian = 0.0;
			for (int row = 1; row < dimension; row++) {
				double remaining = 1.0;
				for (int column = 0; column < row; column++) {
					double z = Math.tanh(source[input++]);
					lower[row * dimension + column] = z * Math.sqrt(remaining);
					double coefficient = matrixJacobian ? 0.5 * (dimension - column)
							: 0.5 * (row - column + 1.0);
					jacobian += coefficient * Math.log1p(-z * z);
					remaining *= 1.0 - z * z;
				}
				lower[row * dimension + row] = Math.sqrt(Math.max(0.0, remaining));
			}
			return jacobian;
		}
		void unconstrainLower(double[] lower, double[] target, int targetOffset) {
			int output = targetOffset;
			for (int row = 1; row < dimension; row++) {
				double remaining = 1.0;
				for (int column = 0; column < row; column++) {
					double z = lower[row * dimension + column] / Math.sqrt(remaining);
					if (!(Math.abs(z) < 1.0)) throw new IllegalArgumentException("invalid correlation Cholesky factor");
					target[output++] = 0.5 * (Math.log1p(z) - Math.log1p(-z));
					remaining *= 1.0 - z * z;
				}
			}
		}
		void pullbackLower(double[] u, int uo, double[] lower, double[] lowerGradient,
				double[] ug, boolean matrixJacobian) {
			int input = uo;
			for (int row = 1; row < dimension; row++) {
				double tail = lowerGradient[row * dimension + row] * lower[row * dimension + row];
				int rowStart = input; input += row;
				for (int column = row - 1; column >= 0; column--) {
					double z = Math.tanh(u[rowStart + column]);
					double remainingScale = lower[row * dimension + column] / z;
					if (z == 0.0) {
						double remaining = 1.0;
						for (int previous = 0; previous < column; previous++) {
							double previousZ = Math.tanh(u[rowStart + previous]);
							remaining *= 1.0 - previousZ * previousZ;
						}
						remainingScale = Math.sqrt(remaining);
					}
					double likelihood = lowerGradient[row * dimension + column]
							* remainingScale * (1.0 - z * z) - z * tail;
					double jacobian = -(matrixJacobian ? dimension - column : row - column + 1.0) * z;
					ug[rowStart + column] += likelihood + jacobian;
					tail += lowerGradient[row * dimension + column] * lower[row * dimension + column];
				}
			}
		}
	}

	private static final class CholeskyFactorCorrelation extends CorrelationBase {
		CholeskyFactorCorrelation(int dimension) { super(dimension); }
		@Override public String description() { return "cholesky_factor_corr[" + dimension + "]"; }
		@Override public double constrain(double[] source, int sourceOffset, double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double[] lower = new double[constrained]; double jacobian = fillLower(source, sourceOffset, lower, false);
			System.arraycopy(lower, 0, target, targetOffset, constrained); return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset, double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			validateCorrelationCholesky(source, sourceOffset, dimension);
			double[] lower = new double[constrained]; System.arraycopy(source, sourceOffset, lower, 0, constrained);
			unconstrainLower(lower, target, targetOffset);
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co, double[] cg, double[] ug) {
			double[] lower = new double[constrained], gradient = new double[constrained];
			System.arraycopy(c, co, lower, 0, constrained); System.arraycopy(cg, co, gradient, 0, constrained);
			pullbackLower(u, uo, lower, gradient, ug, false);
		}
	}

	private static final class CorrelationMatrix extends CorrelationBase {
		CorrelationMatrix(int dimension) { super(dimension); }
		@Override public String description() { return "corr_matrix[" + dimension + "]"; }
		@Override public double constrain(double[] source, int sourceOffset, double[] target, int targetOffset) {
			check(source, sourceOffset, unconstrained); check(target, targetOffset, constrained);
			double[] lower = new double[constrained]; double jacobian = fillLower(source, sourceOffset, lower, true);
			lowerProduct(lower, dimension, target, targetOffset); return jacobian;
		}
		@Override public void unconstrain(double[] source, int sourceOffset, double[] target, int targetOffset) {
			check(source, sourceOffset, constrained); check(target, targetOffset, unconstrained);
			for (int i = 0; i < dimension; i++)
				if (Math.abs(source[sourceOffset + i * dimension + i] - 1.0) > 1e-8)
					throw new IllegalArgumentException("correlation matrix must have unit diagonal");
			double[] lower = cholesky(source, sourceOffset, dimension, true);
			unconstrainLower(lower, target, targetOffset);
		}
		@Override public void pullback(double[] u, int uo, double[] c, int co, double[] cg, double[] ug) {
			double[] lower = new double[constrained]; fillLower(u, uo, lower, true);
			double[] lowerGradient = new double[constrained];
			for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++) {
				double derivative = 0.0;
				for (int k = 0; k < dimension; k++)
					derivative += (cg[co + row * dimension + k] + cg[co + k * dimension + row])
							* lower[k * dimension + column];
				lowerGradient[row * dimension + column] = derivative;
			}
			pullbackLower(u, uo, lower, lowerGradient, ug, true);
		}
	}

	private static int matrixCount(int rows, int columns) {
		if (rows < 1 || columns < 1 || (long) rows * columns > Integer.MAX_VALUE)
			throw new IllegalArgumentException("positive practical matrix dimensions required");
		return rows * columns;
	}
	private static int triangularCount(int dimension) {
		if (dimension < 1) throw new IllegalArgumentException("positive dimension required");
		return dimension * (dimension + 1) / 2;
	}
	private static int correlationFreeCount(int dimension) {
		if (dimension < 2) throw new IllegalArgumentException("correlation dimension must be at least two");
		return dimension * (dimension - 1) / 2;
	}
	private static int choleskyFreeCount(int rows, int columns) {
		matrixCount(rows, columns);
		if (rows < columns) throw new IllegalArgumentException("Cholesky factor requires rows >= columns");
		return columns * (columns + 1) / 2 + (rows - columns) * columns;
	}
	private static void lowerProduct(double[] lower, int dimension, double[] target, int offset) {
		for (int row = 0; row < dimension; row++) for (int column = 0; column < dimension; column++) {
			double sum = 0.0;
			for (int k = 0; k <= Math.min(row, column); k++)
				sum += lower[row * dimension + k] * lower[column * dimension + k];
			target[offset + row * dimension + column] = sum;
		}
	}
	private static double[] cholesky(double[] matrix, int offset, int dimension, boolean symmetric) {
		double[] lower = new double[dimension * dimension];
		for (int row = 0; row < dimension; row++) for (int column = 0; column <= row; column++) {
			if (symmetric && Math.abs(matrix[offset + row * dimension + column]
					- matrix[offset + column * dimension + row]) > 1e-8)
				throw new IllegalArgumentException("matrix must be symmetric");
			double value = matrix[offset + row * dimension + column];
			for (int k = 0; k < column; k++) value -= lower[row * dimension + k] * lower[column * dimension + k];
			if (row == column) {
				if (!(value > 0.0) || !Double.isFinite(value)) throw new IllegalArgumentException("matrix must be positive definite");
				lower[row * dimension + column] = Math.sqrt(value);
			} else lower[row * dimension + column] = value / lower[column * dimension + column];
		}
		return lower;
	}
	private static void validateCorrelationCholesky(double[] matrix, int offset, int dimension) {
		for (int row = 0; row < dimension; row++) {
			double norm = 0.0;
			for (int column = 0; column < dimension; column++) {
				double value = matrix[offset + row * dimension + column];
				if (column > row && Math.abs(value) > 1e-10) throw new IllegalArgumentException("factor must be lower triangular");
				if (column <= row) norm += value * value;
			}
			if (!(matrix[offset + row * dimension + row] > 0.0) || Math.abs(norm - 1.0) > 1e-8)
				throw new IllegalArgumentException("correlation Cholesky rows must have unit norm and positive diagonal");
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
