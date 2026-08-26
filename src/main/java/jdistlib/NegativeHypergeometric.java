/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.MathFunctions;
import jdistlib.rng.RandomEngine;

/** Number of draws needed to observe {@code r} white balls without replacement. */
public final class NegativeHypergeometric extends GenericDistribution
		implements SupportedDistribution {
	private final int black;
	private final int white;
	private final int requiredWhite;

	public NegativeHypergeometric(int black, int white, int requiredWhite) {
		if (!valid(black, white, requiredWhite)) {
			throw new IllegalArgumentException("counts must satisfy black >= 0 and 0 <= requiredWhite <= white");
		}
		this.black = black;
		this.white = white;
		this.requiredWhite = requiredWhite;
	}

	private static boolean valid(double black, double white, double requiredWhite) {
		return black >= 0.0 && white >= 0.0 && requiredWhite >= 0.0
				&& requiredWhite <= white && black == Math.rint(black)
				&& white == Math.rint(white) && requiredWhite == Math.rint(requiredWhite)
				&& black <= Integer.MAX_VALUE && white <= Integer.MAX_VALUE;
	}

	public static double density(double x, double black, double white,
			double requiredWhite, boolean log) {
		if (Double.isNaN(x) || Double.isNaN(black) || Double.isNaN(white)
				|| Double.isNaN(requiredWhite)) return x + black + white + requiredWhite;
		if (!valid(black, white, requiredWhite)) return Double.NaN;
		if (requiredWhite == 0.0) {
			boolean atom = x == 0.0;
			return log ? (atom ? 0.0 : Double.NEGATIVE_INFINITY) : (atom ? 1.0 : 0.0);
		}
		if (x != Math.rint(x) || x < requiredWhite || x > black + requiredWhite) {
			return log ? Double.NEGATIVE_INFINITY : 0.0;
		}
		double value = MathFunctions.lchoose(x - 1.0, requiredWhite - 1.0)
				+ MathFunctions.lchoose(white + black - x, white - requiredWhite)
				- MathFunctions.lchoose(white + black, black);
		return log ? value : Math.exp(value);
	}

	public static double cumulative(double x, double black, double white,
			double requiredWhite, boolean lowerTail, boolean logP) {
		if (Double.isNaN(x) || Double.isNaN(black) || Double.isNaN(white)
				|| Double.isNaN(requiredWhite)) return x + black + white + requiredWhite;
		if (!valid(black, white, requiredWhite)) return Double.NaN;
		double minimum = requiredWhite == 0.0 ? 0.0 : requiredWhite;
		if (x < minimum) return DistributionUtil.boundary(false, lowerTail, logP);
		if (requiredWhite == 0.0 || x >= black + requiredWhite) {
			return DistributionUtil.boundary(true, lowerTail, logP);
		}
		double draws = Math.floor(x);
		return HyperGeometric.cumulative(requiredWhite - 1.0, white, black,
				draws, !lowerTail, logP);
	}

	public static double quantile(double p, double black, double white,
			double requiredWhite, boolean lowerTail, boolean logP) {
		if (!valid(black, white, requiredWhite)) return Double.NaN;
		double minimum = requiredWhite == 0.0 ? 0.0 : requiredWhite;
		return DistributionUtil.discreteQuantile(p, lowerTail, logP, minimum,
				black + requiredWhite,
				(x, lt, lp) -> cumulative(x, black, white, requiredWhite, lt, lp));
	}

	public static double random(int black, int white, int requiredWhite,
			RandomEngine random) {
		if (!valid(black, white, requiredWhite)) return Double.NaN;
		if (requiredWhite == 0) return 0.0;
		int blackLeft = black;
		int whiteLeft = white;
		int whiteSeen = 0;
		int draws = 0;
		while (whiteSeen < requiredWhite) {
			if (random.nextDouble() * (blackLeft + whiteLeft) < whiteLeft) {
				whiteLeft--;
				whiteSeen++;
			} else blackLeft--;
			draws++;
		}
		return draws;
	}

	@Override public double density(double x, boolean log) {
		return density(x, black, white, requiredWhite, log);
	}
	@Override public double cumulative(double x, boolean lowerTail, boolean logP) {
		return cumulative(x, black, white, requiredWhite, lowerTail, logP);
	}
	@Override public double quantile(double p, boolean lowerTail, boolean logP) {
		return quantile(p, black, white, requiredWhite, lowerTail, logP);
	}
	@Override public double random() { return random(black, white, requiredWhite, random); }
	@Override public double getLowerBound() { return requiredWhite == 0 ? 0.0 : requiredWhite; }
	@Override public double getUpperBound() { return black + (double) requiredWhite; }
}
