/*
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998   Ross Ihaka
 *  Copyright (C) 2000-9 The R Development Core Team
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
 *  http://www.r-project.org/Licenses/
 */
package jdistlib;

import static java.lang.Math.*;
import static jdistlib.math.Constants.DBL_MIN;

import jdistlib.generic.GenericDistribution;
import jdistlib.math.UnivariateFunction;
import jdistlib.math.opt.Optimization;
import jdistlib.rng.RandomEngine;

/**
 * Inverse normal (or Wald) distribution. Taken from package gamlss.dist.
 * Parameterization: mu and sigma. Note: lambda = 1/sigma^2.
 *
 */
public class InvNormal extends GenericDistribution {
    private static boolean invalid(double mu, double sigma) {
        return !(mu > 0.0) || !(sigma > 0.0) || !Double.isFinite(mu) || !Double.isFinite(sigma);
    }
    public static final double density(double x, double mu, double sigma, boolean giveLog) {
        if (invalid(mu, sigma) || Double.isNaN(x)) return Double.NaN;
        if (x <= 0.0 || x == Double.POSITIVE_INFINITY) return giveLog ? Double.NEGATIVE_INFINITY : 0.0;
        double a = (x / mu - 1.0) / (sigma * sqrt(x));
        double v = -0.5 * log(2.0 * PI) - log(sigma) - 1.5 * log(x) - 0.5 * a * a;
        return giveLog ? v : exp(v);
    }
    public static final double cumulative(double x, double mu, double sigma, boolean lowerTail, boolean logP) {
        if (invalid(mu, sigma) || Double.isNaN(x)) return Double.NaN;
        if (x <= 0.0) return DistributionUtil.boundary(false, lowerTail, logP);
        if (x == Double.POSITIVE_INFINITY) return DistributionUtil.boundary(true, lowerTail, logP);
        double denominator = sigma * sqrt(x);
        double a = (x / mu - 1.0) / denominator;
        double b = (x / mu + 1.0) / denominator;
        double second;
        if (b >= 8.0) {
            // Laplace's continued fraction for the normal Mills ratio avoids
            // subtracting the two very large exponents 2/(mu*sigma^2) and b^2/2.
            double fraction = 0.0;
            for (int i = 64; i >= 1; i--) fraction = i / (b + fraction);
            second = -0.5 * a * a - 0.5 * log(2.0 * PI) - log(b + fraction);
        } else second = 2.0 / (mu * sigma * sigma) + Normal.cumulative(-b, 0.0, 1.0, true, true);
        double answer;
        if (lowerTail) {
            answer = DistributionUtil.logAdd(Normal.cumulative(a, 0.0, 1.0, true, true), second);
            answer = min(answer, 0.0);
        } else {
            double first = Normal.cumulative(a, 0.0, 1.0, false, true);
            answer = first + DistributionUtil.logOneMinusExp(min(second - first, 0.0));
        }
        return logP ? answer : exp(answer);
    }
    public static final double quantile(double p, double mu, double sigma, boolean lowerTail, boolean logP) {
        if (invalid(mu, sigma) || Double.isNaN(p) || DistributionUtil.invalidProbability(p, logP)) return Double.NaN;
        double target = logP ? p : log(p);
        if (target == Double.NEGATIVE_INFINITY) return lowerTail ? 0.0 : Double.POSITIVE_INFINITY;
        if (target == 0.0) return lowerTail ? Double.POSITIVE_INFINITY : 0.0;
        // Solve the smaller probability, retaining log inputs all the way.
        if (target > -log(2.0)) { target = DistributionUtil.logOneMinusExp(target); lowerTail = !lowerTail; }
        double low = log(Double.MIN_VALUE), high = log(Double.MAX_VALUE);
        for (int i = 0; i < 100; i++) {
            double middle = low + (high - low) / 2.0;
            if (middle == low || middle == high) break;
            double value = cumulative(exp(middle), mu, sigma, lowerTail, true);
            if (lowerTail ? value >= target : value <= target) high = middle;
            else low = middle;
        }
        return exp(low + (high - low) / 2.0);
    }
    public static final double random(double mu, double sigma, RandomEngine random) {
        if (invalid(mu, sigma)) return Double.NaN;
        // Michael-Schucany-Haas transformation; rationalized to avoid cancellation.
        double z = Normal.random_standard(random);
        double w = 0.5 * mu * sigma * sigma * z * z;
        double ratio = 1.0 / (1.0 + w + sqrt(w) * sqrt(w + 2.0));
        return random.nextDouble() <= 1.0 / (1.0 + ratio) ? mu * ratio : mu / ratio;
    }

	public static final double[] random(int n, double mu, double sigma, RandomEngine random) {
		double[] rand = new double[n];
		for (int i = 0; i < n; i++)
			rand[i] = random(mu, sigma, random);
		return rand;
	}

	protected double mu, sigma;

	public InvNormal(double mu, double sigma) {
		this.mu = mu; this.sigma = sigma;
	}

	@Override
	public double density(double x, boolean log) {
		return density(x, mu, sigma, log);
	}

	@Override
	public double cumulative(double p, boolean lower_tail, boolean log_p) {
		return cumulative(p, mu, sigma, lower_tail, log_p);
	}

	@Override
	public double quantile(double q, boolean lower_tail, boolean log_p) {
		return quantile(q, mu, sigma, lower_tail, log_p);
	}

	@Override
	public double random() {
		return random(mu, sigma, random);
	}

	/*
	public static final void main(String[] args) {
		double x, y, z;
		for (int i = 1; i <= 600; i++) {
			x = i/100.0;
			y = cumulative(x, 1, 0.5, true, false);
			z = quantile(y, 1, 0.5, true, false);
			System.out.println(x + ", " + y + ", " + z);
		}
	}
	//*/
}
