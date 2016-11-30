/*
 * Roby Joehanes
 *  
 * Copyright 2012 Roby Joehanes
 * This file is distributed under the GNU General Public License version 3.0.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package net.sourceforge.jdistlib.math.opt;

import net.sourceforge.jdistlib.math.MultivariableFunction;

/**
 * 
 * @author Roby Joehanes
 *
 */
public class BobyqaConfig extends OptimizationConfig {
	public static final double
		defaultInitialTrustRegionRadius = 0.1,
		defaultStoppingTrustRegionRadius = 1e-7;
	protected double
		initialTrustRegionRadius = defaultInitialTrustRegionRadius, // = rhobeg
		stoppingTrustRegionRadius = defaultStoppingTrustRegionRadius; // = rhoend 
	protected int numInterpolationPoints; // = npt

	public BobyqaConfig() {}

	public BobyqaConfig(double[] initGuess, double[] lo, double[] hi, MultivariableFunction fun) {
		this(initGuess, lo, hi, fun, defaultMaxNumFunctionCall, true, defaultInitialTrustRegionRadius, defaultStoppingTrustRegionRadius, initGuess.length + 6);
	}

	public BobyqaConfig(double[] initGuess, double[] lo, double[] hi, MultivariableFunction fun, int maxCall, boolean isMin) {
		this(initGuess, lo, hi, fun, maxCall, isMin, defaultInitialTrustRegionRadius, defaultStoppingTrustRegionRadius, initGuess.length + 6);
	}

	public BobyqaConfig(double[] initGuess, double[] lo, double[] hi, MultivariableFunction fun, int maxCall, boolean isMin,
			double initRad, double stopRad, int npt) {
		super(initGuess, lo, hi, fun, maxCall, isMin);
		setInitialTrustRegionRadius(initRad);
		setStoppingTrustRegionRadius(stopRad);
		setNumInterpolationPoints(npt);
	}

	public BobyqaConfig(OptimizationConfig cfg) {
		setInitialGuess(cfg.initialGuess);
		setLowerBound(cfg.lowerBound);
		setUpperBound(cfg.upperBound);
		setObjectiveFunction(cfg.objectiveFunction);
		setMaxNumFunctionCall(cfg.maxNumFunctionCall);
		setMinimize(cfg.isMinimize);
		if (cfg instanceof BobyqaConfig) {
			BobyqaConfig bcfg = (BobyqaConfig) cfg;
			setInitialTrustRegionRadius(bcfg.initialTrustRegionRadius);
			setStoppingTrustRegionRadius(bcfg.stoppingTrustRegionRadius);
			setNumInterpolationPoints(bcfg.numInterpolationPoints);
		} else {
			setInitialTrustRegionRadius(defaultInitialTrustRegionRadius);
			setStoppingTrustRegionRadius(defaultStoppingTrustRegionRadius);
			setNumInterpolationPoints(cfg.initialGuess.length + 6);
		}
	}

	public double getInitialTrustRegionRadius() {
		return initialTrustRegionRadius;
	}

	public void setInitialTrustRegionRadius(double initialTrustRegionRadius) {
		this.initialTrustRegionRadius = initialTrustRegionRadius;
	}

	public double getStoppingTrustRegionRadius() {
		return stoppingTrustRegionRadius;
	}

	public void setStoppingTrustRegionRadius(double stoppingTrustRegionRadius) {
		this.stoppingTrustRegionRadius = stoppingTrustRegionRadius;
	}

	public int getNumInterpolationPoints() {
		return numInterpolationPoints;
	}

	public void setNumInterpolationPoints(int numInterpolationPoints) {
		this.numInterpolationPoints = numInterpolationPoints;
	}
}
