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
public abstract class OptimizationConfig {
	public static final int defaultMaxNumFunctionCall = 50000;
	protected double[]
		initialGuess,
		lowerBound,
		upperBound;
	protected int maxNumFunctionCall = defaultMaxNumFunctionCall;
	protected boolean isMinimize = true;
	protected MultivariableFunction objectiveFunction;

	public OptimizationConfig() {}

	public OptimizationConfig(double[] initGuess, double[] lo, double[] hi, MultivariableFunction fun) {
		this(initGuess, lo, hi, fun, defaultMaxNumFunctionCall, true);
	}

	public OptimizationConfig(double[] initGuess, double[] lo, double[] hi, MultivariableFunction fun, int maxCall, boolean isMin) {
		setInitialGuess(initGuess);
		setLowerBound(lo);
		setUpperBound(hi);
		setObjectiveFunction(fun);
		setMaxNumFunctionCall(maxCall);
		setMinimize(isMin);
	}

	public double[] getInitialGuess() {
		return initialGuess;
	}
	public void setInitialGuess(double[] initialGuess) {
		this.initialGuess = initialGuess;
	}
	public double[] getLowerBound() {
		return lowerBound;
	}
	public void setLowerBound(double[] lowerBound) {
		this.lowerBound = lowerBound;
	}
	public double[] getUpperBound() {
		return upperBound;
	}
	public void setUpperBound(double[] upperBound) {
		this.upperBound = upperBound;
	}
	public int getMaxNumFunctionCall() {
		return maxNumFunctionCall;
	}
	public void setMaxNumFunctionCall(int maxNumFunctionCall) {
		this.maxNumFunctionCall = maxNumFunctionCall;
	}
	public boolean isMinimize() {
		return isMinimize;
	}
	public void setMinimize(boolean isMinimize) {
		this.isMinimize = isMinimize;
	}
	public MultivariableFunction getObjectiveFunction() {
		return objectiveFunction;
	}
	public void setObjectiveFunction(MultivariableFunction objectiveFunction) {
		this.objectiveFunction = objectiveFunction;
	}
}
