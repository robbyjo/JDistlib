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

/**
 * Class to hold optimization results
 * @author Roby Joehanes
 *
 */
public class OptimizationResult {
	public double[] mX;
	public double mF;
	public int numFunctionCalls;
	public boolean isMinimum;
	private static final String ln = System.getProperty("line.separator"); //$NON-NLS-1$

	public OptimizationResult() {}

	public OptimizationResult(double[] x, double f, int nf, boolean isMin) {
		mX = x; mF = f; numFunctionCalls = nf; isMinimum = isMin;
	}

	@Override
	public String toString() {
		StringBuilder buf = new StringBuilder();
		buf.append("Number of function calls = ");
		buf.append(numFunctionCalls);
		buf.append(ln);
		buf.append(isMinimum ? "Minimum" : "Maximum");
		buf.append(" value of F = ");
		buf.append(mF);
		buf.append(ln);
		buf.append("Corresponding X is: ");
		int numValues = mX.length;
		if (numValues > 0)
		{
			buf.append(mX[0]);
			for (int i = 1; i < numValues; i++)
				buf.append(", " + mX[i]);
		}
		return buf.toString();
	}
}
