/*
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

public class IntegrationResult {
	public UnivariateFunction f;
	public double result, abserr;
	public int neval, ier, last;
	/** Coordinate associated with a callback or non-finite-value failure. */
	public double failureX = Double.NaN;
	/** Original callback failure, when one was caught by the hardened API. */
	public RuntimeException cause;
	/** Additional context supplied by the hardened API. */
	public String detail;

	public boolean isSuccess() {
		return ier == 0;
	}

	public String message() {
		switch (ier) {
		case 0: return "OK";
		case 1: return "maximum number of subdivisions reached";
		case 2: return "roundoff error was detected";
		case 3: return "extremely bad integrand behaviour";
		case 4: return "roundoff error is detected in the extrapolation table";
		case 5: return "the integral is probably divergent";
		case 6: return "the input is invalid";
		case 7: return "the integrand callback failed";
		case 8: return "integration was cancelled";
		case 9: return "the function evaluation budget was exhausted";
		case 10: return "the integrand returned a non-finite value";
		default: return "unknown integration status";
		}
	}

	/** Returns the status message with any hardened-API context appended. */
	public String detailedMessage() {
		return detail == null || detail.length() == 0
				? message() : message() + ": " + detail;
	}
}
