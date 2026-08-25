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
	/** Callback timing captured by the hardened API. */
	public CallbackProfile callbackProfile = CallbackProfile.empty();

	public boolean isSuccess() {
		return ier == 0;
	}

	public String message() {
		return IntegrationStatus.fromCode(ier).getMessage();
	}

	/** Returns the status message with any hardened-API context appended. */
	public String detailedMessage() {
		return detail == null || detail.length() == 0
				? message() : message() + ": " + detail;
	}

	/** Returns the typed interpretation of {@link #ier}. */
	public IntegrationStatus getStatus() { return IntegrationStatus.fromCode(ier); }

	/** Returns immutable callback timing information. */
	public CallbackProfile getCallbackProfile() {
		return callbackProfile == null ? CallbackProfile.empty() : callbackProfile;
	}

	/** Creates an immutable modern snapshot without retaining {@link #f}. */
	public ImmutableIntegrationResult toImmutable() {
		return new ImmutableIntegrationResult(this);
	}

	/** Returns an RFC 8259 JSON diagnostic record. */
	public String toJson() { return IntegrationJson.toJson(toImmutable()); }
}
