/*
 * Roby Joehanes
 * 
 * Copyright 2007 Roby Joehanes
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
package jdistlib.exception;

public class PrecisionException extends RuntimeException {
  private static final long serialVersionUID = 1L;

	protected double value;

	public double getValue() {
		return value;
	}

	public void setValue(double v) {
		value = v;
	}

	public PrecisionException(double v) {
		setValue(v);
	}

	public PrecisionException(String s, double v) {
		super(s);
	}

	public PrecisionException(Throwable t, double v) {
		super(t);
		setValue(v);
	}

	public PrecisionException(String t1, Throwable t2, double v) {
		super(t1, t2);
		setValue(v);
	}
}
