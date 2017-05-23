/*
 * Roby Joehanes 
 * 
 * Copyright 2007 Roby Joehanes
 * This file is distributed under the GNU General Public License version 2.0.
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jdistlib.math;

/**
 * Abstraction of a function with one parameter
 * @author Roby Joehanes
 */
public interface UnivariateFunction {
	public double eval(double x);
	public void setParameters(double... params);
	public void setObjects(Object... obj);
}
