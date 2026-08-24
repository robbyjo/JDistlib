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
package jdistlib.math.spline;

/**
 * Smooth spline criterion.<br>
 * NO_CRITERION = No additional minimizing criterion<br>
 * GCV = Generalized Cross Validation<br>
 * CV = Cross Validation<br>
 * DF_MATCH = Degree of freedom matching<br>
 * 
 * @author Roby Joehanes
 *
 */
public enum SmoothSplineCriterion
{
	NO_CRITERION,
	GCV,
	CV,
	DF_MATCH
}
