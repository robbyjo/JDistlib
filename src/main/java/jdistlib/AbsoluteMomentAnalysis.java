/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import jdistlib.math.IntegrationStabilityResult;

/** Estimate and independent left/right convergence evidence for E[|X|^p]. */
public final class AbsoluteMomentAnalysis {
	private final double order;
	private final double splitPoint;
	private final double leftValue;
	private final double rightValue;
	private final boolean leftStable;
	private final boolean rightStable;
	private final IntegrationStabilityResult leftStability;
	private final IntegrationStabilityResult rightStability;

	AbsoluteMomentAnalysis(double order, double splitPoint, double leftValue,
			double rightValue, boolean leftStable, boolean rightStable,
			IntegrationStabilityResult leftStability,
			IntegrationStabilityResult rightStability) {
		this.order = order;
		this.splitPoint = splitPoint;
		this.leftValue = leftValue;
		this.rightValue = rightValue;
		this.leftStable = leftStable;
		this.rightStable = rightStable;
		this.leftStability = leftStability;
		this.rightStability = rightStability;
	}

	public double getOrder() { return order; }
	public double getSplitPoint() { return splitPoint; }
	public double getLeftValue() { return leftValue; }
	public double getRightValue() { return rightValue; }
	public double getValue() { return leftValue + rightValue; }
	public boolean isLeftStable() { return leftStable; }
	public boolean isRightStable() { return rightStable; }
	public boolean isStable() {
		return leftStable && rightStable && Double.isFinite(getValue());
	}
	/** Null when the support has no interval on the left side. */
	public IntegrationStabilityResult getLeftStability() { return leftStability; }
	/** Null when the support has no interval on the right side. */
	public IntegrationStabilityResult getRightStability() { return rightStability; }
}
