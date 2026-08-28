/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

/** Checked implied-volatility inversion result. */
public final class ImpliedVolatilityResult {
	public enum Status { CONVERGED, PRICE_BELOW_BOUND, PRICE_ABOVE_BOUND, INVALID_INPUT, NOT_BRACKETED }
	private final double volatility,residual,lowerBracket,upperBracket,lowerPriceBound,upperPriceBound;
	private final int iterations;private final Status status;
	ImpliedVolatilityResult(double volatility,double residual,int iterations,double lowerBracket,
			double upperBracket,double lowerPriceBound,double upperPriceBound,Status status){
		this.volatility=volatility;this.residual=residual;this.iterations=iterations;this.lowerBracket=lowerBracket;
		this.upperBracket=upperBracket;this.lowerPriceBound=lowerPriceBound;this.upperPriceBound=upperPriceBound;this.status=status;
	}
	public double getVolatility(){return volatility;}public double getResidual(){return residual;}
	public int getIterations(){return iterations;}public double getLowerBracket(){return lowerBracket;}
	public double getUpperBracket(){return upperBracket;}public double getLowerPriceBound(){return lowerPriceBound;}
	public double getUpperPriceBound(){return upperPriceBound;}public Status getStatus(){return status;}
	public boolean isConverged(){return status==Status.CONVERGED;}
}
