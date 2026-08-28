/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.NumericalContinuousDistribution;

/** Smooth risk-neutral density plus regularization and differentiation diagnostics. */
public final class SmoothOptionDistributionResult {
	private final NumericalContinuousDistribution distribution;private final double bandwidth,normalizationError,maximumPriceResidual,differentiationUncertainty;
	SmoothOptionDistributionResult(NumericalContinuousDistribution distribution,double bandwidth,double normalizationError,double maximumPriceResidual,double differentiationUncertainty){
		this.distribution=distribution;this.bandwidth=bandwidth;this.normalizationError=normalizationError;this.maximumPriceResidual=maximumPriceResidual;this.differentiationUncertainty=differentiationUncertainty;}
	public NumericalContinuousDistribution getDistribution(){return distribution;}public double getBandwidth(){return bandwidth;}
	public double getNormalizationError(){return normalizationError;}public double getMaximumPriceResidual(){return maximumPriceResidual;}
	public double getDifferentiationUncertainty(){return differentiationUncertainty;}
	public boolean isReliable(double tolerance){return normalizationError<=tolerance&&maximumPriceResidual<=tolerance&&differentiationUncertainty<=tolerance;}
}
