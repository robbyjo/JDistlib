/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

/** Work, truncation, and convergence controls for adaptive Fourier inversion. */
public final class FourierInversionOptions {
	private final double initialFrequency, maximumFrequency, tolerance;
	private final int panels, maximumRefinements;
	private FourierInversionOptions(double initialFrequency,double maximumFrequency,
			double tolerance,int panels,int maximumRefinements){
		if(!(initialFrequency>0.0)||!(maximumFrequency>=initialFrequency)||!(tolerance>0.0)
				||panels<256||maximumRefinements<1)throw new IllegalArgumentException("invalid Fourier inversion options");
		this.initialFrequency=initialFrequency;this.maximumFrequency=maximumFrequency;
		this.tolerance=tolerance;this.panels=panels;this.maximumRefinements=maximumRefinements;
	}
	public static FourierInversionOptions defaults(){return new FourierInversionOptions(20.0,640.0,1e-8,4096,8);}
	public FourierInversionOptions withTolerance(double value){return new FourierInversionOptions(initialFrequency,maximumFrequency,value,panels,maximumRefinements);}
	public FourierInversionOptions withFrequencyRange(double initial,double maximum){return new FourierInversionOptions(initial,maximum,tolerance,panels,maximumRefinements);}
	public FourierInversionOptions withPanels(int value){return new FourierInversionOptions(initialFrequency,maximumFrequency,tolerance,value,maximumRefinements);}
	public FourierInversionOptions withMaximumRefinements(int value){return new FourierInversionOptions(initialFrequency,maximumFrequency,tolerance,panels,value);}
	public double getInitialFrequency(){return initialFrequency;}public double getMaximumFrequency(){return maximumFrequency;}
	public double getTolerance(){return tolerance;}public int getPanels(){return panels;}public int getMaximumRefinements(){return maximumRefinements;}
}
