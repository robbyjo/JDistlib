/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.Normal;

/** Reference Black-Scholes/Bachelier transformations and checked inversion. */
public final class ReferenceOptions {
	private ReferenceOptions() {}
	public static double blackScholes(double forward,double strike,double discount,double maturity,
			double volatility,boolean call){
		if(!(forward>0.0)||!(strike>=0.0)||!(discount>0.0)||!(maturity>=0.0)||!(volatility>=0.0))return Double.NaN;
		double intrinsic=discount*Math.max(call?forward-strike:strike-forward,0.0);
		if(maturity==0.0||volatility==0.0||strike==0.0)return call&&strike==0.0?discount*forward:intrinsic;
		double standardDeviation=volatility*Math.sqrt(maturity);
		double d1=(Math.log(forward/strike)+0.5*standardDeviation*standardDeviation)/standardDeviation;
		double d2=d1-standardDeviation;
		double callPrice=discount*(forward*Normal.cumulative(d1,0.0,1.0,true,false)
				-strike*Normal.cumulative(d2,0.0,1.0,true,false));
		return call?callPrice:callPrice-discount*(forward-strike);
	}
	public static double bachelier(double forward,double strike,double discount,double maturity,
			double normalVolatility,boolean call){
		if(!(discount>0.0)||!(maturity>=0.0)||!(normalVolatility>=0.0))return Double.NaN;
		double scale=normalVolatility*Math.sqrt(maturity),difference=forward-strike;
		double callPrice=scale==0.0?discount*Math.max(difference,0.0):discount*(difference*Normal.cumulative(difference/scale,0,1,true,false)
				+scale*Normal.density(difference/scale,0,1,false));
		return call?callPrice:callPrice-discount*difference;
	}
	public static ImpliedVolatilityResult impliedBlackScholes(double price,double forward,double strike,
			double discount,double maturity,boolean call){
		return implied(price,forward,strike,discount,maturity,call,false);
	}
	public static ImpliedVolatilityResult impliedBachelier(double price,double forward,double strike,
			double discount,double maturity,boolean call){
		return implied(price,forward,strike,discount,maturity,call,true);
	}
	private static ImpliedVolatilityResult implied(double price,double forward,double strike,double discount,
			double maturity,boolean call,boolean normal){
		if(!Double.isFinite(price)||!(forward>0.0)||!(strike>=0.0)||!(discount>0.0)||!(maturity>0.0))
			return new ImpliedVolatilityResult(Double.NaN,Double.NaN,0,0,0,Double.NaN,Double.NaN,ImpliedVolatilityResult.Status.INVALID_INPUT);
		double lower=discount*Math.max(call?forward-strike:strike-forward,0.0);
		double upper=normal?Double.POSITIVE_INFINITY:discount*(call?forward:strike);
		if(price<lower-1e-12)return new ImpliedVolatilityResult(Double.NaN,price-lower,0,0,0,lower,upper,ImpliedVolatilityResult.Status.PRICE_BELOW_BOUND);
		if(price>upper+1e-12)return new ImpliedVolatilityResult(Double.NaN,price-upper,0,0,0,lower,upper,ImpliedVolatilityResult.Status.PRICE_ABOVE_BOUND);
		if(Math.abs(price-lower)<=1e-12)return new ImpliedVolatilityResult(0.0,lower-price,0,0,0,lower,upper,ImpliedVolatilityResult.Status.CONVERGED);
		double low=0.0,high=normal?Math.max(forward,strike):1.0;
		while(value(normal,forward,strike,discount,maturity,high,call)<price&&high<1e6)high*=2.0;
		if(high>=1e6)return new ImpliedVolatilityResult(Double.NaN,Double.NaN,0,low,high,lower,upper,ImpliedVolatilityResult.Status.NOT_BRACKETED);
		int iteration=0;for(;iteration<100;iteration++){double middle=low+(high-low)/2.0;double model=value(normal,forward,strike,discount,maturity,middle,call);
			if(model<price)low=middle;else high=middle;if(high-low<1e-13*Math.max(1.0,high))break;}
		double volatility=low+(high-low)/2.0,residual=value(normal,forward,strike,discount,maturity,volatility,call)-price;
		return new ImpliedVolatilityResult(volatility,residual,iteration+1,low,high,lower,upper,ImpliedVolatilityResult.Status.CONVERGED);
	}
	private static double value(boolean normal,double f,double k,double d,double t,double v,boolean call){
		return normal?bachelier(f,k,d,t,v,call):blackScholes(f,k,d,t,v,call);
	}
}
