/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

/** Immutable European option quote used by the narrow option-implied layer. */
public final class OptionObservation {
	private final double strike, bid, ask, weight;
	private final boolean call;
	public OptionObservation(double strike, boolean call, double price) {
		this(strike,call,price,price,1.0);
	}
	public OptionObservation(double strike, boolean call, double bid, double ask, double weight) {
		if (!(strike >= 0.0) || !(bid >= 0.0) || !(ask >= bid) || !(weight > 0.0)
				|| !Double.isFinite(strike) || !Double.isFinite(bid) || !Double.isFinite(ask)
				|| !Double.isFinite(weight)) throw new IllegalArgumentException("invalid option observation");
		this.strike=strike;this.call=call;this.bid=bid;this.ask=ask;this.weight=weight;
	}
	public double getStrike(){return strike;} public boolean isCall(){return call;}
	public double getBid(){return bid;} public double getAsk(){return ask;}
	public double getMid(){return bid+(ask-bid)/2.0;} public double getWeight(){return weight;}
}
