/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package examples;

import jdistlib.LogNormal;
import jdistlib.finance.*;

/**
 * Worked European-options analysis: validate quotes, infer a risk-neutral law,
 * and compare it with an explicitly separate physical scenario law.
 */
public final class WorkedOptionsTradingExample {
	private WorkedOptionsTradingExample() {}

	public static void main(String[] args) {
		double forward = 102.0;
		double discount = Math.exp(-0.04 * 0.5);
		double maturity = 0.5;

		OptionObservation[] quotes = {
			new OptionObservation(80, true, 22.75, 23.15, 1.0),
			new OptionObservation(90, true, 14.40, 14.75, 1.2),
			new OptionObservation(100, true, 8.25, 8.55, 1.5),
			new OptionObservation(110, true, 4.15, 4.45, 1.2),
			new OptionObservation(120, true, 1.95, 2.20, 1.0),
			new OptionObservation(130, true, 0.85, 1.05, 0.8)
		};

		System.out.println("1. Convert each quote to a checked Black-Scholes implied volatility.");
		for (OptionObservation quote : quotes) {
			ImpliedVolatilityResult result = ReferenceOptions.impliedBlackScholes(
					quote.getMid(), forward, quote.getStrike(), discount, maturity, quote.isCall());
			System.out.printf("K=%5.1f  IV=%6.2f%%  residual=% .3g  status=%s%n",
					quote.getStrike(), 100.0 * result.getVolatility(), result.getResidual(), result.getStatus());
		}

		System.out.println("\n2. Repair the whole curve jointly before extracting probabilities.");
		OptionCurve curve = OptionCurve.build(forward, discount, maturity, quotes);
		OptionCurve.Diagnostics curveDiagnostics = curve.getDiagnostics();
		System.out.printf("repaired=%d  max price change=%.4f  weighted RMSE=%.4f  convex=%s%n",
				curveDiagnostics.getRepairedObservations(), curveDiagnostics.getMaximumPriceResidual(),
				curveDiagnostics.getWeightedRmse(), curveDiagnostics.isConvex());

		OptionImpliedDistribution riskNeutral = curve.getDistribution();
		System.out.println("\n3. Ask probability questions of the recovered risk-neutral law.");
		double below90 = riskNeutral.cumulative(90.0);
		double above120 = riskNeutral.cumulative(120.0, false, false);
		double between = curve.strikeIntervalProbability(95.0, 115.0);
		double median = riskNeutral.quantile(0.5);
		System.out.printf("Q[S<=90]=%.3f  Q[S>120]=%.3f  Q[95<S<=115]=%.3f  median=%.2f%n",
				below90, above120, between, median);

		System.out.println("\n4. Work a covered-call terminal payoff (one share minus a K=110 call). ");
		double strike = 110.0;
		double expectedStock = forward;
		double expectedCallPayoff = FinancialRisk.callPayoff(riskNeutral, strike).valueOrThrow();
		double expectedCoveredCall = expectedStock - expectedCallPayoff;
		double probabilityCapped = riskNeutral.cumulative(strike, false, false);
		System.out.printf("undiscounted E[min(S,110)]=%.3f; Q[payoff capped]=%.3f%n",
				expectedCoveredCall, probabilityCapped);

		System.out.println("\n5. Keep physical forecasts separate from option-implied Q probabilities.");
		double physicalVolatility = 0.24;
		double physicalMean = 106.0;
		double logVariance = Math.log1p(physicalVolatility * physicalVolatility);
		LogNormal physical = new LogNormal(Math.log(physicalMean) - 0.5 * logVariance,
				Math.sqrt(logVariance));
		double physicalLossVar = FinancialRisk.valueAtRisk(physical, 0.05, RiskConvention.LOSS);
		System.out.printf("P[S>120]: risk-neutral %.3f versus physical %.3f; physical 5%% quantile %.2f%n",
				above120, physical.cumulative(120.0, false, false), physical.quantile(0.05));
		System.out.printf("(The illustrative LOSS VaR call on price itself is %.2f; use a P&L law in production.)%n",
				physicalLossVar);

		System.out.println("\nThis is probability analysis, not investment advice or a backtesting/execution system.");
	}
}
