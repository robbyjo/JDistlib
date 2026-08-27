/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/** Factories for trace, rank, autocorrelation, energy, and pair-plot datasets. */
public final class DiagnosticGraphs {
	private DiagnosticGraphs() {}

	public static ChartSpec trace(String parameter, int coordinate,
			ChainResult... chains) {
		validate(coordinate, chains);
		List<ChartSpec.Series> series = new ArrayList<ChartSpec.Series>();
		for (int chain = 0; chain < chains.length; chain++) {
			ChainResult result = chains[chain];
			double[] x = sequence(result.size());
			double[] y = new double[result.size()];
			for (int draw = 0; draw < result.size(); draw++) y[draw] = result.valueAt(draw, coordinate);
			series.add(new ChartSpec.Series("chain " + (chain + 1), x, y));
		}
		return new ChartSpec("Trace: " + parameter, "retained draw", parameter,
				ChartSpec.Type.LINE, series);
	}

	public static ChartSpec ranks(String parameter, int coordinate, int bins,
			ChainResult... chains) {
		validate(coordinate, chains);
		if (bins < 2) throw new IllegalArgumentException("at least two rank bins are required");
		final List<Ranked> pooled = new ArrayList<Ranked>();
		for (int chain = 0; chain < chains.length; chain++) {
			for (int draw = 0; draw < chains[chain].size(); draw++)
				pooled.add(new Ranked(chains[chain].valueAt(draw, coordinate), chain));
		}
		pooled.sort(new Comparator<Ranked>() {
			@Override public int compare(Ranked first, Ranked second) {
				return Double.compare(first.value, second.value);
			}
		});
		double[][] counts = new double[chains.length][bins];
		for (int rank = 0; rank < pooled.size(); rank++) {
			int bin = Math.min(bins - 1, rank * bins / pooled.size());
			counts[pooled.get(rank).chain][bin]++;
		}
		double[] x = new double[bins];
		for (int bin = 0; bin < bins; bin++) x[bin] = bin + 1;
		List<ChartSpec.Series> series = new ArrayList<ChartSpec.Series>();
		for (int chain = 0; chain < chains.length; chain++)
			series.add(new ChartSpec.Series("chain " + (chain + 1), x, counts[chain]));
		return new ChartSpec("Rank histogram: " + parameter, "rank bin", "count",
				ChartSpec.Type.BAR, series);
	}

	public static ChartSpec autocorrelation(String parameter, int coordinate,
			int maximumLag, ChainResult... chains) {
		validate(coordinate, chains);
		if (maximumLag < 1) throw new IllegalArgumentException("maximum lag must be positive");
		List<ChartSpec.Series> series = new ArrayList<ChartSpec.Series>();
		for (int chain = 0; chain < chains.length; chain++) {
			double[] values = new double[chains[chain].size()];
			for (int draw = 0; draw < values.length; draw++) values[draw] = chains[chain].valueAt(draw, coordinate);
			int lags = Math.min(maximumLag, Math.max(1, values.length - 1));
			double[] x = new double[lags + 1];
			double[] y = new double[lags + 1];
			for (int lag = 0; lag <= lags; lag++) {
				x[lag] = lag; y[lag] = McmcDiagnostics.autocorrelation(values, lag);
			}
			series.add(new ChartSpec.Series("chain " + (chain + 1), x, y));
		}
		return new ChartSpec("Autocorrelation: " + parameter, "lag", "correlation",
				ChartSpec.Type.LINE, series);
	}

	public static ChartSpec energy(int bins, ChainResult... chains) {
		if (chains == null || chains.length == 0 || bins < 2)
			throw new IllegalArgumentException("chains and at least two bins are required");
		List<ChartSpec.Series> series = new ArrayList<ChartSpec.Series>();
		for (int chain = 0; chain < chains.length; chain++) {
			double[] values = new double[chains[chain].size()];
			int count = 0;
			for (int draw = 0; draw < chains[chain].size(); draw++) {
				IterationStats stat = chains[chain].statisticsAt(draw);
				if (Double.isFinite(stat.energy())) values[count++] = stat.energy();
			}
			values = Arrays.copyOf(values, count);
			double[][] histogram = histogram(values, bins);
			series.add(new ChartSpec.Series("chain " + (chain + 1), histogram[0], histogram[1]));
		}
		return new ChartSpec("Hamiltonian energy", "energy", "count",
				ChartSpec.Type.LINE, series);
	}

	public static ChartSpec pairs(String firstName, int firstCoordinate,
			String secondName, int secondCoordinate, ChainResult... chains) {
		validate(firstCoordinate, chains); validate(secondCoordinate, chains);
		List<ChartSpec.Series> series = new ArrayList<ChartSpec.Series>();
		for (int chain = 0; chain < chains.length; chain++) {
			ChainResult result = chains[chain];
			double[] x = new double[result.size()];
			double[] y = new double[result.size()];
			for (int draw = 0; draw < result.size(); draw++) {
				x[draw] = result.valueAt(draw, firstCoordinate);
				y[draw] = result.valueAt(draw, secondCoordinate);
			}
			series.add(new ChartSpec.Series("chain " + (chain + 1), x, y));
		}
		return new ChartSpec(firstName + " vs " + secondName, firstName, secondName,
				ChartSpec.Type.SCATTER, series);
	}

	private static final class Ranked {
		final double value; final int chain;
		Ranked(double value, int chain) { this.value = value; this.chain = chain; }
	}
	private static void validate(int coordinate, ChainResult[] chains) {
		if (chains == null || chains.length == 0)
			throw new IllegalArgumentException("at least one chain is required");
		for (ChainResult chain : chains)
			if (chain == null || coordinate < 0 || coordinate >= chain.dimension())
				throw new IllegalArgumentException("coordinate is outside a chain");
	}
	private static double[] sequence(int size) {
		double[] result = new double[size];
		for (int i = 0; i < size; i++) result[i] = i + 1;
		return result;
	}
	private static double[][] histogram(double[] values, int bins) {
		double[] x = new double[bins];
		double[] y = new double[bins];
		if (values.length == 0) return new double[][] {x, y};
		double minimum = values[0], maximum = values[0];
		for (double value : values) { minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }
		double width = maximum == minimum ? 1.0 : (maximum - minimum) / bins;
		for (int bin = 0; bin < bins; bin++) x[bin] = minimum + (bin + 0.5) * width;
		for (double value : values) {
			int bin = maximum == minimum ? bins / 2 : (int) ((value - minimum) / width);
			y[Math.max(0, Math.min(bins - 1, bin))]++;
		}
		return new double[][] {x, y};
	}
}
