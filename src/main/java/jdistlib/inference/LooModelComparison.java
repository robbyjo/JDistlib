/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Pointwise, paired comparison of models evaluated with PSIS-LOO. */
public final class LooModelComparison {
	private LooModelComparison() {}
	public static NamedResult model(String name, PsisLoo.Result result) { return new NamedResult(name, result); }
	public static List<Entry> compare(NamedResult... models) {
		if (models == null || models.length < 2) throw new IllegalArgumentException("at least two models required");
		ObservationMetadata metadata = models[0].result.metadata();
		for (NamedResult model : models) {
			if (model == null || !metadata.equals(model.result.metadata()))
				throw new IllegalArgumentException("models must use identical observation metadata");
		}
		List<NamedResult> sorted = new ArrayList<NamedResult>();
		Collections.addAll(sorted, models);
		Collections.sort(sorted, new Comparator<NamedResult>() {
			@Override public int compare(NamedResult first, NamedResult second) {
				return -Double.compare(first.result.elpd(), second.result.elpd());
			}
		});
		double[] best = sorted.get(0).result.pointwiseElpd();
		List<Entry> entries = new ArrayList<Entry>();
		for (NamedResult model : sorted) {
			double[] candidate = model.result.pointwiseElpd(), differences = new double[best.length];
			for (int i = 0; i < best.length; i++) differences[i] = candidate[i] - best[i];
			entries.add(new Entry(model.name, model.result.elpd(), model.result.elpd() - sorted.get(0).result.elpd(),
					Math.sqrt(best.length * PredictiveMath.sampleVariance(differences)), model.result.reliable()));
		}
		return Collections.unmodifiableList(entries);
	}
	public static final class NamedResult {
		private final String name; private final PsisLoo.Result result;
		private NamedResult(String name, PsisLoo.Result result) {
			if (name == null || name.trim().isEmpty() || result == null) throw new IllegalArgumentException("name and result required");
			this.name = name; this.result = result;
		}
	}
	public static final class Entry {
		private final String name; private final double elpd, difference, differenceStandardError; private final boolean reliable;
		private Entry(String name, double elpd, double difference, double differenceStandardError, boolean reliable) {
			this.name = name; this.elpd = elpd; this.difference = difference;
			this.differenceStandardError = differenceStandardError; this.reliable = reliable;
		}
		public String name() { return name; }
		public double elpd() { return elpd; }
		public double differenceFromBest() { return difference; }
		public double differenceStandardError() { return differenceStandardError; }
		public boolean reliable() { return reliable; }
	}
}
