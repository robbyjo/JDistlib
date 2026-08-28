/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Progress listener retaining lightweight step-size and schedule traces. */
public final class WarmupTrace implements ProgressListener {
	public static final class Entry {
		private final int iteration;
		private final double stepSize;
		private final WarmupSchedule.Phase stage;
		private final double metricConditionNumber;
		Entry(int iteration, double stepSize, WarmupSchedule.Phase stage,
				double metricConditionNumber) {
			this.iteration = iteration; this.stepSize = stepSize; this.stage = stage;
			this.metricConditionNumber = metricConditionNumber;
		}
		public int iteration() { return iteration; }
		public double stepSize() { return stepSize; }
		public WarmupSchedule.Phase stage() { return stage; }
		public double metricConditionNumber() { return metricConditionNumber; }
	}
	private final int warmupIterations;
	private final WarmupSchedule schedule;
	private final List<Entry> entries = new ArrayList<Entry>();
	public WarmupTrace(int warmupIterations, WarmupSchedule schedule) {
		if (warmupIterations < 0 || schedule == null) throw new IllegalArgumentException("invalid trace configuration");
		this.warmupIterations = warmupIterations; this.schedule = schedule;
	}
	@Override public void update(int completedIterations, int totalIterations,
			boolean warmup, IterationStats statistics) {
		if (warmup) entries.add(new Entry(completedIterations, statistics.stepSize(),
				schedule.resolve(warmupIterations).phase(completedIterations - 1),
				statistics.metricConditionNumber()));
	}
	public List<Entry> entries() { return Collections.unmodifiableList(new ArrayList<Entry>(entries)); }
}
