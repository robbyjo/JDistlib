/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Stan-style fast/slow/final warmup schedule with expanding metric windows. */
public final class WarmupSchedule {
	public enum Phase { INITIAL_FAST, SLOW, FINAL_FAST, COMPLETE }
	private final int initialBuffer;
	private final int terminalBuffer;
	private final int initialWindow;

	public WarmupSchedule(int initialBuffer, int terminalBuffer, int initialWindow) {
		if (initialBuffer < 0 || terminalBuffer < 0 || initialWindow < 1)
			throw new IllegalArgumentException("invalid warmup schedule");
		this.initialBuffer = initialBuffer;
		this.terminalBuffer = terminalBuffer;
		this.initialWindow = initialWindow;
	}
	public static WarmupSchedule stanDefault() { return new WarmupSchedule(75, 50, 25); }
	public int initialBuffer() { return initialBuffer; }
	public int terminalBuffer() { return terminalBuffer; }
	public int initialWindow() { return initialWindow; }

	/** Resolves the requested schedule to a particular warmup length. */
	public Resolved resolve(int warmup) {
		if (warmup < 0) throw new IllegalArgumentException("warmup must be nonnegative");
		if (warmup == 0) return new Resolved(0, 0, 0, new int[0]);
		int initial = initialBuffer;
		int terminal = terminalBuffer;
		int firstWindow = initialWindow;
		if (initial + terminal + firstWindow > warmup) {
			initial = (int) Math.floor(0.15 * warmup);
			terminal = (int) Math.floor(0.10 * warmup);
			firstWindow = Math.max(1, warmup - initial - terminal);
		}
		int slow = Math.max(0, warmup - initial - terminal);
		int count = 0;
		for (int used = 0, width = firstWindow; used < slow; width *= 2) {
			used += Math.min(width, slow - used); count++;
		}
		int[] ends = new int[count];
		int used = 0;
		int width = firstWindow;
		for (int i = 0; i < count; i++) {
			int remaining = slow - used;
			int current = Math.min(width, remaining);
			if (remaining > current && remaining < 2 * current) current = remaining;
			used += current;
			ends[i] = initial + used;
			width *= 2;
		}
		return new Resolved(warmup, initial, terminal, ends);
	}

	public static final class Resolved {
		private final int warmup;
		private final int initial;
		private final int terminal;
		private final int[] slowEnds;
		Resolved(int warmup, int initial, int terminal, int[] slowEnds) {
			this.warmup = warmup; this.initial = initial;
			this.terminal = terminal; this.slowEnds = slowEnds;
		}
		public Phase phase(int zeroBasedIteration) {
			if (zeroBasedIteration >= warmup) return Phase.COMPLETE;
			if (zeroBasedIteration < initial) return Phase.INITIAL_FAST;
			if (zeroBasedIteration >= warmup - terminal) return Phase.FINAL_FAST;
			return Phase.SLOW;
		}
		public boolean endsSlowWindow(int completedIterations) {
			for (int end : slowEnds) if (end == completedIterations) return true;
			return false;
		}
		public int[] slowWindowEnds() { return slowEnds.clone(); }
		public int initialBuffer() { return initial; }
		public int terminalBuffer() { return terminal; }
	}
}
