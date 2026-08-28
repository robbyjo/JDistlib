/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Appends generated quantities to each state before forwarding it to another sink. */
public final class GeneratedQuantitySink implements DrawSink {
	private final DrawSink delegate; private final GeneratedQuantity[] quantities;
	public GeneratedQuantitySink(DrawSink delegate, GeneratedQuantity... quantities) {
		if (delegate == null || quantities == null) throw new IllegalArgumentException("delegate and quantities are required");
		this.delegate = delegate; this.quantities = quantities.clone(); for (GeneratedQuantity quantity : quantities) if (quantity == null || quantity.name() == null) throw new IllegalArgumentException("named quantities are required");
	}
	@Override public void accept(int retainedIndex, double[] state, double logDensity, IterationStats statistics) {
		double[] expanded = new double[state.length + quantities.length]; System.arraycopy(state, 0, expanded, 0, state.length);
		for (int i = 0; i < quantities.length; i++) expanded[state.length + i] = quantities[i].evaluate(state.clone());
		delegate.accept(retainedIndex, expanded, logDensity, statistics);
	}
	public String[] names() { String[] result = new String[quantities.length]; for (int i = 0; i < result.length; i++) result[i] = quantities[i].name(); return result; }
}
