/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.autodiff;

import java.util.Arrays;

/**
 * Allocation-conscious reverse-mode automatic-differentiation tape.
 *
 * <p>Nodes are represented by integer handles and stored in reusable primitive
 * arrays.  A caller normally invokes {@link #reset()} before every log-density
 * evaluation, creates the independent variables, builds a scalar result, and
 * calls {@link #reverse(int)}.  Resetting retains the allocated arena.</p>
 */
public final class ReverseTape {
	private static final int CONSTANT = 0;
	private static final int UNARY = 1;
	private static final int BINARY = 2;

	private double[] values;
	private double[] adjoints;
	private double[] leftPartials;
	private double[] rightPartials;
	private int[] leftParents;
	private int[] rightParents;
	private byte[] arities;
	private int size;

	public ReverseTape() { this(1024); }

	public ReverseTape(int initialCapacity) {
		if (initialCapacity < 1) throw new IllegalArgumentException("positive capacity required");
		values = new double[initialCapacity];
		adjoints = new double[initialCapacity];
		leftPartials = new double[initialCapacity];
		rightPartials = new double[initialCapacity];
		leftParents = new int[initialCapacity];
		rightParents = new int[initialCapacity];
		arities = new byte[initialCapacity];
	}

	/** Clears all nodes while retaining the arena capacity. */
	public void reset() {
		Arrays.fill(adjoints, 0, size, 0.0);
		size = 0;
	}

	/** Current node count, useful for instrumentation and capacity planning. */
	public int size() { return size; }

	/** Current reusable arena capacity. */
	public int capacity() { return values.length; }

	/** Returns a mark to which temporary nodes can later be rewound. */
	public int mark() { return size; }

	/** Discards nodes created after {@code mark}, retaining their storage. */
	public void rewind(int mark) {
		if (mark < 0 || mark > size) throw new IllegalArgumentException("invalid tape mark");
		Arrays.fill(adjoints, mark, size, 0.0);
		size = mark;
	}

	public int variable(double value) { return leaf(value); }
	public int constant(double value) { return leaf(value); }

	public double value(int handle) { checkHandle(handle); return values[handle]; }
	public double adjoint(int handle) { checkHandle(handle); return adjoints[handle]; }

	public int add(int a, int b) { return binary(a, b, values[a] + values[b], 1.0, 1.0); }
	public int add(int a, double b) { return unary(a, values[a] + b, 1.0); }
	public int subtract(int a, int b) { return binary(a, b, values[a] - values[b], 1.0, -1.0); }
	public int subtract(double a, int b) { return unary(b, a - values[b], -1.0); }
	public int multiply(int a, int b) {
		return binary(a, b, values[a] * values[b], values[b], values[a]);
	}
	public int multiply(int a, double b) { return unary(a, values[a] * b, b); }
	public int divide(int a, int b) {
		double denominator = values[b];
		return binary(a, b, values[a] / denominator, 1.0 / denominator,
				-values[a] / (denominator * denominator));
	}
	public int divide(int a, double b) { return unary(a, values[a] / b, 1.0 / b); }
	public int negate(int a) { return unary(a, -values[a], -1.0); }
	public int exp(int a) { double v = Math.exp(values[a]); return unary(a, v, v); }
	public int expm1(int a) { return unary(a, Math.expm1(values[a]), Math.exp(values[a])); }
	public int log(int a) { return unary(a, Math.log(values[a]), 1.0 / values[a]); }
	public int log1p(int a) { return unary(a, Math.log1p(values[a]), 1.0 / (1.0 + values[a])); }
	public int sqrt(int a) { double v = Math.sqrt(values[a]); return unary(a, v, 0.5 / v); }
	public int sin(int a) { return unary(a, Math.sin(values[a]), Math.cos(values[a])); }
	public int cos(int a) { return unary(a, Math.cos(values[a]), -Math.sin(values[a])); }
	public int tanh(int a) {
		double v = Math.tanh(values[a]); return unary(a, v, 1.0 - v * v);
	}
	public int pow(int a, double exponent) {
		double v = Math.pow(values[a], exponent);
		return unary(a, v, exponent * Math.pow(values[a], exponent - 1.0));
	}
	public int pow(int a, int b) {
		double v = Math.pow(values[a], values[b]);
		return binary(a, b, v, values[b] * Math.pow(values[a], values[b] - 1.0),
				v * Math.log(values[a]));
	}

	/** Runs one reverse sweep with unit seed at {@code output}. */
	public void reverse(int output) {
		checkHandle(output);
		Arrays.fill(adjoints, 0, size, 0.0);
		adjoints[output] = 1.0;
		for (int node = output; node >= 0; node--) {
			double seed = adjoints[node];
			if (seed == 0.0 || arities[node] == CONSTANT) continue;
			adjoints[leftParents[node]] += seed * leftPartials[node];
			if (arities[node] == BINARY)
				adjoints[rightParents[node]] += seed * rightPartials[node];
		}
	}

	private int leaf(double value) {
		int node = allocate(); values[node] = value; arities[node] = CONSTANT; return node;
	}
	private int unary(int parent, double value, double partial) {
		checkHandle(parent); int node = allocate(); values[node] = value;
		arities[node] = UNARY; leftParents[node] = parent; leftPartials[node] = partial;
		return node;
	}
	private int binary(int left, int right, double value, double leftPartial,
			double rightPartial) {
		checkHandle(left); checkHandle(right); int node = allocate(); values[node] = value;
		arities[node] = BINARY; leftParents[node] = left; rightParents[node] = right;
		leftPartials[node] = leftPartial; rightPartials[node] = rightPartial;
		return node;
	}
	private int allocate() {
		if (size == values.length) grow();
		adjoints[size] = 0.0; return size++;
	}
	private void grow() {
		int capacity = values.length < 1_048_576 ? values.length * 2
				: values.length + values.length / 2;
		values = Arrays.copyOf(values, capacity); adjoints = Arrays.copyOf(adjoints, capacity);
		leftPartials = Arrays.copyOf(leftPartials, capacity);
		rightPartials = Arrays.copyOf(rightPartials, capacity);
		leftParents = Arrays.copyOf(leftParents, capacity);
		rightParents = Arrays.copyOf(rightParents, capacity);
		arities = Arrays.copyOf(arities, capacity);
	}
	private void checkHandle(int handle) {
		if (handle < 0 || handle >= size) throw new IllegalArgumentException("invalid tape handle");
	}
}
