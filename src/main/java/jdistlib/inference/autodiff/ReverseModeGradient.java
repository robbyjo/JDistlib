/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.autodiff;

/** Reusable reverse-mode value-and-gradient evaluator. */
public final class ReverseModeGradient {
	private final ReverseTape tape;
	private int[] handles = new int[0];

	public ReverseModeGradient() { this(new ReverseTape()); }
	public ReverseModeGradient(ReverseTape tape) {
		if (tape == null) throw new IllegalArgumentException("tape is required");
		this.tape = tape;
	}

	public ReverseTape tape() { return tape; }

	/** Evaluates the function and writes its gradient without allocating tape nodes. */
	public double evaluate(ReverseDifferentiableFunction function, double[] position,
			double[] gradient) {
		if (function == null || position == null || gradient == null
				|| gradient.length != position.length)
			throw new IllegalArgumentException("function, position, and matching gradient are required");
		tape.reset();
		if (handles.length < position.length) handles = new int[position.length];
		for (int i = 0; i < position.length; i++) handles[i] = tape.variable(position[i]);
		int output = function.evaluate(tape, handles);
		tape.reverse(output);
		for (int i = 0; i < position.length; i++) gradient[i] = tape.adjoint(handles[i]);
		return tape.value(output);
	}
}
