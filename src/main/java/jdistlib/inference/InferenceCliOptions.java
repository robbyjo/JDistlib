/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.List;

import jdistlib.accelerator.Compute;

/** Parses reusable compute switches for command-line applications embedding JDistlib. */
public final class InferenceCliOptions {
	private final Compute compute;
	private final ComputeNuts nuts;
	private final String[] remaining;
	private InferenceCliOptions(Compute compute, ComputeNuts nuts, String[] remaining) {
		this.compute = compute; this.nuts = nuts; this.remaining = remaining;
	}
	/**
	 * Recognizes {@code --compute}, {@code --nuts-offload}, and {@code --gpu-nuts};
	 * unrecognized arguments are retained for the host application.
	 */
	public static InferenceCliOptions parse(String[] arguments) {
		if (arguments == null) throw new IllegalArgumentException("arguments are required");
		Compute compute = Compute.parse(System.getProperty("jdistlib.compute.backend", "auto"));
		ComputeNuts nuts = ComputeNuts.parse(System.getProperty("jdistlib.compute.nuts", "auto"));
		List<String> remaining = new ArrayList<String>();
		for (int index = 0; index < arguments.length; index++) {
			String argument = arguments[index];
			if ("--gpu-nuts".equals(argument)) { compute = Compute.GPU; nuts = ComputeNuts.FORCE; }
			else if (argument.startsWith("--compute="))
				compute = Compute.parse(argument.substring("--compute=".length()));
			else if ("--compute".equals(argument)) {
				if (++index == arguments.length) throw new IllegalArgumentException("--compute requires a value");
				compute = Compute.parse(arguments[index]);
			} else if (argument.startsWith("--nuts-offload="))
				nuts = ComputeNuts.parse(argument.substring("--nuts-offload=".length()));
			else if ("--nuts-offload".equals(argument)) {
				if (++index == arguments.length) throw new IllegalArgumentException("--nuts-offload requires a value");
				nuts = ComputeNuts.parse(arguments[index]);
			} else remaining.add(argument);
		}
		return new InferenceCliOptions(compute, nuts,
				remaining.toArray(new String[remaining.size()]));
	}
	public Compute computeBackend() { return compute; }
	public ComputeNuts nutsBackend() { return nuts; }
	public String[] remainingArguments() { return remaining.clone(); }
	/** Applies parsed compute switches without replacing other builder settings. */
	public SamplingOptions.Builder applyTo(SamplingOptions.Builder builder) {
		if (builder == null) throw new IllegalArgumentException("sampling builder is required");
		return builder.computeBackend(compute).nutsBackend(nuts);
	}
}
