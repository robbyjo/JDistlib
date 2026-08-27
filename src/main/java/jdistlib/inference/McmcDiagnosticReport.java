/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable parameter and sampler diagnostics with machine-readable output. */
public final class McmcDiagnosticReport {
	private final List<ParameterDiagnostics> parameters;
	private final SamplerDiagnostics sampler;
	private final List<String> warnings;
	private final int chains;
	private final int drawsPerChain;

	McmcDiagnosticReport(List<ParameterDiagnostics> parameters,
			SamplerDiagnostics sampler, List<String> warnings, int chains,
			int drawsPerChain) {
		this.parameters = Collections.unmodifiableList(
				new ArrayList<ParameterDiagnostics>(parameters));
		this.sampler = sampler;
		this.warnings = Collections.unmodifiableList(new ArrayList<String>(warnings));
		this.chains = chains; this.drawsPerChain = drawsPerChain;
	}
	public List<ParameterDiagnostics> parameters() { return parameters; }
	public ParameterDiagnostics parameter(String name) {
		for (ParameterDiagnostics result : parameters)
			if (result.name().equals(name)) return result;
		throw new IllegalArgumentException("unknown parameter diagnostic: " + name);
	}
	public SamplerDiagnostics sampler() { return sampler; }
	public List<String> warnings() { return warnings; }
	public int chains() { return chains; }
	public int drawsPerChain() { return drawsPerChain; }
	public boolean reliable() {
		if (!sampler.healthy()) return false;
		for (ParameterDiagnostics parameter : parameters)
			if (!parameter.reliable()) return false;
		return true;
	}
	public String toJson() { return McmcJson.toJson(this); }
}
