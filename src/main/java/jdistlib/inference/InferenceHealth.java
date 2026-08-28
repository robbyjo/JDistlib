/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Actionable policy over sampler and posterior diagnostics. */
public final class InferenceHealth {
	private final List<HealthIssue> issues;
	private InferenceHealth(List<HealthIssue> issues) { this.issues = Collections.unmodifiableList(issues); }
	public static InferenceHealth assess(McmcDiagnosticReport report) {
		return assess(report, null, null);
	}
	public static InferenceHealth assess(McmcDiagnosticReport report, PathfinderFit pathfinder,
			GradientCheckResult gradientCheck) {
		if (report == null) throw new IllegalArgumentException("diagnostic report is required"); List<HealthIssue> result = new ArrayList<HealthIssue>(); SamplerDiagnostics sampler = report.sampler();
		if (sampler.numericalFailures() > 0) result.add(issue("NUMERICAL_FAILURE", HealthSeverity.ERROR, sampler.numericalFailures() + " failed chains", "Inspect initial values, support constraints, and non-finite factor profiles."));
		if (sampler.divergences() > 0) result.add(issue("DIVERGENCES", HealthSeverity.ERROR, sampler.divergences() + " divergent retained transitions", "Inspect divergence locations; reparameterize or reduce step size before interpreting draws."));
		if (sampler.treeDepthSaturations() > 0) result.add(issue("TREE_DEPTH", HealthSeverity.WARNING, sampler.treeDepthSaturations() + " saturated transitions", "Inspect geometry and ESS before increasing maximum tree depth."));
		if (Double.isFinite(sampler.energyBayesianFractionMissingInformation()) && sampler.energyBayesianFractionMissingInformation() < 0.3)
			result.add(issue("LOW_EBFMI", HealthSeverity.WARNING, "minimum E-BFMI=" + sampler.energyBayesianFractionMissingInformation(), "Check parameterization and marginal energy behavior; longer sampling alone may not help."));
		for (ParameterDiagnostics parameter : report.parameters()) { if (Double.isFinite(parameter.rHat()) && parameter.rHat() >= 1.01)
			result.add(issue("RHAT", HealthSeverity.ERROR, parameter.name() + " R-hat=" + parameter.rHat(), "Run more warmup from dispersed starts and inspect multimodality or non-identifiability."));
			if (parameter.bulkEffectiveSampleSize() < 100.0 || parameter.tailEffectiveSampleSize() < 100.0)
				result.add(issue("LOW_ESS", HealthSeverity.WARNING, parameter.name() + " bulk/tail ESS=" + parameter.bulkEffectiveSampleSize() + "/" + parameter.tailEffectiveSampleSize(), "Continue sampling to a quantity-specific MCSE goal after resolving convergence failures.")); }
		if (pathfinder != null && pathfinder.paretoK() > 0.7) result.add(issue("PARETO_K", pathfinder.paretoK() > 1.0 ? HealthSeverity.ERROR : HealthSeverity.WARNING,
				"Pathfinder Pareto-k=" + pathfinder.paretoK(), "Do not treat the weighted approximation as reliable; use its draws only as dispersed MCMC starts."));
		if (gradientCheck != null && !gradientCheck.passed()) result.add(issue("GRADIENT_CHECK", HealthSeverity.ERROR,
				gradientCheck.message(), "Correct the analytic gradient at coordinate " + gradientCheck.worstCoordinate() + " before using HMC."));
		return new InferenceHealth(result);
	}
	private static HealthIssue issue(String code, HealthSeverity severity, String evidence, String remediation) { return new HealthIssue(code, severity, evidence, remediation); }
	public List<HealthIssue> issues() { return issues; } public boolean healthy() { for (HealthIssue issue : issues) if (issue.severity() != HealthSeverity.INFO) return false; return true; }
}
