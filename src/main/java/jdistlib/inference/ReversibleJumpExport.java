/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Tidy CSV export for ragged model-specific parameters. */
public final class ReversibleJumpExport {
	private ReversibleJumpExport() {}
	public static String toTidyCsv(ReversibleJumpResult result, ReversibleJumpTarget target) {
		if (result == null || target == null) throw new IllegalArgumentException("result and target required");
		StringBuilder out = new StringBuilder("draw,model_id,model_name,parameter,value,log_joint\n");
		for (int draw = 0; draw < result.size(); draw++) {
			ReversibleJumpState state = result.draw(draw); ReversibleJumpModelSpace space = target.modelSpace(state.modelId());
			if (state.dimension() == 0) row(out, draw, state.modelId(), space.name(), "", Double.NaN, result.logJointAt(draw));
			else for (int parameter = 0; parameter < state.dimension(); parameter++) row(out, draw, state.modelId(), space.name(),
					space.parameterName(parameter), state.parameter(parameter), result.logJointAt(draw));
		}
		return out.toString();
	}
	private static void row(StringBuilder out, int draw, long model, String modelName, String parameter, double value, double logJoint) {
		out.append(draw).append(',').append(model).append(',').append(csv(modelName)).append(',').append(csv(parameter)).append(',');
		if (Double.isFinite(value)) out.append(value); out.append(',').append(logJoint).append('\n');
	}
	private static String csv(String value) { return '"' + value.replace("\"", "\"\"") + '"'; }
}
