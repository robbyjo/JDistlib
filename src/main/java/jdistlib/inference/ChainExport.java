/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Versioned JSON and tidy CSV interchange for retained chain draws. */
public final class ChainExport {
	private ChainExport() {}
	public static String toCsv(String[] names, ChainResult... chains) {
		validate(names, chains);
		StringBuilder csv = new StringBuilder("chain,draw");
		for (String name : names) csv.append(',').append(csv(name));
		csv.append(",log_density,acceptance,divergent,tree_depth,leapfrog_steps\n");
		for (int chain = 0; chain < chains.length; chain++) {
			ChainResult result = chains[chain];
			for (int draw = 0; draw < result.size(); draw++) {
				csv.append(chain + 1).append(',').append(draw + 1);
				for (int coordinate = 0; coordinate < result.dimension(); coordinate++)
					csv.append(',').append(result.valueAt(draw, coordinate));
				IterationStats stats = result.statisticsAt(draw);
				csv.append(',').append(result.logDensityAt(draw)).append(',')
						.append(stats.acceptanceProbability()).append(',')
						.append(stats.divergent()).append(',')
						.append(stats.treeDepth()).append(',')
						.append(stats.leapfrogSteps()).append('\n');
			}
		}
		return csv.toString();
	}
	public static String toJson(String[] names, ChainResult... chains) {
		validate(names, chains);
		StringBuilder json = new StringBuilder();
		json.append('{'); McmcJson.string(json, "schema").append(':');
		McmcJson.string(json, "jdistlib.chains/1").append(',');
		McmcJson.string(json, "parameters").append(':').append('[');
		for (int i = 0; i < names.length; i++) {
			if (i > 0) json.append(','); McmcJson.string(json, names[i]);
		}
		json.append(']').append(','); McmcJson.string(json, "chains").append(':').append('[');
		for (int chain = 0; chain < chains.length; chain++) {
			if (chain > 0) json.append(','); json.append('[');
			ChainResult result = chains[chain];
			for (int draw = 0; draw < result.size(); draw++) {
				if (draw > 0) json.append(','); json.append('[');
				for (int coordinate = 0; coordinate < result.dimension(); coordinate++) {
					if (coordinate > 0) json.append(',');
					json.append(result.valueAt(draw, coordinate));
				}
				json.append(']');
			}
			json.append(']');
		}
		return json.append(']').append('}').toString();
	}
	private static void validate(String[] names, ChainResult[] chains) {
		if (names == null || chains == null || chains.length == 0)
			throw new IllegalArgumentException("parameter names and chains are required");
		for (ChainResult chain : chains)
			if (chain == null || chain.dimension() != names.length)
				throw new IllegalArgumentException("parameter names and chain dimensions must match");
	}
	private static String csv(String value) { return "\"" + value.replace("\"", "\"\"") + "\""; }
}
