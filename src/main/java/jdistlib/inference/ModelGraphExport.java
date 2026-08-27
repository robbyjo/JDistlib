/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Graphviz DOT and versioned JSON export for model dependency graphs. */
public final class ModelGraphExport {
	private ModelGraphExport() {}
	public static String toDot(ModelGraph graph) {
		if (graph == null) throw new IllegalArgumentException("graph is required");
		StringBuilder dot = new StringBuilder("digraph model {\n  rankdir=LR;\n");
		for (ModelGraph.Node node : graph.nodes()) {
			dot.append("  \"").append(dot(node.id())).append("\" [label=\"")
					.append(dot(node.label())).append("\", shape=")
					.append(node.kind() == ModelGraph.NodeKind.FACTOR ? "box" : "ellipse")
					.append(node.kind() == ModelGraph.NodeKind.DATA ? ", style=filled, fillcolor=\"#eeeeee\"" : "")
					.append("];\n");
		}
		for (ModelGraph.Edge edge : graph.edges()) dot.append("  \"")
				.append(dot(edge.from())).append("\" -> \"").append(dot(edge.to())).append("\";\n");
		return dot.append("}\n").toString();
	}
	public static String toJson(ModelGraph graph) {
		if (graph == null) return "null";
		StringBuilder json = new StringBuilder();
		json.append('{'); McmcJson.string(json, "schema").append(':');
		McmcJson.string(json, "jdistlib.model-graph/1").append(',');
		McmcJson.string(json, "nodes").append(':').append('[');
		for (int i = 0; i < graph.nodes().size(); i++) {
			if (i > 0) json.append(','); ModelGraph.Node node = graph.nodes().get(i);
			json.append('{'); field(json, "id", node.id()).append(',');
			field(json, "label", node.label()).append(','); field(json, "kind", node.kind().name()).append('}');
		}
		json.append(']').append(','); McmcJson.string(json, "edges").append(':').append('[');
		for (int i = 0; i < graph.edges().size(); i++) {
			if (i > 0) json.append(','); ModelGraph.Edge edge = graph.edges().get(i);
			json.append('{'); field(json, "from", edge.from()).append(',');
			field(json, "to", edge.to()).append('}');
		}
		return json.append(']').append('}').toString();
	}
	private static StringBuilder field(StringBuilder json, String name, String value) {
		McmcJson.string(json, name).append(':'); return McmcJson.string(json, value);
	}
	private static String dot(String value) {
		return value.replace("\\", "\\\\").replace("\"", "\\\"")
				.replace("\n", "\\n").replace("\r", "");
	}
}
