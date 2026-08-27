/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable bipartite parameter/factor graph for inspection and rendering. */
public final class ModelGraph {
	public enum NodeKind { PARAMETER, FACTOR, DATA }
	public static final class Node {
		private final String id;
		private final String label;
		private final NodeKind kind;
		Node(String id, String label, NodeKind kind) {
			this.id = id; this.label = label; this.kind = kind;
		}
		public String id() { return id; }
		public String label() { return label; }
		public NodeKind kind() { return kind; }
	}
	public static final class Edge {
		private final String from;
		private final String to;
		Edge(String from, String to) { this.from = from; this.to = to; }
		public String from() { return from; }
		public String to() { return to; }
	}
	private final List<Node> nodes;
	private final List<Edge> edges;
	ModelGraph(List<Node> nodes, List<Edge> edges) {
		this.nodes = Collections.unmodifiableList(new ArrayList<Node>(nodes));
		this.edges = Collections.unmodifiableList(new ArrayList<Edge>(edges));
	}
	public List<Node> nodes() { return nodes; }
	public List<Edge> edges() { return edges; }
}
