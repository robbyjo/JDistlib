/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

/** Crash-safe segment export for ragged sparse draws. */
public final class SparseSubsetExport {
	private SparseSubsetExport() {}
	public static void writeTidySegmentAtomic(Path path, SparseSubsetResult result,
			SparseSubsetTarget target, long firstRetainedIndex) throws IOException {
		if (path == null || result == null || target == null || firstRetainedIndex < 0L) throw new IllegalArgumentException("path, result, target, and first index required");
		Path absolute = path.toAbsolutePath(), parent = absolute.getParent(); if (parent != null) Files.createDirectories(parent);
		Path temporary = absolute.resolveSibling(absolute.getFileName().toString() + ".tmp");
		try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.CREATE,
				StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
				BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
						Channels.newOutputStream(channel), StandardCharsets.UTF_8))) {
			out.write("retained_draw\tmodel_size\tmodel_key\tparameter_type\tparameter\tvalue\tlog_joint\n");
			for (int draw = 0; draw < result.size(); draw++) {
				SparseSubsetState state = result.draw(draw); long retained = firstRetainedIndex + draw;
				for (int common = 0; common < state.commonDimension(); common++) row(out, retained, state, "common", target.commonParameterName(common), state.commonParameter(common), result.logJointAt(draw));
				for (int coefficient = 0; coefficient < state.size(); coefficient++) row(out, retained, state, "candidate", target.candidateName(state.activeCandidate(coefficient)), state.coefficient(coefficient), result.logJointAt(draw));
			}
			out.flush(); channel.force(true);
		}
		try { Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING); }
		catch (AtomicMoveNotSupportedException unsupported) { Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING); }
	}
	private static void row(BufferedWriter out, long retained, SparseSubsetState state, String type,
			String parameter, double value, double logJoint) throws IOException {
		out.write(Long.toString(retained)); out.write('\t'); out.write(Integer.toString(state.size())); out.write('\t');
		out.write(state.modelKey()); out.write('\t'); out.write(type); out.write('\t'); out.write(escape(parameter)); out.write('\t');
		out.write(Double.toString(value)); out.write('\t'); out.write(Double.toString(logJoint)); out.write('\n');
	}
	private static String escape(String value) { return value.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r"); }
}
