/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.InflaterInputStream;

/** Selected-coordinate streaming sink with independently compressed, recoverable chunks. */
public final class ChunkedDrawSink implements DrawSink, AutoCloseable {
	private static final int MAGIC = 0x4a444453, VERSION = 1;
	private final DataOutputStream output; private final int[] coordinates; private final int chunkSize;
	private final List<Row> pending = new ArrayList<Row>(); private boolean closed;
	public ChunkedDrawSink(Path path, int[] coordinates, int chunkSize) throws IOException {
		if (path == null || coordinates == null || coordinates.length == 0 || chunkSize < 1) throw new IllegalArgumentException("path, coordinates, and chunk size are required");
		this.coordinates = coordinates.clone(); for (int coordinate : coordinates) if (coordinate < 0) throw new IllegalArgumentException("coordinates must be nonnegative");
		this.chunkSize = chunkSize; output = new DataOutputStream(Files.newOutputStream(path)); output.writeInt(MAGIC); output.writeInt(VERSION);
		output.writeInt(coordinates.length); for (int coordinate : coordinates) output.writeInt(coordinate);
	}
	@Override public synchronized void accept(int retainedIndex, double[] state, double logDensity, IterationStats statistics) {
		if (closed) throw new IllegalStateException("draw sink is closed"); double[] selected = new double[coordinates.length];
		for (int i = 0; i < coordinates.length; i++) { if (coordinates[i] >= state.length) throw new IllegalArgumentException("selected coordinate out of range"); selected[i] = state[coordinates[i]]; }
		pending.add(new Row(retainedIndex, selected, logDensity, statistics.accepted(), statistics.divergent()));
		if (pending.size() >= chunkSize) try { flushChunk(); } catch (IOException exception) { throw new UncheckedIOException(exception); }
	}
	private void flushChunk() throws IOException { if (pending.isEmpty()) return; ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		try (DataOutputStream chunk = new DataOutputStream(new DeflaterOutputStream(bytes))) { for (Row row : pending) { chunk.writeInt(row.index); chunk.writeDouble(row.logDensity);
			chunk.writeBoolean(row.accepted); chunk.writeBoolean(row.divergent); for (double value : row.values) chunk.writeDouble(value); } }
		byte[] compressed = bytes.toByteArray(); output.writeInt(pending.size()); output.writeInt(compressed.length); output.write(compressed); pending.clear(); }
	@Override public synchronized void close() throws IOException { if (!closed) { flushChunk(); output.writeInt(-1); output.close(); closed = true; } }
	public static ColumnarDraws read(Path path) throws IOException { try (DataInputStream input = new DataInputStream(Files.newInputStream(path))) {
		if (input.readInt() != MAGIC || input.readInt() != VERSION) throw new IOException("unsupported draw store"); int columns = input.readInt();
		if (columns < 1 || columns > 1_000_000) throw new IOException("invalid column count"); int[] coordinates = new int[columns]; for (int i = 0; i < columns; i++) coordinates[i] = input.readInt();
		List<Row> rows = new ArrayList<Row>(); while (true) { int count; try { count = input.readInt(); } catch (EOFException completeChunksRemainReadable) { break; }
			if (count == -1) break; int length = input.readInt(); if (count < 0 || length < 0 || length > 1_000_000_000) throw new IOException("invalid draw chunk"); byte[] compressed = new byte[length]; input.readFully(compressed);
			try (DataInputStream chunk = new DataInputStream(new InflaterInputStream(new ByteArrayInputStream(compressed)))) { for (int row = 0; row < count; row++) { int index = chunk.readInt(); double density = chunk.readDouble();
				boolean accepted = chunk.readBoolean(), divergent = chunk.readBoolean(); double[] values = new double[columns]; for (int column = 0; column < columns; column++) values[column] = chunk.readDouble(); rows.add(new Row(index, values, density, accepted, divergent)); } } }
		int[] indices = new int[rows.size()]; double[][] values = new double[columns][rows.size()]; double[] densities = new double[rows.size()]; boolean[] accepted = new boolean[rows.size()], divergent = new boolean[rows.size()];
		for (int row = 0; row < rows.size(); row++) { Row value = rows.get(row); indices[row] = value.index; densities[row] = value.logDensity; accepted[row] = value.accepted; divergent[row] = value.divergent;
			for (int column = 0; column < columns; column++) values[column][row] = value.values[column]; }
		return new ColumnarDraws(coordinates, indices, values, densities, accepted, divergent); } }
	private static final class Row { final int index; final double[] values; final double logDensity; final boolean accepted, divergent;
		Row(int index, double[] values, double logDensity, boolean accepted, boolean divergent) { this.index = index; this.values = values; this.logDensity = logDensity; this.accepted = accepted; this.divergent = divergent; } }
}
