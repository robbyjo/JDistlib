/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

/** Fixed-capacity selected-coordinate draw sink and zero-copy memory-mapped reader. */
public final class MappedDrawStore implements DrawSink, AutoCloseable {
	private static final int MAGIC = 0x4a444d53, VERSION = 1, COUNT_OFFSET = 12;
	private final RandomAccessFile file; private final FileChannel channel; private final MappedByteBuffer buffer;
	private final int[] coordinates; private final int maximumDraws, recordStart, recordSize; private int count; private boolean closed;
	public MappedDrawStore(Path path, int[] coordinates, int maximumDraws) throws IOException {
		if (path == null || coordinates == null || coordinates.length == 0 || maximumDraws < 1) throw new IllegalArgumentException("path, coordinates, and capacity are required");
		this.coordinates = coordinates.clone(); for (int coordinate : coordinates) if (coordinate < 0) throw new IllegalArgumentException("coordinates must be nonnegative");
		this.maximumDraws = maximumDraws; recordStart = 20 + 4 * coordinates.length; recordSize = 14 + 8 * coordinates.length;
		long bytes = recordStart + (long) recordSize * maximumDraws; if (bytes > Integer.MAX_VALUE) throw new IllegalArgumentException("one mapped segment cannot exceed 2 GiB");
		file = new RandomAccessFile(path.toFile(), "rw"); file.setLength(bytes); channel = file.getChannel(); buffer = channel.map(FileChannel.MapMode.READ_WRITE, 0, bytes); buffer.order(ByteOrder.LITTLE_ENDIAN);
		buffer.putInt(MAGIC).putInt(VERSION).putInt(maximumDraws).putInt(0).putInt(coordinates.length); for (int coordinate : coordinates) buffer.putInt(coordinate);
	}
	@Override public synchronized void accept(int retainedIndex, double[] state, double logDensity, IterationStats statistics) {
		if (closed) throw new IllegalStateException("mapped store is closed"); if (count >= maximumDraws) throw new IllegalStateException("mapped draw capacity exceeded");
		int offset = recordStart + count * recordSize; buffer.position(offset); buffer.putInt(retainedIndex).putDouble(logDensity).put((byte) (statistics.accepted() ? 1 : 0)).put((byte) (statistics.divergent() ? 1 : 0));
		for (int coordinate : coordinates) { if (coordinate >= state.length) throw new IllegalArgumentException("selected coordinate out of range"); buffer.putDouble(state[coordinate]); }
		count++; buffer.putInt(COUNT_OFFSET, count);
	}
	public synchronized void force() { if (closed) throw new IllegalStateException("mapped store is closed"); buffer.force(); }
	@Override public synchronized void close() throws IOException { if (!closed) { buffer.force(); unmap(buffer); channel.close(); file.close(); closed = true; } }
	public static ColumnarDraws read(Path path) throws IOException { try (RandomAccessFile file = new RandomAccessFile(path.toFile(), "r"); FileChannel channel = file.getChannel()) {
		long size = channel.size(); if (size < 20 || size > Integer.MAX_VALUE) throw new IOException("invalid mapped draw store size"); MappedByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size); ByteBuffer buffer = mapped.order(ByteOrder.LITTLE_ENDIAN);
		try {
		if (buffer.getInt() != MAGIC || buffer.getInt() != VERSION) throw new IOException("unsupported mapped draw store"); int maximum = buffer.getInt(), count = buffer.getInt(), columns = buffer.getInt();
		if (maximum < 1 || count < 0 || count > maximum || columns < 1) throw new IOException("invalid mapped draw header"); int[] coordinates = new int[columns]; for (int i = 0; i < columns; i++) coordinates[i] = buffer.getInt();
		int[] indices = new int[count]; double[] densities = new double[count]; boolean[] accepted = new boolean[count], divergent = new boolean[count]; double[][] values = new double[columns][count];
		for (int row = 0; row < count; row++) { indices[row] = buffer.getInt(); densities[row] = buffer.getDouble(); accepted[row] = buffer.get() != 0; divergent[row] = buffer.get() != 0;
			for (int column = 0; column < columns; column++) values[column][row] = buffer.getDouble(); }
		return new ColumnarDraws(coordinates, indices, values, densities, accepted, divergent);
		} finally { unmap(mapped); } } }
	private static void unmap(ByteBuffer value) {
		try { Class<?> unsafeClass = Class.forName("sun.misc.Unsafe"); Field field = unsafeClass.getDeclaredField("theUnsafe"); field.setAccessible(true);
			Object unsafe = field.get(null); Method cleaner = unsafeClass.getMethod("invokeCleaner", ByteBuffer.class); cleaner.invoke(unsafe, value); return;
		} catch (ReflectiveOperationException unavailableOnThisVm) { /* Java 8 fallback below. */ }
		try { Method cleanerMethod = value.getClass().getMethod("cleaner"); cleanerMethod.setAccessible(true); Object cleaner = cleanerMethod.invoke(value);
			if (cleaner != null) cleaner.getClass().getMethod("clean").invoke(cleaner);
		} catch (ReflectiveOperationException unavailableOnThisVm) { /* GC will eventually release the mapping. */ }
	}
}
