/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

import jdistlib.rng.RandomEngine;

/** Checksummed, forced, atomic persistence for complete sparse RJ restart state. */
public final class SparseSubsetCheckpointIO {
	private static final int MAGIC = 0x4a445352, VERSION = 1;
	private SparseSubsetCheckpointIO() {}
	public static void writeAtomic(Path path, SparseSubsetCheckpoint checkpoint, String modelFingerprint,
			String optionsFingerprint) throws IOException {
		if (path == null || checkpoint == null || modelFingerprint == null || optionsFingerprint == null)
			throw new IllegalArgumentException("path, checkpoint, and fingerprints required");
		Path absolute = path.toAbsolutePath(), parent = absolute.getParent(); if (parent != null) Files.createDirectories(parent);
		Path temporary = absolute.resolveSibling(absolute.getFileName().toString() + ".tmp"); byte[] fileBytes = encode(checkpoint, modelFingerprint, optionsFingerprint);
		try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
				DataOutputStream output = new DataOutputStream(Channels.newOutputStream(channel))) {
			output.write(fileBytes); output.flush(); channel.force(true);
		}
		try { Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING); }
		catch (AtomicMoveNotSupportedException unsupported) { Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING); }
	}
	public static PortableSparseSubsetCheckpoint read(Path path, String expectedModelFingerprint,
			String expectedOptionsFingerprint) throws IOException {
		try (DataInputStream file = new DataInputStream(Files.newInputStream(path))) {
			if (file.readInt() != MAGIC) throw new IOException("not a JDistlib sparse RJ checkpoint"); int length = file.readInt();
			if (length < 0 || length > 1_000_000_000) throw new IOException("invalid sparse RJ checkpoint length"); byte[] payload = new byte[length]; file.readFully(payload);
			byte[] checksum = new byte[32]; file.readFully(checksum); if (!Arrays.equals(checksum, sha256(payload))) throw new IOException("sparse RJ checkpoint checksum mismatch");
			DataInputStream in = new DataInputStream(new ByteArrayInputStream(payload)); if (in.readInt() != VERSION) throw new IOException("unsupported sparse RJ checkpoint version");
			String model = in.readUTF(), options = in.readUTF(), platform = in.readUTF();
			if (expectedModelFingerprint != null && !expectedModelFingerprint.equals(model)) throw new IOException("sparse RJ checkpoint model fingerprint mismatch");
			if (expectedOptionsFingerprint != null && !expectedOptionsFingerprint.equals(options)) throw new IOException("sparse RJ checkpoint options fingerprint mismatch");
			SparseSubsetState state = new SparseSubsetState(readInts(in, 1000000, "active candidates"), readDoubles(in, 1000000, "common parameters"), readDoubles(in, 1000000, "coefficients"));
			double logJoint = in.readDouble(); long completed = in.readLong(), retained = in.readLong(), warmupIterations = in.readLong(); boolean warmupComplete = in.readBoolean();
			String randomClass = in.readUTF(); int randomLength = checkedLength(in.readInt(), 100_000_000, "RNG state"); byte[] randomBytes = new byte[randomLength]; in.readFully(randomBytes);
			RandomEngine random = deserialize(randomBytes, randomClass); String[] names = readStrings(in, 1000, "move names");
			SparseSubsetCheckpoint checkpoint = new SparseSubsetCheckpoint(state, logJoint, completed, retained, random, names,
					readDoubles(in, 1000, "move weights"), readDoubles(in, 1000000, "scales"), readLongs(in, 1000000, "scale updates"),
					readLongs(in, 1000, "move attempts"), readLongs(in, 1000, "move accepts"), readLongs(in, 1000, "invalid proposals"),
					readLongs(in, 1000000, "model sizes"), readLongs(in, 10000000, "inclusion counts"), readLongs(in, 10000000, "coefficient counts"),
					readDoubles(in, 10000000, "coefficient sums"), readDoubles(in, 10000000, "coefficient squares"),
					readDoubles(in, 1000000, "common sums"), readDoubles(in, 1000000, "common squares"), warmupIterations, warmupComplete);
			return new PortableSparseSubsetCheckpoint(checkpoint, model, options, platform);
		}
	}
	private static byte[] encode(SparseSubsetCheckpoint checkpoint, String model, String options) throws IOException {
		ByteArrayOutputStream payloadBytes = new ByteArrayOutputStream(); DataOutputStream out = new DataOutputStream(payloadBytes);
		out.writeInt(VERSION); out.writeUTF(model); out.writeUTF(options); out.writeUTF(System.getProperty("os.name", "unknown") + "/" + System.getProperty("os.arch", "unknown") + "/Java " + System.getProperty("java.version", "unknown"));
		writeInts(out, checkpoint.state().activeCandidates()); writeDoubles(out, checkpoint.state().commonParameters()); writeDoubles(out, checkpoint.state().coefficients());
		out.writeDouble(checkpoint.logJoint()); out.writeLong(checkpoint.completedTransitions()); out.writeLong(checkpoint.retainedDraws()); out.writeLong(checkpoint.warmupIterations()); out.writeBoolean(checkpoint.warmupComplete());
		RandomEngine random = checkpoint.random(); out.writeUTF(random.getClass().getName()); byte[] randomBytes = serialize(random); out.writeInt(randomBytes.length); out.write(randomBytes);
		writeStrings(out, checkpoint.moveNames()); writeDoubles(out, checkpoint.moveWeights()); writeDoubles(out, checkpoint.logScales()); writeLongs(out, checkpoint.scaleUpdates());
		writeLongs(out, checkpoint.moveAttempts()); writeLongs(out, checkpoint.moveAccepts()); writeLongs(out, checkpoint.invalidProposals()); writeLongs(out, checkpoint.modelSizeCounts());
		writeLongs(out, checkpoint.inclusionCounts()); writeLongs(out, checkpoint.coefficientCounts()); writeDoubles(out, checkpoint.coefficientSums()); writeDoubles(out, checkpoint.coefficientSquareSums());
		writeDoubles(out, checkpoint.commonSums()); writeDoubles(out, checkpoint.commonSquareSums()); out.flush(); byte[] payload = payloadBytes.toByteArray(), checksum = sha256(payload);
		ByteArrayOutputStream fileBytes = new ByteArrayOutputStream(); DataOutputStream file = new DataOutputStream(fileBytes); file.writeInt(MAGIC); file.writeInt(payload.length); file.write(payload); file.write(checksum); file.flush(); return fileBytes.toByteArray();
	}
	private static byte[] serialize(RandomEngine random) throws IOException { ByteArrayOutputStream bytes = new ByteArrayOutputStream(); try (ObjectOutputStream out = new ObjectOutputStream(bytes)) { out.writeObject(random); } return bytes.toByteArray(); }
	private static RandomEngine deserialize(byte[] bytes, final String expectedClass) throws IOException {
		if (!expectedClass.startsWith("jdistlib.rng.")) throw new IOException("untrusted sparse RJ RNG class");
		try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes)) {
			@Override protected Class<?> resolveClass(ObjectStreamClass descriptor) throws IOException, ClassNotFoundException {
				String name = descriptor.getName(); if (!(name.startsWith("jdistlib.rng.") || name.equals("java.util.Random") || name.startsWith("["))) throw new IOException("unexpected serialized class " + name); return super.resolveClass(descriptor);
			}
		}) { Object value = in.readObject(); if (!expectedClass.equals(value.getClass().getName()) || !(value instanceof RandomEngine)) throw new IOException("sparse RJ RNG class mismatch"); return (RandomEngine) value;
		} catch (ClassNotFoundException exception) { throw new IOException("sparse RJ RNG class unavailable", exception); }
	}
	private static void writeInts(DataOutputStream out, int[] values) throws IOException { out.writeInt(values.length); for (int value : values) out.writeInt(value); }
	private static void writeLongs(DataOutputStream out, long[] values) throws IOException { out.writeInt(values.length); for (long value : values) out.writeLong(value); }
	private static void writeDoubles(DataOutputStream out, double[] values) throws IOException { out.writeInt(values.length); for (double value : values) out.writeDouble(value); }
	private static void writeStrings(DataOutputStream out, String[] values) throws IOException { out.writeInt(values.length); for (String value : values) out.writeUTF(value); }
	private static int[] readInts(DataInputStream in, int maximum, String label) throws IOException { int length = checkedLength(in.readInt(), maximum, label); int[] values = new int[length]; for (int i = 0; i < length; i++) values[i] = in.readInt(); return values; }
	private static long[] readLongs(DataInputStream in, int maximum, String label) throws IOException { int length = checkedLength(in.readInt(), maximum, label); long[] values = new long[length]; for (int i = 0; i < length; i++) values[i] = in.readLong(); return values; }
	private static double[] readDoubles(DataInputStream in, int maximum, String label) throws IOException { int length = checkedLength(in.readInt(), maximum, label); double[] values = new double[length]; for (int i = 0; i < length; i++) values[i] = in.readDouble(); return values; }
	private static String[] readStrings(DataInputStream in, int maximum, String label) throws IOException { int length = checkedLength(in.readInt(), maximum, label); String[] values = new String[length]; for (int i = 0; i < length; i++) values[i] = in.readUTF(); return values; }
	private static int checkedLength(int length, int maximum, String label) throws IOException { if (length < 0 || length > maximum) throw new IOException("invalid sparse RJ " + label); return length; }
	private static byte[] sha256(byte[] value) { try { return MessageDigest.getInstance("SHA-256").digest(value); } catch (NoSuchAlgorithmException exception) { throw new IllegalStateException(exception); } }
}
