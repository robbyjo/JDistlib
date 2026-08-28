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
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import jdistlib.rng.RandomEngine;

/** Checksummed portable persistence for complete reversible-jump restart state. */
public final class ReversibleJumpCheckpointIO {
	private static final int MAGIC = 0x4a44524a, VERSION = 1;
	private ReversibleJumpCheckpointIO() {}
	public static void write(Path path, ReversibleJumpCheckpoint checkpoint, String modelFingerprint,
			String optionsFingerprint) throws IOException {
		if (path == null || checkpoint == null || modelFingerprint == null || optionsFingerprint == null)
			throw new IllegalArgumentException("path, checkpoint, and fingerprints required");
		ByteArrayOutputStream bytes = new ByteArrayOutputStream(); DataOutputStream out = new DataOutputStream(bytes);
		out.writeInt(VERSION); out.writeUTF(modelFingerprint); out.writeUTF(optionsFingerprint);
		out.writeUTF(System.getProperty("os.name", "unknown") + "/" + System.getProperty("os.arch", "unknown") + "/Java " + System.getProperty("java.version", "unknown"));
		out.writeLong(checkpoint.state().modelId()); writeArray(out, checkpoint.state().parameters()); out.writeDouble(checkpoint.logJoint());
		out.writeInt(checkpoint.completedIterations()); out.writeBoolean(checkpoint.warmupComplete());
		RandomEngine random = checkpoint.random(); out.writeUTF(random.getClass().getName()); byte[] randomBytes = serialize(random);
		out.writeInt(randomBytes.length); out.write(randomBytes); String[] moveNames = checkpoint.moveNames(); double[] moveWeights = checkpoint.moveWeights();
		out.writeInt(moveNames.length); for (int i = 0; i < moveNames.length; i++) { out.writeUTF(moveNames[i]); out.writeDouble(moveWeights[i]); }
		Map<String, double[]> adaptation = checkpoint.adaptationState(); out.writeInt(adaptation.size());
		for (Map.Entry<String, double[]> entry : adaptation.entrySet()) { out.writeUTF(entry.getKey()); writeArray(out, entry.getValue()); }
		out.flush(); byte[] payload = bytes.toByteArray(), checksum = sha256(payload);
		try (DataOutputStream file = new DataOutputStream(Files.newOutputStream(path))) { file.writeInt(MAGIC); file.writeInt(payload.length); file.write(payload); file.write(checksum); }
	}
	public static PortableReversibleJumpCheckpoint read(Path path, String expectedModelFingerprint,
			String expectedOptionsFingerprint) throws IOException {
		try (DataInputStream file = new DataInputStream(Files.newInputStream(path))) {
			if (file.readInt() != MAGIC) throw new IOException("not a JDistlib RJ checkpoint"); int length = file.readInt();
			if (length < 0 || length > 1_000_000_000) throw new IOException("invalid RJ checkpoint length"); byte[] payload = new byte[length]; file.readFully(payload);
			byte[] expectedChecksum = new byte[32]; file.readFully(expectedChecksum); if (!Arrays.equals(expectedChecksum, sha256(payload))) throw new IOException("RJ checkpoint checksum mismatch");
			DataInputStream in = new DataInputStream(new ByteArrayInputStream(payload)); if (in.readInt() != VERSION) throw new IOException("unsupported RJ checkpoint version");
			String model = in.readUTF(), options = in.readUTF(), platform = in.readUTF();
			if (expectedModelFingerprint != null && !expectedModelFingerprint.equals(model)) throw new IOException("RJ checkpoint model fingerprint mismatch");
			if (expectedOptionsFingerprint != null && !expectedOptionsFingerprint.equals(options)) throw new IOException("RJ checkpoint options fingerprint mismatch");
			ReversibleJumpState state = new ReversibleJumpState(in.readLong(), readArray(in)); double logJoint = in.readDouble(); int completed = in.readInt(); boolean warmupComplete = in.readBoolean();
			String randomClass = in.readUTF(); int randomLength = in.readInt(); if (randomLength < 0 || randomLength > 100_000_000) throw new IOException("invalid RJ RNG state length");
			byte[] randomBytes = new byte[randomLength]; in.readFully(randomBytes); RandomEngine random = deserialize(randomBytes, randomClass);
			int moveCount = checkedLength(in.readInt(), 100000, "move count"); String[] moveNames = new String[moveCount]; double[] moveWeights = new double[moveCount];
			for (int i = 0; i < moveCount; i++) { moveNames[i] = in.readUTF(); moveWeights[i] = in.readDouble(); }
			int entries = checkedLength(in.readInt(), 1000000, "adaptation count"); Map<String, double[]> adaptation = new LinkedHashMap<String, double[]>();
			for (int i = 0; i < entries; i++) adaptation.put(in.readUTF(), readArray(in));
			ReversibleJumpCheckpoint checkpoint = new ReversibleJumpCheckpoint(state, logJoint, completed, random, moveNames, moveWeights, adaptation, warmupComplete);
			return new PortableReversibleJumpCheckpoint(checkpoint, model, options, platform);
		}
	}
	private static byte[] serialize(RandomEngine random) throws IOException {
		ByteArrayOutputStream bytes = new ByteArrayOutputStream(); try (ObjectOutputStream out = new ObjectOutputStream(bytes)) { out.writeObject(random); } return bytes.toByteArray();
	}
	private static RandomEngine deserialize(byte[] bytes, final String expectedClass) throws IOException {
		if (!expectedClass.startsWith("jdistlib.rng.")) throw new IOException("untrusted RJ RNG class");
		try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes)) {
			@Override protected Class<?> resolveClass(ObjectStreamClass descriptor) throws IOException, ClassNotFoundException {
				String name = descriptor.getName(); if (!(name.startsWith("jdistlib.rng.") || name.equals("java.util.Random") || name.startsWith("["))) throw new IOException("unexpected serialized class " + name); return super.resolveClass(descriptor);
			}
		}) {
			Object value = in.readObject(); if (!expectedClass.equals(value.getClass().getName()) || !(value instanceof RandomEngine)) throw new IOException("RJ RNG class mismatch"); return (RandomEngine) value;
		} catch (ClassNotFoundException exception) { throw new IOException("RJ RNG class unavailable", exception); }
	}
	private static void writeArray(DataOutputStream out, double[] values) throws IOException { out.writeInt(values.length); for (double value : values) out.writeDouble(value); }
	private static double[] readArray(DataInputStream in) throws IOException { int length = checkedLength(in.readInt(), 10000000, "array length"); double[] result = new double[length]; for (int i = 0; i < length; i++) result[i] = in.readDouble(); return result; }
	private static int checkedLength(int length, int maximum, String label) throws IOException { if (length < 0 || length > maximum) throw new IOException("invalid RJ " + label); return length; }
	private static byte[] sha256(byte[] value) { try { return MessageDigest.getInstance("SHA-256").digest(value); } catch (NoSuchAlgorithmException exception) { throw new IllegalStateException(exception); } }
}
