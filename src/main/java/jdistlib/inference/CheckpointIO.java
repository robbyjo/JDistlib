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

import jdistlib.rng.RandomEngine;

/** Checksummed, versioned binary checkpoint envelope with compatibility fingerprints. */
public final class CheckpointIO {
	private static final int MAGIC = 0x4a444350, VERSION = 1;
	private CheckpointIO() {}
	public static void write(Path path, ChainCheckpoint checkpoint, String modelFingerprint,
			String optionsFingerprint) throws IOException {
		if (path == null || checkpoint == null || modelFingerprint == null || optionsFingerprint == null)
			throw new IllegalArgumentException("path, checkpoint, and fingerprints are required");
		ByteArrayOutputStream bytes = new ByteArrayOutputStream(); DataOutputStream out = new DataOutputStream(bytes);
		out.writeInt(VERSION); out.writeUTF(modelFingerprint); out.writeUTF(optionsFingerprint);
		out.writeUTF(System.getProperty("os.name", "unknown") + "/" + System.getProperty("os.arch", "unknown") + "/Java " + System.getProperty("java.version", "unknown"));
		writeArray(out, checkpoint.state()); out.writeDouble(checkpoint.logDensity()); out.writeInt(checkpoint.completedIterations());
		RandomEngine random = checkpoint.random(); out.writeUTF(random.getClass().getName()); byte[] randomBytes = serialize(random);
		out.writeInt(randomBytes.length); out.write(randomBytes); writeSampler(out, checkpoint.samplerCheckpoint()); out.flush();
		byte[] payload = bytes.toByteArray(), checksum = sha256(payload);
		try (DataOutputStream file = new DataOutputStream(Files.newOutputStream(path))) { file.writeInt(MAGIC); file.writeInt(payload.length); file.write(payload); file.write(checksum); }
	}
	public static PortableCheckpoint read(Path path, String expectedModelFingerprint,
			String expectedOptionsFingerprint) throws IOException {
		try (DataInputStream file = new DataInputStream(Files.newInputStream(path))) {
			if (file.readInt() != MAGIC) throw new IOException("not a JDistlib checkpoint"); int length = file.readInt();
			if (length < 0 || length > 1_000_000_000) throw new IOException("invalid checkpoint length"); byte[] payload = new byte[length]; file.readFully(payload);
			byte[] expectedChecksum = new byte[32]; file.readFully(expectedChecksum); if (!Arrays.equals(expectedChecksum, sha256(payload))) throw new IOException("checkpoint checksum mismatch");
			DataInputStream in = new DataInputStream(new ByteArrayInputStream(payload)); if (in.readInt() != VERSION) throw new IOException("unsupported checkpoint version");
			String model = in.readUTF(), options = in.readUTF(), platform = in.readUTF();
			if (expectedModelFingerprint != null && !expectedModelFingerprint.equals(model)) throw new IOException("checkpoint model fingerprint mismatch");
			if (expectedOptionsFingerprint != null && !expectedOptionsFingerprint.equals(options)) throw new IOException("checkpoint options fingerprint mismatch");
			double[] state = readArray(in); double logDensity = in.readDouble(); int completed = in.readInt(); String randomClass = in.readUTF();
			int randomLength = in.readInt(); if (randomLength < 0 || randomLength > 100_000_000) throw new IOException("invalid RNG state length");
			byte[] randomBytes = new byte[randomLength]; in.readFully(randomBytes); RandomEngine random = deserialize(randomBytes, randomClass);
			SamplerCheckpoint sampler = readSampler(in); return new PortableCheckpoint(new ChainCheckpoint(state, logDensity, completed, random, sampler), model, options, platform);
		}
	}
	private static byte[] serialize(RandomEngine random) throws IOException { ByteArrayOutputStream bytes = new ByteArrayOutputStream();
		try (ObjectOutputStream out = new ObjectOutputStream(bytes)) { out.writeObject(random); } return bytes.toByteArray(); }
	private static RandomEngine deserialize(byte[] bytes, final String expectedClass) throws IOException { if (!expectedClass.startsWith("jdistlib.rng.")) throw new IOException("untrusted RNG class");
		try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes)) { @Override protected Class<?> resolveClass(ObjectStreamClass descriptor) throws IOException, ClassNotFoundException {
			String name = descriptor.getName(); if (!(name.startsWith("jdistlib.rng.") || name.equals("java.util.Random") || name.startsWith("["))) throw new IOException("unexpected serialized class " + name); return super.resolveClass(descriptor); } }) {
			Object value = in.readObject(); if (!expectedClass.equals(value.getClass().getName()) || !(value instanceof RandomEngine)) throw new IOException("RNG class mismatch"); return (RandomEngine) value;
		} catch (ClassNotFoundException exception) { throw new IOException("RNG class unavailable", exception); } }
	private static void writeSampler(DataOutputStream out, SamplerCheckpoint value) throws IOException { out.writeBoolean(value != null); if (value == null) return;
		out.writeUTF(value.sampler()); out.writeInt(value.version()); out.writeInt(value.warmupIteration()); out.writeDouble(value.initialStepSize()); out.writeDouble(value.stepSize());
		writeMatrix(out, value.inverseMassMatrix()); writeArray(out, value.dualAveragingState()); out.writeInt(value.covarianceCount()); writeArray(out, value.covarianceMean());
		writeMatrix(out, value.covarianceProducts()); out.writeDouble(value.warmupAcceptanceSum()); }
	private static SamplerCheckpoint readSampler(DataInputStream in) throws IOException { if (!in.readBoolean()) return null; String sampler = in.readUTF(); int version = in.readInt(), warmup = in.readInt();
		double initial = in.readDouble(), step = in.readDouble(); double[][] mass = readMatrix(in); double[] dual = readArray(in); int count = in.readInt(); double[] mean = readArray(in);
		double[][] products = readMatrix(in); double sum = in.readDouble(); return new SamplerCheckpoint(sampler, version, warmup, initial, step, mass, dual, count, mean, products, sum); }
	private static void writeArray(DataOutputStream out, double[] values) throws IOException { out.writeInt(values == null ? -1 : values.length); if (values != null) for (double value : values) out.writeDouble(value); }
	private static double[] readArray(DataInputStream in) throws IOException { int length = in.readInt(); if (length < 0) return null; if (length > 10_000_000) throw new IOException("array too large");
		double[] result = new double[length]; for (int i = 0; i < length; i++) result[i] = in.readDouble(); return result; }
	private static void writeMatrix(DataOutputStream out, double[][] values) throws IOException { out.writeInt(values == null ? -1 : values.length); if (values != null) for (double[] row : values) writeArray(out, row); }
	private static double[][] readMatrix(DataInputStream in) throws IOException { int rows = in.readInt(); if (rows < 0) return null; if (rows > 100_000) throw new IOException("matrix too large");
		double[][] result = new double[rows][]; for (int i = 0; i < rows; i++) result[i] = readArray(in); return result; }
	private static byte[] sha256(byte[] value) { try { return MessageDigest.getInstance("SHA-256").digest(value); } catch (NoSuchAlgorithmException exception) { throw new IllegalStateException(exception); } }
}
