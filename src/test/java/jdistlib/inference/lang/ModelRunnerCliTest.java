/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import static org.junit.Assert.*;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class ModelRunnerCliTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();
    private String messages;
    private Path file(String name, String text) throws Exception {
        Path path = temporary.getRoot().toPath().resolve(name);
        Files.write(path, text.getBytes(StandardCharsets.UTF_8)); return path;
    }
    private int run(String... args) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (PrintStream stream = new PrintStream(bytes, true, "UTF-8")) {
            int result = ModelRunnerCli.run(args, stream, stream);
            messages = bytes.toString("UTF-8"); return result;
        }
    }
    @Test public void mergesFilesAndBindsCsvColumnsWithAnalyticMatrixGradient() throws Exception {
        Map<String, double[]> data = new LinkedHashMap<String, double[]>();
        ScriptDataFiles.load(data, file("size.json", "\ufeff{\"N\":2,\"K\":2}").toString());
        ScriptDataFiles.load(data, "X=" + file("matrix.csv", "1,2\n3,4\n"));
        ScriptDataFiles.column(data, "y=" + file("response.tsv", "id\tresponse\na\t5\nb\t6\n") + ":response");
        ScriptDataFiles.literal(data, "sigma=1");
        String source = "data { int N; int K; matrix[N,K] X; vector[N] y; real sigma; } "
                + "parameters { vector[K] beta; } model { beta ~ normal(0,1); y ~ normal(X*beta,sigma); }";
        double[] gradient = new double[2];
        ModelScript.compileStan(source, data).model().logDensityAndGradient(new double[2], gradient);
        assertArrayEquals(new double[] {23,34}, gradient, 1e-12); // X' y, independently calculated
        Path model = file("matrix.stan", source);
        assertEquals(0, run("--input", model.toString(), "--data", temporary.getRoot() + "/size.json",
                "--data", "X=" + temporary.getRoot() + "/matrix.csv", "--data-column",
                "y=" + temporary.getRoot() + "/response.tsv:response", "--set", "sigma=1", "--validate"));
        assertTrue(messages.contains("Validated"));
    }
    @Test public void existingThreeArgumentSourceGeneratorRemainsAvailable() throws Exception {
        Path model = file("source.stan", "parameters { real x; } model { x ~ normal(0,1); }");
        Path java = temporary.getRoot().toPath().resolve("Example.java");
        ModelScriptCli.main(new String[] {model.toString(), "generated.Example", java.toString()});
        assertTrue(ScriptDataFiles.read(java).contains("class Example"));
        Path output = temporary.getRoot().toPath().resolve("run.txt");
        int status = run("--run", "--input", model.toString(), "--output", output.toString(),
                "--warmup", "50", "--samples", "2", "--chains", "1", "--compute", "cpu");
        assertEquals(messages, 0, status);
        assertTrue(ScriptDataFiles.read(output).contains("\"x\""));
    }
    @Test public void numericJsonAndTablesFlattenRowMajor() throws Exception {
        Map<String, double[]> data = new LinkedHashMap<String, double[]>();
        ScriptDataFiles.load(data, file("first.json", "{\"N\":2,\"X\":[[1,2],[3,4]]}").toString());
        ScriptDataFiles.load(data, file("second.json", "{\"y\":[5,6]}").toString());
        ScriptDataFiles.load(data, "z=" + file("values.json", "[7,8]"));
        assertArrayEquals(new double[] {1,2,3,4}, data.get("X"), 0);
        assertArrayEquals(new double[] {5,6}, data.get("y"), 0);
        assertArrayEquals(new double[] {7,8}, data.get("z"), 0);
    }
    @Test public void rejectsMalformedDuplicateAndNonfiniteData() throws Exception {
        String[] invalid = {"{\"N\":1,\"N\":2}", "{\"x\":[[1,2],[3]]}", "{\"x\":null}",
                "{\"x\":true}", "{\"x\":1e999}", "{\"x\":01}", "{\"x\":1,}", "{\"x\":1} junk"};
        for (int i = 0; i < invalid.length; i++) {
            try {
                ScriptDataFiles.load(new LinkedHashMap<String, double[]>(), file("invalid" + i + ".json", invalid[i]).toString());
                fail(invalid[i]);
            } catch (IllegalArgumentException expected) { assertFalse(expected.getMessage().isEmpty()); }
        }
        Map<String, double[]> data = new LinkedHashMap<String, double[]>();
        ScriptDataFiles.literal(data, "N=1");
        try { ScriptDataFiles.literal(data, "N=2"); fail(); }
        catch (IllegalArgumentException expected) { assertTrue(expected.getMessage().contains("duplicate")); }
        for (String content : Arrays.asList("1,\n", "1,2\n3\n", "NaN\n", "1,Infinity\n")) {
            try { ScriptDataFiles.load(new LinkedHashMap<String, double[]>(), "x=" + file("bad.csv", content)); fail(); }
            catch (IllegalArgumentException expected) { assertTrue(expected.getMessage().contains("bad.csv")); }
        }
    }
    @Test public void inputErrorsProtectExistingOutputAndReportSourceDiagnostics() throws Exception {
        Path model = file("normal.stan", "data { int<lower=1> N; vector[N] y; } parameters { real mu; } model { mu ~ normal(0,1); }");
        Path output = file("output.txt", "preserve me");
        assertEquals(2, run("--input", model.toString(), "--output", output.toString()));
        assertTrue(messages.contains("missing required data"));
        assertEquals("preserve me", ScriptDataFiles.read(output));
        assertEquals(2, run("--input", model.toString(), "--validate", "--set", "typo=3"));
        assertTrue(messages.contains("not declared"));
        assertEquals(2, run("--input", model.toString(), "--validate", "--set", "N=2", "--set", "y=[1]"));
        assertTrue(messages.contains("declared length"));
        assertEquals(2, run("--input", model.toString(), "--output", model.toString()));
        assertTrue(messages.contains("overwrite"));
        assertEquals(2, run("--input", model.toString(), "--validate", "--chains", "0"));
        assertEquals(2, run("--input", model.toString(), "--validate", "--samples", "2147483647", "--thin", "2"));
        assertEquals(2, run("--input", model.toString(), "--typo", "1"));
        assertEquals(0, run("--help"));
    }
    @Test public void samplesConstrainedBetaPosteriorAndGeneratedQuantitiesReproducibly() throws Exception {
        Path model = file("beta.stan", "data { int<lower=0> n; int<lower=0> k; } "
                + "parameters { real<lower=0,upper=1> theta; } "
                + "model { theta ~ beta(1,1); k ~ binomial(n,theta); } "
                + "generated quantities { real twice = 2*theta; real predictive = bernoulli_rng(theta); }");
        Path output = temporary.getRoot().toPath().resolve("draws.txt");
        String[] args = {"--input", model.toString(), "--output", output.toString(), "--set", "n=10", "--set", "k=7",
                "--warmup", "300", "--samples", "500", "--thin", "2", "--chains", "2", "--seed", "987", "--compute", "cpu"};
        int status = run(args);
        assertEquals(messages, 0, status);
        String first = ScriptDataFiles.read(output);
        List<String> rows = new ArrayList<String>();
        for (String line : first.split("\n")) if (!line.startsWith("#")) rows.add(line);
        assertEquals(1001, rows.size());
        assertTrue(rows.get(0).contains("\"theta\",\"twice\",\"predictive\""));
        double sum = 0;
        for (int i = 1; i < rows.size(); i++) {
            String[] cells = rows.get(i).split(",");
            double theta = Double.parseDouble(cells[6]);
            assertTrue(theta > 0 && theta < 1);
            assertEquals(2*theta, Double.parseDouble(cells[7]), 0);
            assertTrue(cells[8].equals("0.0") || cells[8].equals("1.0"));
            sum += theta;
        }
        assertEquals(8.0/12, sum/1000, 0.035); // analytic Beta(8,4) posterior
        assertTrue(first.contains("# diagnostics="));
        assertEquals(0, run(args));
        assertEquals(first, ScriptDataFiles.read(output));
        String[] parallel = Arrays.copyOf(args, args.length + 2);
        parallel[args.length] = "--threads"; parallel[args.length + 1] = "2";
        assertEquals(0, run(parallel));
        assertEquals(first, ScriptDataFiles.read(output));
    }
    @Test public void failedSamplingDoesNotReplaceOutput() throws Exception {
        Path model = file("invalid.stan", "parameters { real x; } model { target += negative_infinity(); }");
        Path output = file("old.txt", "previous result");
        assertEquals(1, run("--input", model.toString(), "--output", output.toString(), "--warmup", "0",
                "--samples", "1", "--chains", "1", "--compute", "cpu"));
        assertEquals("previous result", ScriptDataFiles.read(output));
    }
}
