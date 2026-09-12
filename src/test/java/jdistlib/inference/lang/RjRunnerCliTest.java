/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import static org.junit.Assert.*;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import jdistlib.inference.*;
import jdistlib.rng.MersenneTwister;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class RjRunnerCliTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();
    private String messages;
    private Path file(String name, String content) throws Exception {
        Path path = temporary.getRoot().toPath().resolve(name);
        Files.write(path, content.getBytes(StandardCharsets.UTF_8)); return path;
    }
    private int run(String... args) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (PrintStream stream = new PrintStream(bytes, true, "UTF-8")) {
            int status = ModelRunnerCli.run(args, stream, stream); messages = bytes.toString("UTF-8"); return status;
        }
    }
    private static String model(String script, double prior) {
        return "{\"script\":\"" + script + "\",\"prior_probability\":" + prior + "}";
    }
    private static String edge(String to, String added) {
        return "{\"type\":\"add_drop\",\"from\":\"base\",\"to\":\"" + to + "\",\"copy\":{\"alpha\":\"alpha\"},"
                + "\"add\":{\"" + added + "\":{\"proposal\":\"normal\",\"mean\":0,\"sd\":2}},\"space\":\"unconstrained\"}";
    }
    private Path family() throws Exception {
        file("base.stan", "parameters { real alpha; } model { target += normal_lpdf(alpha | 0,5); }");
        file("beta.stan", "parameters { real beta; real alpha; } model { target += normal_lpdf(alpha | 0,5) + normal_lpdf(beta | 0,2); }");
        file("gamma.stan", "parameters { real alpha; real gamma; } model { target += normal_lpdf(alpha | 0,5) + normal_lpdf(gamma | 0,2); }");
        return file("selection.rj.json", "{\"schema\":\"jdistlib.rj/1\",\"models\":{\"base\":" + model("base.stan", .2)
                + ",\"beta\":" + model("beta.stan", .3) + ",\"gamma\":" + model("gamma.stan", .5)
                + "},\"transitions\":[" + edge("beta", "beta") + "," + edge("gamma", "gamma") + "]}");
    }
    @Test public void scalarInsertionRoundTripsAndIncludesProposalDensity() throws Exception {
        RjScriptModelSpace target = new RjScriptModelSpace(family(), new LinkedHashMap<String, double[]>());
        ReversibleJumpState initial = new ReversibleJumpState(0, new double[] {1.25});
        ReversibleJumpProposal birth = target.moves.get(0).propose(initial, target, new MersenneTwister(73));
        assertEquals(1, birth.proposedState().modelId());
        assertEquals(1.25, birth.proposedState().parameter(1), 0); // reordered destination
        double beta = birth.proposedState().parameter(0);
        double logNormal = -.5 * beta * beta / 4 - Math.log(2) - .5 * Math.log(2*Math.PI);
        assertEquals(logNormal, birth.logForwardDensity(), 1e-14);
        ReversibleJumpProposal death = target.moves.get(1).propose(birth.proposedState(), target, new MersenneTwister(9));
        assertArrayEquals(initial.parameters(), death.proposedState().parameters(), 0);
        assertEquals(birth.logForwardDensity(), death.logReverseDensity(), 0);
        assertEquals(0, birth.logAbsJacobian(), 0);
        // Independent target difference: extra normalized beta prior plus log model-prior ratio.
        assertEquals(logNormal + Math.log(.3/.2), target.logJoint(birth.proposedState()) - target.logJoint(initial), 1e-14);
    }
    @Test public void unequalDegreeGraphRecoversPriorModelAndInclusionProbabilities() throws Exception {
        RjScriptModelSpace target = new RjScriptModelSpace(family(), new LinkedHashMap<String, double[]>());
        ReversibleJumpSamplingOptions options = ReversibleJumpSamplingOptions.builder().warmupIterations(100)
                .sampleIterations(12000).adaptMoveWeights(false).build();
        ReversibleJumpResult[] chains = ReversibleJumpChains.parallel(() -> target.sampler(.2,.3), target,
                new ReversibleJumpState[] {target.initial(0),target.initial(1),target.initial(2)}, options, 1987, 2);
        ReversibleJumpDiagnosticReport report = ReversibleJumpDiagnostics.analyze(target, chains);
        double[] expected = {.2,.3,.5};
        for (int i = 0; i < report.modelIds().length; i++) assertEquals(expected[(int) report.modelIds()[i]], report.modelProbabilities()[i], .02);
        assertArrayEquals(new String[] {"beta","gamma"}, report.candidateNames());
        assertArrayEquals(new double[] {.3,.5}, report.inclusionProbabilities(), .02);
    }
    @Test public void validatesSchemaNormalizationMappingsAndDataWithoutReplacingOutput() throws Exception {
        Path manifest = family(); String original = ScriptDataFiles.read(manifest);
        Path output = file("output.txt", "old result");
        String[] invalid = {original.replace("\"schema\":", "\"sampler\":1,\"schema\":"),
                original.replace("jdistlib.rj/1", "jdistlib.rj/99"), original.replace("\"sd\":2", "\"sd\":0"),
                original.replace("\"normal\"", "\"uniform\""), original.replace("\"beta\",\"copy\"", "\"missing\",\"copy\""),
                original.replace("\"alpha\":\"alpha\"", "\"typo\":\"alpha\""),
                original.replace("\"prior_probability\":0.2", "\"prior_probability\":0.8"),
                original.replace("," + edge("gamma","gamma"), "")};
        for (String bad : invalid) {
            Files.write(manifest, bad.getBytes(StandardCharsets.UTF_8));
            int status = run("--run", "--sampler", "rjmcmc", "--model-space", manifest.toString(), "--output", output.toString(), "--validate");
            assertEquals(messages, 2, status); assertEquals("old result", ScriptDataFiles.read(output));
        }
        Files.write(manifest, original.getBytes(StandardCharsets.UTF_8));
        for (String source : new String[] {"parameters { real alpha; } model { alpha ~ normal(0,5); }",
                "parameters { real alpha; } model { target += normal_lupdf(alpha | 0,5); }",
                "parameters { real<lower=0> alpha; } model { target += normal_lpdf(alpha | 0,5); }",
                "parameters { vector[1] alpha; } model { target += normal_lpdf(alpha | 0,5); }"}) {
            file("base.stan", source);
            assertEquals(2, run("--run", "--sampler", "rjmcmc", "--model-space", manifest.toString(), "--validate"));
        }
        manifest = family();
        assertEquals(2, run("--run", "--sampler", "rjmcmc", "--model-space", manifest.toString(), "--set", "typo=1", "--validate"));
        assertEquals(2, run("--run", "--sampler", "rjmcmc", "--model-space", manifest.toString(), "--output", manifest.toString()));
        assertEquals(2, run("--run", "--sampler", "rjmcmc", "--model-space", manifest.toString(), "--max-depth", "9", "--validate"));
        assertEquals(0, run("--run", "--sampler", "rjmcmc", "--help")); assertTrue(messages.contains("--model-space"));
    }
    @Test public void jsonRejectsDuplicateKeysAndInvalidSyntax() {
        for (String text : new String[] {"{\"x\":1,\"x\":2}", "{\"x\":1,}", "[1,]", "01", "1e999", "\"\\q\"", "{} junk"}) {
            try { RjJson.parse(text); fail(text); } catch (IllegalArgumentException expected) { assertFalse(expected.getMessage().isEmpty()); }
        }
        assertEquals("a\\b", RjJson.parse("\"a\\\\b\""));
        assertEquals("x", RjJson.parse("\"\\u0078\""));
    }
    @Test public void checkpointResumeMatchesUninterruptedRetainedStatesAndRejectsChangedModel() throws Exception {
        Path manifest = family(), prefix = temporary.getRoot().toPath().resolve("checkpoint");
        Path shortOutput = temporary.getRoot().toPath().resolve("short.csv"), longOutput = temporary.getRoot().toPath().resolve("long.csv"),
                resumedOutput = temporary.getRoot().toPath().resolve("resumed.csv");
        String[] common = {"--run", "--sampler", "rjmcmc", "--model-space", manifest.toString(), "--chains", "2", "--thin", "2"};
        int status = run(join(common, "--output", shortOutput.toString(), "--warmup", "50", "--samples", "100", "--seed", "452", "--checkpoint", prefix.toString()));
        assertEquals(messages, 0, status);
        status = run(join(common, "--output", longOutput.toString(), "--warmup", "50", "--samples", "200", "--seed", "452", "--threads", "2"));
        assertEquals(messages, 0, status);
        status = run(join(common, "--output", resumedOutput.toString(), "--samples", "100", "--resume", prefix.toString()));
        assertEquals(messages, 0, status);
        assertEquals(rows(longOutput, 100), rows(resumedOutput, 0));
        Path firstCheckpoint = Paths.get(prefix + ".chain-1.rjckpt"), secondCheckpoint = Paths.get(prefix + ".chain-2.rjckpt");
        byte[] savedFirst = Files.readAllBytes(firstCheckpoint);
        Files.write(firstCheckpoint, Files.readAllBytes(secondCheckpoint));
        assertEquals(1, run(join(common, "--output", resumedOutput.toString(), "--samples", "100", "--resume", prefix.toString())));
        assertTrue(messages.contains("fingerprint mismatch"));
        Files.write(firstCheckpoint, savedFirst);
        file("base.stan", "parameters { real alpha; } model { target += normal_lpdf(alpha | 0,4); }");
        assertEquals(1, run(join(common, "--output", resumedOutput.toString(), "--samples", "100", "--resume", prefix.toString())));
        assertTrue(messages.contains("fingerprint mismatch"));
    }
    private static String[] join(String[] first, String... rest) {
        String[] result = Arrays.copyOf(first, first.length + rest.length); System.arraycopy(rest,0,result,first.length,rest.length); return result;
    }
    private static List<String> rows(Path path, int skipDraws) throws Exception {
        List<String> result = new ArrayList<String>();
        for (String line : ScriptDataFiles.read(path).split("\n")) if (!line.startsWith("#") && !line.startsWith("chain,")) {
            int first = line.indexOf(','), second = line.indexOf(',', first + 1);
            if (Integer.parseInt(line.substring(first + 1, second)) > skipDraws) result.add(line.substring(0,first) + line.substring(second));
        }
        return result;
    }
    @Test public void workedJavaExampleHasIdenticalLogJointAndMatchesExactGaussianEvidence() throws Exception {
        Map<String, double[]> data = new LinkedHashMap<String, double[]>();
        ScriptDataFiles.load(data, "examples/cli/rjmcmc/observations.json");
        RjScriptModelSpace target = new RjScriptModelSpace(Paths.get("examples/cli/rjmcmc/selection.rj.json"), data);
        double[] x = data.get("X"), y = data.get("y");
        double[] residuals = {.10,-.18,.24,-.05,-.20,.12,.04,-.16,.19,-.08,.05,.15,-.14,.09,-.03,-.11};
        for (int i = 0; i < y.length; i++) assertEquals(1.1 + 1.4*x[i*3] + .45*x[i*3+1] + residuals[i], y[i], 1e-14);
        double[] evidence = new double[8]; double max = Double.NEGATIVE_INFINITY;
        for (int mask = 0; mask < 8; mask++) {
            double[] theta = new double[1 + Integer.bitCount(mask)]; Arrays.fill(theta, .4); theta[0] = 1.2;
            double expected = logNormal(theta[0],0,5) + Integer.bitCount(mask)*Math.log(.3) + (3-Integer.bitCount(mask))*Math.log(.7);
            for (int j = 1; j < theta.length; j++) expected += logNormal(theta[j],0,2);
            for (int i = 0; i < y.length; i++) {
                double mean = theta[0]; int j = 1;
                for (int c = 0; c < 3; c++) if ((mask & (1 << c)) != 0) mean += theta[j++] * x[i*3+c];
                expected += logNormal(y[i],mean,.75);
            }
            assertEquals(expected, target.logJoint(new ReversibleJumpState(mask,theta)), 1e-10);
            evidence[mask] = logEvidence(mask,x,y) + Integer.bitCount(mask)*Math.log(.3) + (3-Integer.bitCount(mask))*Math.log(.7);
            max = Math.max(max,evidence[mask]);
        }
        double total = 0;
        for (int i = 0; i < 8; i++) { evidence[i] = Math.exp(evidence[i]-max); total += evidence[i]; }
        for (int i = 0; i < 8; i++) evidence[i] /= total;
        ReversibleJumpSamplingOptions options = ReversibleJumpSamplingOptions.builder().warmupIterations(1500).sampleIterations(6000).build();
        ReversibleJumpResult[] chains = ReversibleJumpChains.parallel(() -> target.sampler(.2,.3),target,
                new ReversibleJumpState[] {target.initial(0),target.initial(1),target.initial(2),target.initial(4)},options,20260829,2);
        double[] observed = new double[8];
        for (ReversibleJumpResult chain : chains) {
            assertEquals(ReversibleJumpResult.Status.SUCCESS,chain.status());
            for (ReversibleJumpState state : chain.draws()) observed[(int)state.modelId()] += 1.0/24000;
        }
        assertArrayEquals(evidence,observed,.045);
    }
    private static double logNormal(double x, double mean, double sd) { return -.5*Math.pow((x-mean)/sd,2)-Math.log(sd)-.5*Math.log(2*Math.PI); }
    // Independent exact integration: y ~ N(0, .75^2 I + 25 11' + 4 X_M X_M').
    private static double logEvidence(int mask, double[] x, double[] y) {
        int n = y.length; double[][] lower = new double[n][n]; double logDet = 0;
        for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) {
            double value = 25 + (i == j ? .75*.75 : 0);
            for (int c = 0; c < 3; c++) if ((mask & (1 << c)) != 0) value += 4*x[i*3+c]*x[j*3+c];
            for (int k = 0; k < j; k++) value -= lower[i][k]*lower[j][k];
            lower[i][j] = i == j ? Math.sqrt(value) : value/lower[j][j];
            if (i == j) logDet += 2*Math.log(lower[i][i]);
        }
        double[] solved = new double[n]; double quadratic = 0;
        for (int i = 0; i < n; i++) {
            double value = y[i]; for (int j = 0; j < i; j++) value -= lower[i][j]*solved[j];
            solved[i] = value/lower[i][i]; quadratic += solved[i]*solved[i];
        }
        return -.5*(n*Math.log(2*Math.PI)+logDet+quadratic);
    }
}
