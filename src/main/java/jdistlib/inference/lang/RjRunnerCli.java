/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import jdistlib.inference.*;

/** Command-line adapter for finite script model spaces and reversible add/drop moves. */
final class RjRunnerCli {
    private RjRunnerCli() {}
    static int run(String[] args, PrintStream out, PrintStream err) {
        try {
            for (String arg : args) if ("--help".equals(arg) || "-h".equals(arg)) { help(out); return 0; }
            Map<String, double[]> data = new LinkedHashMap<String, double[]>();
            List<Path> inputs = new ArrayList<Path>(); Set<String> seen = new HashSet<String>();
            Path manifest = null, output = null, checkpoint = null, resume = null;
            int chains = 4, threads = 1, warmup = 1000, samples = 1000, thin = 1;
            long seed = 12345;
            double withinScale = .2, withinTarget = .3, jumpTarget = .25;
            boolean validate = false, adaptWeights = true;
            for (int i = 0; i < args.length; i++) {
                String option = args[i], value;
                if ("--run".equals(option) || "--validate".equals(option)) {
                    if (!seen.add(option)) throw new IllegalArgumentException("duplicate option: " + option);
                    if ("--validate".equals(option)) validate = true; continue;
                }
                int equals = option.indexOf('=');
                if (equals >= 0) { value = option.substring(equals + 1); option = option.substring(0, equals); }
                else {
                    if (i + 1 >= args.length || args[i + 1].startsWith("--")) throw new IllegalArgumentException("missing value for " + option);
                    value = args[++i];
                }
                if (!"--data".equals(option) && !"--data-column".equals(option) && !"--set".equals(option) && !seen.add(option))
                    throw new IllegalArgumentException("duplicate option: " + option);
                switch (option) {
                case "--sampler": if (!"rjmcmc".equals(value)) throw new IllegalArgumentException("conflicting sampler options"); break;
                case "--model-space": manifest = Paths.get(value); break;
                case "--output": output = Paths.get(value); break;
                case "--checkpoint": checkpoint = Paths.get(value); break;
                case "--resume": resume = Paths.get(value); break;
                case "--data": ScriptDataFiles.load(data, value); inputs.add(Paths.get(value.substring(value.indexOf('=') + 1))); break;
                case "--data-column":
                    ScriptDataFiles.column(data, value);
                    inputs.add(Paths.get(value.substring(value.indexOf('=') + 1, value.lastIndexOf(':')))); break;
                case "--set": ScriptDataFiles.literal(data, value); break;
                case "--chains": chains = Integer.parseInt(value); break;
                case "--threads": threads = Integer.parseInt(value); break;
                case "--warmup": warmup = Integer.parseInt(value); break;
                case "--samples": samples = Integer.parseInt(value); break;
                case "--thin": thin = Integer.parseInt(value); break;
                case "--seed": seed = Long.parseLong(value); break;
                case "--within-scale": withinScale = Double.parseDouble(value); break;
                case "--within-target-accept": withinTarget = Double.parseDouble(value); break;
                case "--target-jump-accept": jumpTarget = Double.parseDouble(value); break;
                case "--adapt-move-weights":
                    if (!"true".equals(value) && !"false".equals(value)) throw new IllegalArgumentException("--adapt-move-weights requires true or false");
                    adaptWeights = Boolean.parseBoolean(value); break;
                case "--compute":
                    if (!"cpu".equals(value) && !"auto".equals(value)) throw new IllegalArgumentException("RJ script runner supports CPU only"); break;
                default: throw new IllegalArgumentException("unknown RJ option: " + option);
                }
            }
            if (manifest == null) throw new IllegalArgumentException("--model-space is required for RJMCMC");
            if (!validate && output == null) throw new IllegalArgumentException("--output is required");
            if (chains < 1 || threads < 1) throw new IllegalArgumentException("chains and threads must be positive");
            if (resume != null) {
                if (seen.contains("--seed")) throw new IllegalArgumentException("--resume restores random streams; do not specify --seed");
                if (seen.contains("--warmup") && warmup != 0) throw new IllegalArgumentException("--resume requires zero new warmup");
                warmup = 0;
            }
            ReversibleJumpSamplingOptions options = ReversibleJumpSamplingOptions.builder()
                    .warmupIterations(warmup).sampleIterations(samples).thinning(thin)
                    .adaptMoveWeights(resume == null && adaptWeights).targetJumpAcceptance(jumpTarget).build();
            RjScriptModelSpace target = new RjScriptModelSpace(manifest, data); inputs.addAll(target.inputs);
            target.sampler(withinScale, withinTarget); // validate kernel controls before touching output
            String optionFingerprint = "rj-cli-v1;chains=" + chains + ";thin=" + thin + ";scale=" + withinScale
                    + ";within=" + withinTarget + ";jump=" + jumpTarget + ";adapt=" + adaptWeights;
            List<Path> destinations = new ArrayList<Path>();
            if (output != null) destinations.add(output);
            if (checkpoint != null) for (int chain = 0; chain < chains; chain++) destinations.add(checkpointPath(checkpoint, chain));
            for (int i = 0; i < destinations.size(); i++) {
                for (Path input : inputs) distinct(destinations.get(i), input);
                for (int j = 0; j < i; j++) distinct(destinations.get(i), destinations.get(j));
            }
            if (resume != null && output != null) for (int chain = 0; chain < chains; chain++) distinct(output, checkpointPath(resume, chain));
            ReversibleJumpCheckpoint[] restored = new ReversibleJumpCheckpoint[chains];
            if (resume != null) for (int chain = 0; chain < chains; chain++) {
                restored[chain] = ReversibleJumpCheckpointIO.read(checkpointPath(resume, chain), target.fingerprint, optionFingerprint + ";chain=" + (chain + 1)).checkpoint();
                if ((long) restored[chain].completedIterations() + (long) samples * thin > Integer.MAX_VALUE)
                    throw new IllegalArgumentException("resumed iteration count exceeds integer range");
            }
            if (validate) { out.println("Validated RJ model space: " + target.models.size() + " models, " + target.moves.size() + " directed moves"); return 0; }
            err.println("Running RJMCMC: models=" + target.models.size() + ", chains=" + chains + ", warmup=" + warmup + ", samples=" + samples);
            err.println("RJ source checks do not prove proper priors or normalization of arbitrary target expressions.");
            final double scale = withinScale, acceptance = withinTarget;
            ReversibleJumpResult[] results;
            if (resume == null) {
                ReversibleJumpState[] initial = new ReversibleJumpState[chains];
                for (int chain = 0; chain < chains; chain++) initial[chain] = target.initial(chain);
                results = ReversibleJumpChains.parallel(() -> target.sampler(scale, acceptance), target, initial, options, seed, threads);
            } else {
                results = new ReversibleJumpResult[chains];
                for (int chain = 0; chain < chains; chain++) results[chain] = target.sampler(scale, acceptance).resume(target, restored[chain], options);
            }
            for (int chain = 0; chain < chains; chain++) if (results[chain].status() != ReversibleJumpResult.Status.SUCCESS || results[chain].size() != samples)
                throw new IllegalStateException("RJ chain " + (chain + 1) + " failed: " + results[chain].status());
            write(output, checkpoint, target, results, options, optionFingerprint, resume == null ? Long.toString(seed) : "restored", err);
            out.println("Wrote " + output.toAbsolutePath()); return 0;
        } catch (IllegalArgumentException exception) {
            err.println("Input error: " + exception.getMessage()); return 2;
        } catch (IOException | RuntimeException exception) {
            err.println("RJ run failed: " + exception.getMessage());
            for (Throwable cause = exception.getCause(); cause != null; cause = cause.getCause()) err.println("  Caused by: " + cause.getMessage());
            return 1;
        } catch (LinkageError exception) {
            err.println("Runtime dependency unavailable: " + exception.getMessage() + "; use the all-in-one JAR"); return 1;
        }
    }
    private static Path checkpointPath(Path prefix, int chain) { return Paths.get(prefix + ".chain-" + (chain + 1) + ".rjckpt"); }
    private static void distinct(Path first, Path second) throws IOException {
        if (first.toAbsolutePath().normalize().equals(second.toAbsolutePath().normalize())
                || Files.exists(first) && Files.exists(second) && Files.isSameFile(first, second))
            throw new IllegalArgumentException("output/input paths must be distinct: " + first);
    }
    private static Path stage(Path destination, Map<Path, Path> staged) throws IOException {
        Path absolute = destination.toAbsolutePath(); Files.createDirectories(absolute.getParent());
        Path temporary = Files.createTempFile(absolute.getParent(), ".jdistlib-rj-", ".tmp"); staged.put(temporary, absolute); return temporary;
    }
    private static void write(Path output, Path checkpoint, RjScriptModelSpace target, ReversibleJumpResult[] results,
            ReversibleJumpSamplingOptions options, String optionFingerprint, String seed, PrintStream err) throws IOException {
        ReversibleJumpDiagnosticReport report = ReversibleJumpDiagnostics.analyze(target, results);
        Map<Path, Path> staged = new LinkedHashMap<Path, Path>();
        try {
            Path temporary = stage(output, staged);
            try (BufferedWriter writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) {
                writer.write("# jdistlib.rj-cli/1; CPU; unbounded scalar parameters; generated quantities use separate streams\n");
                writer.write("# model_fingerprint=" + target.fingerprint + "; seed=" + seed + "; warmup=" + options.warmupIterations()
                        + "; samples=" + options.sampleIterations() + "; " + optionFingerprint + "\n");
                Set<Long> visited = new HashSet<Long>();
                for (ReversibleJumpResult result : results) for (ReversibleJumpState draw : result.draws()) visited.add(draw.modelId());
                for (int model = 0; model < target.models.size(); model++) {
                    writer.write("# model=" + model + "; name=" + target.spaces.get(model).name() + "; prior_probability=" + target.priors.get(model) + "\n");
                    if (!visited.contains((long) model)) {
                        String warning = "model " + model + " was never visited; zero observed occupancy does not establish zero posterior probability";
                        writer.write("# Warning: " + warning + "\n"); err.println("Diagnostic warning: " + warning);
                    }
                }
                writer.write("# diagnostics=" + report.toJson() + "\n");
                for (String warning : report.warnings()) err.println("Diagnostic warning: " + warning);
                if (results.length < 2) { writer.write("# R-hat unavailable: only one chain\n"); err.println("Diagnostic warning: R-hat requires multiple chains"); }
                writer.write("chain,draw,model_id,model_name,kind,parameter,value,log_joint\n");
                for (int chain = 0; chain < results.length; chain++) {
                    ReversibleJumpResult result = results[chain];
                    // Clone the final checkpoint stream: generated quantities never advance the sampler stream.
                    jdistlib.rng.RandomEngine random = result.checkpoint().random();
                    for (int draw = 0; draw < result.size(); draw++) {
                        ReversibleJumpState state = result.draw(draw); ReversibleJumpModelSpace space = target.modelSpace(state.modelId());
                        if (state.dimension() == 0) row(writer, chain, draw, space, "model", "", "", result.logJointAt(draw));
                        for (int parameter = 0; parameter < state.dimension(); parameter++)
                            row(writer, chain, draw, space, "parameter", space.parameterName(parameter), Double.toString(state.parameter(parameter)), result.logJointAt(draw));
                        Map<String, double[]> generated = target.models.get((int) state.modelId()).generate(state.parameters(), random);
                        for (Map.Entry<String, double[]> entry : generated.entrySet()) for (int i = 0; i < entry.getValue().length; i++)
                            row(writer, chain, draw, space, "generated", entry.getKey() + (entry.getValue().length == 1 ? "" : "[" + (i + 1) + "]"),
                                    Double.toString(entry.getValue()[i]), result.logJointAt(draw));
                    }
                }
            }
            if (checkpoint != null) for (int chain = 0; chain < results.length; chain++)
                ReversibleJumpCheckpointIO.write(stage(checkpointPath(checkpoint, chain), staged), results[chain].checkpoint(), target.fingerprint, optionFingerprint + ";chain=" + (chain + 1));
            for (Map.Entry<Path, Path> entry : staged.entrySet()) Files.move(entry.getKey(), entry.getValue(), StandardCopyOption.REPLACE_EXISTING);
        } finally { for (Path temporary : staged.keySet()) Files.deleteIfExists(temporary); }
    }
    private static void row(BufferedWriter writer, int chain, int draw, ReversibleJumpModelSpace space,
            String kind, String name, String value, double logJoint) throws IOException {
        writer.write((chain + 1) + "," + (draw + 1) + "," + space.modelId() + "," + csv(space.name()) + "," + kind + "," + csv(name) + "," + value + "," + logJoint + "\n");
    }
    private static String csv(String value) { return "\"" + value.replace("\"", "\"\"") + "\""; }
    private static void help(PrintStream out) {
        out.println("Usage: java -jar jdistlib-all.jar --run --sampler rjmcmc --model-space selection.rj.json --output results.txt\n"
                + "  --data file.json | variable=file.csv (repeatable)\n"
                + "  --data-column y=file.csv:header --set N=100 (repeatable)\n"
                + "  --chains 4 --threads 1 --warmup 1000 --samples 1000 --thin 1 --seed 12345\n"
                + "  --within-scale 0.2 --within-target-accept 0.3 --target-jump-accept 0.25\n"
                + "  --adapt-move-weights true|false (default true; warmup only)\n"
                + "  --checkpoint prefix --resume prefix (prefix.chain-N.rjckpt per chain)\n"
                + "  --validate (compile models/data/moves and check resume files, no sampling/output)\n"
                + "  --compute auto|cpu (CPU only; within-model adaptive random walk)\n"
                + "Scripts resolve relative to the model-space JSON; CLI data paths resolve from the working directory.\n"
                + "Schema 1: connected finite model graph; unbounded scalar real parameters; name-preserving copy;\n"
                + "paired add_drop moves with Normal proposals and a unit Jacobian. Model priors must sum to 1.\n"
                + "Use explicit normalized _lpdf/_lpmf terms, with proper priors and all model-dependent constants.\n"
                + "Output is tidy CSV with model/inclusion diagnostics in # comments. Samples means retained draws per chain.\n"
                + "Resume uses zero new warmup and saved RNGs; omit --seed and retain the original kernel controls.\n"
                + "Resume currently executes chains sequentially; output contains only the new draws.");
    }
}
