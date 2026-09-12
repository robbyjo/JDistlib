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
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import jdistlib.inference.BayesianModel;
import jdistlib.accelerator.Compute;
import jdistlib.inference.ComputeNuts;
import jdistlib.inference.ChainResult;
import jdistlib.inference.Chains;
import jdistlib.inference.InferenceCliOptions;
import jdistlib.inference.IterationStats;
import jdistlib.inference.McmcDiagnosticReport;
import jdistlib.inference.McmcDiagnostics;
import jdistlib.inference.NoUTurnSampler;
import jdistlib.inference.ParameterSpec;
import jdistlib.inference.SamplingOptions;
import jdistlib.rng.MersenneTwister;

/** Executable-JAR entry point for supported Stan/JDM scripts and numeric data files. */
public final class ModelRunnerCli {
    private ModelRunnerCli() {}

    public static void main(String[] arguments) {
        int status = run(arguments, System.out, System.err);
        if (status != 0) System.exit(status);
    }

    /** Returns 0 on completion/help, 2 on input errors, and 1 on execution/I/O failure. */
    public static int run(String[] arguments, PrintStream out, PrintStream err) {
        try {
            for (int i = 0; i < arguments.length; i++)
                if ("--sampler=rjmcmc".equals(arguments[i]) || "--sampler".equals(arguments[i])
                        && i + 1 < arguments.length && "rjmcmc".equals(arguments[i + 1]))
                    return RjRunnerCli.run(arguments, out, err);
            for (String argument : arguments) if ("--help".equals(argument) || "-h".equals(argument)) {
                help(out); return 0;
            }
            InferenceCliOptions compute = InferenceCliOptions.parse(arguments);
            if (compute.computeBackend() != Compute.AUTO && compute.computeBackend() != Compute.CPU
                    || compute.nutsBackend() == ComputeNuts.FORCE)
                throw new IllegalArgumentException("script NUTS currently runs on CPU; use --compute auto or cpu and --nuts-offload auto or off");
            String[] args = compute.remainingArguments();
            Map<String, double[]> data = new LinkedHashMap<String, double[]>();
            List<Path> inputs = new ArrayList<Path>();
            Set<String> seen = new HashSet<String>();
            SamplingOptions.Builder builder = SamplingOptions.builder().computeBackend(Compute.CPU).nutsBackend(ComputeNuts.OFF);
            Path input = null, output = null;
            int chainCount = 4, threads = 1;
            long seed = 12345;
            boolean validateOnly = false;
            for (int i = 0; i < args.length; i++) {
                String option = args[i], value;
                if ("--run".equals(option)) {
                    if (!seen.add(option)) throw new IllegalArgumentException("duplicate option: --run");
                    continue;
                }
                if ("--validate".equals(option)) { validateOnly = true; continue; }
                int equals = option.indexOf('=');
                if (equals >= 0) { value = option.substring(equals + 1); option = option.substring(0, equals); }
                else {
                    if (i + 1 >= args.length || args[i + 1].startsWith("--"))
                        throw new IllegalArgumentException("missing value for " + option);
                    value = args[++i];
                }
                if (!Arrays.asList("--data", "--data-column", "--set").contains(option) && !seen.add(option))
                    throw new IllegalArgumentException("duplicate option: " + option);
                switch (option) {
                case "--sampler":
                    if (!"nuts".equals(value)) throw new IllegalArgumentException("--sampler must be nuts or rjmcmc");
                    break;
                case "--input": input = Paths.get(value); inputs.add(input); break;
                case "--output": output = Paths.get(value); break;
                case "--data":
                    ScriptDataFiles.load(data, value);
                    inputs.add(Paths.get(value.substring(value.indexOf('=') + 1))); break;
                case "--data-column":
                    ScriptDataFiles.column(data, value);
                    inputs.add(Paths.get(value.substring(value.indexOf('=') + 1, value.lastIndexOf(':')))); break;
                case "--set": ScriptDataFiles.literal(data, value); break;
                case "--chains": chainCount = Integer.parseInt(value); break;
                case "--threads": threads = Integer.parseInt(value); break;
                case "--seed": seed = Long.parseLong(value); break;
                case "--warmup": builder.warmupIterations(Integer.parseInt(value)); break;
                case "--samples": builder.sampleIterations(Integer.parseInt(value)); break;
                case "--thin": builder.thinning(Integer.parseInt(value)); break;
                case "--target-accept": builder.targetAcceptance(Double.parseDouble(value)); break;
                case "--max-depth": builder.maximumTreeDepth(Integer.parseInt(value)); break;
                case "--step-size": builder.stepSize(Double.parseDouble(value)); break;
                default: throw new IllegalArgumentException("unknown option: " + option);
                }
            }
            if (input == null) throw new IllegalArgumentException("--input is required; use --help for usage");
            if (!validateOnly && output == null) throw new IllegalArgumentException("--output is required");
            if (chainCount < 1 || threads < 1) throw new IllegalArgumentException("--chains and --threads must be positive");
            SamplingOptions options = builder.build();
            if ((long) options.warmupIterations() + (long) options.sampleIterations() * options.thinning() > Integer.MAX_VALUE)
                throw new IllegalArgumentException("total iterations exceed the supported integer range");
            if (output != null) for (Path path : inputs)
                if (output.toAbsolutePath().normalize().equals(path.toAbsolutePath().normalize())
                        || Files.exists(output) && Files.isSameFile(output, path))
                    throw new IllegalArgumentException("output must not overwrite an input file: " + path);
            String source = ScriptDataFiles.read(input);
            Set<String> names = ModelScript.dataNames(source);
            for (String name : data.keySet()) if (!names.contains(name))
                throw new IllegalArgumentException("data variable is not declared in the script: " + name);
            CompiledModelScript compiled = ModelScript.compile(source, data);
            BayesianModel model = compiled.model();
            if (validateOnly) { out.println("Validated " + input + " with data " + data.keySet()); return 0; }
            if (model.dimension() == 0) throw new IllegalArgumentException("NUTS requires at least one continuous parameter");
            double[][] initial = new double[chainCount][];
            for (int chain = 0; chain < chainCount; chain++) initial[chain] = model.initialState();
            err.println("Running NUTS: chains=" + chainCount + ", warmup=" + options.warmupIterations()
                    + ", samples=" + options.sampleIterations() + ", thin=" + options.thinning() + ", seed=" + seed);
            ChainResult[] chains = Chains.parallel(new NoUTurnSampler(), model, initial, options, seed, threads);
            for (int chain = 0; chain < chains.length; chain++) {
                for (String warning : chains[chain].warnings()) err.println("Chain " + (chain + 1) + ": " + warning);
                if (chains[chain].status() != ChainResult.Status.SUCCESS || chains[chain].size() == 0)
                    throw new IllegalStateException("chain " + (chain + 1) + " failed: " + chains[chain].status());
            }
            write(output, compiled, chains, options, seed, err);
            out.println("Wrote " + output.toAbsolutePath()); return 0;
        } catch (IllegalArgumentException exception) {
            err.println("Input error: " + exception.getMessage()); return 2;
        } catch (LinkageError exception) {
            err.println("Runtime dependency unavailable: " + exception.getMessage()
                    + ". Use jdistlib-all.jar, or keep the core JAR beside its generated lib/ directory.");
            return 1;
        } catch (IOException | RuntimeException exception) {
            err.println("Run failed: " + exception.getMessage());
            Throwable cause = exception.getCause();
            while (cause != null) {
                err.println("  Caused by: " + cause.getClass().getSimpleName() + ": " + cause.getMessage());
                cause = cause.getCause();
            }
            return 1;
        }
    }

    private static void write(Path output, CompiledModelScript compiled, ChainResult[] chains,
            SamplingOptions options, long seed, PrintStream err) throws IOException {
        BayesianModel model = compiled.model();
        List<String> parameterNames = new ArrayList<String>();
        for (ParameterSpec parameter : model.parameters().values())
            addNames(parameterNames, parameter.name(), parameter.constrainedDimension());
        ChainResult[] constrained = new ChainResult[chains.length];
        for (int i = 0; i < chains.length; i++) {
            ChainResult chain = chains[i];
            constrained[i] = new ChainResult(chain.constrainedSamples(model), chain.logDensities(),
                    chain.statistics(), null, null, chain.status(), chain.warnings());
        }
        Path destination = output.toAbsolutePath();
        Files.createDirectories(destination.getParent());
        Path temporary = Files.createTempFile(destination.getParent(), ".jdistlib-run-", ".tmp");
        try {
            try (BufferedWriter writer = Files.newBufferedWriter(temporary, StandardCharsets.UTF_8)) {
                writer.write("# JDistlib model runner; NUTS; constrained parameters and generated quantities\n");
                writer.write("# seed=" + seed + "; chains=" + chains.length + "; warmup=" + options.warmupIterations()
                        + "; samples=" + options.sampleIterations() + "; thin=" + options.thinning()
                        + "; target_accept=" + options.targetAcceptance() + "; max_depth=" + options.maximumTreeDepth()
                        + "; compute=" + options.computeBackend() + "; nuts_offload=" + options.nutsBackend() + "\n");
                writer.write("# Container columns use one-based flat row-major indices; log_density__ is unconstrained (includes Jacobian).\n");
                writer.write("# All chains start at the model's default initial state; seeds are independently derived.\n");
                if (chains[0].size() >= 4) {
                    McmcDiagnosticReport report = McmcDiagnostics.analyze(parameterNames.toArray(new String[0]), constrained);
                    writer.write("# diagnostics=" + report.toJson() + "\n");
                    for (String warning : report.warnings()) err.println("Diagnostic warning: " + warning);
                } else writer.write("# Diagnostics unavailable: fewer than four retained draws per chain.\n");
                if (chains.length == 1) {
                    writer.write("# R-hat unavailable: only one chain.\n");
                    err.println("Diagnostic warning: R-hat requires multiple chains.");
                }
                List<String> generatedNames = null;
                for (int chain = 0; chain < chains.length; chain++) {
                    MersenneTwister random = new MersenneTwister(seed ^ (0x632be59bd9b4e019L * (chain + 1L)));
                    for (int draw = 0; draw < chains[chain].size(); draw++) {
                        Map<String, double[]> generated = compiled.generate(chains[chain].sample(draw), random);
                        List<String> currentNames = new ArrayList<String>();
                        for (Map.Entry<String, double[]> entry : generated.entrySet())
                            addNames(currentNames, entry.getKey(), entry.getValue().length);
                        if (generatedNames == null) {
                            generatedNames = currentNames;
                            writer.write("chain__,draw__,log_density__,acceptance__,divergent__,tree_depth__");
                            for (String name : parameterNames) writer.write(",\"" + name + "\"");
                            for (String name : generatedNames) writer.write(",\"" + name + "\"");
                            writer.write('\n');
                        } else if (!generatedNames.equals(currentNames))
                            throw new IllegalStateException("generated quantity dimensions changed between draws");
                        IterationStats stats = chains[chain].statisticsAt(draw);
                        writer.write((chain + 1) + "," + (draw + 1) + "," + chains[chain].logDensityAt(draw)
                                + "," + stats.acceptanceProbability() + "," + stats.divergent() + "," + stats.treeDepth());
                        for (double value : constrained[chain].sample(draw)) writer.write("," + value);
                        for (double[] values : generated.values()) for (double value : values) writer.write("," + value);
                        writer.write('\n');
                    }
                }
            }
            Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING);
        } finally { Files.deleteIfExists(temporary); }
    }

    private static void addNames(List<String> names, String name, int count) {
        for (int i = 0; i < count; i++) names.add(count == 1 ? name : name + "[" + (i + 1) + "]");
    }

    private static void help(PrintStream out) {
        out.println("Usage: java -jar jdistlib-all.jar --run --input script.stan --output output.txt [options]\n"
                + "Runs the supported Stan/JDM core with JDistlib NUTS (not CmdStan).\n"
                + "For RJMCMC: --run --sampler rjmcmc --model-space selection.rj.json --output output.txt\n"
                + "Use --run --sampler rjmcmc --help for RJ options.\n"
                + "  --data file.json             Merge named numeric JSON values (repeatable)\n"
                + "  --data variable=file         Bind numeric JSON or headerless CSV/TSV (repeatable)\n"
                + "  --data-column y=file.csv:y    Bind a named CSV/TSV column (repeatable)\n"
                + "  --set N=100                  Bind a number or JSON array (repeatable)\n"
                + "  --validate                   Compile with data; skip sampling/output\n"
                + "  --chains 4 --threads 1 --seed 12345\n"
                + "  --warmup 1000 --samples 1000 --thin 1\n"
                + "  --target-accept 0.8 --max-depth 10 --step-size 0.25\n"
                + "  --compute auto|cpu --nuts-offload auto|off (scripts currently use CPU)\n"
                + "Output is CSV text with # metadata/diagnostics, constrained parameters and generated quantities.\n"
                + "File paths are relative to the working directory. Duplicate variables are errors.\n"
                + "Numeric arrays/tables flatten row-major; script declarations supply dimensions.\n"
                + "--samples counts retained draws per chain; --thin multiplies post-warmup iterations.\n"
                + "Use --help for this text.");
    }
}
