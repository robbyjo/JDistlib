/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import jdistlib.Normal;
import jdistlib.inference.*;
import jdistlib.rng.RandomEngine;

/** Compiled finite model family and bijective scalar insertion/removal proposals. */
final class RjScriptModelSpace implements ReversibleJumpInclusionTarget {
    final List<Path> inputs = new ArrayList<Path>();
    final List<CompiledModelScript> models = new ArrayList<CompiledModelScript>();
    final List<ReversibleJumpModelSpace> spaces = new ArrayList<ReversibleJumpModelSpace>();
    final List<Double> priors = new ArrayList<Double>();
    final List<ReversibleJumpMove> moves = new ArrayList<ReversibleJumpMove>();
    final String fingerprint;
    private final String[] candidates;

    RjScriptModelSpace(Path path, Map<String, double[]> data) throws IOException {
        path = path.toAbsolutePath().normalize(); inputs.add(path);
        String document = ScriptDataFiles.read(path);
        MessageDigest digest = digest(); update(digest, "jdistlib.rj/1-scalar-normal-v1"); update(digest, document);
        Map<String, Object> root = RjJson.object(RjJson.parse(document), "model space");
        RjJson.fields(root, "schema", "models", "transitions");
        if (!"jdistlib.rj/1".equals(root.get("schema"))) throw new IllegalArgumentException("unsupported RJ schema");
        Map<String, Object> definitions = RjJson.object(root.get("models"), "models");
        if (definitions.size() < 2) throw new IllegalArgumentException("at least two RJ models are required");
        Map<String, Integer> ids = new LinkedHashMap<String, Integer>();
        Set<String> usedData = new LinkedHashSet<String>(), allParameters = new LinkedHashSet<String>();
        double totalPrior = 0;
        for (Map.Entry<String, Object> entry : definitions.entrySet()) {
            String name = entry.getKey();
            if (!name.matches("[A-Za-z_][A-Za-z_0-9-]*")) throw new IllegalArgumentException("invalid model name: " + name);
            Map<String, Object> definition = RjJson.object(entry.getValue(), name);
            RjJson.fields(definition, "script", "prior_probability");
            Path script = path.getParent().resolve(RjJson.string(definition.get("script"), "script")).normalize();
            inputs.add(script); String source = ScriptDataFiles.read(script); update(digest, source);
            double prior = RjJson.number(definition.get("prior_probability"), "prior_probability");
            if (!(prior > 0 && prior <= 1)) throw new IllegalArgumentException("model priors must be positive probabilities");
            totalPrior += prior;
            Map<String, double[]> bound = new LinkedHashMap<String, double[]>();
            for (String key : ModelScript.dataNames(source)) {
                usedData.add(key); if (data.containsKey(key)) bound.put(key, data.get(key));
            }
            try {
                ModelScript.validateRjSource(source);
                CompiledModelScript compiled = ModelScript.compile(source, bound);
                String[] names = compiled.model().parameters().keySet().toArray(new String[0]);
                ids.put(name, models.size());
                spaces.add(new ReversibleJumpModelSpace(models.size(), name, names));
                models.add(compiled); priors.add(prior); allParameters.addAll(Arrays.asList(names));
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException(script + ": " + exception.getMessage(), exception);
            }
        }
        if (Math.abs(totalPrior - 1) > 1e-10) throw new IllegalArgumentException("model prior probabilities must sum to 1");
        for (Map.Entry<String, double[]> entry : new TreeMap<String, double[]>(data).entrySet()) {
            if (!usedData.contains(entry.getKey())) throw new IllegalArgumentException("data variable is not declared in any model: " + entry.getKey());
            update(digest, entry.getKey());
            for (double value : entry.getValue()) update(digest, Double.toHexString(value));
        }
        boolean[][] edges = new boolean[models.size()][models.size()];
        Set<String> pairs = new LinkedHashSet<String>();
        for (Object value : RjJson.list(root.get("transitions"), "transitions")) {
            Map<String, Object> transition = RjJson.object(value, "transition");
            RjJson.fields(transition, "type", "from", "to", "copy", "add", "space");
            if (!"add_drop".equals(transition.get("type")) || !"unconstrained".equals(transition.get("space")))
                throw new IllegalArgumentException("RJ schema 1 supports add_drop transitions in unconstrained space");
            Integer from = ids.get(RjJson.string(transition.get("from"), "from"));
            Integer to = ids.get(RjJson.string(transition.get("to"), "to"));
            if (from == null || to == null || from.equals(to)) throw new IllegalArgumentException("transition needs two distinct declared models");
            if (!pairs.add(from + ":" + to)) throw new IllegalArgumentException("duplicate transition");
            Map<String, Object> copy = RjJson.object(transition.get("copy"), "copy");
            Map<String, Object> add = RjJson.object(transition.get("add"), "add");
            String[] sourceNames = spaces.get(from).parameterNames(), targetNames = spaces.get(to).parameterNames();
            if (copy.size() != sourceNames.length || add.isEmpty() || targetNames.length != copy.size() + add.size())
                throw new IllegalArgumentException("copy must cover every source parameter; copy plus add must exactly cover the destination");
            int[] copied = new int[sourceNames.length];
            Set<String> destinations = new LinkedHashSet<String>();
            for (int i = 0; i < copied.length; i++) {
                String targetName = RjJson.string(copy.get(sourceNames[i]), "copy." + sourceNames[i]);
                // Stable names make pooled conditional summaries and inclusion indicators meaningful.
                if (!sourceNames[i].equals(targetName)) throw new IllegalArgumentException("RJ schema 1 copy mappings must preserve parameter names");
                copied[i] = index(targetNames, targetName); destinations.add(targetName);
            }
            int[] added = new int[add.size()]; double[] means = new double[add.size()], sds = new double[add.size()];
            int i = 0;
            for (Map.Entry<String, Object> entry : add.entrySet()) {
                added[i] = index(targetNames, entry.getKey());
                if (!destinations.add(entry.getKey())) throw new IllegalArgumentException("parameter is both copied and added: " + entry.getKey());
                Map<String, Object> proposal = RjJson.object(entry.getValue(), "proposal");
                RjJson.fields(proposal, "proposal", "mean", "sd");
                if (!"normal".equals(proposal.get("proposal"))) throw new IllegalArgumentException("RJ schema 1 requires normal birth proposals");
                means[i] = RjJson.number(proposal.get("mean"), "mean"); sds[i] = RjJson.number(proposal.get("sd"), "sd");
                if (!(sds[i] > 0)) throw new IllegalArgumentException("birth sd must be positive"); i++;
            }
            String label = spaces.get(from).name() + "->" + spaces.get(to).name();
            moves.add(new AddDrop(label + "/add", label + "/drop", from, to, copied, added, means, sds, true));
            moves.add(new AddDrop(label + "/drop", label + "/add", from, to, copied, added, means, sds, false));
            edges[from][to] = edges[to][from] = true;
        }
        boolean[] reachable = new boolean[models.size()]; reachable[0] = true;
        for (int pass = 0; pass < models.size(); pass++) for (int i = 0; i < models.size(); i++)
            if (reachable[i]) for (int j = 0; j < models.size(); j++) reachable[j] |= edges[i][j];
        for (boolean reached : reachable) if (!reached) throw new IllegalArgumentException("RJ transition graph is disconnected");
        List<String> optional = new ArrayList<String>();
        for (String parameter : allParameters) {
            int count = 0;
            for (ReversibleJumpModelSpace space : spaces) if (Arrays.asList(space.parameterNames()).contains(parameter)) count++;
            if (count < spaces.size()) optional.add(parameter);
        }
        candidates = optional.toArray(new String[0]); fingerprint = hex(digest.digest());
    }
    static MessageDigest digest() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException exception) { throw new IllegalStateException(exception); }
    }
    private static void update(MessageDigest digest, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        digest.update(Integer.toString(bytes.length).getBytes(StandardCharsets.US_ASCII)); digest.update((byte) ':'); digest.update(bytes);
    }
    private static String hex(byte[] bytes) {
        StringBuilder result = new StringBuilder();
        for (byte value : bytes) result.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
        return result.toString();
    }
    private static int index(String[] names, String name) {
        for (int i = 0; i < names.length; i++) if (names[i].equals(name)) return i;
        throw new IllegalArgumentException("unknown destination parameter: " + name);
    }
    ReversibleJumpSampler sampler(double scale, double target) {
        double[] weights = new double[moves.size()]; Arrays.fill(weights, 1);
        return new ReversibleJumpSampler(moves.toArray(new ReversibleJumpMove[0]), weights,
                new AdaptiveRjRandomWalkKernel("within-model-random-walk", scale, target));
    }
    ReversibleJumpState initial(int chain) {
        int model = chain % models.size(); return new ReversibleJumpState(model, models.get(model).model().initialState());
    }
    @Override public ReversibleJumpModelSpace modelSpace(long id) {
        if (id < 0 || id >= spaces.size()) throw new IllegalArgumentException("unknown model ID: " + id);
        return spaces.get((int) id);
    }
    @Override public double logJoint(ReversibleJumpState state) {
        modelSpace(state.modelId());
        return models.get((int) state.modelId()).model().logDensity(state.parameters()) + Math.log(priors.get((int) state.modelId()));
    }
    @Override public String[] candidateNames() { return candidates.clone(); }
    @Override public boolean active(long model, int candidate) {
        return Arrays.asList(modelSpace(model).parameterNames()).contains(candidates[candidate]);
    }
    private static final class AddDrop implements ReversibleJumpMove {
        final String name, reverse; final int from, to; final int[] copy, add;
        final double[] mean, sd; final boolean birth;
        AddDrop(String name, String reverse, int from, int to, int[] copy, int[] add, double[] mean, double[] sd, boolean birth) {
            this.name = name; this.reverse = reverse; this.from = from; this.to = to;
            this.copy = copy; this.add = add; this.mean = mean; this.sd = sd; this.birth = birth;
        }
        @Override public String name() { return name; }
        @Override public boolean applicable(ReversibleJumpState state, ReversibleJumpTarget target) { return state.modelId() == (birth ? from : to); }
        @Override public ReversibleJumpProposal propose(ReversibleJumpState state, ReversibleJumpTarget target, RandomEngine random) {
            double[] result = new double[birth ? copy.length + add.length : copy.length];
            for (int i = 0; i < copy.length; i++) if (birth) result[copy[i]] = state.parameter(i); else result[i] = state.parameter(copy[i]);
            double logProposal = 0;
            for (int i = 0; i < add.length; i++) {
                double value = birth ? mean[i] + sd[i] * random.nextGaussian() : state.parameter(add[i]);
                if (birth) result[add[i]] = value;
                logProposal += Normal.density(value, mean[i], sd[i], true);
            }
            if (!Double.isFinite(logProposal)) return ReversibleJumpProposal.invalid("nonfinite Normal birth density");
            return ReversibleJumpProposal.valid(new ReversibleJumpState(birth ? to : from, result), reverse,
                    birth ? logProposal : 0, birth ? 0 : logProposal, 0);
        }
    }
}
