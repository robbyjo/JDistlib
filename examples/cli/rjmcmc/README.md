# Run the Java RJMCMC example through the CLI

These files reproduce the **statistical target** in
`examples/WorkedReversibleJumpSelectionExample.java`: 16 observations, dose,
genotype and proxy-marker candidates, observation SD 0.75, intercept prior
N(0,5), coefficient priors N(0,2), and independent inclusion probability 0.30.
The eight Stan files enumerate every candidate subset. The JSON contains only
model definitions, model prior probabilities and reversible transitions; data
and sampler controls stay on the command line.

## 1. Build the executable JAR

Run from the JDistlib repository root. In PowerShell:

```powershell
.\gradlew.bat :jar :jdistlib-all:jar
$jar = (Get-ChildItem distribution/all/build/libs/jdistlib-all-*-SNAPSHOT.jar | Select-Object -First 1).FullName
```

The all-in-one JAR includes dependencies. You can instead use the core JAR from
`build/libs/` with its generated `lib/` directory alongside it. This addition is
available in the current source checkout; older published JARs do not include it.

## 2. Inspect the model-space JSON

`selection.rj.json` lists all eight models. For example, the intercept-only
subset has probability `0.7^3 = 0.343`; the dose-only subset has probability
`0.3 * 0.7^2 = 0.147`. Probabilities describe **individual subsets**, not subset
sizes, and sum to one. For a smaller two-model analysis, a complete document is:

```json
{
  "schema": "jdistlib.rj/1",
  "models": {
    "intercept": {"script": "intercept.stan", "prior_probability": 0.7},
    "dose": {"script": "dose.stan", "prior_probability": 0.3}
  },
  "transitions": [
    {
      "type": "add_drop",
      "from": "intercept",
      "to": "dose",
      "copy": {"alpha": "alpha"},
      "add": {"dose": {"proposal": "normal", "mean": 0, "sd": 2}},
      "space": "unconstrained"
    }
  ]
}
```

Script paths are relative to the JSON document's directory. An add move copies
`alpha` and draws `dose` from the declared Normal **proposal**; the paired drop
move removes it and evaluates its reverse proposal density. This proposal is
distinct from the coefficient prior, which lives in `dose.stan`. The adapter
supplies a unit insertion/removal Jacobian. The sampler separately includes
forward/reverse move-selection probabilities, including graph boundaries.

The full example has 12 add/drop pairs (24 directed moves). The Java example
also has direct swap moves. These schedules target the same posterior but need
not have the same mixing or seeded draws. The CLI does not claim identical
trajectories to the Java example.

## 3. Supply data and validate

```powershell
java -jar $jar --run --sampler rjmcmc --model-space examples/cli/rjmcmc/selection.rj.json --data examples/cli/rjmcmc/observations.json --validate
```

Expected: `Validated RJ model space: 8 models, 24 directed moves`.
The JSON data contains `N`, a row-major `X` matrix, and `y`. Data flags are the
same as the NUTS CLI: repeat `--data file.json` for disjoint named variables,
use `--data X=file.csv` for a headerless numeric matrix, select a column with
`--data-column`, or supply literals with `--set`.

The equivalent separate-file bindings are:

```text
--set N=16 --data X=examples/cli/rjmcmc/design.csv --data-column y=examples/cli/rjmcmc/response.csv:response
```

CLI paths resolve from the working directory. Each script receives only its
declared data; a variable unused by every model is rejected. Duplicate names,
missing data, invalid sizes/bounds, unknown JSON fields, incomplete copy maps,
unsupported proposals, and disconnected model graphs fail before sampling.

## 4. Run four chains

```powershell
java -jar $jar --run --sampler rjmcmc --model-space examples/cli/rjmcmc/selection.rj.json --data examples/cli/rjmcmc/observations.json --chains 4 --threads 4 --warmup 1500 --samples 4000 --seed 20260829 --within-scale 0.20 --within-target-accept 0.30 --target-jump-accept 0.25 --output build/rj-cli/results.txt --checkpoint build/rj-cli/state
```

`--samples` means retained draws **per chain**. `--thin 2` doubles the
post-warmup iteration count. Every chain owns its adaptive random-walk kernel;
move weights and within-model scales freeze after warmup. Birth proposal
parameters remain fixed. All directed edges start with equal selection weights;
`--adapt-move-weights false` keeps those weights fixed throughout. Chains start
at default parameter values in successive declared models, wrapping if needed.
Changing `--threads` does not change fresh-run seeded results.

## 5. Read output and diagnostics

`results.txt` is tidy UTF-8 CSV, with one parameter or generated-quantity row
per retained draw and columns:

```text
chain,draw,model_id,model_name,kind,parameter,value,log_joint
```

Model IDs follow JSON declaration order. An absent coefficient has no row;
absence is **not** a sampled zero. Parameters sharing a name must represent
the same scientific quantity and scale. A parameter appearing in some but not
all models receives an inclusion indicator. Parameter summaries are conditional
on presence. The `# diagnostics=` comment contains JSON with model occupancy,
model and inclusion ESS/R-hat/MCSE, move attempts/acceptance, invalid proposals,
model changes, and conditional parameter summaries. Unvisited models are listed
in the catalog and warned about; the diagnostics API lists visited models only.

In PowerShell, extract the embedded report:

```powershell
$line = Get-Content build/rj-cli/results.txt | Where-Object { $_.StartsWith('# diagnostics=') }
$report = $line.Substring('# diagnostics='.Length) | ConvertFrom-Json
$report | ConvertTo-Json -Depth 20
```

Inspect warnings and mixing before using the probabilities. A single chain
cannot estimate R-hat, and zero observed occupancy does not prove zero posterior
probability. The file includes a fingerprint of the manifest, scripts and data.
Ordinary generated quantities are evaluated per retained state on a cloned
post-run random stream; they do not alter RJ transitions. Their random values
can differ between an uninterrupted export and split/resumed exports even when
the sampler states agree exactly.

## 6. Resume from checkpoints

The run writes `state.chain-1.rjckpt` through `state.chain-4.rjckpt`. Continue with
the same model/data, chain count, thinning and kernel/adaptation controls:

```powershell
java -jar $jar --run --sampler rjmcmc --model-space examples/cli/rjmcmc/selection.rj.json --data examples/cli/rjmcmc/observations.json --chains 4 --samples 1000 --within-scale 0.20 --within-target-accept 0.30 --target-jump-accept 0.25 --resume build/rj-cli/state --output build/rj-cli/continued.txt --checkpoint build/rj-cli/continued-state
```

Omit `--seed`: RNG states are restored. New warmup is zero. Resume currently
executes chains sequentially; each checkpoint binds the model/data fingerprint,
kernel controls and chain index. Changing model source/data or these controls
rejects the restore. Output contains only the new draws, numbered from one in
the new segment. Continuation of sampled states is exact for the same runtime;
this is not a cross-version portability guarantee. Keep the original model,
data and software available with long-lived checkpoints.

Compilation, sampling and output generation finish before replacing existing
outputs. Files are staged then replaced individually; an I/O failure during
publication can leave a partially updated set, so prefer a fresh checkpoint
prefix when preserving an earlier run matters. Exit codes are 0 for success,
2 for invalid arguments/model-space/data, and 1 for execution/I/O failure.

## Supported scope and normalization contract

Schema 1 supports a **finite explicitly enumerated model space**, at least one
unbounded scalar `real` parameter per model, name-preserving copy mappings, and independent Normal
proposals for one or more added coordinates. No bounds, vectors, parameter
offsets/multipliers, renaming, custom Jacobians, split/merge moves, or direct
swap moves are accepted in this version. Large sparse candidate spaces still
use `SparseSubsetRjSampler` through Java; do not enumerate exponentially many
models to reproduce that API. Within-model CLI updates are CPU adaptive random
walks, not NUTS. NUTS options such as `--max-depth` fail in RJ mode.

Scripts must define a complete joint density with proper parameter priors and
all model-dependent normalization terms. The manifest adds the model prior;
do **not** also put that prior in each script. Use explicit normalized calls:

```stan
target += normal_lpdf(alpha | 0, 5);
target += normal_lpdf(dose | 0, 2);
```

The adapter rejects `~` and `_lupdf`/`_lupmf` to make this requirement visible.
It cannot prove that arbitrary `target +=` expressions or user functions are
normalized, or that a prior is proper. Passing validation is not certification
of those mathematical properties. The worked Gaussian family is independently
tested against its normalized Java log joint and exact integrated model
evidence. Test fixtures also verify paired-move densities, boundary selection
probabilities, deterministic parallel execution and checkpoint continuation.

Use `--run --sampler rjmcmc --help` for controls. Existing NUTS `--run --input`
and three-positional-argument Java source generation remain available.
